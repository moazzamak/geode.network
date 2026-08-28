"""Positive-definite precision metrics for fixed-space GEODE components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


LOCAL_FAMILIES = (
    "spherical",
    "diagonal",
    "full",
    "diagonal_low_rank",
)
SHARED_FAMILIES = (
    "shared_full_diagonal",
    "shared_low_rank_diagonal",
)
METRIC_FAMILIES = LOCAL_FAMILIES + SHARED_FAMILIES


def _as_matrix(value: np.ndarray | None, rows: int) -> np.ndarray:
    if value is None:
        return np.zeros((rows, 0), dtype=np.float64)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != rows:
        raise ValueError(f"Expected a ({rows}, rank) matrix.")
    return matrix


def _as_vector(value: np.ndarray | None, dimension: int) -> np.ndarray:
    if value is None:
        return np.zeros(dimension, dtype=np.float64)
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (dimension,):
        raise ValueError(f"Expected a ({dimension},) vector.")
    return vector


@dataclass(frozen=True)
class PrecisionMetric:
    """Factorized precision ``P = diag(d) + U U^T + L L^T``.

    Local diagonal and low-rank factors are stored separately from an optional
    shared dense factor so inference can score low-rank families without
    constructing a dense matrix.
    """

    family: str
    dimension: int
    diagonal: np.ndarray | None = None
    factors: np.ndarray | None = None
    dense_factor: np.ndarray | None = None
    shared_parameter_count: int = 0
    eigenvalue_floor: float = 1e-6

    def __post_init__(self) -> None:
        if self.family not in METRIC_FAMILIES:
            raise ValueError(f"Unsupported metric family {self.family!r}.")
        if self.dimension < 1:
            raise ValueError("dimension must be positive.")
        if not np.isfinite(self.eigenvalue_floor) or self.eigenvalue_floor <= 0.0:
            raise ValueError("eigenvalue_floor must be finite and positive.")
        diagonal = _as_vector(self.diagonal, self.dimension)
        factors = _as_matrix(self.factors, self.dimension)
        dense_factor = _as_matrix(self.dense_factor, self.dimension)
        if np.any(~np.isfinite(diagonal)) or np.any(diagonal < 0.0):
            raise ValueError("diagonal entries must be finite and nonnegative.")
        if np.any(~np.isfinite(factors)) or np.any(~np.isfinite(dense_factor)):
            raise ValueError("metric factors must be finite.")
        if self.shared_parameter_count < 0:
            raise ValueError("shared_parameter_count cannot be negative.")
        object.__setattr__(self, "diagonal", diagonal)
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "dense_factor", dense_factor)
        eigenvalues = np.linalg.eigvalsh(self.dense_precision())
        if eigenvalues[0] < self.eigenvalue_floor * (1.0 - 1e-8):
            raise ValueError(
                "metric is not positive definite at the declared eigenvalue floor."
            )

    @property
    def rank(self) -> int:
        return int(self.factors.shape[1])

    @property
    def local_parameter_count(self) -> int:
        if self.family == "spherical":
            return 1
        if self.family == "diagonal":
            return self.dimension
        if self.family == "full":
            return self.dimension * (self.dimension + 1) // 2
        if self.family == "diagonal_low_rank":
            return self.dimension + int(self.factors.size)
        return self.dimension

    @property
    def total_parameter_count(self) -> int:
        return self.local_parameter_count + int(self.shared_parameter_count)

    @property
    def array_bytes(self) -> int:
        return int(
            self.diagonal.nbytes + self.factors.nbytes + self.dense_factor.nbytes
        )

    def dense_precision(self) -> np.ndarray:
        precision = np.diag(self.diagonal)
        if self.factors.shape[1]:
            precision = precision + self.factors @ self.factors.T
        if self.dense_factor.shape[1]:
            precision = precision + self.dense_factor @ self.dense_factor.T
        return (precision + precision.T) * 0.5

    def quadratic_form(self, deltas: np.ndarray) -> np.ndarray:
        values = np.asarray(deltas, dtype=np.float64)
        if values.shape[-1] != self.dimension:
            raise ValueError("deltas have the wrong final dimension.")
        result = np.sum(values * values * self.diagonal, axis=-1)
        if self.factors.shape[1]:
            result = result + np.sum(np.square(values @ self.factors), axis=-1)
        if self.dense_factor.shape[1]:
            result = result + np.sum(
                np.square(values @ self.dense_factor), axis=-1
            )
        return result

    def gradient(self, deltas: np.ndarray) -> np.ndarray:
        values = np.asarray(deltas, dtype=np.float64)
        if values.shape[-1] != self.dimension:
            raise ValueError("deltas have the wrong final dimension.")
        return 2.0 * (values @ self.dense_precision())

    def log_determinant(self) -> float:
        sign, value = np.linalg.slogdet(self.dense_precision())
        if sign <= 0.0:
            raise ValueError("precision determinant must be positive.")
        return float(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "family": self.family,
            "dimension": self.dimension,
            "diagonal": self.diagonal.tolist(),
            "factors": self.factors.tolist(),
            "dense_factor": self.dense_factor.tolist(),
            "shared_parameter_count": self.shared_parameter_count,
            "eigenvalue_floor": self.eigenvalue_floor,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrecisionMetric":
        required = {
            "schema_version",
            "family",
            "dimension",
            "diagonal",
            "factors",
            "dense_factor",
            "shared_parameter_count",
            "eigenvalue_floor",
        }
        if set(payload) != required or payload.get("schema_version") != 1:
            raise ValueError("Unsupported precision-metric schema.")
        dimension = int(payload["dimension"])
        factors = np.asarray(payload["factors"], dtype=np.float64)
        dense_factor = np.asarray(payload["dense_factor"], dtype=np.float64)
        if factors.size == 0:
            factors = np.zeros((dimension, 0), dtype=np.float64)
        if dense_factor.size == 0:
            dense_factor = np.zeros((dimension, 0), dtype=np.float64)
        return cls(
            family=str(payload["family"]),
            dimension=dimension,
            diagonal=np.asarray(payload["diagonal"], dtype=np.float64),
            factors=factors,
            dense_factor=dense_factor,
            shared_parameter_count=int(payload["shared_parameter_count"]),
            eigenvalue_floor=float(payload["eigenvalue_floor"]),
        )


@dataclass(frozen=True)
class MetricFit:
    metric: PrecisionMetric
    center: np.ndarray
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        if center.shape != (self.metric.dimension,) or np.any(~np.isfinite(center)):
            raise ValueError("center must be a finite vector matching the metric.")
        object.__setattr__(self, "center", center)


def _regularized_covariance(
    points: np.ndarray,
    eigenvalue_floor: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("points must contain at least two observations.")
    if np.any(~np.isfinite(values)):
        raise ValueError("points must be finite.")
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(eigenvalues[-1]), 1.0)
    floor = eigenvalue_floor * scale
    clipped = np.maximum(eigenvalues, floor)
    warnings: list[str] = []
    if np.any(eigenvalues < floor):
        warnings.append("covariance_eigenvalues_floored")
    regularized = (eigenvectors * clipped) @ eigenvectors.T
    return (regularized + regularized.T) * 0.5, tuple(warnings)


def _precision_from_covariance(
    covariance: np.ndarray,
    eigenvalue_floor: float,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    precision_eigenvalues = np.maximum(1.0 / eigenvalues, eigenvalue_floor)
    precision = (eigenvectors * precision_eigenvalues) @ eigenvectors.T
    return (precision + precision.T) * 0.5


def _low_rank_precision(
    precision: np.ndarray,
    rank: int,
    eigenvalue_floor: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    dimension = precision.shape[0]
    if rank < 0:
        raise ValueError("rank cannot be negative.")
    selected_rank = min(rank, dimension)
    warnings: list[str] = []
    if selected_rank != rank:
        warnings.append("rank_clipped_to_dimension")
    target_diagonal = np.diag(precision)
    floor_diagonal = np.full(dimension, eigenvalue_floor, dtype=np.float64)
    diagonal_span = np.maximum(target_diagonal - floor_diagonal, 0.0)
    low = 0.0
    high = 1.0
    for _ in range(60):
        fraction = (low + high) * 0.5
        residual = precision - np.diag(
            floor_diagonal + fraction * diagonal_span
        )
        if np.linalg.eigvalsh(residual)[0] >= -1e-12:
            low = fraction
        else:
            high = fraction
    diagonal = floor_diagonal + low * diagonal_span
    residual_matrix = (precision - np.diag(diagonal))
    residual_matrix = (residual_matrix + residual_matrix.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(residual_matrix)
    residual = np.maximum(eigenvalues, 0.0)
    order = np.argsort(residual)[::-1][:selected_rank]
    retained = residual[order]
    nonzero = retained > np.finfo(np.float64).eps
    factors = eigenvectors[:, order[nonzero]] * np.sqrt(retained[nonzero])
    if selected_rank == 0:
        factors = np.zeros((dimension, 0), dtype=np.float64)
    return (
        diagonal,
        factors,
        tuple(warnings),
    )


def fit_precision_metric(
    points: np.ndarray,
    family: str,
    *,
    rank: int = 4,
    eigenvalue_floor: float = 1e-6,
) -> MetricFit:
    if family not in LOCAL_FAMILIES:
        raise ValueError(f"{family!r} is not a local metric family.")
    values = np.asarray(points, dtype=np.float64)
    covariance, covariance_warnings = _regularized_covariance(
        values, eigenvalue_floor
    )
    center = values.mean(axis=0)
    dimension = values.shape[1]
    precision = _precision_from_covariance(covariance, eigenvalue_floor)
    warnings = list(covariance_warnings)

    if family == "spherical":
        variance = max(float(np.trace(covariance) / dimension), eigenvalue_floor)
        metric = PrecisionMetric(
            family=family,
            dimension=dimension,
            diagonal=np.full(
                dimension, max(1.0 / variance, eigenvalue_floor)
            ),
            eigenvalue_floor=eigenvalue_floor,
        )
    elif family == "diagonal":
        variances = np.maximum(np.diag(covariance), eigenvalue_floor)
        metric = PrecisionMetric(
            family=family,
            dimension=dimension,
            diagonal=np.maximum(1.0 / variances, eigenvalue_floor),
            eigenvalue_floor=eigenvalue_floor,
        )
    elif family == "full":
        metric = PrecisionMetric(
            family=family,
            dimension=dimension,
            dense_factor=np.linalg.cholesky(precision),
            eigenvalue_floor=eigenvalue_floor,
        )
    else:
        diagonal, factors, rank_warnings = _low_rank_precision(
            precision, rank, eigenvalue_floor
        )
        warnings.extend(rank_warnings)
        metric = PrecisionMetric(
            family=family,
            dimension=dimension,
            diagonal=diagonal,
            factors=factors,
            eigenvalue_floor=eigenvalue_floor,
        )
    return MetricFit(metric=metric, center=center, warnings=tuple(warnings))


def fit_class_precision_metrics(
    class_points: Mapping[int | str, np.ndarray],
    family: str,
    *,
    rank: int = 4,
    eigenvalue_floor: float = 1e-6,
) -> dict[int | str, MetricFit]:
    if not class_points:
        raise ValueError("class_points cannot be empty.")
    if family in LOCAL_FAMILIES:
        return {
            class_id: fit_precision_metric(
                points,
                family,
                rank=rank,
                eigenvalue_floor=eigenvalue_floor,
            )
            for class_id, points in class_points.items()
        }
    if family not in SHARED_FAMILIES:
        raise ValueError(f"Unsupported metric family {family!r}.")

    ordered_ids = sorted(class_points, key=str)
    arrays = [np.asarray(class_points[class_id], dtype=np.float64) for class_id in ordered_ids]
    dimension = arrays[0].shape[1]
    if any(values.ndim != 2 or values.shape[1] != dimension for values in arrays):
        raise ValueError("All classes must use the same feature dimension.")
    pooled = np.vstack(
        [values - values.mean(axis=0, keepdims=True) for values in arrays]
    )

    if family == "shared_full_diagonal":
        shared_fit = fit_precision_metric(
            pooled,
            "full",
            rank=rank,
            eigenvalue_floor=eigenvalue_floor,
        )
        shared = shared_fit.metric
        shared_precision = shared.dense_precision()
        shared_count = shared.local_parameter_count
        result: dict[int | str, MetricFit] = {}
        for class_id, values in zip(ordered_ids, arrays):
            local = fit_precision_metric(
                values,
                "diagonal",
                eigenvalue_floor=eigenvalue_floor,
            )
            correction = np.maximum(
                local.metric.diagonal - np.diag(shared_precision), 0.0
            )
            result[class_id] = MetricFit(
                metric=PrecisionMetric(
                    family=family,
                    dimension=dimension,
                    diagonal=correction,
                    dense_factor=shared.dense_factor,
                    shared_parameter_count=shared_count,
                    eigenvalue_floor=eigenvalue_floor,
                ),
                center=local.center,
                warnings=tuple(sorted(set(shared_fit.warnings + local.warnings))),
            )
        return result

    shared_fit = fit_precision_metric(
        pooled,
        "diagonal_low_rank",
        rank=rank,
        eigenvalue_floor=eigenvalue_floor,
    )
    shared = shared_fit.metric
    shared_count = shared.factors.size
    result = {}
    for class_id, values in zip(ordered_ids, arrays):
        local = fit_precision_metric(
            values,
            "diagonal",
            eigenvalue_floor=eigenvalue_floor,
        )
        result[class_id] = MetricFit(
            metric=PrecisionMetric(
                family=family,
                dimension=dimension,
                diagonal=local.metric.diagonal,
                factors=shared.factors,
                shared_parameter_count=shared_count,
                eigenvalue_floor=eigenvalue_floor,
            ),
            center=local.center,
            warnings=tuple(sorted(set(shared_fit.warnings + local.warnings))),
        )
    return result
