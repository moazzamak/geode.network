from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp

from src.subspace_primitive import SubspacePrimitive
from src.subspace_primitive import fit_subspace_primitive


STRATA = ("near_surface", "deep_interior", "exterior")


def normalized_shell(fields: np.ndarray) -> np.ndarray:
    values = np.asarray(fields, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("fields must be finite")
    return np.abs(values)


def metric_corrected_field(
    primitive: SubspacePrimitive,
    points: np.ndarray,
    *,
    eta: float,
) -> np.ndarray:
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be finite and positive")
    values = np.asarray(points, dtype=np.float64)
    quadratic = primitive.quadratic_form(values)
    radial = np.sqrt(np.maximum(quadratic, 0.0))
    gradient = primitive.quadratic_gradient(values)
    denominators = 2.0 * np.maximum(radial, np.finfo(np.float64).tiny)
    gradient_norm = np.linalg.norm(gradient / denominators[:, None], axis=1)
    gradient_norm = np.where(radial > 0.0, gradient_norm, 0.0)
    corrected = (radial - 1.0) / (gradient_norm + eta)
    if not np.all(np.isfinite(corrected)):
        raise FloatingPointError("metric-corrected fields must remain finite")
    return corrected


def metric_field_matrix(
    primitives: Sequence[SubspacePrimitive],
    points: np.ndarray,
    *,
    eta: float,
) -> np.ndarray:
    if not primitives:
        raise ValueError("at least one primitive is required")
    return np.column_stack(
        [metric_corrected_field(item, points, eta=eta) for item in primitives]
    )


def normalized_softmin(
    fields: np.ndarray,
    labels: Sequence[int],
    classes: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    values = np.asarray(fields, dtype=np.float64)
    component_weights = np.asarray(weights, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.int64)
    component_labels = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != len(component_labels):
        raise ValueError("fields and labels have incompatible shapes")
    if component_weights.shape != (len(component_labels),):
        raise ValueError("weights have the wrong shape")
    result = np.empty((len(values), len(class_values)), dtype=np.float64)
    for column, class_label in enumerate(class_values):
        indices = np.flatnonzero(component_labels == class_label)
        if not len(indices):
            raise ValueError("every class requires a component")
        class_weights = component_weights[indices]
        if np.any(class_weights < 0.0) or not np.isclose(
            class_weights.sum(), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("weights must form a simplex within each class")
        positive = class_weights > 0.0
        result[:, column] = -logsumexp(
            np.log(class_weights[positive])[None, :]
            - values[:, indices[positive]],
            axis=1,
        )
    return result


def class_signed_depths(
    fields: np.ndarray,
    labels: Sequence[int],
    classes: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return normalized_softmin(fields, labels, classes, weights)


def class_minimum_fields(
    fields: np.ndarray,
    labels: Sequence[int],
    classes: np.ndarray,
) -> np.ndarray:
    values = np.asarray(fields, dtype=np.float64)
    component_labels = np.asarray(labels, dtype=np.int64)
    class_values = np.asarray(classes, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != len(component_labels):
        raise ValueError("fields and labels have incompatible shapes")
    result = np.empty((len(values), len(class_values)), dtype=np.float64)
    for column, class_label in enumerate(class_values):
        indices = np.flatnonzero(component_labels == class_label)
        if not len(indices):
            raise ValueError("every class requires a component")
        result[:, column] = np.min(values[:, indices], axis=1)
    return result


def deterministic_equal_mass_bands(
    own_signed_depths: np.ndarray,
    *,
    fraction: float,
) -> dict[str, float]:
    values = np.asarray(own_signed_depths, dtype=np.float64)
    interior = np.sort(values[np.isfinite(values) & (values < 0.0)], kind="stable")
    if not 0.0 < fraction <= 0.5:
        raise ValueError("fraction must be in (0, 0.5]")
    count = int(np.floor(len(interior) * fraction))
    if count < 1:
        raise ValueError("insufficient negative interior for equal-mass bands")
    return {
        "mass_count": count,
        "deep_upper": float(interior[count - 1]),
        "near_lower": float(interior[-count]),
    }


def assign_strata(signed_depths: np.ndarray, bands: Mapping[str, float]) -> np.ndarray:
    values = np.asarray(signed_depths, dtype=np.float64)
    strata = np.full(len(values), "middle_interior", dtype=object)
    strata[values >= 0.0] = "exterior"
    strata[values <= float(bands["deep_upper"])] = "deep_interior"
    near = (values < 0.0) & (values >= float(bands["near_lower"]))
    strata[near] = "near_surface"
    return strata


def validate_disjoint_partitions(partitions: Mapping[str, Sequence[str]]) -> None:
    observed: set[str] = set()
    for name, identifiers in partitions.items():
        current = set(identifiers)
        if len(current) != len(identifiers):
            raise ValueError(f"partition {name} contains duplicate IDs")
        overlap = observed & current
        if overlap:
            raise ValueError(f"partition overlap detected: {sorted(overlap)[:3]}")
        observed.update(current)


def random_orientation(
    primitive: SubspacePrimitive,
    *,
    seed: int,
) -> SubspacePrimitive:
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(primitive.dimension, primitive.rank)))
    return SubspacePrimitive(
        center=primitive.center,
        basis=basis[:, : primitive.rank],
        tangent_variances=primitive.tangent_variances,
        residual_variance=primitive.residual_variance,
        class_label=primitive.class_label,
        anchor_index=primitive.anchor_index,
        support_size=primitive.support_size,
    )


def permuted_labels(labels: Sequence[int], *, seed: int) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    unique = np.unique(values)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    mapping = dict(zip(unique.tolist(), shuffled.tolist(), strict=True))
    return np.asarray([mapping[int(value)] for value in values], dtype=np.int64)


def replay_digest(payload: Any) -> str:
    from experiments.common.experiment_manifest import canonical_json

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BoundedTubePrimitive:
    center: np.ndarray
    basis: np.ndarray
    residual_variance: float
    tangent_extents: np.ndarray
    tangent_scales: np.ndarray
    penalty_weight: float
    class_label: int

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        basis = np.asarray(self.basis, dtype=np.float64)
        extents = np.asarray(self.tangent_extents, dtype=np.float64)
        scales = np.asarray(self.tangent_scales, dtype=np.float64)
        if center.ndim != 1 or not np.all(np.isfinite(center)):
            raise ValueError("center must be a finite vector")
        if (
            basis.ndim != 2
            or basis.shape[0] != len(center)
            or basis.shape[1] < 1
            or not np.all(np.isfinite(basis))
        ):
            raise ValueError("basis must be a finite rank-positive matrix")
        if not np.allclose(
            basis.T @ basis, np.eye(basis.shape[1]), rtol=0.0, atol=1e-8
        ):
            raise ValueError("basis columns must be orthonormal")
        if extents.shape != (basis.shape[1],) or np.any(extents <= 0.0):
            raise ValueError("tangent extents must be positive and match rank")
        if scales.shape != extents.shape or np.any(scales <= 0.0):
            raise ValueError("tangent scales must be positive and match rank")
        if (
            not np.isfinite(self.residual_variance)
            or self.residual_variance <= 0.0
        ):
            raise ValueError("residual_variance must be finite and positive")
        if not np.isfinite(self.penalty_weight) or self.penalty_weight <= 0.0:
            raise ValueError("penalty_weight must be finite and positive")
        if isinstance(self.class_label, bool) or self.class_label < 0:
            raise ValueError("class_label must be nonnegative")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "tangent_extents", extents)
        object.__setattr__(self, "tangent_scales", scales)

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    @property
    def parameter_count(self) -> int:
        return int(
            self.center.size
            + self.basis.size
            + self.tangent_extents.size
            + self.tangent_scales.size
            + 2
        )

    def coordinates(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.center):
            raise ValueError("points have the wrong shape")
        deltas = values - self.center
        tangent = deltas @ self.basis
        residual = deltas - tangent @ self.basis.T
        residual_squared = np.sum(residual * residual, axis=1)
        return tangent, residual_squared

    def unbounded_score(self, points: np.ndarray) -> np.ndarray:
        _, residual_squared = self.coordinates(points)
        return residual_squared / self.residual_variance

    def score(self, points: np.ndarray) -> np.ndarray:
        tangent, residual_squared = self.coordinates(points)
        outside = np.maximum(
            (np.abs(tangent) - self.tangent_extents[None, :])
            / self.tangent_scales[None, :],
            0.0,
        )
        return (
            residual_squared / self.residual_variance
            + self.penalty_weight * np.sum(outside * outside, axis=1)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "center": self.center.tolist(),
            "basis": self.basis.tolist(),
            "residual_variance": float(self.residual_variance),
            "tangent_extents": self.tangent_extents.tolist(),
            "tangent_scales": self.tangent_scales.tolist(),
            "penalty_weight": float(self.penalty_weight),
            "class_label": self.class_label,
        }


def fit_bounded_tube(
    geometry_points: np.ndarray,
    calibration_points: np.ndarray,
    *,
    rank: int,
    extent_quantile: float,
    scale_quantile: float,
    penalty_weight: float,
    class_label: int,
) -> BoundedTubePrimitive:
    if not 0.0 < scale_quantile < extent_quantile < 1.0:
        raise ValueError("tube quantiles must satisfy 0 < scale < extent < 1")
    primitive = fit_subspace_primitive(
        geometry_points, rank, class_label=class_label
    )
    calibration = np.asarray(calibration_points, dtype=np.float64)
    if (
        calibration.ndim != 2
        or calibration.shape[1] != primitive.dimension
        or not len(calibration)
    ):
        raise ValueError("calibration_points have the wrong shape")
    tangent = np.abs((calibration - primitive.center) @ primitive.basis)
    floor = np.sqrt(np.finfo(np.float64).eps)
    extents = np.maximum(
        np.quantile(tangent, extent_quantile, axis=0, method="higher"), floor
    )
    scales = np.maximum(
        np.quantile(tangent, scale_quantile, axis=0, method="higher"), floor
    )
    return BoundedTubePrimitive(
        center=primitive.center,
        basis=primitive.basis,
        residual_variance=primitive.residual_variance,
        tangent_extents=extents,
        tangent_scales=scales,
        penalty_weight=penalty_weight,
        class_label=class_label,
    )
