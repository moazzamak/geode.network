from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.directional_primitive import l2_normalize
from src.subspace_primitive import deterministic_basis_signs


def sphere_log_map(mean_direction: np.ndarray, features: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean_direction, dtype=np.float64)
    if (
        mean.ndim != 1
        or not np.all(np.isfinite(mean))
        or not np.isclose(np.linalg.norm(mean), 1.0, rtol=0.0, atol=1e-10)
    ):
        raise ValueError("mean_direction must be a finite unit vector.")
    unit = l2_normalize(features)
    if unit.shape[1] != len(mean):
        raise ValueError("features have the wrong ambient dimension.")
    cosine = np.clip(unit @ mean, -1.0, 1.0)
    tangent = unit - cosine[:, None] * mean
    tangent_norm = np.linalg.norm(tangent, axis=1)
    angles = np.arctan2(tangent_norm, cosine)
    mapped = np.zeros_like(unit)
    regular = tangent_norm > 1e-12
    mapped[regular] = (
        angles[regular, None]
        * tangent[regular]
        / tangent_norm[regular, None]
    )
    antipodes = ~regular & (cosine < 0.0)
    if np.any(antipodes):
        pivot = int(np.argmin(np.abs(mean)))
        direction = np.zeros_like(mean)
        direction[pivot] = 1.0
        direction -= float(direction @ mean) * mean
        direction /= np.linalg.norm(direction)
        mapped[antipodes] = np.pi * direction
    return mapped


@dataclass(frozen=True)
class TangentCapPrimitive:
    mean_direction: np.ndarray
    basis: np.ndarray
    tangent_variances: np.ndarray
    residual_variance: float
    angular_radius: float
    class_label: int | None = None
    anchor_index: int | None = None
    support_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean_direction, dtype=np.float64)
        basis = np.asarray(self.basis, dtype=np.float64)
        variances = np.asarray(self.tangent_variances, dtype=np.float64)
        if (
            mean.ndim != 1
            or not np.all(np.isfinite(mean))
            or not np.isclose(np.linalg.norm(mean), 1.0, rtol=0.0, atol=1e-10)
        ):
            raise ValueError("mean_direction must be a finite unit vector.")
        if (
            basis.ndim != 2
            or basis.shape[0] != len(mean)
            or basis.shape[1] < 1
            or not np.all(np.isfinite(basis))
        ):
            raise ValueError("basis must have shape (dimension, positive rank).")
        if not np.allclose(
            basis.T @ basis, np.eye(basis.shape[1]), rtol=0.0, atol=1e-8
        ):
            raise ValueError("basis columns must be orthonormal.")
        if not np.allclose(mean @ basis, 0.0, rtol=0.0, atol=1e-8):
            raise ValueError("basis columns must lie in the mean tangent plane.")
        if variances.shape != (basis.shape[1],) or np.any(
            ~np.isfinite(variances) | (variances <= 0.0)
        ):
            raise ValueError("tangent_variances must be finite and positive.")
        if not np.isfinite(self.residual_variance) or self.residual_variance <= 0.0:
            raise ValueError("residual_variance must be finite and positive.")
        if (
            not np.isfinite(self.angular_radius)
            or self.angular_radius <= 0.0
            or self.angular_radius > np.pi
        ):
            raise ValueError("angular_radius must lie in (0, pi].")
        if self.anchor_index is not None and self.anchor_index < 0:
            raise ValueError("anchor_index cannot be negative.")
        support = tuple(int(index) for index in self.support_indices)
        if (
            len(support) < basis.shape[1] + 2
            or any(index < 0 for index in support)
            or len(set(support)) != len(support)
        ):
            raise ValueError("support_indices must be unique and satisfy r+2.")
        object.__setattr__(self, "mean_direction", mean)
        object.__setattr__(self, "basis", deterministic_basis_signs(basis))
        object.__setattr__(self, "tangent_variances", variances)
        object.__setattr__(self, "support_indices", support)

    @property
    def dimension(self) -> int:
        return len(self.mean_direction)

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    @property
    def parameter_count(self) -> int:
        return self.dimension + self.dimension * self.rank + self.rank + 2

    @property
    def array_bytes(self) -> int:
        return int(
            self.mean_direction.nbytes
            + self.basis.nbytes
            + self.tangent_variances.nbytes
            + 2 * np.dtype(np.float64).itemsize
        )

    def _coordinates(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mapped = sphere_log_map(self.mean_direction, features)
        tangent = mapped @ self.basis
        residual_squared = np.maximum(
            np.sum(mapped * mapped, axis=1)
            - np.sum(tangent * tangent, axis=1),
            0.0,
        )
        return mapped, tangent, residual_squared

    def angles(self, features: np.ndarray) -> np.ndarray:
        return np.linalg.norm(sphere_log_map(self.mean_direction, features), axis=1)

    def quadratic_form(self, features: np.ndarray) -> np.ndarray:
        _, tangent, residual_squared = self._coordinates(features)
        return np.sum(
            tangent * tangent / self.tangent_variances[None, :], axis=1
        ) + residual_squared / self.residual_variance

    def radial_field(self, features: np.ndarray) -> np.ndarray:
        covariance_field = np.sqrt(
            np.maximum(self.quadratic_form(features), 0.0)
        ) - 1.0
        angular_field = self.angles(features) / self.angular_radius - 1.0
        return np.maximum(covariance_field, angular_field)

    def log_likelihood(self, features: np.ndarray) -> np.ndarray:
        tangent_dimension = self.dimension - 1
        residual_dimensions = tangent_dimension - self.rank
        log_determinant = float(
            np.log(self.tangent_variances).sum()
            + residual_dimensions * np.log(self.residual_variance)
        )
        return -0.5 * (
            self.quadratic_form(features)
            + log_determinant
            + tangent_dimension * np.log(2.0 * np.pi)
        )

    def radial_gradient(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        unit = l2_normalize(values)
        norms = np.linalg.norm(values, axis=1)
        mapped, coordinates, _ = self._coordinates(values)
        cosine = np.clip(unit @ self.mean_direction, -1.0, 1.0)
        sine = np.linalg.norm(
            unit - cosine[:, None] * self.mean_direction, axis=1
        )
        covariance_gradient = np.zeros_like(values)
        angular_gradient = np.zeros_like(values)
        quadratic = self.quadratic_form(values)
        regular = sine > 1e-10
        for row in np.flatnonzero(regular):
            tangent = unit[row] - cosine[row] * self.mean_direction
            theta = float(np.arctan2(sine[row], cosine[row]))
            scale = theta / sine[row]
            scale_derivative = (
                -1.0 / (sine[row] ** 2)
                + theta * cosine[row] / (sine[row] ** 3)
            )
            projected = coordinates[row] @ self.basis.T
            metric_mapped = (
                (coordinates[row] / self.tangent_variances) @ self.basis.T
                + (mapped[row] - projected) / self.residual_variance
            )
            jacobian = scale * (
                np.eye(self.dimension)
                - np.outer(self.mean_direction, self.mean_direction)
            ) + scale_derivative * np.outer(tangent, self.mean_direction)
            input_projection = (
                np.eye(self.dimension) - np.outer(unit[row], unit[row])
            ) / norms[row]
            if quadratic[row] > 1e-20:
                covariance_gradient[row] = (
                    input_projection
                    @ jacobian.T
                    @ metric_mapped
                    / np.sqrt(quadratic[row])
                )
            angular_gradient[row] = (
                -input_projection @ self.mean_direction
                / (sine[row] * self.angular_radius)
            )
        covariance_field = np.sqrt(np.maximum(quadratic, 0.0)) - 1.0
        angular_field = self.angles(values) / self.angular_radius - 1.0
        use_angular = angular_field > covariance_field
        covariance_gradient[use_angular] = angular_gradient[use_angular]
        return covariance_gradient

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mean_direction": self.mean_direction.tolist(),
            "basis": self.basis.tolist(),
            "tangent_variances": self.tangent_variances.tolist(),
            "residual_variance": float(self.residual_variance),
            "angular_radius": float(self.angular_radius),
            "class_label": self.class_label,
            "anchor_index": self.anchor_index,
            "support_indices": list(self.support_indices),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TangentCapPrimitive":
        required = {
            "schema_version",
            "mean_direction",
            "basis",
            "tangent_variances",
            "residual_variance",
            "angular_radius",
            "class_label",
            "anchor_index",
            "support_indices",
        }
        if set(payload) != required or payload.get("schema_version") != 1:
            raise ValueError("Unsupported tangent-cap schema.")
        return cls(
            mean_direction=np.asarray(payload["mean_direction"], dtype=np.float64),
            basis=np.asarray(payload["basis"], dtype=np.float64),
            tangent_variances=np.asarray(
                payload["tangent_variances"], dtype=np.float64
            ),
            residual_variance=float(payload["residual_variance"]),
            angular_radius=float(payload["angular_radius"]),
            class_label=(
                None
                if payload["class_label"] is None
                else int(payload["class_label"])
            ),
            anchor_index=(
                None
                if payload["anchor_index"] is None
                else int(payload["anchor_index"])
            ),
            support_indices=tuple(int(value) for value in payload["support_indices"]),
        )


def fit_tangent_cap(
    features: np.ndarray,
    rank: int,
    *,
    variance_floor_fraction: float = 1e-3,
    residual_floor_fraction: float = 0.05,
    class_label: int | None = None,
    anchor_index: int | None = None,
    support_indices: tuple[int, ...] = (),
) -> TangentCapPrimitive:
    unit = l2_normalize(features)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer.")
    if len(unit) < rank + 2:
        raise ValueError(
            f"Rank {rank} requires r+2={rank + 2} support points; got {len(unit)}."
        )
    if rank >= unit.shape[1] - 1:
        raise ValueError("rank must be smaller than the tangent dimension.")
    if variance_floor_fraction <= 0.0 or residual_floor_fraction <= 0.0:
        raise ValueError("variance floor fractions must be positive.")
    if support_indices and len(support_indices) != len(unit):
        raise ValueError("support_indices must identify every support point.")

    mean = unit.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean))
    if mean_norm <= np.finfo(np.float64).tiny:
        raise ValueError("Support directions have a degenerate mean.")
    mean /= mean_norm
    mapped = sphere_log_map(mean, unit)
    _, singular_values, right_vectors = np.linalg.svd(mapped, full_matrices=False)
    raw_basis = right_vectors[:rank].T
    raw_basis -= np.outer(mean, mean @ raw_basis)
    basis, _ = np.linalg.qr(raw_basis, mode="reduced")
    basis = deterministic_basis_signs(basis)
    all_variances = singular_values * singular_values / (len(unit) - 1)
    total_variance = float(np.sum(mapped * mapped) / (len(unit) - 1))
    tangent_dimension = unit.shape[1] - 1
    ambient_scale = max(
        total_variance / tangent_dimension, np.finfo(np.float64).eps
    )
    variance_floor = variance_floor_fraction * ambient_scale
    tangent_variances = np.maximum(all_variances[:rank], variance_floor)
    retained_variance = float(np.sum(all_variances[:rank]))
    residual_dimensions = tangent_dimension - rank
    empirical_residual = max(
        total_variance - retained_variance, 0.0
    ) / residual_dimensions
    residual_variance = max(
        empirical_residual,
        residual_floor_fraction * ambient_scale,
        variance_floor,
    )
    angles = np.linalg.norm(mapped, axis=1)
    angular_radius = max(
        float(np.sqrt(np.sum(angles * angles) / (len(unit) - 1))),
        1e-8,
    )
    indices = support_indices or tuple(range(len(unit)))
    return TangentCapPrimitive(
        mean_direction=mean,
        basis=basis,
        tangent_variances=tangent_variances,
        residual_variance=residual_variance,
        angular_radius=angular_radius,
        class_label=class_label,
        anchor_index=anchor_index,
        support_indices=indices,
    )
