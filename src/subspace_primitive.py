from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def deterministic_basis_signs(basis: np.ndarray) -> np.ndarray:
    result = np.asarray(basis, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


@dataclass(frozen=True)
class SubspacePrimitive:
    """Bounded affine subspace with an isotropic orthogonal residual."""

    center: np.ndarray
    basis: np.ndarray
    tangent_variances: np.ndarray
    residual_variance: float
    class_label: int | None = None
    anchor_index: int | None = None
    support_size: int | None = None

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        basis = np.asarray(self.basis, dtype=np.float64)
        tangent_variances = np.asarray(self.tangent_variances, dtype=np.float64)
        if center.ndim != 1 or not np.all(np.isfinite(center)):
            raise ValueError("center must be a finite vector.")
        if (
            basis.ndim != 2
            or basis.shape[0] != len(center)
            or basis.shape[1] < 1
            or not np.all(np.isfinite(basis))
        ):
            raise ValueError("basis must have shape (dimension, positive rank).")
        if tangent_variances.shape != (basis.shape[1],):
            raise ValueError("tangent_variances must match the basis rank.")
        if np.any(~np.isfinite(tangent_variances)) or np.any(tangent_variances <= 0.0):
            raise ValueError("tangent variances must be finite and positive.")
        if not np.isfinite(self.residual_variance) or self.residual_variance <= 0.0:
            raise ValueError("residual_variance must be finite and positive.")
        gram = basis.T @ basis
        if not np.allclose(gram, np.eye(basis.shape[1]), rtol=0.0, atol=1e-8):
            raise ValueError("basis columns must be orthonormal.")
        if self.anchor_index is not None and self.anchor_index < 0:
            raise ValueError("anchor_index cannot be negative.")
        if self.support_size is not None and self.support_size < basis.shape[1] + 2:
            raise ValueError("support_size must satisfy the r+2 contract.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "basis", deterministic_basis_signs(basis))
        object.__setattr__(self, "tangent_variances", tangent_variances)

    @property
    def dimension(self) -> int:
        return len(self.center)

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    @property
    def parameter_count(self) -> int:
        return self.dimension + self.dimension * self.rank + self.rank + 1

    @property
    def array_bytes(self) -> int:
        return int(
            self.center.nbytes
            + self.basis.nbytes
            + self.tangent_variances.nbytes
            + np.dtype(np.float64).itemsize
        )

    def _coordinates(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("points have the wrong shape.")
        deltas = values - self.center
        tangent = deltas @ self.basis
        residual_squared = np.maximum(
            np.sum(deltas * deltas, axis=1)
            - np.sum(tangent * tangent, axis=1),
            0.0,
        )
        return deltas, tangent, residual_squared

    def quadratic_form(self, points: np.ndarray) -> np.ndarray:
        _, tangent, residual_squared = self._coordinates(points)
        return np.sum(
            tangent * tangent / self.tangent_variances[None, :], axis=1
        ) + residual_squared / self.residual_variance

    def radial_field(self, points: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(self.quadratic_form(points), 0.0)) - 1.0

    def log_likelihood(self, points: np.ndarray) -> np.ndarray:
        log_determinant = float(
            np.log(self.tangent_variances).sum()
            + (self.dimension - self.rank) * np.log(self.residual_variance)
        )
        normalization = self.dimension * np.log(2.0 * np.pi)
        return -0.5 * (
            self.quadratic_form(points) + log_determinant + normalization
        )

    def quadratic_gradient(self, points: np.ndarray) -> np.ndarray:
        deltas, tangent, _ = self._coordinates(points)
        tangent_projection = tangent @ self.basis.T
        residual = deltas - tangent_projection
        tangent_gradient = (tangent / self.tangent_variances[None, :]) @ self.basis.T
        return 2.0 * (
            tangent_gradient + residual / self.residual_variance
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "center": self.center.tolist(),
            "basis": self.basis.tolist(),
            "tangent_variances": self.tangent_variances.tolist(),
            "residual_variance": float(self.residual_variance),
            "class_label": self.class_label,
            "anchor_index": self.anchor_index,
            "support_size": self.support_size,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SubspacePrimitive":
        required = {
            "schema_version",
            "center",
            "basis",
            "tangent_variances",
            "residual_variance",
            "class_label",
            "anchor_index",
            "support_size",
        }
        if set(payload) != required or payload.get("schema_version") != 1:
            raise ValueError("Unsupported subspace-primitive schema.")
        return cls(
            center=np.asarray(payload["center"], dtype=np.float64),
            basis=np.asarray(payload["basis"], dtype=np.float64),
            tangent_variances=np.asarray(
                payload["tangent_variances"], dtype=np.float64
            ),
            residual_variance=float(payload["residual_variance"]),
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
            support_size=(
                None
                if payload["support_size"] is None
                else int(payload["support_size"])
            ),
        )


def fit_subspace_primitive(
    points: np.ndarray,
    rank: int,
    *,
    variance_floor_fraction: float = 1e-3,
    residual_floor_fraction: float = 0.05,
    class_label: int | None = None,
    anchor_index: int | None = None,
) -> SubspacePrimitive:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("points must be a two-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must be finite.")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer.")
    if len(values) < rank + 2:
        raise ValueError(
            f"Rank {rank} requires r+2={rank + 2} support points; got {len(values)}."
        )
    if rank >= values.shape[1]:
        raise ValueError("rank must be smaller than the ambient dimension.")
    if variance_floor_fraction <= 0.0 or residual_floor_fraction <= 0.0:
        raise ValueError("variance floor fractions must be positive.")

    center = values.mean(axis=0)
    centered = values - center
    _, singular_values, right_vectors = np.linalg.svd(
        centered, full_matrices=False
    )
    basis = deterministic_basis_signs(right_vectors[:rank].T)
    all_variances = singular_values * singular_values / (len(values) - 1)
    total_variance = float(np.sum(centered * centered) / (len(values) - 1))
    ambient_scale = max(total_variance / values.shape[1], np.finfo(np.float64).eps)
    variance_floor = variance_floor_fraction * ambient_scale
    tangent_variances = np.maximum(all_variances[:rank], variance_floor)
    retained_variance = float(np.sum(all_variances[:rank]))
    residual_dimensions = values.shape[1] - rank
    empirical_residual = max(total_variance - retained_variance, 0.0) / residual_dimensions
    residual_variance = max(
        empirical_residual,
        residual_floor_fraction * ambient_scale,
        variance_floor,
    )
    return SubspacePrimitive(
        center=center,
        basis=basis,
        tangent_variances=tangent_variances,
        residual_variance=residual_variance,
        class_label=class_label,
        anchor_index=anchor_index,
        support_size=len(values),
    )
