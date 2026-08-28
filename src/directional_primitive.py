from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def l2_normalize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("features must be a finite two-dimensional array.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).tiny):
        raise ValueError("Directional geometry does not accept zero vectors.")
    return values / norms


@dataclass(frozen=True)
class SphericalCapPrimitive:
    mean_direction: np.ndarray
    angular_radius: float
    class_label: int | None = None
    anchor_index: int | None = None
    support_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        direction = np.asarray(self.mean_direction, dtype=np.float64)
        if direction.ndim != 1 or not np.all(np.isfinite(direction)):
            raise ValueError("mean_direction must be a finite vector.")
        norm = float(np.linalg.norm(direction))
        if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-10):
            raise ValueError("mean_direction must have unit norm.")
        if (
            not np.isfinite(self.angular_radius)
            or self.angular_radius <= 0.0
            or self.angular_radius > np.pi
        ):
            raise ValueError("angular_radius must lie in (0, pi].")
        if self.anchor_index is not None and self.anchor_index < 0:
            raise ValueError("anchor_index cannot be negative.")
        support = tuple(int(index) for index in self.support_indices)
        if any(index < 0 for index in support) or len(set(support)) != len(support):
            raise ValueError("support_indices must be unique and nonnegative.")
        object.__setattr__(self, "mean_direction", direction)
        object.__setattr__(self, "support_indices", support)

    @property
    def dimension(self) -> int:
        return len(self.mean_direction)

    @property
    def parameter_count(self) -> int:
        return self.dimension + 1

    @property
    def array_bytes(self) -> int:
        return int(self.mean_direction.nbytes + np.dtype(np.float64).itemsize)

    def angles(self, features: np.ndarray) -> np.ndarray:
        unit = l2_normalize(features)
        cosine = np.clip(unit @ self.mean_direction, -1.0, 1.0)
        return np.arccos(cosine)

    def angular_field(self, features: np.ndarray) -> np.ndarray:
        return self.angles(features) / self.angular_radius - 1.0

    def angular_gradient(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        unit = l2_normalize(values)
        norms = np.linalg.norm(values, axis=1)
        cosine = np.clip(unit @ self.mean_direction, -1.0, 1.0)
        tangent = self.mean_direction[None, :] - cosine[:, None] * unit
        denominator = (
            norms
            * np.sqrt(np.maximum(1.0 - cosine * cosine, 0.0))
            * self.angular_radius
        )
        gradient = np.zeros_like(values)
        stable = denominator > 1e-12
        gradient[stable] = -tangent[stable] / denominator[stable, None]
        return gradient

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mean_direction": self.mean_direction.tolist(),
            "angular_radius": float(self.angular_radius),
            "class_label": self.class_label,
            "anchor_index": self.anchor_index,
            "support_indices": list(self.support_indices),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SphericalCapPrimitive":
        required = {
            "schema_version",
            "mean_direction",
            "angular_radius",
            "class_label",
            "anchor_index",
            "support_indices",
        }
        if set(payload) != required or payload.get("schema_version") != 1:
            raise ValueError("Unsupported spherical-cap schema.")
        return cls(
            mean_direction=np.asarray(payload["mean_direction"], dtype=np.float64),
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
            support_indices=tuple(int(index) for index in payload["support_indices"]),
        )


def fit_spherical_cap(
    features: np.ndarray,
    *,
    class_label: int | None = None,
    anchor_index: int | None = None,
    support_indices: tuple[int, ...] = (),
) -> SphericalCapPrimitive:
    unit = l2_normalize(features)
    mean = unit.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= np.finfo(np.float64).tiny:
        raise ValueError("Support directions have a degenerate mean.")
    direction = mean / norm
    angles = np.arccos(np.clip(unit @ direction, -1.0, 1.0))
    denominator = max(len(unit) - 1, 1)
    radius = float(np.sqrt(np.sum(angles * angles) / denominator))
    radius = max(radius, 1e-8)
    return SphericalCapPrimitive(
        mean_direction=direction,
        angular_radius=radius,
        class_label=class_label,
        anchor_index=anchor_index,
        support_indices=support_indices,
    )
