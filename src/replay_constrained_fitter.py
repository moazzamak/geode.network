from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sdf_engine import EllipsoidExpert, Expert


@dataclass(frozen=True)
class ReplayConstrainedFit:
    expert: Expert
    radius_scale: float
    positive_coverage: float
    exclusion_violations: int
    minimum_exclusion_sdf: float


def fit_replay_constrained_expert(
    positive_points: np.ndarray,
    exclusion_points: np.ndarray,
    *,
    exclusion_margin: float = 0.1,
    radius_padding: float = 1.1,
    minimum_initial_radius: float = 0.5,
) -> ReplayConstrainedFit:
    """Fit one ellipsoid while keeping replay negatives outside its surface."""
    positives = np.asarray(positive_points, dtype=np.float64)
    exclusions = np.asarray(exclusion_points, dtype=np.float64)
    if positives.ndim != 2 or not len(positives):
        raise ValueError("positive_points must be a non-empty matrix.")
    if exclusions.ndim != 2 or exclusions.shape[1] != positives.shape[1]:
        raise ValueError("exclusion_points must be an aligned matrix.")
    if exclusion_margin < 0.0 or radius_padding <= 0.0:
        raise ValueError("margin must be non-negative and padding positive.")
    if minimum_initial_radius <= 0.0:
        raise ValueError("minimum_initial_radius must be positive.")

    center = positives.mean(axis=0)
    initial_radii = np.maximum(
        np.max(np.abs(positives - center), axis=0) * radius_padding,
        minimum_initial_radius,
    )
    radius_scale = 1.0
    if len(exclusions):
        normalized_distances = np.sqrt(np.sum(
            ((exclusions - center) / initial_radii) ** 2,
            axis=1,
        ))
        radius_scale = min(
            radius_scale,
            float(np.min(normalized_distances) / (1.0 + exclusion_margin)),
        )
    radius_scale = max(radius_scale * (1.0 - 1e-9), 1e-6)
    ellipsoid = EllipsoidExpert(
        center=center,
        radii=initial_radii * radius_scale,
    )
    expert = Expert(alpha=2.0)
    expert.add_ellipsoid(ellipsoid)
    positive_sdf = ellipsoid.compute_sdf(positives)
    exclusion_sdf = ellipsoid.compute_sdf(exclusions)
    return ReplayConstrainedFit(
        expert=expert,
        radius_scale=radius_scale,
        positive_coverage=float(np.mean(positive_sdf < 0.0)),
        exclusion_violations=int(np.sum(exclusion_sdf < exclusion_margin)),
        minimum_exclusion_sdf=(
            float(np.min(exclusion_sdf)) if len(exclusion_sdf) else float("inf")
        ),
    )