"""Deterministic diagnostics for frozen representation geometry."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_representation_diagnostics(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_neighbors: int = 10,
) -> dict[str, float]:
    """Measure local purity, intrinsic dimension, and class separation."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 2 or labels.ndim != 1:
        raise ValueError("Features must be rank 2 and labels must be rank 1.")
    if len(features) != len(labels):
        raise ValueError("Features and labels must contain the same sample count.")
    if len(features) < 3:
        raise ValueError("At least three samples are required.")
    if not np.isfinite(features).all():
        raise ValueError("Features must be finite.")
    if n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2.")

    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("At least two classes are required.")

    neighbor_count = min(n_neighbors, len(features) - 1)
    neighbors = NearestNeighbors(n_neighbors=neighbor_count + 1).fit(features)
    distances, indices = neighbors.kneighbors(features)
    neighbor_labels = labels[indices[:, 1:]]
    purity = float(np.mean(neighbor_labels == labels[:, None]))

    local_distances = np.maximum(distances[:, 1:], np.finfo(float).eps)
    neighborhood_radius = local_distances[:, -1]
    denominator = np.sum(
        np.log(local_distances / neighborhood_radius[:, None]),
        axis=1,
    )
    finite_lid = denominator < -np.finfo(float).eps
    local_intrinsic_dimension = (
        float(np.median(-neighbor_count / denominator[finite_lid]))
        if np.any(finite_lid)
        else 0.0
    )

    centroids = np.vstack(
        [features[labels == class_id].mean(axis=0) for class_id in classes]
    )
    within_class_radius = float(
        np.mean(
            [
                np.mean(
                    np.linalg.norm(
                        features[labels == class_id] - centroid,
                        axis=1,
                    )
                )
                for class_id, centroid in zip(classes, centroids, strict=True)
            ]
        )
    )
    centroid_distances = np.linalg.norm(
        centroids[:, None, :] - centroids[None, :, :],
        axis=2,
    )
    np.fill_diagonal(centroid_distances, np.inf)
    minimum_centroid_separation = float(np.min(centroid_distances))
    compactness_ratio = (
        within_class_radius / minimum_centroid_separation
        if minimum_centroid_separation > 0.0
        else float("inf")
    )
    return {
        "neighborhood_purity": purity,
        "local_intrinsic_dimension": local_intrinsic_dimension,
        "within_class_radius": within_class_radius,
        "minimum_centroid_separation": minimum_centroid_separation,
        "compactness_ratio": compactness_ratio,
    }
