"""Basis-identifiability measurement for the v13 sample-adequacy milestone.

The v12 geometry is fitted in two stages: a global PCA projection to
`output_dimension`, then a per-class subspace of `rank` inside that projection.
Measuring whether the *basis* is identified by the data therefore requires the
two halves being compared to share a projection frame. Fitting
`initialize_projected_metric_fields` independently on each half does not do
this: each half derives its own principal axes, and principal angles computed
across the two results measure projection variance rather than basis stability.

This module fits the projection once on the full geometry split, projects both
halves through that single shared frame, and only then fits per-class
subspaces. It also reports the Monte-Carlo expected principal angle between
independent random subspaces of the same shape, so that a measured angle can be
read against the value that carries no information.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.common.v12_metric_fields import initialize_metric_fields


def fit_shared_projection(
    features: np.ndarray, *, output_dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the v12 PCA projection.

    Mirrors `initialize_projected_metric_fields` exactly, including the sign
    convention, so that the halves are compared in the frame the milestone
    actually trains in.
    """
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not 1 <= output_dimension < values.shape[1]:
        raise ValueError("shared projection dimensions are invalid")
    mean = np.mean(values, axis=0)
    _, _, right_vectors = np.linalg.svd(values - mean, full_matrices=False)
    projection = right_vectors[:output_dimension].copy()
    for row in range(len(projection)):
        pivot = int(np.argmax(np.abs(projection[row])))
        if projection[row, pivot] < 0.0:
            projection[row] *= -1.0
    return mean, projection


def mean_principal_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    """Mean principal angle between two orthonormal bases, in degrees."""
    singular = np.linalg.svd(left.T @ right, compute_uv=False)
    return float(np.mean(np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))))


def random_subspace_angle_degrees(
    *, dimension: int, rank: int, trials: int, seed: int
) -> float:
    """Expected mean principal angle between independent random subspaces.

    This is the reference value for a basis that carries no information about
    the data. A measured angle at or near this value means the fitted subspace
    is not identified.
    """
    generator = np.random.default_rng(seed)
    angles = []
    for _ in range(trials):
        left = np.linalg.qr(generator.normal(size=(dimension, rank)))[0]
        right = np.linalg.qr(generator.normal(size=(dimension, rank)))[0]
        angles.append(mean_principal_angle_degrees(left, right))
    return float(np.mean(angles))


def _disjoint_halves(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first: list[int] = []
    second: list[int] = []
    for class_label in np.unique(labels):
        rows = np.flatnonzero(labels == class_label)
        midpoint = len(rows) // 2
        first.extend(rows[:midpoint])
        second.extend(rows[midpoint : 2 * midpoint])
    return (
        np.asarray(first, dtype=np.int64),
        np.asarray(second, dtype=np.int64),
    )


def basis_stability(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    *,
    output_dimension: int,
    rank: int,
    random_trials: int = 64,
    random_seed: int = 7801,
) -> dict[str, Any]:
    """Measure whether per-class subspaces are identified by the sample.

    The projection is fitted once on the whole geometry split; each disjoint
    half is then projected through it and fitted independently. The resulting
    bases are directly comparable.
    """
    features = np.asarray(fit_x, dtype=np.float64)
    labels = np.asarray(fit_y, dtype=np.int64)
    first_rows, second_rows = _disjoint_halves(labels)
    if len(first_rows) == 0:
        raise ValueError("basis stability requires at least two samples per class")

    mean, projection = fit_shared_projection(
        features, output_dimension=output_dimension
    )
    projected = (features - mean) @ projection.T

    halves = [
        initialize_metric_fields(projected[rows], labels[rows], rank=rank)
        for rows in (first_rows, second_rows)
    ]
    first, second = halves
    effective_rank = int(first.bases.shape[2])
    if int(second.bases.shape[2]) != effective_rank:
        raise ValueError("disjoint halves produced bases of different rank")

    angles = [
        mean_principal_angle_degrees(
            first.bases[class_index], second.bases[class_index]
        )
        for class_index in range(len(first.classes))
    ]
    reference = random_subspace_angle_degrees(
        dimension=output_dimension,
        rank=effective_rank,
        trials=random_trials,
        seed=random_seed,
    )
    mean_angle = float(np.mean(angles))
    return {
        "mean_principal_angle_degrees": mean_angle,
        "max_principal_angle_degrees": float(np.max(angles)),
        "random_subspace_angle_degrees": reference,
        # 1.0 when the halves agree exactly, 0.0 when the fitted subspace is
        # indistinguishable from an uninformative random one.
        "identifiability": float(max(0.0, 1.0 - mean_angle / reference)),
        "requested_rank": int(rank),
        "effective_rank": effective_rank,
        "half_sample_count_per_class": int(len(first_rows) // len(first.classes)),
        "shared_projection": True,
    }
