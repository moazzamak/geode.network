"""GEODE erasure (v25 M179) — closed-form LEACE on frozen components.

Provenance: `leace_eraser`, `AffineMap`, `_symmetric_powers`, and
`_one_hot` are VERBATIM from the sealed M90.2 implementation
(`archive/experiments/tier4_legacy/eval_v14_m90_2_domain_erasure.py`,
14 Aug 2026), copied into the core package so v25 cells import one
canonical source. The float64-throughout discipline (N90.2.16) and the
rank cap at group_count-1 (N90.2.17) travel with the code.

`erasure_certificate` is new here (M179): the before/after pair of the
largest pairwise group-mean gap and the largest absolute cross-
covariance with the one-hot concept, judged on a relative residual.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class AffineMap:
    """A fixed map fitted on the fit rows and applied unchanged everywhere.

    Held and applied in float64 (N90.2.16); the result is cast back to
    the caller's dtype so downstream geometry sees the same type the
    untransformed baseline does.
    """

    def __init__(self, matrix: np.ndarray, offset: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.offset = np.asarray(offset, dtype=np.float64)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        promoted = np.asarray(features, dtype=np.float64)
        mapped = promoted @ self.matrix.T + self.offset
        return mapped.astype(features.dtype, copy=False)


def _symmetric_powers(
    covariance: np.ndarray, *, floor: float
) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    clipped = np.clip(eigenvalues, floor, None)
    inverse_root = (eigenvectors * (clipped ** -0.5)) @ eigenvectors.T
    forward_root = (eigenvectors * (clipped ** 0.5)) @ eigenvectors.T
    return inverse_root, forward_root


def _one_hot(groups: np.ndarray, group_count: int) -> np.ndarray:
    encoded = np.zeros((len(groups), group_count), dtype=np.float64)
    encoded[np.arange(len(groups)), groups] = 1.0
    return encoded


def leace_eraser(
    features: np.ndarray,
    groups: np.ndarray,
    *,
    group_count: int,
    floor: float,
    singular_tolerance: float,
) -> tuple[AffineMap, int]:
    """LEACE, arXiv:2306.03819, in closed form (M90.2 verbatim).

    Returns the eraser and the rank actually removed. A centred one-hot
    over g groups has rank g-1, so the removed rank is the check that
    the erasure did what it claims rather than a free parameter.
    """
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0)
    centred = features - mean
    concept = _one_hot(groups, group_count)
    concept -= concept.mean(axis=0)
    n = len(features)
    covariance = centred.T @ centred / n
    cross = centred.T @ concept / n

    inverse_root, forward_root = _symmetric_powers(covariance, floor=floor)
    whitened_cross = inverse_root @ cross
    basis, singular_values, _ = np.linalg.svd(whitened_cross,
                                              full_matrices=False)
    # Rank cap at group_count-1 (N90.2.17): the budget is known in
    # advance, so a tolerance is the wrong instrument for the cap.
    limit = max(group_count - 1, 0)
    above = int(np.count_nonzero(
        singular_values > singular_values[0] * singular_tolerance))
    retained = basis[:, : min(limit, above)]
    projector = retained @ retained.T

    eraser = forward_root @ projector @ inverse_root
    matrix = np.eye(features.shape[1]) - eraser
    return AffineMap(matrix, eraser @ mean), int(retained.shape[1])


def erasure_certificate(features: np.ndarray, groups: np.ndarray,
                        group_count: int,
                        eraser: AffineMap) -> dict[str, Any]:
    """M90.2-style certificate (N90.2.15): the largest pairwise
    group-mean gap and the largest absolute cross-covariance with the
    one-hot concept, before and after the transform, plus the relative
    residuals. Floats promoted to float64 for the certificate."""
    features = np.asarray(features, dtype=np.float64)
    erased = eraser(features)

    def stats(x: np.ndarray) -> dict[str, float]:
        concept = _one_hot(groups, group_count)
        concept = concept - concept.mean(axis=0)
        means = np.zeros((group_count, x.shape[1]))
        for g in range(group_count):
            rows = np.flatnonzero(groups == g)
            means[g] = x[rows].mean(axis=0) if len(rows) else 0.0
        diffs = means[None, :, :] - means[:, None, :]
        gap = float(np.max(np.linalg.norm(diffs, axis=-1))) \
            if group_count > 1 else 0.0
        cross = np.abs(x.T @ concept / len(x)).max()
        return {"mean_gap": gap, "cross_covariance": float(cross)}

    before = stats(features)
    after = stats(erased)
    base = max(before["mean_gap"], 1e-12)
    base_cross = max(before["cross_covariance"], 1e-12)
    return {
        "before": before,
        "after": after,
        "relative_mean_gap_residual": after["mean_gap"] / base,
        "relative_cross_covariance_residual":
            after["cross_covariance"] / base_cross,
    }
