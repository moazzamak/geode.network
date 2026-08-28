"""M301 module - closed-form alignment (orthogonal Procrustes and
CCA) as registered frozen artifacts.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md``
§8.37 (27 Aug 2026, before any build). Alignment is one exact
solve - no optimizer, no random seed, replayable under the M306
oracle's numerics discipline (float64 promotion, pinned kernels).

- ``orthogonal_procrustes``: the orthogonal map minimizing
  ||A R - B||_F via the exact SVD construction
  R = U V^T with U S V^T = B^T A.
- ``cca_align``: the canonical-correlation projections that
  decorrelate the two projected spaces (the whitened cross-
  covariance SVD).

The H26-4 measurement (aligned > concatenated > single encoder on
a registered multi-encoder cell) is a tier4 experiment, recorded
as pending - this module registers the machinery, not the gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from geode.core.replay_oracle import head_digest


class AlignmentError(RuntimeError):
    """An alignment solve failed its registered instrument."""


@dataclass(frozen=True)
class ProcrustesArtifact:
    """The registered frozen alignment artifact: the orthogonal
    map and its instrument report."""
    rotation: np.ndarray
    report: dict[str, Any]

    def map(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float64) @ self.rotation

    def digest(self) -> str:
        return head_digest(self.rotation,
                           np.zeros(self.rotation.shape[1]))


def orthogonal_procrustes(a: np.ndarray, b: np.ndarray
                          ) -> ProcrustesArtifact:
    """The orthogonal map R minimizing ||A R - B||_F. Exact SVD
    construction: U S V^T = B^T A, R = U V^T (the Barron 2019
    lemma). Float64 throughout; the orthogonality and objective
    instruments are recorded, not assumed."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise AlignmentError(
            f"shape mismatch: {a.shape} vs {b.shape}")
    m = a.T @ b
    u, _s, vt = np.linalg.svd(m, full_matrices=False)
    rotation = u @ vt
    residual = np.linalg.norm(rotation.T @ rotation
                              - np.eye(rotation.shape[1]))
    before = np.linalg.norm(a - b)
    after = np.linalg.norm(a @ rotation - b)
    report = {
        "objective_before": float(before),
        "objective_after": float(after),
        "orthogonality_residual": float(residual),
        "objective_improves": bool(after <= before + 1e-12),
    }
    if residual > 1e-10:
        raise AlignmentError(
            f"Procrustes map is not orthogonal: residual {residual}")
    return ProcrustesArtifact(rotation=rotation, report=report)


@dataclass(frozen=True)
class CcaArtifact:
    """Two canonical projections, each pinned by its own digest."""
    projection_a: np.ndarray
    projection_b: np.ndarray
    report: dict[str, Any]

    def digest(self) -> str:
        return head_digest(
            self.projection_a,
            np.zeros(self.projection_a.shape[1]))[:16] + \
            head_digest(self.projection_b,
                        np.zeros(self.projection_b.shape[1]))[:16]


def _inv_sqrt_symmetric(cov: np.ndarray, ridge: float) -> np.ndarray:
    """The symmetric inverse square root in float64: eig-based
    (never a float32 path - the exact-guarantee lesson), ridge on
    the spectrum for low-rank safety."""
    vals, vecs = np.linalg.eigh(cov)
    inv = 1.0 / np.sqrt(np.maximum(vals, 0.0) + ridge)
    return (vecs * inv[None, :]) @ vecs.T


def cca_from_moments(cov_a: np.ndarray, cov_b: np.ndarray,
                     cross: np.ndarray, components: int,
                     ridge: float = 1e-8) -> CcaArtifact:
    """CCA from precomputed sufficient statistics (covariances and
    cross-covariance). The streaming entry point: callers
    accumulate the moments in float64 chunks and never materialise
    the design matrix. The ridge is TRACE-RELATIVE (ridge x
    mean eigenvalue), so the construction is invariant to the
    caller's covariance normalization (divided by n or not) - the
    direct and the streaming paths agree by construction. The
    projections come from the standard construction
    U S V^T = inv_sqrt(S_aa) S_ab inv_sqrt(S_bb);
    decorrelation of the projected variates is a data property the
    caller measures and reports."""
    cov_a = np.asarray(cov_a, dtype=np.float64)
    cov_b = np.asarray(cov_b, dtype=np.float64)
    cross = np.asarray(cross, dtype=np.float64)
    ridge_a = ridge * np.trace(cov_a) / cov_a.shape[0]
    ridge_b = ridge * np.trace(cov_b) / cov_b.shape[0]
    inv_a = _inv_sqrt_symmetric(cov_a, ridge_a)
    inv_b = _inv_sqrt_symmetric(cov_b, ridge_b)
    target = inv_a @ cross @ inv_b
    u, s, vt = np.linalg.svd(target, full_matrices=False)
    keep = min(components, len(u), vt.shape[1])
    proj_a = inv_a @ u[:, :keep]
    proj_b = inv_b @ vt[:keep, :].T
    report = {
        "components": keep,
        "canonical_correlations": s[:keep].tolist(),
        "all_nonnegative": bool((s[:keep] >= 0.0).all()),
        "note": ("decorrelation of the projected variates is "
                 "measured by the caller on data"),
    }
    return CcaArtifact(projection_a=proj_a, projection_b=proj_b,
                       report=report)


def cca_align(a: np.ndarray, b: np.ndarray, components: int,
              ridge: float = 1e-8) -> CcaArtifact:
    """Closed-form CCA projections: the standard construction
    U S V^T = inv_sqrt(S_aa) S_ab inv_sqrt(S_bb), with the
    a-projection inv_sqrt(S_aa) U and the b-projection
    inv_sqrt(S_bb) V. The canonical correlations are the singular
    values; the projected variates are decorrelated within each
    space (the instrument, measured not assumed)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise AlignmentError(
            f"row mismatch: {len(a)} vs {len(b)}")
    centre_a = a - a.mean(axis=0, keepdims=True)
    centre_b = b - b.mean(axis=0, keepdims=True)
    n = max(len(a) - 1, 1)
    cross = (centre_a.T @ centre_b) / n
    cov_a = (centre_a.T @ centre_a) / n
    cov_b = (centre_b.T @ centre_b) / n
    artifact = cca_from_moments(cov_a, cov_b, cross, components,
                                ridge=ridge)
    # the correlation instrument: the canonical correlations are
    # the singular values of the whitened cross-covariance, and
    # the projected variates decorrelate within each space
    proj_a = artifact.projection_a
    proj_b = artifact.projection_b
    za = centre_a @ proj_a
    zb = centre_b @ proj_b
    off_a = np.abs(np.triu(np.corrcoef(za.T), k=1)).max()
    off_b = np.abs(np.triu(np.corrcoef(zb.T), k=1)).max()
    report = dict(artifact.report)
    report["decorrelated"] = bool(max(off_a, off_b) < 0.05)
    report["max_within_offdiag"] = float(max(off_a, off_b))
    return CcaArtifact(projection_a=proj_a, projection_b=proj_b,
                       report=report)
