"""GEODE OOD input guard (v25 M251) — distribution-shift escalation.

A deterministic, structure-only gate: it fits a reference profile
(centroid + diagonal scale) from quorum-admitted reference vectors
and scores a query by its normalized distance. A query beyond the
threshold is REJECTED by the guard, which feeds the M241 abstention
path — shifted inputs never reach unvetted arms.

Deterministic: the profile is a mean + elementwise standard
deviation (zero-variance dimensions get scale 1 so they never
divide by zero); no RNG, no wall clocks. This is the input-side
sibling of the M242 drift gate; the trained/shifted-data detectors
behind it are future data artifacts (gated, like the empirical
encoder).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class OodGate:
    """Diagonal-Mahalanobis-style distance gate in feature space."""

    def __init__(self, threshold: float = 3.0):
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        self.threshold = float(threshold)
        self._centroid: list[float] | None = None
        self._scale: list[float] | None = None
        self._dim: int | None = None

    def fit_profile(self, reference_vectors: Sequence[Sequence[float]]
                    ) -> None:
        """Fit centroid + diagonal scale from the reference set
        (assumed quorum-admitted by the caller)."""
        refs = [list(v) for v in reference_vectors]
        if not refs:
            raise ValueError("fit_profile requires at least one vector")
        dim = len(refs[0])
        for i, v in enumerate(refs):
            if len(v) != dim:
                raise ValueError(f"vector {i} has length {len(v)}, "
                                 f"expected {dim}")
        n = len(refs)
        centroid = [sum(v[d] for v in refs) / n for d in range(dim)]
        scale = []
        for d in range(dim):
            var = sum((v[d] - centroid[d]) ** 2 for v in refs) / n
            scale.append(var ** 0.5 if var > 0.0 else 1.0)
        self._centroid = centroid
        self._scale = scale
        self._dim = dim

    @property
    def fitted(self) -> bool:
        return self._centroid is not None

    def score(self, vector: Sequence[float]) -> float:
        """Normalized distance: sqrt(mean((x-c)/s)^2). Raises when
        unfit or dimension-mismatched."""
        if not self.fitted:
            raise RuntimeError("OodGate is not fitted")
        if len(vector) != self._dim:
            raise ValueError(f"vector length {len(vector)}, expected "
                             f"{self._dim}")
        assert self._centroid is not None and self._scale is not None
        sq = sum(((v - self._centroid[d]) / self._scale[d]) ** 2
                 for d, v in enumerate(vector))
        return (sq / self._dim) ** 0.5

    def admits(self, vector: Sequence[float],
               threshold: float | None = None) -> dict[str, Any]:
        """(admitted, reason) — the escalation decision."""
        t = self.threshold if threshold is None else float(threshold)
        try:
            s = self.score(vector)
        except RuntimeError:
            # unfit guard refuses everything (fail-closed, registered:
            # an ungated input path is not a safety path)
            return {"admitted": False, "reason": "guard_unfitted",
                    "score": None}
        return {"admitted": bool(s <= t),
                "reason": "in_distribution" if s <= t else "out_of_distribution",
                "score": s}
