"""GEODE measured routing (v25 M272) — the routing rules the M268
wave measured, shipped as the live policy.

Registered 22 Aug 2026 in
``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` (the M272-M281 wave).
What this module carries, and only what was measured:

- the embedding nearest-centroid router: cosine to frozen centroid
  fingerprints in BERT feature space (cell 4: 0 misroutes on 700
  natural-query items, routed 0.960 vs generalist 0.764);
- the measured per-type arm rules (cell 1 / cell 4):
  sentiment -> the generalist (0.959 vs the 66M specialist's 0.908),
  arithmetic/logic -> the programmatic primitives (1.0 vs 0.17/0.55),
  code -> the coder arm (0.598 vs 0.506).

Boundaries, declared: the routing instrument is deterministic
(centroids are frozen references, no RNG); the descriptor-DSL
fingerprint path is unchanged; a learned router may only replace
this policy behind the M281 gate.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

# The canonical centroid reference queries (the cell-4 _SAMPLES,
# now the registered single source of truth).
FAMILY_SAMPLES: dict[str, list[str]] = {
    "sentiment": [
        "This movie is an absolute masterpiece of modern cinema.",
        "The acting was wooden and the plot went nowhere.",
        "I loved every minute of this film.",
        "A complete waste of two hours, poorly written.",
        "The best film I have seen all year, brilliant cast.",
    ],
    "arithmetic": [
        "What is twelve plus seven?",
        "If I take twenty and multiply by three, what do I get?",
        "What is forty five minus nine?",
        "Compute eight times six.",
        "What does ninety one plus two equal?",
    ],
    "logic": [
        "Suppose A is true and B is false. Is (A and B) true or false?",
        "Given P true, Q true, R false: is (P or (Q and R)) true or false?",
        "A is false, B is true. Is (A or B) true or false?",
        "X true, Y false, Z true: is (X and (Y or Z)) true or false?",
        "Is (not A) true or false when A is false?",
    ],
}

FAMILY_ORDER: list[str] = ["sentiment", "arithmetic", "logic"]

# The measured per-type arm rules (cell 1 / cell 4 sealed numbers).
MEASURED_ARM_RULES: dict[str, dict[str, object]] = {
    "sentiment": {"arm": "generalist", "measured": 0.959,
                  "vs_specialist": 0.908,
                  "source": "M268 cell 1 (rows 0..999) / cell 1b full split 0.941"},
    "arithmetic": {"arm": "primitive", "measured": 1.0,
                   "vs_generalist": 0.1667,
                   "source": "M268 cell 1"},
    "logic": {"arm": "primitive", "measured": 1.0,
              "vs_generalist": 0.55, "source": "M268 cell 1"},
    "code": {"arm": "coder", "measured": 0.5976,
             "vs_generalist": 0.5061, "source": "M268 cell 3"},
}

Embedder = Callable[[Sequence[str]], "np.ndarray"]


def _unit_norm(vectors: "np.ndarray") -> "np.ndarray":
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class EmbeddingRouter:
    """Cosine nearest-centroid over frozen reference embeddings.

    Deterministic: the centroids are means of the canonical sample
    embeddings, unit-normalised; ties resolve to the earlier family
    in FAMILY_ORDER (the cell-4 strict-greater rule)."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._centroids: dict[str, "np.ndarray"] = {}
        self._fit()

    def _fit(self) -> None:
        for family in FAMILY_ORDER:
            emb = np.asarray(self._embedder(FAMILY_SAMPLES[family]),
                             dtype=np.float64)
            unit = _unit_norm(emb)
            centroid = unit.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            self._centroids[family] = centroid / max(norm, 1e-12)

    @property
    def centroids(self) -> dict[str, "np.ndarray"]:
        return {k: v.copy() for k, v in self._centroids.items()}

    def route(self, text: str) -> str:
        """Nearest family by cosine (dot of unit vectors)."""
        return self.route_batch([text])[0]

    def route_batch(self, texts: Sequence[str]) -> list[str]:
        """The same rule over a batch (one embedding pass — the
        registered single-call semantics; identical centroids,
        identical tie rule)."""
        vecs = _unit_norm(np.asarray(self._embedder(list(texts)),
                                     dtype=np.float64))
        routes: list[str] = []
        for vec in vecs:
            best, best_cos = None, -2.0
            for family in FAMILY_ORDER:
                cos = float(np.dot(vec, self._centroids[family]))
                if cos > best_cos:  # strict: ties to the earlier family
                    best, best_cos = family, cos
            routes.append(best if best is not None else "sentiment")
        return routes


def route_policy(router: EmbeddingRouter, text: str
                 ) -> dict[str, object]:
    """The shipped policy: (1) the M284 claim pre-pass (boolean
    claims with arithmetic surface route to the logic primitive,
    answered exactly); (2) the M284b verdict rule (review context
    + a true/false token routes to sentiment); then the embedding
    route and the measured arm rule for the routed family
    (sentiment -> generalist, exact families -> primitives).
    Returns {family, arm, rule} with an optional claim field on
    claim matches."""
    from geode.core.claim_route import (  # local: no cycle
        claim_answer, detect_verdict)
    claim = claim_answer(text)
    if claim is not None:
        return {"family": "logic", "arm": "primitive",
                "claim": claim,
                "rule": {"arm": "primitive", "measured": 1.0,
                         "source": "M284 claim evaluator "
                                   "(M281b/M283 measured gap)"}}
    if detect_verdict(text):
        rule = MEASURED_ARM_RULES["sentiment"]
        return {"family": "sentiment", "arm": rule["arm"],
                "rule": dict(rule),
                "verdict_rule": True}
    family = router.route(text)
    rule = MEASURED_ARM_RULES.get(family, {"arm": "generalist",
                                           "note": "unmeasured family"})
    return {"family": family, "arm": rule["arm"], "rule": rule}
