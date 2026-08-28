"""GEODE guard composition (v25 M274) — per-arm guards with the
measured two-axis structure.

Registered 22 Aug 2026 in
``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` (the M272-M281 wave).
The M263 lesson, shipped as policy:

- a geometric guard (diagonal Mahalanobis) alone SCORED ITS OWN OOD
  probes inside distribution (1.05-1.51 vs threshold 3.0); the
  structural vocab-coverage primitive caught them;
- therefore every arm ships a COMPOSED guard: geometric (fit on the
  arm's own train features — per-arm, never pooled) PLUS structural
  primitives, and no guard is ADMITTED until its own authored OOD
  probes are rejected (the standing instrument rule).

Deterministic; no RNG, no wall clocks.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

from geode.core.ood import OodGate


def abstention_floor_from_scores(scores: Sequence[float],
                                 coverage: float = 0.95) -> float:
    """M275: the registered per-arm floor-setting rule — the
    split-conformal order statistic (rank = ceil((n+1) * coverage))
    of the arm's own in-distribution guard scores. An empirical
    quantile undershoots coverage (the registered statistics lesson);
    the conformal rank does not. The floor is a measured number of
    the arm's own profile, never a hand-picked constant."""
    if coverage <= 0.0 or coverage >= 1.0:
        raise ValueError("coverage must lie in (0, 1)")
    vals = sorted(float(s) for s in scores)
    if not vals:
        raise ValueError("at least one score is required")
    rank = max(1, min(len(vals),
                      int(math.ceil((len(vals) + 1) * coverage))))
    return vals[rank - 1]


class VocabCoveragePrimitive:
    """The M263 structural primitive: fraction of whitespace tokens
    present in a reference vocabulary. Token soup / base64 / log
    dumps score near zero; natural text scores high."""

    def __init__(self, vocab: set[str], threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        self._vocab = set(vocab)
        self.threshold = float(threshold)

    def check(self, text: str) -> dict[str, Any]:
        tokens = text.split()
        if not tokens:
            return {"admitted": False, "reason": "empty",
                    "coverage": 0.0}
        coverage = sum(1 for t in tokens
                       if t in self._vocab) / len(tokens)
        return {"admitted": bool(coverage >= self.threshold),
                "reason": ("covered" if coverage >= self.threshold
                           else "low_vocab_coverage"),
                "coverage": coverage}


class ComposedGuard:
    """Geometric gate + structural primitives. An input is admitted
    iff every component admits it."""

    def __init__(self, geometric: OodGate,
                 primitives: Sequence[tuple[str, Callable[[str],
                                                          dict[str,
                                                               Any]]]]
                 ) -> None:
        self.geometric = geometric
        self._primitives = list(primitives)

    def admit(self, text: str, vector: Sequence[float],
              threshold: float | None = None) -> dict[str, Any]:
        geo = self.geometric.admits(vector, threshold=threshold)
        if not geo["admitted"]:
            return {"admitted": False, "reason": geo["reason"],
                    "geometric": geo, "primitives": []}
        prim: list[dict[str, Any]] = []
        for name, check in self._primitives:
            result = check(text)
            prim.append({"name": name, **result})
            if not result["admitted"]:
                return {"admitted": False, "reason": result["reason"],
                        "geometric": geo, "primitives": prim}
        return {"admitted": True, "reason": "in_distribution",
                "geometric": geo, "primitives": prim}


class GuardRegistry:
    """Per-arm composed guards. A guard is ADMITTED only after its
    own authored OOD probes are rejected by it (the standing
    instrument rule): register_guard raises if any probe slips
    through — a broken guard never ships."""

    def __init__(self) -> None:
        self._guards: dict[str, ComposedGuard] = {}

    def register_guard(self, arm_id: str, guard: ComposedGuard,
                       probes: Sequence[tuple[str,
                                              Sequence[float]]]
                       ) -> None:
        """Register an arm's guard; raises ValueError listing every
        probe the guard failed to reject."""
        leaks = []
        for text, vector in probes:
            if guard.admit(text, vector)["admitted"]:
                leaks.append(text[:60])
        if leaks:
            raise ValueError(
                f"guard for {arm_id!r} admitted its own authored "
                f"probes ({len(leaks)}): {leaks}")
        self._guards[str(arm_id)] = guard

    def guard(self, arm_id: str) -> ComposedGuard | None:
        return self._guards.get(str(arm_id))

    def admit(self, arm_id: str, text: str,
              vector: Sequence[float]) -> dict[str, Any]:
        guard = self._guards.get(str(arm_id))
        if guard is None:
            return {"admitted": False, "reason": "guard_missing"}
        return guard.admit(text, vector)
