"""GEODE attribution estimators (v25 M180) — Shapley, leave-one-out,
Beta Shapley, fingerprint coverage, and the H2 ranking-stability check.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(18 Aug 2026). Imported methods, not re-derivations: Shapley in ML
(arXiv:2202.05594), Beta Shapley (arXiv:2110.14049), EcoVal-style
measurement-first attribution (arXiv:2402.09288). Exact formulas for
small coalition games (our registries are small); permutation sampling
is NOT used — exactness is the audit property (M177).
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from typing import Any

Coalition = frozenset[str]


def _marginals(V: Mapping[Coalition, float],
               players: list[str]) -> dict[str, float]:
    """Marginal of each player to the grand coalition minus itself (LOO)."""
    all_p = frozenset(players)
    return {p: V.get(all_p, 0.0) - V.get(all_p - {p}, 0.0)
            for p in players}


def leave_one_out(V: Mapping[Coalition, float],
                  players: list[str]) -> dict[str, float]:
    """LOO attribution: the drop when one player is removed."""
    return _marginals(V, players)


def _coalition_weights(n: int, beta: float) -> list[float]:
    """Symmetric Beta Shapley per-coalition weights (arXiv:2110.14049).

    Size distribution p_s ∝ C(n-1, s) B(s+beta, n-s-1+beta); the
    per-coalition weight is p_s / C(n-1, s) = B(s+beta, n-s-1+beta),
    normalized over sizes. beta = 1 is classic Shapley exactly: the
    per-coalition weight becomes s!(n-s-1)!/n! (the textbook form,
    checked in the unit tests against hand-computed values)."""
    from math import comb, gamma

    raw_p = [comb(n - 1, s) * gamma(s + beta) * gamma(n - s - 1 + beta)
             / gamma(n - 1 + 2 * beta)
             for s in range(n)]
    total = sum(raw_p)
    return [(p / comb(n - 1, s)) / total
            for s, p in enumerate(raw_p)]


def beta_shapley(V: Mapping[Coalition, float], players: list[str],
                 beta: float = 1.0) -> dict[str, float]:
    """Symmetric Beta Shapley over exact coalition values;
    beta = 1 is classic Shapley."""
    n = len(players)
    weights = _coalition_weights(n, beta)
    out: dict[str, float] = {}
    for p in players:
        value = 0.0
        for s in range(n):
            weight = weights[s]
            for subset in itertools.combinations(
                    [q for q in players if q != p], s):
                coalition = frozenset(subset)
                value += weight * (
                    V.get(coalition | {p}, 0.0) - V.get(coalition, 0.0))
        out[p] = value
    return out


def shapley(V: Mapping[Coalition, float],
            players: list[str]) -> dict[str, float]:
    """Exact Shapley values = symmetric Beta Shapley(beta=1)."""
    return beta_shapley(V, players, beta=1.0)


def fingerprint_coverage(fp_task: Iterable[float],
                         arm_fingerprints: Mapping[str, Iterable[float]]
                         ) -> dict[str, float]:
    """Coverage attribution: each arm's share of the task fingerprint,
    by cosine similarity normalized across arms (negative cosines count
    as zero share — an anti-correlated arm covers nothing)."""
    def _cos(a: Iterable[float], b: Iterable[float]) -> float:
        a = list(a)
        b = list(b)
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    positives = {k: max(_cos(fp_task, v), 0.0)
                 for k, v in arm_fingerprints.items()}
    total = sum(positives.values())
    if total <= 0.0:
        return {k: 0.0 for k in positives}
    return {k: v / total for k, v in positives.items()}


def ranking_stability(rankings: list[dict[str, float]],
                      top: int | None = None) -> float:
    """H2: mean pairwise Kendall tau over the induced rankings (higher
    = more stable). Values -> ranks per dict (descending)."""
    if len(rankings) < 2:
        return 1.0

    def _ranks(scores: dict[str, float]) -> dict[str, int]:
        order = sorted(scores, key=lambda k: (-scores[k], k))
        return {k: i for i, k in enumerate(order)}

    def _tau(a: dict[str, int], b: dict[str, int]) -> float:
        players = sorted(set(a) | set(b))
        pairs = list(itertools.combinations(players, 2))
        if not pairs:
            return 1.0
        concordant = discordant = 0
        for x, y in pairs:
            ax = a[x] - a[y]
            bx = b[x] - b[y]
            if ax * bx > 0:
                concordant += 1
            elif ax * bx < 0:
                discordant += 1
        return (concordant - discordant) / len(pairs)

    rs = [_ranks(r) for r in rankings]
    taus = [_tau(rs[i], rs[j]) for i in range(len(rs))
            for j in range(i + 1, len(rs))]
    return sum(taus) / len(taus)


def rank_order(scores: dict[str, float]) -> list[str]:
    """Descending score ranking with lexicographic tie-break."""
    return sorted(scores, key=lambda k: (-scores[k], k))
