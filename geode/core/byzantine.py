"""GEODE Byzantine measurement aggregation (v25 M245).

The empirical-facts layer admits a measurement only with k-of-n
independent verifier attestation, and vector measurements aggregate by
the elementwise MEDIAN — a minority of Byzantine verifiers cannot move
the admitted value. This is the Byzantine backbone under M242 (the
drift gate) and M244 (demerit attestation).

Determinism by construction: no RNG anywhere. Even-n medians take the
LOWER middle element (the conservative choice, registered); ties
resolve in input order. No wall clocks — freshness is expressed in
ledger indices, never timestamps.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def median_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Elementwise median across verifier vectors.

    Even n takes the LOWER middle value (conservative: the median
    can never be moved UP by a pair of colluding high reports). Empty
    input raises ValueError; ragged vectors raise ValueError.
    """
    if not vectors:
        raise ValueError("median_vector requires at least one vector")
    n = len(vectors)
    dim = len(vectors[0])
    for i, v in enumerate(vectors):
        if len(v) != dim:
            raise ValueError(f"vector {i} has length {len(v)}, "
                             f"expected {dim}")
    out = []
    for d in range(dim):
        col = sorted(v[d] for v in vectors)
        # even n: lower middle -> floor((n-1)/2) of the sorted column
        out.append(col[(n - 1) // 2])
    return out


def quorum(attestations: dict[str, frozenset[str]],
           k_of_n: int) -> dict[str, dict[str, Any]]:
    """Which facts clear the k-of-n independent-attestation bar.

    ``attestations`` maps a fact key to the set of verifier ids that
    attested it. A fact is ADMITTED iff at least ``k_of_n`` distinct
    verifiers attested it. Deterministic output order follows the
    insertion order of the input dict.
    """
    out: dict[str, dict[str, Any]] = {}
    for fact, verifiers in attestations.items():
        count = len(verifiers)
        out[fact] = {
            "count": count,
            "quorum": count >= k_of_n,
            "verifiers": sorted(verifiers),
        }
    return out


def admitted_facts(attestations: dict[str, frozenset[str]],
                   k_of_n: int) -> list[str]:
    """The admitted fact keys (insertion order), quorum only."""
    return [f for f, r in quorum(attestations, k_of_n).items()
            if r["quorum"]]
