"""M368 (G28) — minimum executor pool size; the operative instrument
is published per artifact.

Registered 29 Aug 2026, before the build. G28's defect: "a corrupt
fraction of the pool raised to the executor count" assumes a large
pool. With k_e = 2 and a pool of 2-3 — the realistic case for a
niche artifact — collusion probability is 1 or near it, and the
guarantee is void.

The repair (registered in the review): register a minimum pool size
PI = 8 (giving C(8,2) = 28 samples and a corrupt-pair probability of
(c/8)^2). Below PI the artifact falls back to behavioural identity as
the operative mechanism, and its bond is sized on the weaker
instrument. The pool size per artifact is published so a buyer can
see which instrument is operative.

Design (registered before the build):

- MIN_EXECUTOR_POOL = 8.
- operative_instrument(pool_size) returns the mechanism: sampled
  executors at/above PI, behavioural-identity fallback below.
- corrupt_sample_probability(pool, corrupt, sample): the probability
  that every sampled executor is corrupt, C(corrupt, sample) /
  C(pool, sample) — exactly G28's "corrupt fraction raised to the
  executor count" but exact (hypergeometric), not approximate.
- The registry entry carries pool_size and the operative instrument,
  so a buyer reads which guarantee is live.

Gate (M368): below PI the operative mechanism falls back and the
registry shows it.
"""
from __future__ import annotations

import math


# The registered minimum executor pool size (G28's proposal).
MIN_EXECUTOR_POOL = 8


def operative_instrument(pool_size: int,
                         min_pool: int = MIN_EXECUTOR_POOL) -> dict:
    """Which mechanism actually protects a probe on this artifact.

    At/above the minimum pool the operative instrument is sampled
    reference executors. Below it, the artifact falls back to
    behavioural identity on a fixed probe set (the weaker
    instrument), and the registry MUST show the fallback so a buyer
    can price it."""
    if int(pool_size) < 0:
        raise ValueError("pool_size must be non-negative")
    if int(pool_size) >= int(min_pool):
        return {"instrument": "sampled_executors",
                "pool_size": int(pool_size),
                "fallback": False}
    return {"instrument": "behavioral_identity_fallback",
            "pool_size": int(pool_size),
            "fallback": True}


def corrupt_sample_probability(pool_size: int, corrupt: int,
                               sample: int) -> float:
    """Exact probability that every sampled executor is corrupt:
    C(corrupt, sample) / C(pool, sample) — the hypergeometric form of
    G28's "corrupt fraction raised to the executor count". 0 when the
    sample exceeds the pool; 1 when the corrupt set covers the pool."""
    n, c, k = int(pool_size), int(corrupt), int(sample)
    if n < 0 or c < 0 or k < 0:
        raise ValueError("pool, corrupt, and sample must be "
                         "non-negative")
    if c > n or k > n:
        raise ValueError("corrupt and sample cannot exceed the pool")
    if c < k:
        return 0.0
    if k == 0:
        return 1.0
    return math.comb(c, k) / math.comb(n, k)


def registry_entry(artifact_id: str, pool_size: int,
                   min_pool: int = MIN_EXECUTOR_POOL) -> dict:
    """The published per-artifact registry row: the pool size and the
    operative instrument, so a buyer never has to infer which
    guarantee is live."""
    operative = operative_instrument(pool_size, min_pool)
    return {"artifact_id": str(artifact_id),
            "executor_pool_size": int(pool_size),
            "operative_instrument": operative["instrument"],
            "fallback_active": operative["fallback"]}
