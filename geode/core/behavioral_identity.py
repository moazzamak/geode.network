"""M307 — behavioural artifact identity for finding A1 (26 Aug 2026).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M307
before any build. A1: verification requires handing weights to a
rotating set of pseudonymous reference executors, and hash dedup does
not see a one-bit-flip copy. Repairs (R-A1a + R-A1c):

- **Committed probe set.** At admission the contributor commits to a
  Merkle root over f(x) on a sealed probe set; a fresh beacon-seeded
  slice is revealed each epoch and must open against the root. The
  serving host can prove behaviour without distributing weights.
- **Locality checks.** Perturbed neighbours of probe points: a stored
  lookup table answers exact probes but not their neighbours; a real
  model answers both.
- **Behavioural dedup.** The registry key includes the response
  profile on the sealed reference set. Two artifacts agreeing above
  the registered threshold are the same artifact whatever their
  weight hashes.
"""
from __future__ import annotations

import hashlib
from typing import Any

DEDUP_AGREEMENT = 0.95      # R-A1c: profile agreement threshold
LOCALITY_SCALE = 1e-3       # perturbation size relative to unit norm


def _leaf(value: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + value).digest()


def merkle_root(values: list[bytes]) -> str:
    """The commitment: a SHA-256 Merkle root over the leaf hashes of
    the response values (one leaf per probe). Deterministic."""
    if not values:
        raise ValueError("the probe set cannot be empty")
    level = [_leaf(v) for v in values]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])     # the odd leaf pairs with itself
        level = [hashlib.sha256(b"\x01" + a + b).digest()
                 for a, b in zip(level[0::2], level[1::2])]
    return level[0].hex()


def probe_slice(beacon_seed: str, epoch: int, slice_size: int,
                total: int) -> list[int]:
    """R-A1a: the fresh per-epoch slice - a deterministic,
    beacon-seeded sample of probe indices. The seed postdates the
    commitment (M311: external beacon), so the contributor cannot
    grind which slice gets revealed."""
    if int(epoch) < 0:
        raise ValueError("epoch must be non-negative")
    if int(slice_size) > int(total):
        raise ValueError("slice cannot exceed the probe set")
    digest = hashlib.sha256(
        f"geode:probe-slice:{beacon_seed}:{epoch}".encode("utf-8")
    ).digest()
    out: list[int] = []
    counter = 0
    while len(out) < int(slice_size):
        h = hashlib.sha256(digest + counter.to_bytes(8, "big")).digest()
        index = int.from_bytes(h[:8], "big") % int(total)
        if index not in out:
            out.append(index)
        counter += 1
    return out


def verify_slice_answers(leaves: list[bytes], commitment: str,
                         slice_indices: list[int],
                         answers: list[bytes]) -> bool:
    """R-A1a: re-derive the Merkle root from the full leaf list, then
    check every revealed answer matches its leaf. Returns True iff
    both hold."""
    if merkle_root(leaves) != str(commitment):
        return False
    if len(slice_indices) != len(answers):
        return False
    for index, answer in zip(slice_indices, answers):
        if not 0 <= int(index) < len(leaves):
            return False
        if _leaf(answer) != leaves[int(index)]:
            return False
    return True


def locality_perturbations(probe: list[float], count: int,
                           scale: float = LOCALITY_SCALE,
                           seed: int = 0) -> list[list[float]]:
    """R-A1a: perturbed neighbours of a probe point (per-component
    relative jitter). A lookup table storing exact answers cannot
    serve these consistently; a real model can."""
    import numpy as np
    if int(count) < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(int(seed))
    base = np.asarray(probe, dtype=np.float64)
    return [list(base * (1.0 + scale * rng.standard_normal(base.shape)))
            for _ in range(int(count))]


def behavioural_dedup_key(profile: list[int]) -> str:
    """R-A1c: the behavioural signature - the hash of the response
    profile on the sealed reference set. The registry key carries it
    alongside the weight hash."""
    payload = b"".join(int(i).to_bytes(4, "big", signed=True)
                       for i in profile)
    return hashlib.sha256(b"geode:behaviour:" + payload).hexdigest()


def profile_agreement(profile_a: list[int], profile_b: list[int]
                      ) -> float:
    """R-A1c: fraction of reference-set responses that agree."""
    if len(profile_a) != len(profile_b) or not profile_a:
        raise ValueError("profiles must be non-empty and equal length")
    return sum(a == b for a, b in zip(profile_a, profile_b)) \
        / len(profile_a)


def same_artifact(profile_a: list[int], profile_b: list[int],
                  threshold: float = DEDUP_AGREEMENT) -> dict[str, Any]:
    """R-A1c: two artifacts whose behavioural profiles agree above
    the registered threshold are the same artifact for registration
    purposes, whatever their weight hashes."""
    agreement = profile_agreement(profile_a, profile_b)
    return {"same_artifact": agreement >= float(threshold),
            "agreement": agreement,
            "threshold": float(threshold)}
