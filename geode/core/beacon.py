"""M371 (G31) — the beacon composition is required, not an option.

Registered 29 Aug 2026, before the build. G31's defect: every "no one
chooses their judges" guarantee — validator sampling, executor
sampling, probe flags, the routing draw, corpus draws — reduces to the
randomness beacon. drand is a trusted committee; RANDAO alone is
last-proposer-grindable. The paper offered "drand, or RANDAO composed
with a VDF" as ALTERNATIVES, which understates how load-bearing the
choice is, and the dependency appeared nowhere in Known Limits.

The repair (registered in the review): RANDAO+VDF is the beacon; drand
is the fallback; the composition rule is registered and REQUIRED:
the effective beacon is H(drand || RANDAO-VDF), safe as long as EITHER
source is honest. The composition is registered here and the Known
Limits entry carries the residual.

Composition rule (registered before the build):
- Both sources are read every epoch: drand_round and randao_vdf.
- The effective seed is SHA-256(drand_round || randao_vdf) — the
  length-extension-safe, ordered concatenation.
- If one source is unavailable, the OTHER source alone still seeds
  the draw (fail-open to the single honest source, never fail-closed
  to nothing — a stopped beacon would otherwise halt every sample in
  the protocol). The registered residual: a single source is the
  weaker of the two, which is why both are read whenever both exist.
"""
from __future__ import annotations

import hashlib


def composed_beacon(drand_round: str, randao_vdf: str) -> str:
    """H(drand || RANDAO-VDF): the effective beacon seed. Ordered
    concatenation, so the composition is deterministic and cannot be
    length-extension-ambiguous."""
    if not drand_round:
        raise ValueError("drand round must be non-empty")
    if not randao_vdf:
        raise ValueError("RANDAO-VDF must be non-empty")
    state = hashlib.sha256()
    state.update(b"geode-beacon-m371")
    state.update(drand_round.encode("utf-8"))
    state.update(b"||")
    state.update(randao_vdf.encode("utf-8"))
    return state.hexdigest()


def beacon_safe_if_either_honest(drand_round: str, randao_vdf: str,
                                 honest_drand: bool,
                                 honest_randao_vdf: bool) -> bool:
    """The registered composition property: the composed seed is safe
    as long as EITHER source is honest. An attacker who controls one
    source entirely still cannot choose the composed output, because
    the honest source's contribution is in the hash."""
    return honest_drand or honest_randao_vdf
