"""GEODE stake sizing (v25 M256 cell 1) — the honesty condition.

The per-attestation bond that makes lying unprofitable: for a
measurement class with expected gain ``g`` from an admitted lie and
detection probability ``p``, the minimum bond is ``S = g / p`` (with
a safety margin): a detected lie costs the whole bond, so expected
loss ``p*S`` covers the expected gain ``g``. Deterministic, closed
form; a seeded simulation verifies the bound (the M184 harness
discipline: synthetic instrument, not a deployment claim).

The collusion side of the question (a lie needs n-k+1 colluding
attesters, each risking a bond) is the caller's composition of this
bound — the bound is per-attester, so collective lying costs at
least (n-k+1)*S.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MeasurementClass:
    """One measurement class: what a lie pays and how often it is
    caught."""
    name: str
    gain_from_lie: float     # expected per-attestation gain if the
                             # lie is ADMITTED
    detection_prob: float    # probability the lie is detected


def minimum_bond(cls: MeasurementClass,
                 safety_margin: float = 1.0) -> float:
    """S = (g / p) * margin — the minimum slashable bond."""
    if not 0.0 < cls.detection_prob <= 1.0:
        raise ValueError("detection_prob must lie in (0, 1]")
    if safety_margin <= 0.0:
        raise ValueError("safety_margin must be positive")
    return (max(0.0, cls.gain_from_lie) / cls.detection_prob
            * safety_margin)


def stake_schedule(classes: list[MeasurementClass],
                   safety_margin: float = 1.0) -> dict[str, float]:
    """name -> minimum bond, per measurement class."""
    return {c.name: minimum_bond(c, safety_margin) for c in classes}


def simulate_liar(cls: MeasurementClass, bond: float, rounds: int,
                  fee: float = 0.0, seed: int = 11) -> dict[str, Any]:
    """Seeded synthetic scenario: a verifier who lies every round.

    With the minimum bond in place, the liar's cumulative cash must
    end <= 0 (the honesty gate); an honest verifier earning ``fee``
    per round ends positive (the baseline). Deterministic (seeded)."""
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    rng = np.random.default_rng(seed)
    liar_cash = 0.0
    for _ in range(rounds):
        detected = rng.random() < cls.detection_prob
        if detected:
            liar_cash -= bond          # slashed
        else:
            liar_cash += cls.gain_from_lie
    honest_cash = fee * rounds
    return {"rounds": rounds, "liar_cash": round(liar_cash, 6),
            "honest_cash": round(honest_cash, 6),
            "liar_unprofitable": liar_cash <= 0.0}
