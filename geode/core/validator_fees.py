"""M335 - the validator fee-schedule measurement (R-F7).

Registered in ``analysis/FEASIBILITY_THREAT_REVIEW_2026-08-28.md``
(F7, R-F7). The gap: the fee schedule (pre-launch gate #2) is
unset, and the two constraints on it conflict - R-A9a says
"validation is a service, not a yield source", while the payment
rule says validators must earn enough to participate. The moment
honest earnings exceed the amortized identity cost, identities are
cash-flow-positive again and a k-identity Sybil fleet becomes a
profitable business (A9). Pedigree gates raise the time cost; they
do not resolve the monetary sign.

The registered measurement: compute the minimum fee schedule at
which an honest validator's per-epoch return is positive, then the
fraction of that schedule a k-identity Sybil fleet recovers over
the activation horizon (the k cancels - the fraction is
per-identity). The two constraints are:

- honest-participation: fee > cost_per_challenge (the validator's
  own per-challenge cost);
- sybil-safety: fee * challenges_per_epoch * horizon <
  registration_fee (the fleet's per-identity recovery < 1).

Both can hold only when cost_per_challenge < registration_fee /
(challenges_per_epoch * horizon). The admissible fee window is
(break_even, ceiling]; the margin is ceiling / break_even. When no
window exists, the eligibility apparatus needs a stake-like
addition (a validator bond forfeitable on delisting - economic,
not identity) or an explicit acceptance of the residual.
"""
from __future__ import annotations

import math
from typing import Any


def break_even_fee(cost_per_challenge: float) -> float:
    """The minimum fee at which an honest validator's per-epoch
    return is positive (the participation constraint)."""
    cost = float(cost_per_challenge)
    if cost <= 0.0:
        raise ValueError("cost_per_challenge must be positive")
    return cost


def sybil_recovery_fraction(fee: float, challenges_per_epoch: float,
                            horizon_epochs: int,
                            registration_fee: float) -> float:
    """The per-identity recovery fraction: a k-identity fleet's
    gross earnings over the activation horizon, divided by its
    k registration fees (k cancels). >= 1 means identities are
    cash-flow-positive at this fee."""
    f = float(fee)
    if f < 0.0:
        raise ValueError("the fee must be non-negative")
    if float(challenges_per_epoch) < 0.0:
        raise ValueError("challenges_per_epoch must be non-negative")
    if int(horizon_epochs) <= 0:
        raise ValueError("the horizon must be positive")
    if float(registration_fee) <= 0.0:
        raise ValueError("the registration fee must be positive")
    return (f * float(challenges_per_epoch)
            * int(horizon_epochs)) / float(registration_fee)


def sybil_safety_ceiling(challenges_per_epoch: float,
                         horizon_epochs: int,
                         registration_fee: float) -> float:
    """The fee at which the Sybil recovery fraction reaches exactly
    1.0 - the upper end of the admissible window."""
    if float(challenges_per_epoch) <= 0.0:
        raise ValueError("challenges_per_epoch must be positive")
    if int(horizon_epochs) <= 0:
        raise ValueError("the horizon must be positive")
    if float(registration_fee) <= 0.0:
        raise ValueError("the registration fee must be positive")
    return float(registration_fee) / (float(challenges_per_epoch)
                                      * int(horizon_epochs))


def fee_schedule_verdict(cost_per_challenge: float,
                         challenges_per_epoch: float,
                         horizon_epochs: int,
                         registration_fee: float,
                         fee_ladder: list[float],
                         ) -> dict[str, Any]:
    """The registered measurement: the admissible window, the
    recovery fraction beside every fee on the ladder, and the
    verdict. When the window is empty the verdict requires the
    stake-like addition (or an accepted residual)."""
    be = break_even_fee(cost_per_challenge)
    ceiling = sybil_safety_ceiling(challenges_per_epoch,
                                   horizon_epochs, registration_fee)
    window_exists = be < ceiling
    table: dict[str, float] = {}
    for fee in fee_ladder:
        table[f"{fee}"] = sybil_recovery_fraction(
            fee, challenges_per_epoch, horizon_epochs,
            registration_fee)
    if window_exists:
        margin = ceiling / be
        admissible = [f for f in fee_ladder if be < f < ceiling]
        verdict = (
            "an admissible fee window exists "
            f"({be:.6g}, {ceiling:.6g}], margin {margin:.3g}x; "
            "fees inside the window pay validators to show up "
            "without making identities cash-flow-positive")
        if not admissible:
            verdict += ("; NO ladder point sits inside the window "
                        "- the schedule must add one before launch")
    else:
        margin = 0.0
        admissible = []
        verdict = (
            "no admissible fee window exists: every fee that pays "
            "an honest validator to show up makes a Sybil fleet "
            "cash-flow-positive - the eligibility apparatus needs "
            "a stake-like addition (a validator bond forfeitable "
            "on delisting) or an explicitly accepted residual")
    return {
        "break_even_fee": be,
        "sybil_safety_ceiling": ceiling,
        "window_exists": bool(window_exists),
        "margin_factor": margin,
        "admissible_ladder_points": admissible,
        "recovery_fractions": table,
        "verdict": verdict,
    }
