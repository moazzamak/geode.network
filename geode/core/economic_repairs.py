"""M313 — economic repairs for the protocol attack surface A11 and
A20 (26 Aug 2026).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M313
before any build. Four repairs, one module:

- **R-A11a per-axis bond.** Sized to the compute saving the axis
  makes available over the open-exposure window — the quantity
  actually being arbitraged. Forfeitable on conviction. Bonds are
  economic, so the no-identity design principle is untouched.
- **R-A11b claim delay.** Claims on credits earned under open probe
  exposure are delayed until the exposure drains, so vested credits
  remain reachable while detection is pending (the reachability L3
  was missing).
- **R-A11c honest L3.** L0 warning, L1 freeze claims, L2 delist,
  L3 delist + burn vested-but-unclaimed credits. Reachable because
  of the claim delay.
- **R-A20 verified-work-only tenure.** Tenure and activity credit
  accrue only from sampled, verified work. Self-generated volume —
  any address, including the wash ring — accrues zero. This retires
  the payout-address self-payment exclusion, which A20 showed is a
  speed bump against accidents.
"""
from __future__ import annotations

import math
from typing import Any

# Activity record sources that accrue tenure credit. Anything else —
# self-generated volume under any address — accrues zero.
VERIFIED_SOURCES = frozenset(("sampled_challenge", "probe_reference"))

# The registered slash ladder, redescribed honestly (R-A11c).
SLASH_LADDER = ("L0 warning", "L1 freeze claims", "L2 delist",
                "L3 delist + burn vested-unclaimed")


def per_axis_bond(saving_per_unit: float, exposure_units: float
                  ) -> float:
    """R-A11a: the per-axis bond, sized to the compute saving the
    axis makes available over the open-exposure window — the quantity
    actually being arbitraged. Forfeitable on conviction.

    ``exposure_units`` is the open-exposure window in units of work
    (the registered detection horizon is 1/(rho*delta) units; the
    frozen window from R-A11b is its lower bound)."""
    saving = float(saving_per_unit)
    units = float(exposure_units)
    if saving < 0.0:
        raise ValueError("saving_per_unit must be non-negative")
    if units <= 0.0:
        raise ValueError("exposure_units must be positive")
    return saving * units


def claim_delay_epochs(open_exposure_units: float,
                       units_per_epoch: float) -> int:
    """R-A11b: claims on credits earned under open probe exposure are
    delayed until the exposure drains. Returns the delay in whole
    epochs (ceil), so vested credits stay reachable while detection
    is pending."""
    exposure = float(open_exposure_units)
    rate = float(units_per_epoch)
    if exposure < 0.0:
        raise ValueError("open_exposure_units must be non-negative")
    if rate <= 0.0:
        raise ValueError("units_per_epoch must be positive")
    if exposure == 0.0:
        return 0
    return int(math.ceil(exposure / rate))


def conviction_burn(vested: float, claimed: float) -> float:
    """R-A11c: L3 burns vested-but-unclaimed credits only. Reachable
    because R-A11b froze claims while exposure was open: at conviction
    at least the frozen window's accrual is unclaimed. Returns the
    burn amount; never negative, never above vested."""
    v = float(vested)
    c = float(claimed)
    if v < 0.0 or c < 0.0:
        raise ValueError("vested and claimed must be non-negative")
    if c > v:
        raise ValueError("claimed cannot exceed vested")
    return max(0.0, v - c)


def slash_decision(level: int, vested: float, claimed: float
                   ) -> dict[str, Any]:
    """Apply the redescribed ladder. Returns the graded decision and
    (for L3) the burn amount. ``level`` is 0-3."""
    if not 0 <= int(level) <= 3:
        raise ValueError("level must be in 0..3")
    if level < 3:
        return {"level": int(level), "description": SLASH_LADDER[level],
                "burn": 0.0, "delist": level >= 2,
                "freeze_claims": level >= 1}
    return {"level": 3, "description": SLASH_LADDER[3],
            "burn": conviction_burn(vested, claimed),
            "delist": True, "freeze_claims": True}


def verified_activity(records: list[dict[str, Any]]
                      ) -> list[dict[str, Any]]:
    """R-A20: keep only sampled, verified work. ``records`` is a list
    of activity records with a ``source`` field; ``sampled_challenge``
    (a challenge the beacon sampled and the contributor answered
    correctly) and ``probe_reference`` (a reference run on a probed
    session initiated by another party) accrue credit. Everything
    else — self-generated volume under any address — accrues zero."""
    kept: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("source")) in VERIFIED_SOURCES:
            kept.append(record)
    return kept


def tenure_weight(records: list[dict[str, Any]],
                  weight_field: str = "weight") -> float:
    """R-A20: tenure weight over verified activity only. Each
    verified record contributes its ``weight`` (default 1.0)."""
    total = 0.0
    for record in verified_activity(records):
        weight = float(record.get(weight_field, 1.0))
        if weight < 0.0:
            raise ValueError("record weights must be non-negative")
        total += weight
    return total


# M338 (F10-2): the bond's input is the substitute's compute saving -
# the contributor's private information. The registry must estimate
# it from public quantities. The registered estimator: the saving is
# at most the axis unit price minus the reference hosting cost (the
# substitute charges the posted price and pays its own hosting), and
# the reference cost is the developer's posted ladder, public by
# construction. The conservatism factor covers a substitute more
# efficient than the reference (its saving is LARGER than the
# estimate): the registered factor over-states the saving, which
# over-sizes the bond - the defense-pessimistic direction.
BOND_SAVING_CONSERVATISM = 2.0


def saving_estimate(unit_price: float, reference_hosting_cost: float,
                    conservatism: float = BOND_SAVING_CONSERVATISM
                    ) -> float:
    """M338 (F10-2): the per-unit saving estimate from public
    quantities. ``unit_price`` is the axis's posted price;
    ``reference_hosting_cost`` is the developer's posted reference
    hosting cost per unit (public). The estimate is
    conservatism * max(0, price - cost): a substitute cannot save
    more than the price, and a substitute cheaper than the
    reference is covered by the factor."""
    price = float(unit_price)
    cost = float(reference_hosting_cost)
    if price < 0.0 or cost < 0.0:
        raise ValueError("price and cost must be non-negative")
    if conservatism < 1.0:
        raise ValueError("the conservatism factor must be >= 1")
    return conservatism * max(0.0, price - cost)


def bond_from_publics(unit_price: float, reference_hosting_cost: float,
                      exposure_units: float,
                      conservatism: float = BOND_SAVING_CONSERVATISM
                      ) -> float:
    """M338 (F10-2): the per-axis bond sized from public quantities
    only (the R-A11a form with the registered estimator)."""
    return per_axis_bond(
        saving_estimate(unit_price, reference_hosting_cost,
                        conservatism),
        exposure_units)
