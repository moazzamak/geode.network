"""GEODE core economics — the whitepaper-aligned constants and rules
(24 Aug 2026). Single source of truth for the Python side; the EVM
contracts mirror these values (``infrastructure/evm/contracts/``).

Registered rules (``docs/WHITEPAPER_GEODE.tex``,
``archive/research_2026-08-28/analysis/GEODE_ECONOMIC_DESIGN_v1.md``):

- one registration form for arms and primitives: operator key +
  payout address (may differ) + price per unit of work + sealed
  claim; a primitive's royalty is simply its payout address;
- the unit of work is DERIVED from the (input type, output type)
  pair by a registered table — never chosen per registration;
- session fields: max unit price (routing filter) and max spend
  (total charge cap; serving stops when the remaining cap is less
  than one unit);
- self-payment exclusion keys on the PAYOUT address (C1);
- vesting: linear over N = 4 epochs, pull claims;
- slash = burn, graded L0-L3, replay-gated.

Nothing here authorizes deployment; all parameters are the
registered defaults and carry their adjustment path in the plan.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

# -- registered constants (mirror CreditLedger.sol) --------------------
DEV_FUND_BPS = 25        # 2.5% of 1000
EPOCH_SECONDS = 7 * 24 * 3600
VESTING_EPOCHS = 4       # N=4: first tranche after epoch 1
MAX_BATCH = 64           # mirrors CreditLedger.MAX_BATCH
CHANGE_DELAY_SECONDS = 2 * 24 * 3600        # admin timelocks
PRICE_CHANGE_DELAY_SECONDS = EPOCH_SECONDS  # 1-epoch notice period
SHADOW_PROBE_RATE = 0.05                     # rho: serve-and-compare

# M360 (G21): the registered per-epoch price drift band. A contributor
# may register price as a FIXED amount (today's rule, band = 0) or as
# a fixed amount with this per-epoch drift band, so it can track a
# large move without a governance action per epoch. The band is a CAP
# on the per-epoch change, never a forced change. The router replays
# against the day's effective price either way.
PRICE_DRIFT_BAND_BPS = 1000   # +/-10% per epoch, registered proposal

# M361 (G22): the reference hosting cost becomes a measured,
# multi-party statistic. The old rule let the developer's single
# posted figure simultaneously set the price floor, the developer's
# own bootstrap price, and every competitor's bond. The repair: the
# reference cost is the MEDIAN of the posted hosting costs of all
# admitted arms on the axis with at least one epoch of verified
# traffic, floored at the developer's figure only while the axis has
# fewer than this many such arms.
REFERENCE_COST_MIN_ARMS = 3

# -- security floors (v26 M314: outside ordinary governance) ------------
# The registered defaults ARE the floors. A timelocked adjustment may
# raise a value with notice; it may never lower one. The floors sit
# outside ordinary governance alongside the zakat rule. The EVM mirror
# of these floors belongs to the contract-stack audit, which is out of
# scope for v26 and registered as such.
SECURITY_FLOORS: dict[str, float] = {
    "shadow_probe_rate": SHADOW_PROBE_RATE,
    "vesting_epochs": VESTING_EPOCHS,
    "admission_validator_sample": 3,
    "reference_executor_sample": 2,
    "audit_fraction": 0.1,
    # v26 M315 (registered before build): the takedown quorum floor
    "takedown_min_responders": 3,
}


def assert_at_or_above_floor(name: str, value: float) -> float:
    """Raise ValueError if ``value`` would take a security parameter
    below its registered floor; return the value otherwise. Every
    governance/timelock path that adjusts a floored parameter must
    call this before applying the change."""
    floor = SECURITY_FLOORS.get(str(name))
    if floor is None:
        raise KeyError(f"unregistered security parameter {name!r}: "
                       "a floored parameter must appear in "
                       "SECURITY_FLOORS before any adjustment path "
                       "may touch it")
    value_f = float(value)
    if not value_f >= floor:
        raise ValueError(
            f"{name}={value_f} is below the registered floor "
            f"{floor}; security floors sit outside ordinary "
            "governance and cannot be lowered")
    return value_f

# -- the unified registration form --------------------------------------
REGISTRATION_FIELDS = ("operator_key", "payout_address",
                       "price_per_unit", "sealed_claim")

# -- the unit-of-work pair table ---------------------------------------
# Whitepaper rows: (input type, output type) -> unit. First rule that
# applies wins; a pair without a row is not an admissible task until
# the table is extended (a registered rule change).
UNIT_TABLE: dict[tuple[str, str], str] = {
    ("image", "class label"): "query",
    ("text", "class label"): "query",
    ("number series", "number"): "query",
    ("audio", "transcript"): "audio second",
    ("text", "transcript"): "token",
    ("text", "audio"): "audio second",
}
PRIMITIVE_UNIT = "execution attempt"


def unit_of_work(input_type: str, output_type: str, kind: str) -> str:
    """The pricing denominator, derived from the task's shape.

    ``kind`` is ``"primitive"`` or ``"arm"``. Raises ValueError for a
    pair without a registered row — a different unit is a different
    task, and extending the table is a registered rule change."""
    if kind == "primitive":
        return PRIMITIVE_UNIT
    unit = UNIT_TABLE.get((str(input_type), str(output_type)))
    if unit is None:
        raise ValueError(
            f"no registered unit of work for "
            f"({input_type!r}, {output_type!r}): the pair table must "
            "be extended by a registered rule change before the task "
            "is admissible")
    return unit


# -- addresses and identifiers -----------------------------------------
def address_of(name: str) -> str:
    """Deterministic 20-byte address from a name (no identity)."""
    digest = hashlib.sha256(f"geode:{name}".encode("utf-8")).hexdigest()
    return "0x" + digest[:40]


def artifact_id_of(arm_id: str) -> str:
    """Deterministic 32-byte artifact id for a registration."""
    digest = hashlib.sha256(
        f"geode:artifact:{arm_id}".encode("utf-8")).hexdigest()
    return "0x" + digest[:64]


def deposit_split(amount: int) -> tuple[int, int]:
    """The contract's own deposit arithmetic: 2.5% dev cut first,
    the rest to the attribution pool (floor division, as in
    Solidity)."""
    dev = amount * DEV_FUND_BPS // 1000
    return amount - dev, dev


# -- session budget rules ----------------------------------------------
def served_units(max_spend: int, price_per_unit: int) -> int:
    """Units affordable under a total spend cap (floor division); the
    session stops when the remaining cap is less than one unit."""
    if price_per_unit <= 0:
        raise ValueError("price_per_unit must be positive")
    if max_spend < 0:
        raise ValueError("max_spend must be non-negative")
    return max_spend // price_per_unit


def within_cap(units: int, price_per_unit: int, max_spend: int) -> bool:
    """True iff serving ``units`` units stays inside the cap (an
    optional cap of 0 means unlimited)."""
    if max_spend <= 0:
        return True
    return units * price_per_unit <= max_spend


# M360 (G21): the per-epoch price drift band ----------------------------
def price_within_drift(previous_price: int, proposed_price: int,
                       band_bps: int = PRICE_DRIFT_BAND_BPS) -> bool:
    """True iff ``proposed_price`` is inside the registered drift band
    of ``previous_price``: |proposed - previous| <= previous * band.
    A band of 0 is today's fixed-price rule (only an equal price is
    inside the band)."""
    if previous_price <= 0:
        raise ValueError("previous price must be positive")
    if proposed_price <= 0:
        raise ValueError("proposed price must be positive")
    if band_bps < 0:
        raise ValueError("drift band must be non-negative")
    limit = previous_price * band_bps // 10000
    return abs(proposed_price - previous_price) <= limit


def clamp_to_drift(previous_price: int, proposed_price: int,
                   band_bps: int = PRICE_DRIFT_BAND_BPS) -> int:
    """Clamp ``proposed_price`` into the drift band of the previous
    epoch's price. The router replays against the clamped (effective)
    price of the day; the contributor's declared price may exceed the
    band, but the effective price never moves more than the band per
    epoch."""
    if previous_price <= 0:
        raise ValueError("previous price must be positive")
    if proposed_price <= 0:
        raise ValueError("proposed price must be positive")
    if band_bps < 0:
        raise ValueError("drift band must be non-negative")
    limit = previous_price * band_bps // 10000
    lo = previous_price - limit
    hi = previous_price + limit
    return max(lo, min(hi, proposed_price))


def effective_price_path(base_price: int, declarations: list[int],
                         band_bps: int = PRICE_DRIFT_BAND_BPS
                         ) -> list[int]:
    """The deterministic, replayable price table of the day: the
    effective price at each epoch, built by clamping the contributor's
    per-epoch declarations into the drift band of the previous
    effective price. ``declarations`` are the declared prices for
    epochs 1..n (index 0 is the first epoch after the base). With no
    declaration for an epoch, the previous effective price carries
    over. The router replays THIS path, never a live feed. With band 0
    this is the base price at every epoch (today's rule)."""
    if base_price <= 0:
        raise ValueError("base price must be positive")
    if band_bps < 0:
        raise ValueError("drift band must be non-negative")
    path = [int(base_price)]
    prev = int(base_price)
    for declared in declarations:
        if declared is None:
            path.append(prev)
            continue
        if declared <= 0:
            raise ValueError("a declared price must be positive")
        eff = clamp_to_drift(prev, int(declared), band_bps)
        path.append(eff)
        prev = eff
    return path


# -- M361 (G22): the multi-party reference hosting cost ------------------
def reference_hosting_cost(admitted_costs: list[float],
                           developer_cost: float,
                           min_arms: int = REFERENCE_COST_MIN_ARMS
                           ) -> dict:
    """M361 (G22): the reference hosting cost as a measured,
    multi-party statistic.

    ``admitted_costs`` are the posted hosting costs of all admitted
    arms on the axis with at least one epoch of verified traffic.
    ``developer_cost`` is the developer's posted figure, which is the
    floor only while the axis has fewer than ``min_arms`` such arms.

    Returns the reference cost, which rule produced it (median /
    developer floor), and the count — so the registry can publish
    which instrument is operative, exactly as the <3-arm fallback
    requires."""
    costs = [float(c) for c in admitted_costs]
    if any(c < 0.0 for c in costs):
        raise ValueError("admitted costs must be non-negative")
    if developer_cost < 0.0:
        raise ValueError("developer cost must be non-negative")
    if min_arms < 1:
        raise ValueError("min_arms must be >= 1")
    if len(costs) < int(min_arms):
        return {"reference": float(developer_cost),
                "rule": "developer floor (<min_arms)",
                "admitted_arms": len(costs)}
    ordered = sorted(costs)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else \
        (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    return {"reference": float(median), "rule": "median of admitted",
            "admitted_arms": len(costs)}


# -- the slash ladder (graded, burn, replay-gated) ----------------------
SLASH_LADDER: dict[int, str] = {
    0: "underperformance or downtime: no slash (the market penalizes "
       "through reduced attribution)",
    1: "provably fraudulent output or deviation from the sealed "
       "artifact: burn the unvested promise remainder",
    2: "provable adversarial attack: burn the full promise and delist "
       "the artifact",
    3: "coordinated attack: delist and burn vested credits by "
       "replay-gated dispute",
}


def registration_record(arm_id: str, payout_address: str,
                        operator_key: str, price_per_unit: int,
                        sealed_claim: str) -> dict[str, Any]:
    """The unified registration form as a plain record."""
    return {
        "arm_id": str(arm_id),
        "artifact_id": artifact_id_of(str(arm_id)),
        "operator_key": str(operator_key),
        "payout_address": str(payout_address),
        "price_per_unit": int(price_per_unit),
        "sealed_claim": str(sealed_claim),
    }


# -- M304: expected units on the reference workload --------------------
# At admission the artifact's expected unit count ubar on a registered
# reference workload is MEASURED and sealed; routing ranks on
# s/(p*ubar) (M303). Re-measured on re-registration. The live meter
# drift (observed mean units / ubar) is a ledger-visible statistic
# with a registered deviation band.
DRIFT_BAND = (0.5, 2.0)   # registered: observed/ubar must stay inside


def reference_workload_units(units_per_query: list[float]) -> float:
    """The expected unit count on the reference workload: the mean of
    the per-query unit counts. Deterministic; no seed, no sampling."""
    if not units_per_query:
        raise ValueError("the reference workload must be non-empty")
    if any(float(v) <= 0.0 for v in units_per_query):
        raise ValueError("unit counts must be positive")
    return float(np.mean([float(v) for v in units_per_query]))


def meter_drift(observed_mean_units: float, ubar: float) -> float:
    """Live meter drift: observed mean units / sealed ubar. A value
    far above 1 is the bloat signature the routing score penalises;
    far below 1 is under-serving. Both are reported, never hidden."""
    if float(ubar) <= 0.0:
        raise ValueError("ubar must be positive")
    return float(observed_mean_units) / float(ubar)


def drift_in_band(drift: float, band: tuple[float, float]
                  = DRIFT_BAND) -> bool:
    """True iff the drift sits inside the registered band."""
    lo, hi = float(band[0]), float(band[1])
    if not 0.0 < lo <= hi:
        raise ValueError("the drift band must be positive and ordered")
    return lo <= float(drift) <= hi
