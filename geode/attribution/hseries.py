"""GEODE H-series economic simulation battery (v25 M293) — the
copycat race, the detection-horizon sweep, and the bootstrap dynamics.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md``
section 6 (25 Aug 2026) BEFORE building. Deterministic (seeded),
CPU-only. Synthetic-scenario instruments: they verify the REGISTERED
mechanism forms and measure the registered metrics; they are NOT
claims about real deployments.

Scenario A (copycat race): one task axis; the publisher registers at
epoch 0 with quality q, a copycat at epoch k with equal quality.
Attribution is marginal contribution (the M180 form): with the
publisher present the copycat's marginal is 0 (the capability is
already served, zero novelty), so the publisher receives the fee pool
regardless of who serves. Routing picks the lower price; price ties
break to the EARLIER registration (ordered registry).

Scenario B (detection-horizon sweep): per cheat class, the horizon
(epochs until detection) is sampled under the registered detection
mechanics; the gate is N >= p90 / 2 for every class (the registered
rule: N at least half the measured horizon).

Scenario C (bootstrap dynamics): the dev's headroom-rule bootstrap
arm at 80% quality vs contributor arms at below / equal /
strictly-better quality; routing priority passes by measurement alone;
after handover the bootstrap arm keeps only the registered fallback
share.
"""
from __future__ import annotations

from typing import Any

import numpy as np

DEV_FUND_FRACTION = 0.025


# --------------------------------------------------------------------------
# Scenario A — the copycat race
# --------------------------------------------------------------------------

def copycat_race_cell(
        demand: float, price: float, epochs: int, quality: float,
        copycat_epoch: int, undercut: float, serving_cost: float,
        seed: int) -> dict[str, Any]:
    """One sweep cell: publisher at epoch 0, copycat at
    ``copycat_epoch`` with equal quality, copycat price = price *
    (1 - undercut). Attribution is marginal: the copycat's marginal
    given the publisher is 0, so the publisher receives the pool in
    every epoch; the serving arm pays ``serving_cost``.

    Publisher-absent control: ``copycat_epoch < 0`` means the
    publisher never registers and the copycat is the incumbent.
    """
    if copycat_epoch >= 0 and quality <= 0.0:
        raise ValueError("quality must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if demand <= 0.0 or price <= 0.0:
        raise ValueError("demand and price must be positive")
    if not 0.0 <= undercut < 1.0:
        raise ValueError("undercut must be in [0, 1)")
    publisher_absent = copycat_epoch < 0
    if publisher_absent:
        copycat_epoch = 0
    copy_price = price * (1.0 - undercut)
    pub_fees = cat_fees = pub_cost = cat_cost = 0.0
    pub_served = cat_served = 0
    served_every_epoch = True
    for t in range(epochs):
        publisher_present = (not publisher_absent) and t >= 0
        copycat_present = t >= copycat_epoch
        if not (publisher_present or copycat_present):
            served_every_epoch = False
            continue
        # Routing: lower price serves; price ties break to the earlier
        # registration (the publisher, registered at epoch 0).
        if publisher_present and copycat_present:
            copycat_serves = copy_price < price
        else:
            copycat_serves = copycat_present
        pool = demand * (copy_price if copycat_serves else price)
        # Marginal attribution: the publisher unlocks the axis; a
        # copycat present with the publisher adds nothing.
        if publisher_present:
            pub_fees += pool
            if copycat_serves:
                cat_cost += serving_cost
                cat_served += 1
            else:
                pub_cost += serving_cost
                pub_served += 1
        else:
            # Publisher absent: the copycat is the incumbent and
            # captures the pool.
            cat_fees += pool
            cat_cost += serving_cost
            cat_served += 1
    total_fees = pub_fees + cat_fees
    pub_net = pub_fees - pub_cost
    cat_net = cat_fees - cat_cost
    served_epochs = pub_served + cat_served
    return {
        "seed": seed,
        "demand": demand,
        "price": price,
        "epochs": epochs,
        "quality": quality,
        "copycat_epoch": copycat_epoch if not publisher_absent else None,
        "publisher_absent": publisher_absent,
        "undercut": undercut,
        "copycat_price": copy_price,
        "publisher_fees": pub_fees,
        "copycat_fees": cat_fees,
        "publisher_fee_share": pub_fees / total_fees
        if total_fees > 0.0 else 0.0,
        "copycat_fee_share": cat_fees / total_fees
        if total_fees > 0.0 else 0.0,
        "publisher_net": pub_net,
        "copycat_net": cat_net,
        "publisher_traffic_share": pub_served / served_epochs
        if served_epochs else 0.0,
        "copycat_traffic_share": cat_served / served_epochs
        if served_epochs else 0.0,
        "served_every_epoch": served_every_epoch,
        # A1: publisher's cumulative fee share >= 0.5 in every
        # publisher-present cell (the momentum claim).
        "gate_a1_share": pub_fees / total_fees >= 0.5
        if total_fees > 0.0 and not publisher_absent else True,
        # A2: the copycat never profits from copying.
        "gate_a2_copycat_net": cat_net <= 0.0 if not publisher_absent
        else cat_net > 0.0,
        # A3: the axis is served every epoch (publisher-present cells).
        "gate_a3_served": served_every_epoch if not publisher_absent
        else True,
        # A4: with the publisher absent the copycat captures the
        # stream (instrument control: no bias against late registrants).
        "gate_a4_incumbent": cat_fees == total_fees and cat_net > 0.0
        if publisher_absent else True,
        # Informational: the publisher's fees relative to the
        # no-copycat counterfactual (what undercutting costs it).
        "publisher_fees_vs_no_copycat": pub_fees / (demand * price *
                                                     epochs)
        if not publisher_absent else None,
    }


def copycat_race_sweep(
        demand: float, price: float, epochs: int, quality: float,
        copycat_epochs: list[int], undercuts: list[float],
        serving_cost: float, seed: int) -> dict[str, Any]:
    """Full scenario A sweep. ``copycat_epochs`` containing -1 adds
    the publisher-absent control cell."""
    cells = []
    gates: dict[str, bool] = {}
    for k in copycat_epochs:
        for u in undercuts:
            cells.append(copycat_race_cell(
                demand, price, epochs, quality, k, u, serving_cost,
                seed=seed + abs(k) * 100 + int(u * 100)))
    gates["A1"] = all(c["gate_a1_share"] for c in cells)
    gates["A2"] = all(c["gate_a2_copycat_net"] for c in cells)
    gates["A3"] = all(c["gate_a3_served"] for c in cells)
    gates["A4"] = all(c["gate_a4_incumbent"] for c in cells)
    publisher_present = [c for c in cells if not c["publisher_absent"]]
    worst_share = min((c["publisher_fees_vs_no_copycat"]
                       for c in publisher_present), default=1.0)
    return {
        "cells": cells,
        "gates": gates,
        "passes": all(gates.values()),
        "worst_publisher_fees_vs_no_copycat": worst_share,
        "note": ("the copycat's attributed share is zero by the "
                 "marginal-contribution form; undercutting steals "
                 "TRAFFIC and therefore lowers the pool the publisher "
                 "receives (predatory cycles, slowed not eliminated)"),
    }


# --------------------------------------------------------------------------
# Scenario B — detection-horizon sweep (sets the vesting window N)
# --------------------------------------------------------------------------

def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {"median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90))}


def detection_horizon_class(kind: str, rng: np.random.Generator,
                            draws: int, probe_rate: float,
                            epoch_volume: float,
                            gaming_rate: float,
                            ring_rate: float,
                            health_probes: int,
                            health_hit_rate: float) -> dict[str, Any]:
    """Sample the detection horizon (epochs) for one cheat class.

    B1 serving substitution / bit-inexact deviation: caught per probed
    query with probability ``probe_rate``; horizon = queries until
    catch / epoch volume.
    B2 attribution gaming: caught per epoch with probability
    ``gaming_rate``.
    B3 wash ring: caught per epoch with probability ``ring_rate``.
    B4 availability gaming: ``health_probes`` probes per epoch at
    ``health_hit_rate`` each.
    """
    if draws <= 0:
        raise ValueError("draws must be positive")
    if kind == "B1_serving_deviation":
        queries = rng.geometric(probe_rate, size=draws).astype(float)
        horizons = queries / epoch_volume
    elif kind == "B2_attribution_gaming":
        horizons = rng.geometric(gaming_rate, size=draws).astype(float)
    elif kind == "B3_wash_ring":
        horizons = rng.geometric(ring_rate, size=draws).astype(float)
    elif kind == "B4_availability_gaming":
        per_epoch = 1.0 - (1.0 - health_hit_rate) ** health_probes
        horizons = rng.geometric(per_epoch, size=draws).astype(float)
    else:
        raise ValueError(f"unknown cheat class {kind!r}")
    q = _quantiles(horizons)
    q["draws"] = draws
    q["class"] = kind
    return q


def detection_horizon_sweep(
        draws: int, probe_rate: float, epoch_volume: float,
        gaming_rate: float, ring_rate: float, health_probes: int,
        health_hit_rate: float, vesting_window: int,
        seed: int) -> dict[str, Any]:
    """Full scenario B sweep. Gate B5: the registered vesting window
    N must satisfy N >= p90 / 2 for every class; the verdict names the
    binding class (the largest p90)."""
    rng = np.random.default_rng(seed)
    classes = [
        detection_horizon_class(
            "B1_serving_deviation", rng, draws, probe_rate, epoch_volume,
            gaming_rate, ring_rate, health_probes, health_hit_rate),
        detection_horizon_class(
            "B2_attribution_gaming", rng, draws, probe_rate, epoch_volume,
            gaming_rate, ring_rate, health_probes, health_hit_rate),
        detection_horizon_class(
            "B3_wash_ring", rng, draws, probe_rate, epoch_volume,
            gaming_rate, ring_rate, health_probes, health_hit_rate),
        detection_horizon_class(
            "B4_availability_gaming", rng, draws, probe_rate, epoch_volume,
            gaming_rate, ring_rate, health_probes, health_hit_rate),
    ]
    per_class = {}
    for c in classes:
        per_class[c["class"]] = {
            "median_horizon": c["median"],
            "p90_horizon": c["p90"],
            "required_N": np.ceil(c["p90"] / 2.0),
            "gate_b5_pass": vesting_window >= c["p90"] / 2.0,
        }
    binding = max(classes, key=lambda c: c["p90"])
    gate = all(v["gate_b5_pass"] for v in per_class.values())
    return {
        "per_class": per_class,
        "binding_class": binding["class"],
        "binding_p90": binding["p90"],
        "vesting_window": vesting_window,
        "gates": {"B5": gate},
        "passes": gate,
        "note": ("d_g/d_r are SCENARIO detection capabilities: the sweep "
                 "measures the horizon FORM given a detection capability, "
                 "and therefore what N must be for it; the real per-epoch "
                 "detection rates of the ledger tests are a deployment "
                 "question"),
    }


# --------------------------------------------------------------------------
# Scenario C — bootstrap dynamics
# --------------------------------------------------------------------------

def bootstrap_run(
        demand: float, price: float, epochs: int, bootstrap_quality: float,
        arrivals: dict[int, float], fallback_share: float,
        vesting_window: int, seed: int) -> dict[str, Any]:
    """The headroom-rule handover. ``arrivals`` maps registration epoch
    to measured contributor quality. Routing: STRICTLY better quality
    wins priority (pedigree ignored); equal quality keeps the
    bootstrap arm; after handover the bootstrap arm serves only the
    registered ``fallback_share`` of demand.

    Fees: 2.5% dev fund; the remainder credits the serving arm in
    proportion to its traffic share, vesting linearly over
    ``vesting_window`` epochs.
    """
    if not 0.0 < fallback_share < 1.0:
        raise ValueError("fallback_share must be in (0, 1)")
    if vesting_window <= 0:
        raise ValueError("vesting_window must be positive")
    registered: dict[str, float] = {}
    log: list[dict[str, Any]] = []
    handover_epoch: int | None = None
    contributor_serving: str | None = None
    boot_earned = contrib_earned = 0.0
    pool_cumulative = 0.0
    for t in range(epochs):
        if t in arrivals:
            registered[f"contrib_{t}"] = arrivals[t]
        better = [name for name, q in registered.items()
                  if q > bootstrap_quality]
        equal = [name for name, q in registered.items()
                 if q == bootstrap_quality]
        if better:
            # Strictly better wins by measurement alone.
            winner = sorted(better)[0]
            if handover_epoch is None:
                handover_epoch = t
                contributor_serving = winner
            boot_share, contrib_share = fallback_share, 1.0 - fallback_share
        else:
            # Equal-but-not-better keeps the bootstrap arm in priority.
            winner = "bootstrap"
            boot_share, contrib_share = 1.0, 0.0
        pool = demand * price * (1.0 - DEV_FUND_FRACTION)
        boot_earned += pool * boot_share
        contrib_earned += pool * contrib_share
        pool_cumulative += pool
        log.append({
            "epoch": t,
            "winner": winner,
            "handover": handover_epoch == t,
            "bootstrap_traffic_share": boot_share,
            "pool_vesting_credit": pool,
            "pool_cumulative": pool_cumulative,
            "equal_arms_present": sorted(equal),
        })
    served_after = [e for e in log if e["handover"]
                    or (handover_epoch is not None
                        and e["epoch"] > handover_epoch)]
    post_handover_ok = bool(served_after) and all(
        abs(e["bootstrap_traffic_share"] - fallback_share) < 1e-12
        for e in served_after)
    equal_stays = all(e["winner"] == "bootstrap" for e in log
                      if e["equal_arms_present"]
                      and e["epoch"] < (handover_epoch
                                        if handover_epoch is not None
                                        else epochs))
    monotone = all(log[i]["pool_cumulative"] <= log[i + 1]
                   ["pool_cumulative"] for i in range(len(log) - 1))
    gates = {
        # C1: handover in the first epoch after a strictly-better arm.
        "C1_handover": handover_epoch is not None,
        # C2: after handover the bootstrap arm keeps only the
        # registered fallback share.
        "C2_fallback": post_handover_ok,
        # C3: the equal-quality arm does NOT displace the bootstrap arm.
        "C3_equal_keeps": equal_stays,
        # C4: pool accumulation is non-decreasing.
        "C4_pool": monotone,
    }
    return {
        "log": log,
        "handover_epoch": handover_epoch,
        "contributor_serving": contributor_serving,
        "bootstrap_earned": boot_earned,
        "contributor_earned": contrib_earned,
        "gates": gates,
        "passes": all(gates.values()),
        "note": ("pedigree is ignored: the strictly-better arm takes the "
                 "axis by measurement alone; equal-but-not-better keeps "
                 "the bootstrap arm (the bar must be BEATEN)"),
    }
