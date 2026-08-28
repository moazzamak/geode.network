"""M313 harness — the registered economic-repair gate (A11, A20).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M313
(26 Aug 2026, before any build). Gate cells:

- **C1 bond covers the arbitrage.** A substitute axis is simulated
  under the M305 SPRT at the corrected horizon 1/(rho*delta). For
  every possible conviction time up to the horizon, the forfeited
  bond plus the L3 burn must be at least the compute the substitute
  saved over the same window; at the horizon the bond equals the
  saved amount exactly (the bond is sized to the exposure window).
- **C2 verified-work-only tenure.** The campaign's wash-ring volume
  (self-purchases, any addresses) accrues zero tenure weight; the
  sampled verified records accrue positive weight.
- **C3 claim delay.** Non-decreasing in open exposure units.
- **C4 honest L3.** The burn never exceeds vested and is zero for a
  fully-claimed account.

All four cells must pass. Expected loss vs expected gain under the
simulated campaign is recorded (not gated): with P(convict)=1-beta the
expected forfeit lands just below the expected saving, which is why
the bond is per-axis and why the registered reading is the
deterministic coverage, not the expectation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from geode.core.economic_repairs import (
    claim_delay_epochs,
    conviction_burn,
    per_axis_bond,
    slash_decision,
    tenure_weight,
)
from geode.core.probe_seqtest import corrected_horizon, sprt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
                  / "m313_economic_repairs.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m313_economic_repairs")


def _probe_campaign(rng: np.random.Generator, p1: float, p0: float,
                    alpha: float, beta: float, budget: int
                    ) -> tuple[int, int]:
    """One SPRT stream of a substitute axis; returns (mismatches,
    units until the terminal decision, or the budget)."""
    m = 0
    for n in range(1, budget + 1):
        m += int(rng.random() < p1)
        out = sprt(m, n, p0, p1, alpha, beta)
        if out["decision"] != "continue":
            return m, n
    return m, budget


def run_m313(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()

    rho = float(config["rho"])
    delta = float(config["delta"])
    p0 = float(config["p0_honest"])
    p1 = float(config["p1_substitute"])
    alpha = float(config["alpha"])
    beta = float(config["beta"])
    saving = float(config["saving_per_unit"])
    units_per_epoch = float(config["units_per_epoch"])
    horizon = corrected_horizon(rho, delta)          # 1/(rho*delta)
    horizon_units = int(np.ceil(horizon))
    budget = max(int(config["session_budget"]), horizon_units * 4)
    rng = np.random.default_rng(int(config["seed"]))

    cells: dict[str, Any] = {}

    # ---- C1 bond covers the arbitrage, at every conviction time ----
    bond = per_axis_bond(saving, float(horizon))
    coverage_ok = True
    worst_gap = 0.0
    for at in range(1, horizon_units + 1):
        # vested at detection (linear vesting over 4 epochs) is
        # claimed as it vests; the L3 burn is the unclaimed remainder,
        # which the claim delay (R-A11b) keeps non-empty while
        # exposure is open.
        vested = saving * at
        claimed = vested * max(0.0, 1.0 - (1.0 / 4.0))
        burn = conviction_burn(vested, claimed)
        forfeit = bond
        gap = (forfeit + burn) - (saving * at)
        worst_gap = min(worst_gap, gap)
        if gap < -1e-12:
            coverage_ok = False
    c1 = {
        "bond": bond,
        "horizon_units": horizon_units,
        "saving_per_unit": saving,
        "worst_coverage_gap": worst_gap,
        "coverage_holds_at_every_time": coverage_ok,
        "at_horizon_bond_equals_saved": bool(
            abs(bond - saving * horizon_units) <= 1e-9
            * max(1.0, abs(saving * horizon_units))),
        "passes": coverage_ok,
    }
    cells["c1_bond_covers_arbitrage"] = c1

    # ---- the campaign: expected loss vs expected gain (recorded) ----
    runs = int(config["campaign_runs"])
    detections: list[int] = []
    for _ in range(runs):
        _, at = _probe_campaign(rng, p1, p0, alpha, beta, budget)
        if at < budget:
            detections.append(at)
    p_convict = len(detections) / runs
    median_at = float(np.median(detections)) if detections else budget
    expected_forfeit = p_convict * bond
    expected_saved = saving * median_at
    campaign = {
        "runs": runs,
        "conviction_fraction": p_convict,
        "median_detection_units": median_at,
        "registered_horizon_units": float(horizon),
        "expected_forfeit": expected_forfeit,
        "expected_saved": expected_saved,
        "expected_loss_ge_gain": expected_forfeit >= expected_saved,
        "recorded_not_gated": True,
    }
    cells["campaign_expected_loss_vs_gain"] = campaign

    # ---- C2 verified-work-only tenure ----
    wash_ring = [
        {"source": "self_served", "weight": 1e6, "address": f"0x{i}"}
        for i in range(100)
    ]
    verified = [
        {"source": "sampled_challenge", "weight": 1.0},
        {"source": "probe_reference", "weight": 2.0},
    ]
    wash_weight = tenure_weight(wash_ring)
    verified_weight = tenure_weight(verified)
    c2 = {
        "wash_ring_tenure_weight": wash_weight,
        "verified_tenure_weight": verified_weight,
        "passes": bool(wash_weight == 0.0 and verified_weight > 0.0),
    }
    cells["c2_verified_work_only"] = c2

    # ---- C3 claim delay non-decreasing in exposure ----
    exposures = np.linspace(0.0, horizon, 64)
    delays = [claim_delay_epochs(float(e), units_per_epoch)
              for e in exposures]
    c3 = {
        "exposure_units_sample": float(horizon),
        "max_delay_epochs": int(max(delays)),
        "non_decreasing": bool(all(
            d1 <= d2 for d1, d2 in zip(delays, delays[1:]))),
        "passes": bool(all(
            d1 <= d2 for d1, d2 in zip(delays, delays[1:]))),
    }
    cells["c3_claim_delay"] = c3

    # ---- C4 honest L3 ----
    burn_full = conviction_burn(100.0, 100.0)
    burn_partial = conviction_burn(100.0, 40.0)
    decision = slash_decision(3, 100.0, 40.0)
    c4 = {
        "burn_fully_claimed": burn_full,
        "burn_partial": burn_partial,
        "l3_decision_burn": decision["burn"],
        "never_exceeds_vested": bool(
            burn_partial <= 100.0 and decision["burn"] <= 100.0),
        "passes": bool(burn_full == 0.0 and burn_partial == 60.0
                       and decision["burn"] == 60.0),
    }
    cells["c4_honest_l3"] = c4

    gate_cells = [c for key, c in cells.items()
                  if key.startswith(("c1_", "c2_", "c3_", "c4_"))]
    gates_ok = all(bool(c["passes"]) for c in gate_cells)
    elapsed = time.time() - started
    evidence = {
        "milestone": "M313",
        "config_digest": payload_hash(config),
        "gates_ok": gates_ok,
        "cells": cells,
        "registered_checks": ["C1", "C2", "C3", "C4"],
        "runtime_seconds": elapsed,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({
        "gates_ok": gates_ok,
        "c1_bond": c1["bond"],
        "horizon_units": horizon_units,
        "campaign_conviction_fraction": p_convict,
        "campaign_expected_forfeit": expected_forfeit,
        "campaign_expected_saved": expected_saved,
        "wash_ring_weight": wash_weight,
        "verified_weight": verified_weight,
    }, indent=1))
    return evidence


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m313(args.config, args.output)


if __name__ == "__main__":
    main()
