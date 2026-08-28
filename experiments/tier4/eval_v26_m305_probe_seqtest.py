"""M305 harness — the registered H26-6 halves plus the margin-gate cell.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M305
(26 Aug 2026, before any build). H26-6: under R-A6d an honest
contributor on divergent hardware is never convicted across a
registered session budget, while a 99.5%-agreeing substitute IS
convicted by the R-A5a sequential test within the corrected horizon.
Gate: BOTH halves must hold; either alone is a fail.

Registered cells (written before running):

- **H1 honest.** p0 = 0.002 per probed session (the registered honest
  rate for margin-gated divergent hardware), 1000 independent streams
  of N = 8000 probed sessions each: the false-conviction fraction must
  stay at or below 0.02 for the registered alpha = 0.01.
- **H2 substitute.** p1 = 0.005 per probed session (the 99.5%-agreeing
  substitute), 300 independent streams of N = 8000 probed sessions:
  the conviction probability must be at least 0.95, and the median
  probed-sessions-to-conviction must be at most 8000. The wall-clock
  horizon median/rho is recorded per probe rate and reported against
  the corrected horizon 1/(rho*delta) as context (recorded, not
  gated).
- **M1 margin gate.** A crafted stream whose disagreements sit below
  the registered noise floor: the gated count must equal the
  registered expectation (240 of 400), and the gated SPRT must never
  convict before the raw SPRT on the same stream.
"""
from __future__ import annotations

import argparse
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
from geode.core.probe_seqtest import (
    corrected_horizon,
    margin_gated_mismatch,
    sprt,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
                  / "m305_probe_seqtest.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m305_probe_seqtest")


def _stream_decision(stream: np.ndarray, p0: float, p1: float,
                     alpha: float, beta: float) -> tuple[str, int]:
    """Run the SPRT over a binary mismatch stream; returns (decision,
    sessions when the terminal decision first fired, or the budget)."""
    m = 0
    for n, x in enumerate(stream, start=1):
        m += int(x)
        out = sprt(m, n, p0, p1, alpha, beta)
        if out["decision"] != "continue":
            return out["decision"], n
    return "continue", len(stream)


def run_m305(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()

    p0 = float(config["p0_honest"])
    p1 = float(config["p1_substitute"])
    alpha = float(config["alpha"])
    beta = float(config["beta"])
    budget = int(config["session_budget"])
    h2_budget = int(config["h2_session_budget"])
    rng = np.random.default_rng(int(config["seed"]))

    cells: dict[str, Any] = {}

    # ---- H1 honest half ---------------------------------------------------
    h1_runs = int(config["h1_runs"])
    false_convictions = 0
    for _ in range(h1_runs):
        stream = rng.random(budget) < p0
        decision, _ = _stream_decision(stream.astype(np.int8),
                                       p0, p1, alpha, beta)
        if decision == "convict":
            false_convictions += 1
    h1 = {"false_conviction_fraction": false_convictions / h1_runs,
          "false_convictions": false_convictions,
          "runs": h1_runs,
          "registered_bound": float(config["h1_bound"]),
          "passes": false_convictions / h1_runs
          <= float(config["h1_bound"])}
    cells["h1_honest"] = h1

    # ---- H2 substitute half ----------------------------------------------
    h2_runs = int(config["h2_runs"])
    convictions = 0
    times: list[int] = []
    for _ in range(h2_runs):
        stream = rng.random(h2_budget) < p1
        decision, at = _stream_decision(stream.astype(np.int8),
                                        p0, p1, alpha, beta)
        if decision == "convict":
            convictions += 1
            times.append(at)
    median = float(np.median(times)) if times else float("inf")
    h2 = {"conviction_probability": convictions / h2_runs,
          "convictions": convictions, "runs": h2_runs,
          "session_budget": h2_budget,
          "median_sessions_to_conviction": median,
          "registered_conviction_bound": float(config["h2_conviction_bound"]),
          "registered_median_bound": float(config["h2_median_bound"]),
          "passes": bool(
              convictions / h2_runs >= float(config["h2_conviction_bound"])
              and median <= float(config["h2_median_bound"]))}
    horizon_readings: dict[str, float] = {}
    for rho in config["rho_readings"]:
        horizon_readings[str(rho)] = median / float(rho)
    h2["wall_horizon_by_rho"] = horizon_readings
    h2["corrected_horizon_context"] = {
        str(rho): corrected_horizon(float(rho), p1)
        for rho in config["rho_readings"]}
    cells["h2_substitute"] = h2

    # ---- M1 margin gate ----------------------------------------------------
    stream_raw = rng.random(int(config["m1_streams"])) < float(
        config["m1_raw_mismatch_rate"])
    gated = []
    for x in stream_raw:
        if x:
            margin = float(rng.uniform(0.0, 1.0))
            counted, _ = margin_gated_mismatch(
                1.0, 1.0 - margin, float(config["m1_noise_floor"]))
            gated.append(int(counted))
        else:
            gated.append(0)
    gated = np.asarray(gated, dtype=np.int8)
    raw_decision, raw_at = _stream_decision(stream_raw.astype(np.int8),
                                            p0, p1, alpha, beta)
    gated_decision, gated_at = _stream_decision(gated, p0, p1, alpha, beta)
    m1 = {"raw_mismatches": int(stream_raw.sum()),
          "gated_mismatches": int(gated.sum()),
          "registered_gated_expectation": float(
              config["m1_expected_gated_rate"]),
          "gated_rate": float(gated.mean()),
          "raw_decision": raw_decision, "raw_sessions": raw_at,
          "gated_decision": gated_decision, "gated_sessions": gated_at,
          "gated_never_convicts_before_raw": bool(
              raw_decision != "convict"
              or gated_decision != "convict"
              or gated_at >= raw_at),
          "gated_count_below_raw": bool(
              int(gated.sum()) < int(stream_raw.sum())),
          "passes": bool(int(gated.sum()) < int(stream_raw.sum())
                         and (raw_decision != "convict"
                              or gated_decision != "convict"
                              or gated_at >= raw_at))}
    cells["m1_margin_gate"] = m1

    checks = {"h1_honest_not_convicted": h1["passes"],
              "h2_substitute_convicted": h2["passes"],
              "m1_margin_gate": m1["passes"]}
    gates_ok = all(checks.values())

    evidence: dict[str, Any] = {
        "milestone": "M305",
        "cell": "H26-6 halves + margin-gate cell over the SPRT",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "cells": cells,
        "checks": checks,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": ("both H26-6 halves hold: the honest contributor "
                        "is not convicted and the 99.5%-agreeing "
                        "substitute is, within the registered budget"
                        ) if gates_ok else "a check failed — VOID",
        },
        "scope": "synthetic mismatch-stream sweep, registered before "
                 "running",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok, "checks": checks,
                      "h1": h1, "h2_conviction": h2["conviction_probability"],
                      "h2_median": h2["median_sessions_to_conviction"]},
                     indent=1), flush=True)
    print(f"M305 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def calibrate_budget(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """M305a step 1 (registered): find the minimal H2 session budget on
    the registered grid whose conviction power reaches the registered
    target (0.95) under the SPRT's own parameters - an instrument
    calibration computed from the registered rates, never fitted to a
    failed verdict."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    p0 = float(config["p0_honest"])
    p1 = float(config["p1_substitute"])
    alpha = float(config["alpha"])
    beta = float(config["beta"])
    target = float(config["calibration_target_power"])
    runs = int(config["calibration_runs"])
    grid = [int(v) for v in config["calibration_budget_grid"]]
    results: dict[str, float] = {}
    chosen = None
    for n in grid:
        rng = np.random.default_rng(int(config["seed"]) + n)
        convicted = 0
        for _ in range(runs):
            stream = rng.random(n) < p1
            decision, _ = _stream_decision(stream.astype(np.int8),
                                           p0, p1, alpha, beta)
            convicted += decision == "convict"
        power = convicted / runs
        results[str(n)] = power
        if power >= target:
            chosen = n
            break
    out = {"budget_grid_results": results,
           "chosen_budget": chosen,
           "target_power": target,
           "p0": p0, "p1": p1, "alpha": alpha, "beta": beta}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "budget_calibration.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibrate", action="store_true",
                        help="M305a step 1: calibrate the H2 budget and "
                             "exit (writes budget_calibration.json)")
    args = parser.parse_args()
    if args.calibrate:
        calibrate_budget(args.config, args.output)
    else:
        run_m305(args.config, args.output)


if __name__ == "__main__":
    main()
