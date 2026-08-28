"""M293 — H-series economic simulation battery evidence: the copycat
race, the detection-horizon sweep, and the bootstrap dynamics.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(25 Aug 2026) before building. Deterministic, CPU-only. The gates are
synthetic-scenario instruments — they verify the mechanism FORMS and
measure the registered metrics, not real deployments.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from geode.attribution.hseries import (
    bootstrap_run,
    copycat_race_sweep,
    detection_horizon_sweep,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m293_hseries.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v25" / "m293_hseries"


def run_m293(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible

    a_cfg = config["scenario_a"]
    a = copycat_race_sweep(
        demand=float(a_cfg["demand"]),
        price=float(a_cfg["price"]),
        epochs=int(a_cfg["epochs"]),
        quality=float(a_cfg["quality"]),
        copycat_epochs=[int(k) for k in a_cfg["copycat_epochs"]],
        undercuts=[float(u) for u in a_cfg["undercuts"]],
        serving_cost=float(a_cfg["serving_cost"]),
        seed=int(a_cfg["seed"]))
    print(f"A copycat race: {a['passes']} "
          f"(worst publisher fees vs no-copycat "
          f"{a['worst_publisher_fees_vs_no_copycat']:.3f})", flush=True)

    b_cfg = config["scenario_b"]
    b = detection_horizon_sweep(
        draws=int(b_cfg["draws"]),
        probe_rate=float(b_cfg["probe_rate"]),
        epoch_volume=float(b_cfg["epoch_volume"]),
        gaming_rate=float(b_cfg["gaming_rate"]),
        ring_rate=float(b_cfg["ring_rate"]),
        health_probes=int(b_cfg["health_probes"]),
        health_hit_rate=float(b_cfg["health_hit_rate"]),
        vesting_window=int(b_cfg["vesting_window"]),
        seed=int(b_cfg["seed"]))
    print(f"B detection-horizon sweep: {b['passes']} "
          f"(binding class {b['binding_class']}, "
          f"p90 {b['binding_p90']:.2f} epochs)", flush=True)

    c_cfg = config["scenario_c"]
    c = bootstrap_run(
        demand=float(c_cfg["demand"]),
        price=float(c_cfg["price"]),
        epochs=int(c_cfg["epochs"]),
        bootstrap_quality=float(c_cfg["bootstrap_quality"]),
        arrivals={int(k): float(q) for k, q in
                  c_cfg["arrivals"].items()},
        fallback_share=float(c_cfg["fallback_share"]),
        vesting_window=int(c_cfg["vesting_window"]),
        seed=int(c_cfg["seed"]))
    print(f"C bootstrap dynamics: {c['passes']} "
          f"(handover epoch {c['handover_epoch']})", flush=True)

    gates_pass = a["passes"] and b["passes"] and c["passes"]

    evidence: dict[str, Any] = {
        "milestone": "M293",
        "cell": ("H-series economic simulation battery: copycat race, "
                 "detection-horizon sweep, bootstrap dynamics"),
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "gates": {
            "A_copycat_race": a["gates"],
            "B_detection_horizon": b["gates"],
            "C_bootstrap_dynamics": c["gates"],
        },
        "results": {
            "A": {
                "passes": a["passes"],
                "worst_publisher_fees_vs_no_copycat":
                    a["worst_publisher_fees_vs_no_copycat"],
                "n_cells": len(a["cells"]),
                "cells": a["cells"],
            },
            "B": {
                "passes": b["passes"],
                "binding_class": b["binding_class"],
                "binding_p90": b["binding_p90"],
                "vesting_window": b["vesting_window"],
                "per_class": b["per_class"],
            },
            "C": {
                "passes": c["passes"],
                "handover_epoch": c["handover_epoch"],
                "bootstrap_earned": c["bootstrap_earned"],
                "contributor_earned": c["contributor_earned"],
                "log": c["log"],
            },
        },
        "verdict": {
            "passes": gates_pass,
            "reading": (
                "all registered H-series scenario gates pass: the "
                "publisher keeps the fee stream against a copycat "
                "sweep (marginal attribution + first-registration "
                "routing), the vesting window N=4 clears half the p90 "
                "detection horizon of every cheat class at the "
                "registered detection capabilities, and the headroom "
                "rule hands the axis over by measurement alone")
            if gates_pass else "a registered H-series gate failed",
        },
        "scope_note": ("synthetic-scenario instruments, NOT claims about "
                       "real deployments; the registered agents, payoff "
                       "forms, and detection capabilities are what is "
                       "tested"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_pass": gates_pass,
                      "A": a["passes"], "B": b["passes"],
                      "C": c["passes"]}, indent=1), flush=True)
    print(f"M293 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m293(args.config, args.output)


if __name__ == "__main__":
    main()
