"""M199 closure — the remaining H3 arms: self-payment wash, dust
storms, and the structural form-checks for selection front-running
and dev-fund laundering.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(25 Aug 2026, M199 CLOSURE ACTIVE entry) before building.
Deterministic, CPU-only, synthetic-scenario instruments.
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
from geode.attribution.incentives import (
    dust_storm_gate,
    self_payment_wash_gate,
    structural_form_checks,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m199_closure.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v25" / "m199_closure"


def run_m199_closure(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible

    sp_cfg = config["self_payment"]
    self_pay = self_payment_wash_gate(
        sessions=int(sp_cfg["sessions"]),
        demand=float(sp_cfg["demand"]),
        seed=int(sp_cfg["seed"]))
    print(f"M199 closure self-payment: {self_pay['passes']} "
          f"net={self_pay['stack_net']:.2f}", flush=True)

    ds_cfg = config["dust_storm"]
    storms = {
        str(size): dust_storm_gate(
            storm_size=int(size),
            min_session_fee=float(ds_cfg["min_session_fee"]),
            seed=int(ds_cfg["seed"]))
        for size in ds_cfg["sizes"]
    }
    for size, storm in storms.items():
        print(f"M199 closure dust storm ({size}): {storm['passes']} "
              f"net={storm['net_with_fee']:.2f}", flush=True)

    structural = structural_form_checks(seed=int(config["structural"]
                                                 ["seed"]))
    print(f"M199 closure structural: {structural['passes']}", flush=True)

    gates_pass = (self_pay["passes"] and all(s["passes"] for s in
                                             storms.values())
                  and structural["passes"])

    evidence: dict[str, Any] = {
        "milestone": "M199-closure",
        "cell": "anti-wash corner-case closure arms (synthetic)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "gates": {
            "self_payment_wash": self_pay,
            "dust_storm": storms,
            "structural_form_checks": structural,
        },
        "verdict": {
            "passes": gates_pass,
            "reading": (
                "the remaining M199 corner cases are closed: "
                "self-payment loses the entire spend under the dock + "
                "payout-address exclusion (baseline returns it), dust "
                "storms pay the minimum fee per session and earn zero "
                "liveness credit, and selection front-running + "
                "dev-fund laundering are closed structurally by "
                "commit-reveal sealing and governance-only spend")
            if gates_pass else "a closure gate failed",
        },
        "scope_note": ("synthetic-scenario instruments, NOT claims about "
                       "real deployments; no identity-based mechanism "
                       "appears anywhere (C1 rule)"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_pass": gates_pass}, indent=1), flush=True)
    print(f"M199 closure complete -> {output_dir / 'evidence.json'}",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m199_closure(args.config, args.output)


if __name__ == "__main__":
    main()
