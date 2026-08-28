"""M184 — agent-based simulation harness evidence: the registered
synthetic scenarios for H1, H3, H8.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(18 Aug 2026). Deterministic, CPU-only. The gates are synthetic-scenario
instruments — they verify the mechanism FORMS, not real deployments.
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
    Agent,
    h1_gate,
    h3_gate,
    h8_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m184_incentives.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v25" / "m184_incentives"


def run_m184(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible

    def _agents(specs: list[dict[str, Any]]) -> list[Agent]:
        return [Agent(str(s["name"]), str(s["kind"]),
                      cost=float(s.get("cost", 0.0)),
                      contribution=float(s.get("contribution", 0.0)),
                      solo_progress=float(s.get("solo_progress", 0.0)),
                      actually_healthy=bool(s.get("actually_healthy",
                                                  True)),
                      self_reported_healthy=bool(
                          s.get("self_reported_healthy", True)))
                for s in specs]

    h1 = h1_gate(_agents(config["h1"]["agents"]),
                 rounds=int(config["h1"]["rounds"]),
                 demand=float(config["h1"]["demand"]),
                 lag_sweep=[int(l) for l in config["h1"]["lag_sweep"]],
                 seed=int(config["h1"]["seed"]))
    print(f"H1 shared-beats-solo: {h1['passes']}", flush=True)

    h3 = h3_gate(Agent("wash", "wash"), Agent("honest", "cooperative"),
                 rounds=int(config["h3"]["rounds"]),
                 demand=float(config["h3"]["demand"]),
                 lag=int(config["h3"]["lag"]),
                 seed=int(config["h3"]["seed"]))
    print(f"H3 wash-loses: {h3['passes']}", flush=True)

    h8 = h8_gate(_agents(config["h8"]["agents"]),
                 seed=int(config["h8"]["seed"]))
    print(f"H8 availability honesty: {h8['passes']}", flush=True)

    gates_pass = h1["passes"] and h3["passes"] and h8["passes"]

    evidence: dict[str, Any] = {
        "milestone": "M184",
        "cell": "agent-based simulation harness (synthetic H1/H3/H8)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "gates": {"h1": h1, "h3": h3, "h8": h8},
        "verdict": {
            "passes": gates_pass,
            "reading": ("the registered mechanism forms pass their "
                        "synthetic gates: shared beats solo under lag, "
                        "wash loses under the anti-wash stack, and "
                        "selection is validator-measured only")
            if gates_pass else "a synthetic gate failed",
        },
        "scope_note": ("synthetic-scenario instruments, NOT claims about "
                       "real deployments; the registered agents and payoff "
                       "forms are what is tested"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_pass": gates_pass,
                      "h1": h1["passes"], "h3": h3["passes"],
                      "h8": h8["passes"]}, indent=1), flush=True)
    print(f"M184 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m184(args.config, args.output)


if __name__ == "__main__":
    main()
