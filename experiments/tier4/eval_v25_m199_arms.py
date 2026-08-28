"""M199 — anti-wash corner-case arms evidence: collusion rings,
inference farms, and Sybil duplicates.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(19 Aug 2026, C1 of the hardening considerations). Deterministic,
CPU-only, synthetic-scenario instruments.
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
    collusion_ring_gate,
    farm_gate,
    sybil_duplicate_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m199_arms.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v25" / "m199_arms"


def run_m199(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible

    ring = collusion_ring_gate(
        [Agent(str(s["name"]), "gamer", cash=float(s.get("cash", 0.0)))
         for s in config["ring"]["members"]],
        rounds=int(config["ring"]["rounds"]),
        demand=float(config["ring"]["demand"]),
        lag=int(config["ring"]["lag"]),
        seed=int(config["ring"]["seed"]))
    print(f"M199 ring: {ring['passes']} "
          f"net={ring['ring_net_change']:.2f}", flush=True)

    farm = farm_gate(
        Agent("farm", "gamer",
              hosting_cost=float(config["farm"]["hosting_cost"])),
        rounds=int(config["farm"]["rounds"]),
        demand=float(config["farm"]["demand"]),
        lag=int(config["farm"]["lag"]),
        quality_floor=float(config["farm"]["quality_floor"]),
        seed=int(config["farm"]["seed"]))
    print(f"M199 farm: {farm['passes']}", flush=True)

    sybil = sybil_duplicate_gate(
        Agent(str(config["sybil"]["original_name"]), "cooperative",
              contribution=float(config["sybil"]["contribution"]),
              content_digest=str(config["sybil"]["digest"])),
        Agent(str(config["sybil"]["sybil_name"]), "cooperative",
              contribution=float(config["sybil"]["contribution"]),
              content_digest=str(config["sybil"]["digest"])),
        seed=int(config["sybil"]["seed"]))
    print(f"M199 sybil: {sybil['passes']}", flush=True)

    gates_pass = ring["passes"] and farm["passes"] and sybil["passes"]

    evidence: dict[str, Any] = {
        "milestone": "M199",
        "cell": "anti-wash corner-case arms (synthetic)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "gates": {"collusion_ring": ring, "inference_farm": farm,
                  "sybil_duplicate": sybil},
        "verdict": {
            "passes": gates_pass,
            "reading": ("the registered countermeasures hold in the "
                        "synthetic scenarios: payment rings lose in "
                        "aggregate, low-quality farms thaw nothing while "
                        "honest arms thaw, and duplicate contribution "
                        "digests credit zero") if gates_pass
            else "a corner-case gate failed",
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
    print(f"M199 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m199(args.config, args.output)


if __name__ == "__main__":
    main()
