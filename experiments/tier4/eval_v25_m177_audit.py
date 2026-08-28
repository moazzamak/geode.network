"""M177 — audit ladder spec + audit API v0; replay two sealed milestones
bit-exact through it (the H6 gate sample).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(18 Aug 2026) and ``analysis/v25_audit_ladder_spec.md``. The two sampled
milestones: M175 cell A0 and M175 cell C (fast, deterministic, no GPU).
Each is re-run by its OWN sealed runner into a scratch directory and
compared against the sealed evidence on the registered rules (content
hash with timing fields excluded; full-dict equality with those fields
excluded). H6 passes iff both bit_exact and equal_excluding_timing hold
for both.
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
from experiments.tier4.eval_v24_m175_cell_a0 import run_m175_cell_a0
from experiments.tier4.eval_v24_m175_cell_c import run_m175_cell_c
from geode.audit import AuditAPI

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m177_audit.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v25" / "m177_audit"


def run_m177(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible
    api = AuditAPI()

    replays: dict[str, Any] = {}
    for name, spec in config["replay_targets"].items():
        runner = (run_m175_cell_a0 if spec["runner"] == "a0"
                  else run_m175_cell_c)
        report = api.replay(
            runner,
            REPO_ROOT / spec["config"],
            REPO_ROOT / spec["evidence"],
            Path(tempdir()) / f"m177_replay_{name}")
        replays[name] = report.to_dict()
        print(f"  {name}: bit_exact={report.bit_exact} "
              f"equal_excluding_timing={report.equal_excluding_timing} "
              f"hash={report.replayed_hash[:16]}...", flush=True)

    # ---- L1 on the two sampled artifacts -----------------------------------
    provenance: dict[str, Any] = {}
    for name, spec in config["replay_targets"].items():
        artifact_dir = Path(REPO_ROOT / spec["evidence"]).parent
        provenance[name] = api.provenance(artifact_dir).to_dict()

    h6_passes = all(r["bit_exact"] and r["equal_excluding_timing"]
                    for r in replays.values())

    evidence: dict[str, Any] = {
        "milestone": "M177",
        "cell": "audit ladder spec + audit API v0 + two bit-exact replays",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "spec": "analysis/v25_audit_ladder_spec.md",
        "api_module": "geode/audit.py",
        "unit_tests": "experiments/common/test_v25_m177_audit.py (5/5)",
        "replays": replays,
        "provenance_reports": provenance,
        "gate_h6": {
            "passes": h6_passes,
            "reading": ("both sampled milestones replay bit-exact and "
                        "equal-excluding-timing")
            if h6_passes else "a sampled replay diverged",
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"h6_passes": h6_passes,
                      "replays": {k: v["bit_exact"] for k, v
                                  in replays.items()}}, indent=1),
          flush=True)
    print(f"M177 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def tempdir() -> str:
    import tempfile

    return tempfile.mkdtemp(prefix="m177_replay_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m177(args.config, args.output)


if __name__ == "__main__":
    main()
