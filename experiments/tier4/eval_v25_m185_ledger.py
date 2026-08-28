"""M185 — ledger prototype evidence: an append-only hash-chained registry
holding references to sealed milestones, each verified by M177 replay.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(18 Aug 2026). LOCAL part only: the chain, the tamper check, and the
anchor_spec fields M194 will submit to a public testnet. The replay tie
is real: every ledger record that references a sealed milestone carries
the audit API's content hash, and the cell re-runs the replay for each
referenced milestone before appending.
"""
from __future__ import annotations

import argparse
import json
import tempfile
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
from geode.core.ledger import AppendOnlyLedger

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m185_ledger.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v25" / "m185_ledger"


def run_m185(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible
    api = AuditAPI()

    ledger = AppendOnlyLedger()
    replays: dict[str, Any] = {}
    for name, spec in config["records"].items():
        runner = (run_m175_cell_a0 if spec["runner"] == "a0"
                  else run_m175_cell_c)
        report = api.replay(
            runner, REPO_ROOT / spec["config"],
            REPO_ROOT / spec["evidence"],
            Path(tempfile.mkdtemp(prefix="m185_replay_")))
        replays[name] = report.to_dict()
        if not report.bit_exact:
            raise SystemExit(f"M185 VOID: {name} replay not bit-exact")
        ledger.append({
            "key": f"milestone:{name}",
            "milestone": spec["milestone"],
            "evidence_content_hash": report.evidence_hash,
            "note": spec["note"],
        })
        print(f"  {name}: replay bit-exact, record appended", flush=True)

    ledger.append({"key": "ledger:genesis_summary",
                   "note": "M185 local chain; testnet anchoring is M194"})
    verify = ledger.verify()
    tamper_check = verify["ok"]

    evidence: dict[str, Any] = {
        "milestone": "M185",
        "cell": "ledger prototype (local chain + replay tie)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "ledger": ledger.to_dict(),
        "verify": verify,
        "replay_reports": replays,
        "anchor_spec": ledger.anchor_spec(),
        "verdict": {
            "passes": tamper_check,
            "reading": ("the local append-only chain verifies, every "
                        "referenced milestone replays bit-exact, and the "
                        "anchor fields are specified for M194")
            if tamper_check else "chain verification failed",
        },
        "scope": "local only; public-testnet anchoring deferred to M194",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"chain_ok": tamper_check, "tip": ledger.tip(),
                      "records": len(ledger.to_dict()["records"])},
                     indent=1), flush=True)
    print(f"M185 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m185(args.config, args.output)


if __name__ == "__main__":
    main()
