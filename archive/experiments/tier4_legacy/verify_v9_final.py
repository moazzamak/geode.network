from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    sha256_file,
    write_canonical_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v9" / "m55_final_replay.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v9" / "m55_final_replay"


def _resolve(path: str) -> Path:
    resolved = (REPO_ROOT / Path(path)).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError("final replay paths must remain inside the repository")
    return resolved


def _load_locked(specification: dict[str, str]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"locked artifact hash mismatch: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _verify_index(specification: dict[str, str]) -> dict[str, Any]:
    index_path, index = _load_locked(specification)
    root = index_path.parent
    verified = []
    for artifact in index["artifacts"]:
        path = root / artifact["path"]
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"indexed artifact hash mismatch: {path}")
        verified.append(artifact["path"])
    return {
        "path": specification["path"],
        "sha256": specification["sha256"],
        "artifact_count": len(verified),
        "verified_artifacts": verified,
    }


def _conclusion_operands(
    m51: dict[str, Any], m53: dict[str, Any]
) -> dict[str, bool]:
    rank_results = m53["rank_results"]
    return {
        "m51_shell_gate_failed": m51["gate"]["m51_passed"] is False,
        "m52_closed": m51["branch_decision"]["m52_shell_scores_open"] is False,
        "m53_tube_was_opened": m51["branch_decision"]["m53_bounded_tube_open"] is True,
        "m53_no_rank_retained": m53["retained_ranks"] == [],
        "m53_s2_blocked": m53["advance_to_m53_s2"] is False,
        "all_ranks_failed_8x_acceptance": all(
            result["stress"]["acceptance_at_8x"] == 1.0
            for result in rank_results.values()
        ),
        "predictive_signal_reproduced": (
            rank_results["16"]["screen_operands"]["balanced_accuracy_gain"] >= 0.01
            and rank_results["32"]["screen_operands"]["balanced_accuracy_gain"] >= 0.01
        ),
        "final_labels_remained_closed": (
            m51["final_labels_opened"] is False
            and m53["final_labels_opened"] is False
        ),
    }


def run_verification(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verified_indexes = [_verify_index(item) for item in config["indexes"]]
    _, m51 = _load_locked(config["evidence"]["m51"])
    _, m53 = _load_locked(config["evidence"]["m53_s1"])
    operands = _conclusion_operands(m51, m53)
    if not all(operands.values()):
        failed = sorted(name for name, passed in operands.items() if not passed)
        raise ValueError(f"v9 conclusion replay failed: {failed}")
    branch_ledger = [
        {
            "branch": "H1 frozen shell occupancy",
            "status": "stopped",
            "reason": "M51 failed all registered gate operands",
        },
        {
            "branch": "M52 frozen shell scores",
            "status": "blocked",
            "reason": "M51 found no meaningful negative interior in 0/48 diagnostics",
        },
        {
            "branch": "M53 fitted shell",
            "status": "blocked",
            "reason": "M51 failed and M52 did not open",
        },
        {
            "branch": "H2 bounded tube S1",
            "status": "stopped",
            "reason": "all ranks accepted 100% of 8x tangent probes",
        },
        {
            "branch": "M53 bounded tube S2",
            "status": "blocked",
            "reason": "no S1 rank passed every kill switch",
        },
        {
            "branch": "M54 lifecycle utility",
            "status": "blocked",
            "reason": "no M53-eligible support model",
        },
        {
            "branch": "M55 artifact-only finalization",
            "status": "complete",
            "reason": "all conclusion operands reproduced from locked artifacts",
        },
    ]
    result = {
        "schema_version": 1,
        "milestone": "M55",
        "configuration_hash": sha256_file(config_path),
        "outcome": config["expected_outcome"],
        "verified_indexes": verified_indexes,
        "verified_index_count": len(verified_indexes),
        "verified_artifact_count": sum(
            item["artifact_count"] for item in verified_indexes
        ),
        "conclusion_operands": operands,
        "branch_ledger": branch_ledger,
        "training_data_loaded": False,
        "final_labels_opened": False,
        "replay_passed": True,
    }
    write_canonical_json(output_dir / "evidence.json", result)
    build_artifact_index(output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = run_verification(arguments.config, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
