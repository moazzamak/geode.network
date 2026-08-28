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
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v10" / "m62_final_replay.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v10" / "m62_final_replay"


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
    m56: dict[str, Any],
    m57: dict[str, Any],
    m58: dict[str, Any],
) -> dict[str, bool]:
    feasible = [cell for cell in m58["cells"] if cell["calibration_feasible"]]
    best_gain = max(
        cell["gate_operands"]["balanced_accuracy_gain"] for cell in feasible
    )
    feasible_safety = all(
        cell["gate_operands"]["eight_x_tangent_acceptance"] == 0.0
        and cell["gate_operands"]["four_x_tangent_acceptance"] <= 0.01
        and cell["gate_operands"]["bridge_acceptance"] <= 0.05
        and cell["gate_operands"]["random_direction_acceptance"] <= 0.05
        and cell["gate_operands"]["mixed_acceptance"] <= 0.05
        for cell in feasible
    )
    return {
        "m56_protocol_passed": m56["gate"]["m56_passed"] is True,
        "m57_identifiability_passed": m57["gate"]["m57_passed"] is True,
        "m58_all_registered_cells_executed": len(m58["cells"]) == 18,
        "m58_sixteen_cells_calibration_infeasible": len(feasible) == 2,
        "m58_feasible_cells_passed_safety": feasible_safety,
        "m58_residual_signal_present": 0.0 < best_gain < 0.01,
        "m58_no_cell_retained": m58["eligible_cell_indices"] == []
        and m58["retained_cell"] is None,
        "m59_blocked": m58["advance_to_m59"] is False,
        "m60_closed": m58["open_m60_curvature_diagnostic"] is False,
        "final_labels_remained_closed": (
            m56["gate"]["final_labels_opened"] is False
            and m57["gate"]["final_labels_opened"] is False
            and m58["final_labels_opened"] is False
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
    _, m56 = _load_locked(config["evidence"]["m56"])
    _, m57 = _load_locked(config["evidence"]["m57"])
    _, m58 = _load_locked(config["evidence"]["m58"])
    operands = _conclusion_operands(m56, m57, m58)
    if not all(operands.values()):
        failed = sorted(name for name, passed in operands.items() if not passed)
        raise ValueError(f"v10 conclusion replay failed: {failed}")
    branch_ledger = [
        {
            "branch": "M56 protocol and score-unit lock",
            "status": "complete",
            "reason": "all protocol, safety, lineage, and replay operands passed",
        },
        {
            "branch": "M57 synthetic identifiability",
            "status": "complete",
            "reason": "rank recovery, coverage, safety, atlas, and controls passed",
        },
        {
            "branch": "M58 global affine screen",
            "status": "stopped",
            "reason": "0/18 cells passed every predictive and safety gate",
        },
        {
            "branch": "M59 three-seed affine confirmation",
            "status": "blocked",
            "reason": "M58 retained no global affine cell",
        },
        {
            "branch": "M60 local atlas",
            "status": "closed",
            "reason": "M58 did not satisfy the curvature-specific opening condition",
        },
        {
            "branch": "M61 lifecycle utility",
            "status": "blocked",
            "reason": "neither M59 nor M60 retained a support model",
        },
        {
            "branch": "M62 artifact-only finalization",
            "status": "complete",
            "reason": "all conclusion operands reproduced from locked artifacts",
        },
    ]
    result = {
        "schema_version": 1,
        "milestone": "M62",
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
