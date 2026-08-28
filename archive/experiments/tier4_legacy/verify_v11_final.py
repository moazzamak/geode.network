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
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v11" / "m68_final_replay.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v11" / "m68_final_replay"


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


def _verify_locked_file(specification: dict[str, str]) -> dict[str, str]:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"locked file hash mismatch: {path}")
    return dict(specification)


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
    m63: dict[str, Any],
    m64: dict[str, Any],
    m65: dict[str, Any],
) -> dict[str, bool]:
    cells = m65["cells"]
    quantile = [cell for cell in cells if cell["extent_policy"] == "quantile"]
    negative_guided = [
        cell for cell in cells if cell["extent_policy"] != "quantile"
    ]
    delegated_head = m63["delegated_head"]
    straight_tubes = [
        tube for seed_result in m64["seed_results"]
        for tube in seed_result["straight_tubes"]
    ]
    return {
        "m63_protocol_passed": m63["gate"]["m63_passed"] is True,
        "m63_known_only_head_lineage_verified": (
            delegated_head["known_classes"] == list(range(8))
            and delegated_head["fit_class_count"] == 8
            and delegated_head["calibration_class_count"] == 8
            and delegated_head["fit_partition"] == "geometry_fit"
            and delegated_head["calibration_partition"] == "score_calibration"
            and delegated_head["proxy_unknown_classes_excluded_from_fit"] is True
            and delegated_head["support_vectors_unchanged_by_calibration"] is True
            and delegated_head["exact_replay"] is True
        ),
        "m64_controlled_h1_passed": (
            m64["gate"]["m64_passed"] is True
            and m64["masking_scene"]["v10_four_x_acceptance"] > 0.2
            and m64["masking_scene"]["v11_four_x_acceptance"] <= 0.01
            and m64["masking_scene"]["negative_guided_four_x_acceptance"] <= 0.01
        ),
        "m64_nine_ranks_recovered_exactly": (
            len(straight_tubes) == 9
            and all(tube["recovered_rank"] == tube["true_rank"] for tube in straight_tubes)
        ),
        "m65_all_registered_cells_executed": len(cells) == 27,
        "m65_calibration_split_independent": (
            m65["calibration_split"]["independent"] is True
            and m65["calibration_split"]["extent_count"] == 800
            and m65["calibration_split"]["conformal_count"] == 800
        ),
        "m65_eighteen_extent_infeasible": (
            len(negative_guided) == 18
            and all(
                not cell["calibration_feasible"]
                and cell["stop_reason"]
                == "negative-guided extent is infeasible above the 0.90 floor"
                for cell in negative_guided
            )
        ),
        "m65_nine_contrast_infeasible": (
            len(quantile) == 9
            and all(
                not cell["calibration_feasible"]
                and cell["stop_reason"]
                == "no registered contrast margin passes calibration safety"
                and [attempt["margin"] for attempt in cell["calibration_audit"]["attempts"]]
                == [0.0, 0.05, 0.1, 0.2]
                for cell in quantile
            )
        ),
        "m65_no_cell_retained": (
            m65["eligible_cell_indices"] == []
            and m65["retained_cell"] is None
            and m65["retained_cell_index"] is None
        ),
        "m66_blocked": m65["advance_to_m66"] is False,
        "m67_blocked": (
            m65["advance_to_m66"] is False and m65["retained_cell"] is None
        ),
        "final_labels_remained_closed": (
            m63["gate"]["final_labels_opened"] is False
            and m64["gate"]["final_labels_opened"] is False
            and m65["final_labels_opened"] is False
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
    verified_ledger = _verify_locked_file(config["final_claim_ledger"])
    _, m63 = _load_locked(config["evidence"]["m63"])
    _, m64 = _load_locked(config["evidence"]["m64"])
    _, m65 = _load_locked(config["evidence"]["m65"])
    operands = _conclusion_operands(m63, m64, m65)
    if not all(operands.values()):
        failed = sorted(name for name, passed in operands.items() if not passed)
        raise ValueError(f"v11 conclusion replay failed: {failed}")
    branch_ledger = [
        {
            "branch": "M63 protocol, role, and conformal lock",
            "status": "complete",
            "reason": "passed after known-only delegated-head lineage repair",
        },
        {
            "branch": "M64 controlled masking and identifiability",
            "status": "complete",
            "reason": "all controlled H1, rank, coverage, and replay operands passed",
        },
        {
            "branch": "M65 seed-11 directional envelope screen",
            "status": "stopped",
            "reason": "0/27 cells were calibration-feasible",
        },
        {
            "branch": "M66 three-seed confirmation",
            "status": "blocked",
            "reason": "M65 retained no directional envelope",
        },
        {
            "branch": "M67 lifecycle evaluation",
            "status": "blocked",
            "reason": "M66 cannot pass without an M65-retained envelope",
        },
        {
            "branch": "M68 artifact-only finalization",
            "status": "complete",
            "reason": "all conclusion operands reproduced from locked artifacts",
        },
    ]
    result = {
        "schema_version": 1,
        "milestone": "M68",
        "configuration_hash": sha256_file(config_path),
        "outcome": config["expected_outcome"],
        "outcome_interpretation": (
            "controlled H1 passed, but the registered mechanism did not transfer "
            "to frozen real features; the directional envelope line closes"
        ),
        "verified_indexes": verified_indexes,
        "verified_index_count": len(verified_indexes),
        "verified_artifact_count": sum(
            item["artifact_count"] for item in verified_indexes
        ),
        "verified_final_claim_ledger": verified_ledger,
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
