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
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v12" / "m76_final_replay.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v12" / "m76_final_replay"


def _resolve(path: str) -> Path:
    resolved = (REPO_ROOT / Path(path)).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError("M76 paths must remain inside the repository")
    return resolved


def _load_locked(
    specification: dict[str, str],
) -> tuple[Path, dict[str, Any]]:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"M76 locked artifact hash mismatch: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _verify_locked_file(specification: dict[str, str]) -> dict[str, str]:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"M76 locked file hash mismatch: {path}")
    return dict(specification)


def _verify_index(specification: dict[str, str]) -> dict[str, Any]:
    index_path, index = _load_locked(specification)
    verified = []
    for artifact in index["artifacts"]:
        path = index_path.parent / artifact["path"]
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"M76 indexed artifact hash mismatch: {path}")
        verified.append(artifact["path"])
    return {
        "path": specification["path"],
        "sha256": specification["sha256"],
        "artifact_count": len(verified),
        "verified_artifacts": verified,
    }


def _conclusion_operands(
    m70_d1: dict[str, Any],
    m70_native: dict[str, Any],
    m71: dict[str, Any],
    m72: dict[str, Any],
    m73: dict[str, Any],
    m74: dict[str, Any],
    m75: dict[str, Any],
) -> dict[str, bool]:
    return {
        "m70_sample_curve_complete": (
            m70_d1["gate"]["all_registered_sample_sizes_reported"] is True
            and m70_d1["gate"]["n800_marked_extrapolated"] is True
        ),
        "m70_native_pathology_persisted": (
            m70_native["gate"]["native_resolution_verified"] is True
            and m70_native["gate"]["all_class_counts_reported"] is True
            and m70_native["gate"]["penetration_non_decreasing"] is True
        ),
        "m70_did_not_require_v11_amendment": (
            m70_native["upsampling_excluded"] is True
            and m70_native["gate"]["four_x_system_acceptance_non_decreasing"]
            is True
        ),
        "m71_gaussian_not_qualified": (
            m71["gate"]["l1_accuracy_parity"] is False
            and m71["gate"]["l2_open_set_competence"] is False
            and m71["gate"]["i4_closed_form_counterfactual"] is False
        ),
        "m71_l2_was_statistical_near_miss": (
            m71["gate"]["l2_interpretation"]
            == "formal_near_miss_statistically_indistinguishable_from_bar"
        ),
        "m72_failed_with_residual_signal": (
            m72["gate"]["m72_passed"] is False
            and m72["gate"]["threshold_ratio_passed"] is True
            and m72["gate"]["held_out_probes_passed"] is False
            and m72["open_m73"] is True
        ),
        "m73_escalation_gate_passed": (
            m73["gate"]["m73_passed"] is True
            and m73["gate"]["collapse_prevention_load_bearing"] is True
            and m73["gate"]["held_out_four_x_acceptance"] == 0.75
        ),
        "m73_was_not_open_space_qualification": (
            m73["arms"]["constrained"]["metrics"]["probe_acceptance"]["masking"][
                "system_acceptance"
            ]
            == 1.0
            and m73["arms"]["constrained"]["metrics"]["probe_acceptance"]["normal"][
                "system_acceptance"
            ]
            == 1.0
        ),
        "m74_primary_accuracy_parity_only": (
            m74["gate"]["l1_accuracy_parity"] is True
            and m74["gate"]["threshold_ratio_confirmed"] is True
            and m74["gate"]["l2_open_set_competence"] is False
        ),
        "m74_held_out_and_real_ood_failed": (
            m74["gate"]["held_out_probe_transfer"] is False
            and m74["gate"]["real_ood_competence"] is False
            and m74["gate"]["primary_maximum_held_out_four_x_acceptance"]
            == 1.0
        ),
        "m74_transfer_failed": (
            m74["gate"]["l5_transfer_accuracy_parity"] is False
            and m74["gate"]["l5_transfer_open_set_competence"] is False
            and m74["gate"]["transfer_unknown_recall"]
            == 0.008333333333333333
        ),
        "m74_outcome_e": (
            m74["gate"]["m74_passed"] is False
            and m74["advance_to_m75"] is False
        ),
        "m75_partial_structural_inspectability": (
            m75["summary"]["i1_passed"] is True
            and m75["summary"]["i2_passed"] is True
            and m75["summary"][
                "i3_all_top_score_reductions_exceed_random_and_bottom"
            ]
            is True
            and m75["summary"]["i5_probe_beats_no_explanation"] is True
        ),
        "m75_full_inspectability_failed": (
            m75["summary"]["i4_passed"] is False
            and m75["summary"]["qualified_inspectability_claim"] is False
            and m75["outcome_can_change"] is False
        ),
        "all_milestone_replays_passed": (
            m72["gate"]["exact_replay"] is True
            and m73["gate"]["exact_replay"] is True
            and m74["gate"]["exact_replay"] is True
            and m75["summary"]["exact_replay"] is True
        ),
        "final_labels_remained_closed": all(
            evidence["final_labels_opened"] is False
            for evidence in (
                m70_d1,
                m70_native,
                m71,
                m72,
                m73,
                m74,
                m75,
            )
        ),
    }


def run_verification(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verified_indexes = [
        _verify_index(specification) for specification in config["indexes"]
    ]
    verified_ledger = _verify_locked_file(config["final_claim_ledger"])
    evidence = {
        name: _load_locked(specification)[1]
        for name, specification in config["evidence"].items()
    }
    operands = _conclusion_operands(
        evidence["m70_d1"],
        evidence["m70_native"],
        evidence["m71"],
        evidence["m72"],
        evidence["m73"],
        evidence["m74"],
        evidence["m75"],
    )
    if not all(operands.values()):
        failed = sorted(name for name, passed in operands.items() if not passed)
        raise ValueError(f"v12 conclusion replay failed: {failed}")
    branch_ledger = [
        {
            "branch": "M69 prior-art refresh",
            "status": "complete",
            "reason": "conjunction retained with narrowed novelty claims",
        },
        {
            "branch": "M70 diagnostic re-examination",
            "status": "complete",
            "reason": "pathology persisted across sample size, native domain, and class count",
        },
        {
            "branch": "M71 Gaussian classifier baseline",
            "status": "complete_not_qualified",
            "reason": "L1 and I4 failed; L2 was a statistical near miss",
        },
        {
            "branch": "M72 frozen-feature Stage 0",
            "status": "failed_with_residual_signal",
            "reason": "held-out safety and accuracy failed but axis/corner signal opened M73",
        },
        {
            "branch": "M73 learned-projection Stage 1",
            "status": "escalation_passed",
            "reason": "material M72 improvement and load-bearing collapse prevention",
        },
        {
            "branch": "M74 confirmation and transfer",
            "status": "failed_outcome_e",
            "reason": "open-set, held-out mixed, real-OOD, and DomainNet transfer gates failed",
        },
        {
            "branch": "M75 inspectability qualification",
            "status": "partial_not_qualified",
            "reason": "I1/I2 passed, but decision faithfulness was weak and I4 unavailable",
        },
        {
            "branch": "M76 artifact-only finalization",
            "status": "complete",
            "reason": "all conclusion operands reproduced from locked artifacts",
        },
    ]
    result = {
        "schema_version": 1,
        "milestone": "M76",
        "configuration_hash": sha256_file(config_path),
        "outcome": config["expected_outcome"],
        "outcome_interpretation": (
            "primary known-class accuracy approached RBF parity, but generalized "
            "open-space rejection, transfer, and full inspectability failed"
        ),
        "verified_indexes": verified_indexes,
        "verified_index_count": len(verified_indexes),
        "verified_artifact_count": sum(
            item["artifact_count"] for item in verified_indexes
        ),
        "verified_final_claim_ledger": verified_ledger,
        "conclusion_operands": operands,
        "conclusion_operand_count": len(operands),
        "branch_ledger": branch_ledger,
        "v11_amendment_required": config["v11_amendment_required"],
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
