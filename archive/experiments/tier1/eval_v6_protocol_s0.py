from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.experiment_manifest import array_fingerprint
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v6_protocol import (
    BudgetSpec,
    PrimitiveMetadata,
    TeacherLineage,
    enumerate_budget_table,
    require_teacher_compatibility,
    select_boundary_cohort,
    validate_baseline_locks,
    validate_prediction_baseline,
    validate_v6_protocol_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v6" / "protocol.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v6" / "m27_s0"


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_v6_protocol_config(config)
    parent = config["parent_protocol"]
    parent_path = REPO_ROOT / parent["path"]
    if sha256_file(parent_path) != parent["sha256"]:
        raise ValueError("The frozen parent protocol hash does not match.")
    return config


def _s0_teacher_probabilities() -> np.ndarray:
    return np.array(
        [
            [0.51, 0.49],
            [0.80, 0.20],
            [0.55, 0.45],
            [0.10, 0.90],
            [0.65, 0.35],
            [0.49, 0.51],
            [0.25, 0.75],
            [0.95, 0.05],
        ],
        dtype=np.float64,
    )


def run_s0(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    baseline_locks = validate_baseline_locks(config["baseline_locks"], REPO_ROOT)
    prediction_baseline = validate_prediction_baseline(
        config["prediction_baseline"], REPO_ROOT
    )

    probabilities = _s0_teacher_probabilities()
    prediction_hash = array_fingerprint(probabilities)
    checkpoint_hash = payload_hash(
        {
            "family": config["teacher"]["primary_family"],
            "fixture": "m27_s0_fixed_teacher",
        }
    )
    teacher = TeacherLineage(
        family=config["teacher"]["primary_family"],
        representation_hash=config["active_representation"]["hash"],
        training_split_hash=config["split_hashes"]["train"],
        development_split_hash=config["split_hashes"]["development"],
        checkpoint_hash=checkpoint_hash,
        prediction_hash=prediction_hash,
        selection_metric=config["teacher"]["selection_metric"],
        test_labels_used_for_selection=config["teacher"][
            "test_labels_used_for_selection"
        ],
    )
    require_teacher_compatibility(
        teacher,
        representation_hash=config["active_representation"]["hash"],
        training_split_hash=config["split_hashes"]["train"],
        development_split_hash=config["split_hashes"]["development"],
    )

    cohort_config = config["boundary_cohort"]
    cohort = select_boundary_cohort(
        probabilities,
        fraction=float(cohort_config["fraction"]),
        minimum_count=int(cohort_config["minimum_count"]),
    )
    primitives = [
        {
            "id": item["id"],
            **PrimitiveMetadata(
                family=item["family"],
                minimum_seed_rule=item["minimum_seed_rule"],
                score_semantics=item["score_semantics"],
                local_rank=item.get("local_rank"),
                residual_scale=item.get("residual_scale"),
                direction=item.get("direction"),
                angular_radius=item.get("angular_radius"),
            ).to_dict(),
        }
        for item in config["primitive_candidates"]
    ]
    budgets = [
        BudgetSpec(
            mode=item["mode"],
            component_limit=item.get("component_limit"),
            parameter_limit=item.get("parameter_limit"),
        )
        for item in config["budgets"]
    ]
    budget_table = enumerate_budget_table(
        budgets, [item["id"] for item in config["primitive_candidates"]]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "protocol_snapshot.json", config)
    write_canonical_json(output_dir / "baseline_locks.json", baseline_locks)
    write_canonical_json(
        output_dir / "prediction_baseline.json", prediction_baseline
    )
    write_canonical_json(output_dir / "teacher_lineage.json", teacher.to_dict())
    write_canonical_json(output_dir / "boundary_cohort.json", cohort)
    write_canonical_json(output_dir / "primitive_table.json", primitives)
    write_canonical_json(output_dir / "budget_table.json", budget_table)
    np.save(output_dir / "teacher_probabilities.npy", probabilities, allow_pickle=False)
    index = build_artifact_index(output_dir)
    return {
        "baseline_lock_count": len(baseline_locks),
        "prediction_head_count": len(prediction_baseline["observed_metrics"]),
        "primitive_count": len(primitives),
        "budget_cell_count": len(budget_table),
        "boundary_selected_count": cohort["selected_count"],
        "teacher_prediction_hash": prediction_hash,
        "artifact_count": len(index["artifacts"]),
    }


def verify_s0(config_path: Path, output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        first = Path(temporary) / "first"
        second = Path(temporary) / "second"
        first_summary = run_s0(config_path, first)
        second_summary = run_s0(config_path, second)
        first_files = {
            path.relative_to(first).as_posix(): path.read_bytes()
            for path in first.rglob("*")
            if path.is_file()
        }
        second_files = {
            path.relative_to(second).as_posix(): path.read_bytes()
            for path in second.rglob("*")
            if path.is_file()
        }
        if first_summary != second_summary or first_files != second_files:
            raise RuntimeError("M27 S0 replay was not byte-identical.")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(first, output_dir)

    summary = {**first_summary, "byte_identical_replay": True}
    write_canonical_json(output_dir / "verification.json", summary)
    preliminary_index = build_artifact_index(output_dir)
    summary["artifact_count"] = len(preliminary_index["artifacts"])
    write_canonical_json(output_dir / "verification.json", summary)
    build_artifact_index(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M27 deterministic S0 gate.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify_s0(args.config, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
