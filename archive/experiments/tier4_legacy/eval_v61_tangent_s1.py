"""Run the v6.1 A1-T seed-11 tangent-cap falsification gate."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.experiment_manifest import canonical_json
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    serialized_size,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v6_factorial import select_predictive_candidates
from experiments.common.v6_protocol import select_boundary_cohort
from experiments.common.v61_tangent import (
    fit_tangent_global_temperature,
    generate_tangent_cap_candidates,
    predict_tangent_student,
    serialize_tangent_student,
    tangent_field_matrix,
    tangent_local_edit_rollback_evidence,
)
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.tier4.eval_v6_factorial_s2 import _candidate_evaluations, _load_array
from experiments.tier4.eval_v6_subspace_primitives import _metrics
from src.directional_primitive import l2_normalize


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v6_1" / "a1_tangent_s1.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v6_1" / "a1_tangent_s1"


def _resolve(path: str) -> Path:
    return REPO_ROOT / Path(path)


def _load_locked_json(item: dict[str, str], name: str) -> tuple[Path, dict[str, Any]]:
    path = _resolve(item["path"])
    expected = item["sha256"]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{name} hash mismatch: expected {expected}, got {actual}."
        )
    return path, json.loads(path.read_text(encoding="utf-8"))


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != 1
        or config.get("amendment") != "v6.1"
        or config.get("milestone") != "A1-T"
        or config.get("stage") != "S1"
        or config.get("seed") != 11
        or config.get("test_labels_opened") is not False
    ):
        raise ValueError("Unsupported or test-open A1-T configuration.")
    primitive = config["primitive"]
    if (
        primitive["family"] != "tangent_cap"
        or primitive["rank"] != 32
        or primitive["score"] != "normalized_tangent_radial"
        or primitive["normalization"] != "explicit_l2"
        or primitive["minimum_support"] != 34
    ):
        raise ValueError("A1-T primitive contract mismatch.")
    selection = config["selection"]
    budget = config["budget"]
    if (
        selection["objective"] != "direct"
        or selection["component_count"] != 46
        or budget != {"component_count": 46, "parameter_limit": 584476}
        or config["temperature"]["policy"] != "one_global"
        or config["controls"]
        != ["m30_cosine_cap", "m31_direct_rank32_affine"]
    ):
        raise ValueError("A1-T comparison or budget contract mismatch.")


def _fit_student(
    *,
    normalized_train: np.ndarray,
    train_labels: np.ndarray,
    teacher_train: np.ndarray,
    classes: np.ndarray,
    cohort_indices: np.ndarray,
    config: dict[str, Any],
    parent_representation_hash: str,
    directional_representation_hash: str,
) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    primitive = config["primitive"]
    generation = config["candidate_generation"]
    candidates = generate_tangent_cap_candidates(
        normalized_train,
        train_labels,
        teacher_train,
        classes,
        rank=int(primitive["rank"]),
        candidates_per_class=int(generation["candidates_per_class"]),
        anchor_fraction=float(generation["anchor_fraction"]),
        variance_floor_fraction=float(primitive["variance_floor_fraction"]),
        residual_floor_fraction=float(primitive["residual_floor_fraction"]),
    )
    selection_config = config["selection"]
    candidate_evaluations = _candidate_evaluations(
        len(candidates),
        int(selection_config["component_count"]),
        int(selection_config["initial_components_per_class"]) * len(classes),
    )
    if candidate_evaluations > int(
        selection_config["maximum_candidate_evaluations"]
    ):
        raise ValueError("A1-T exceeds its candidate-evaluation budget.")
    cohort_fields = tangent_field_matrix(
        candidates,
        normalized_train[cohort_indices],
        "normalized_tangent_radial",
    )
    selection = select_predictive_candidates(
        cohort_fields,
        [int(candidate.class_label) for candidate in candidates],
        teacher_train[cohort_indices],
        train_labels[cohort_indices],
        classes,
        objective="direct",
        score="normalized_radial",
        component_limit=int(selection_config["component_count"]),
        initial_components_per_class=int(
            selection_config["initial_components_per_class"]
        ),
        minimum_improvement=float(selection_config["minimum_improvement"]),
    )
    student = serialize_tangent_student(
        classes=classes,
        candidates=candidates,
        selection=selection,
        parent_representation_hash=parent_representation_hash,
        directional_representation_hash=directional_representation_hash,
        cohort_indices=cohort_indices,
        configuration=config,
    )
    temperature = config["temperature"]
    fit_tangent_global_temperature(
        student,
        normalized_train[cohort_indices],
        train_labels[cohort_indices],
        minimum=float(temperature["minimum"]),
        maximum=float(temperature["maximum"]),
    )
    return student, candidates, {
        "selection": selection,
        "candidate_evaluations": candidate_evaluations,
    }


def run_evaluation(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    a0_path, _ = _load_locked_json(config["a0_parent_index"], "A0 parent index")
    m30_path, m30_evidence = _load_locked_json(
        config["m30_evidence"], "M30 evidence"
    )
    m31_path, m31_evidence = _load_locked_json(
        config["m31_evidence"], "M31 evidence"
    )
    _, m30_config = _load_locked_json(config["m30_config"], "M30 configuration")
    seed_input = m30_config["seed_inputs"]["11"]
    loaded = _load_seed_data(seed_input)
    train_features, train_labels = loaded["datasets"]["train"]
    dev_features, dev_labels = loaded["datasets"]["dev"]
    classes = np.unique(train_labels)
    normalized_train = l2_normalize(train_features)
    parent_seed = m30_evidence["seed_results"]["11"]
    teacher_train = _load_array(
        m30_path.parent,
        parent_seed["array_artifacts"]["teacher_train_probabilities"],
    )
    teacher_dev = _load_array(
        m30_path.parent,
        parent_seed["array_artifacts"]["teacher_development_probabilities"],
    )
    parent_dev_labels = _load_array(
        m30_path.parent, parent_seed["array_artifacts"]["development_labels"]
    )
    if not np.array_equal(parent_dev_labels, dev_labels):
        raise ValueError("A1-T development labels differ from the M30 parent.")
    m31_seed = m31_evidence["seed_results"]["11"]
    if (
        m31_seed["split_hashes"] != parent_seed["split_hashes"]
        or m31_seed["parent_representation_hash"]
        != seed_input["parent_representation_hash"]
        or m31_seed["directional_representation_hash"]
        != seed_input["directional_representation_hash"]
    ):
        raise ValueError("A1-T parent controls do not share one seed-11 lineage.")

    selection_config = config["selection"]
    cohort = select_boundary_cohort(
        teacher_train,
        fraction=float(selection_config["cohort_fraction"]),
        minimum_count=int(selection_config["cohort_minimum_count"]),
    )
    cohort_indices = np.asarray(cohort["selected_indices"], dtype=np.int64)
    started = time.perf_counter()
    student, candidates, fit = _fit_student(
        normalized_train=normalized_train,
        train_labels=train_labels,
        teacher_train=teacher_train,
        classes=classes,
        cohort_indices=cohort_indices,
        config=config,
        parent_representation_hash=seed_input["parent_representation_hash"],
        directional_representation_hash=seed_input[
            "directional_representation_hash"
        ],
    )
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions, probabilities = predict_tangent_student(
        student,
        dev_features,
        parent_representation_hash=seed_input["parent_representation_hash"],
    )
    inference_seconds = time.perf_counter() - started
    metrics = _metrics(
        dev_labels, predictions, probabilities, classes, teacher_dev
    )

    replay_student, _, _ = _fit_student(
        normalized_train=normalized_train,
        train_labels=train_labels,
        teacher_train=teacher_train,
        classes=classes,
        cohort_indices=cohort_indices,
        config=config,
        parent_representation_hash=seed_input["parent_representation_hash"],
        directional_representation_hash=seed_input[
            "directional_representation_hash"
        ],
    )
    replay_predictions, replay_probabilities = predict_tangent_student(
        replay_student,
        dev_features,
        parent_representation_hash=seed_input["parent_representation_hash"],
    )
    exact_replay = (
        canonical_json(student) == canonical_json(replay_student)
        and np.array_equal(predictions, replay_predictions)
        and np.array_equal(probabilities, replay_probabilities)
    )
    lifecycle = tangent_local_edit_rollback_evidence(
        student,
        dev_features,
        parent_representation_hash=seed_input["parent_representation_hash"],
    )
    selected = [
        candidates[index]
        for index in fit["selection"]["selected_candidate_indices"]
    ]
    parameter_count = int(sum(item.parameter_count for item in selected))
    array_bytes = int(sum(item.array_bytes for item in selected))
    serialized_bytes = serialized_size(student)
    controls = {
        "m30_cosine_cap": parent_seed["variants"]["cosine_cap"]["development"],
        "m31_direct_rank32_affine": m31_seed["cells"][
            "direct_subspace_radial_component"
        ]["development"],
    }
    gate = config["advancement_gate"]
    control_accuracy = max(
        float(item["balanced_accuracy"]) for item in controls.values()
    )
    control_nll = min(float(item["nll"]) for item in controls.values())
    accuracy_improvement = float(metrics["balanced_accuracy"]) - control_accuracy
    nll_reduction = (control_nll - float(metrics["nll"])) / control_nll
    accuracy_loss = control_accuracy - float(metrics["balanced_accuracy"])
    accuracy_path = accuracy_improvement >= float(
        gate["minimum_accuracy_improvement"]
    )
    nll_path = (
        nll_reduction >= float(gate["minimum_nll_reduction_fraction"])
        and accuracy_loss <= float(gate["maximum_accuracy_loss_for_nll"])
    )
    budget_passed = (
        len(selected) == int(config["budget"]["component_count"])
        and parameter_count <= int(config["budget"]["parameter_limit"])
        and fit["candidate_evaluations"]
        <= int(selection_config["maximum_candidate_evaluations"])
    )
    lifecycle_passed = (
        lifecycle["unaffected_prediction_preservation"]
        >= float(gate["minimum_unaffected_prediction_preservation"])
        and (
            lifecycle["exact_json_rollback"]
            and lifecycle["rollback_restored_predictions"]
            if gate["require_exact_rollback"]
            else True
        )
    )
    replay_passed = exact_replay if gate["require_exact_replay"] else True
    advancement_passed = (
        (accuracy_path or nll_path)
        and budget_passed
        and lifecycle_passed
        and replay_passed
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    write_canonical_json(output_dir / "student.json", student)
    np.save(output_dir / "development_predictions.npy", predictions, allow_pickle=False)
    np.save(
        output_dir / "development_probabilities.npy",
        probabilities,
        allow_pickle=False,
    )
    evidence = {
        "schema_version": 1,
        "amendment": "v6.1",
        "milestone": "A1-T",
        "stage": "S1",
        "configuration_hash": payload_hash(config),
        "parent_hashes": {
            "a0_parent_index": sha256_file(a0_path),
            "m30_evidence": sha256_file(m30_path),
            "m31_evidence": sha256_file(m31_path),
        },
        "seed": 11,
        "split_hashes": parent_seed["split_hashes"],
        "parent_representation_hash": seed_input["parent_representation_hash"],
        "directional_representation_hash": seed_input[
            "directional_representation_hash"
        ],
        "cohort": cohort,
        "controls": controls,
        "tangent_cap": {
            "development": metrics,
            "selected_component_count": len(selected),
            "component_counts": fit["selection"]["component_counts"],
            "parameter_count": parameter_count,
            "array_bytes": array_bytes,
            "serialized_bytes": serialized_bytes,
            "candidate_evaluations": fit["candidate_evaluations"],
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            "global_temperature": student["global_temperature"],
        },
        "comparison": {
            "strongest_control_balanced_accuracy": control_accuracy,
            "best_control_nll": control_nll,
            "accuracy_improvement": accuracy_improvement,
            "nll_reduction_fraction": nll_reduction,
            "accuracy_loss": accuracy_loss,
        },
        "lifecycle": lifecycle,
        "exact_replay": exact_replay,
        "gate_operands": {
            "accuracy_path": accuracy_path,
            "nll_path": nll_path,
            "budget": budget_passed,
            "lifecycle": lifecycle_passed,
            "replay": replay_passed,
        },
        "advancement_passed": advancement_passed,
        "test_labels_opened": False,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    evidence = run_evaluation(args.config, args.output)
    print(
        json.dumps(
            {
                "advancement_passed": evidence["advancement_passed"],
                "balanced_accuracy": evidence["tangent_cap"]["development"][
                    "balanced_accuracy"
                ],
                "nll": evidence["tangent_cap"]["development"]["nll"],
                "accuracy_improvement": evidence["comparison"][
                    "accuracy_improvement"
                ],
                "exact_replay": evidence["exact_replay"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
