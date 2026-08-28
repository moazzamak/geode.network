"""Run the v6.1 A1-W seed-11 weighted-readout falsification gate."""

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
from experiments.common.v6_factorial import primitive_field_matrix
from experiments.common.v6_protocol import select_boundary_cohort
from experiments.common.v61_weighted_readout import (
    fit_weighted_readout,
    predict_weighted_student,
    readout_collapse_summary,
    serialize_weighted_student,
    weighted_local_edit_rollback_evidence,
    weighted_student_parameter_count,
)
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.tier4.eval_v6_factorial_s2 import _load_array
from experiments.tier4.eval_v6_subspace_primitives import _metrics
from src.subspace_primitive import SubspacePrimitive


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v6_1" / "a1_weighted_s1.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v6_1" / "a1_weighted_s1"


def _resolve(path: str) -> Path:
    return REPO_ROOT / Path(path)


def _load_locked_json(item: dict[str, Any], name: str) -> tuple[Path, dict[str, Any]]:
    path = _resolve(str(item["path"]))
    expected = str(item["sha256"])
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
        or config.get("milestone") != "A1-W"
        or config.get("stage") != "S1"
        or config.get("seed") != 11
        or config.get("test_labels_opened") is not False
    ):
        raise ValueError("Unsupported or test-open A1-W configuration.")
    readout = config["readout"]
    if readout != {
        "family": "nonnegative_component_mixture",
        "constraint": "per_class_simplex",
        "parameterization": "softmax_log_weights",
        "optimizer": "L-BFGS-B",
        "regularization": 0.0001,
        "maximum_iterations": 500,
        "gradient_tolerance": 1e-08,
        "initialization": "zero_equal_weights",
        "temperature_policy": "one_global",
        "minimum_temperature": 0.05,
        "maximum_temperature": 20.0,
    }:
        raise ValueError("A1-W readout contract mismatch.")
    if config["budget"] != {
        "component_count": 46,
        "parameter_limit": 584476,
    }:
        raise ValueError("A1-W budget contract mismatch.")
    tangent = config["tangent_result"]
    if tangent.get("required_advancement_status") is not False:
        raise ValueError("A1-W must preserve the stopped tangent branch.")


def _fit(
    *,
    fields: np.ndarray,
    candidate_labels: list[int],
    train_labels: np.ndarray,
    classes: np.ndarray,
    parent_student: dict[str, Any],
    parent_student_sha256: str,
    config: dict[str, Any],
    weighted: bool,
) -> dict[str, Any]:
    readout = config["readout"]
    fitted = fit_weighted_readout(
        fields,
        candidate_labels,
        train_labels,
        classes,
        regularization=float(readout["regularization"]),
        maximum_iterations=int(readout["maximum_iterations"]),
        gradient_tolerance=float(readout["gradient_tolerance"]),
        minimum_temperature=float(readout["minimum_temperature"]),
        maximum_temperature=float(readout["maximum_temperature"]),
        initial_temperature=float(parent_student["global_temperature"]),
        fit_component_weights=weighted,
    )
    return serialize_weighted_student(
        parent_student,
        fitted,
        parent_student_sha256=parent_student_sha256,
    )


def _median_inference_seconds(
    student: dict[str, Any],
    features: np.ndarray,
    *,
    parent_representation_hash: str,
    warmup_runs: int,
    measured_runs: int,
) -> float:
    for _ in range(warmup_runs):
        predict_weighted_student(
            student,
            features,
            parent_representation_hash=parent_representation_hash,
        )
    timings = []
    for _ in range(measured_runs):
        started = time.perf_counter()
        predict_weighted_student(
            student,
            features,
            parent_representation_hash=parent_representation_hash,
        )
        timings.append(time.perf_counter() - started)
    return float(np.median(timings))


def run_evaluation(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    a0_path, _ = _load_locked_json(config["a0_parent_index"], "A0 parent index")
    _, m30_config = _load_locked_json(config["m30_config"], "M30 configuration")
    m30_path, m30_evidence = _load_locked_json(
        config["m30_evidence"], "M30 evidence"
    )
    m31_path, m31_evidence = _load_locked_json(
        config["m31_evidence"], "M31 evidence"
    )
    parent_path, parent_student = _load_locked_json(
        config["parent_student"], "M31 selected student"
    )
    tangent_path, tangent_evidence = _load_locked_json(
        config["tangent_result"], "A1-T evidence"
    )
    if tangent_evidence["advancement_passed"] is not False:
        raise ValueError("A1-W affine-only scope requires a stopped A1-T parent.")

    seed_input = m30_config["seed_inputs"]["11"]
    loaded = _load_seed_data(seed_input)
    train_features, train_labels = loaded["datasets"]["train"]
    dev_features, dev_labels = loaded["datasets"]["dev"]
    classes = np.unique(train_labels)
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
        raise ValueError("A1-W development labels differ from the frozen parent.")
    m31_seed = m31_evidence["seed_results"]["11"]
    if (
        m31_seed["parent_representation_hash"]
        != seed_input["parent_representation_hash"]
        or m31_seed["split_hashes"] != parent_seed["split_hashes"]
        or parent_student["parent_representation_hash"]
        != seed_input["parent_representation_hash"]
    ):
        raise ValueError("A1-W parent artifacts do not share seed-11 lineage.")

    cohort_config = config["cohort"]
    cohort = select_boundary_cohort(
        teacher_train,
        fraction=float(cohort_config["fraction"]),
        minimum_count=int(cohort_config["minimum_count"]),
    )
    if cohort != m31_seed["cohort"]:
        raise ValueError("A1-W cohort differs from the frozen M31 selection cohort.")
    cohort_indices = np.asarray(cohort["selected_indices"], dtype=np.int64)
    candidates = [
        SubspacePrimitive.from_dict(item["payload"])
        for item in parent_student["selected_candidates"]
    ]
    candidate_labels = [
        int(candidate.class_label) for candidate in candidates
    ]
    cohort_fields = primitive_field_matrix(
        candidates,
        train_features[cohort_indices],
        primitive="subspace_r32",
        score="normalized_radial",
    )
    started = time.perf_counter()
    equal_student = _fit(
        fields=cohort_fields,
        candidate_labels=candidate_labels,
        train_labels=train_labels[cohort_indices],
        classes=classes,
        parent_student=parent_student,
        parent_student_sha256=sha256_file(parent_path),
        config=config,
        weighted=False,
    )
    equal_fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    weighted_student = _fit(
        fields=cohort_fields,
        candidate_labels=candidate_labels,
        train_labels=train_labels[cohort_indices],
        classes=classes,
        parent_student=parent_student,
        parent_student_sha256=sha256_file(parent_path),
        config=config,
        weighted=True,
    )
    weighted_fit_seconds = time.perf_counter() - started
    equal_predictions, equal_probabilities = predict_weighted_student(
        equal_student,
        dev_features,
        parent_representation_hash=seed_input["parent_representation_hash"],
    )
    weighted_predictions, weighted_probabilities = predict_weighted_student(
        weighted_student,
        dev_features,
        parent_representation_hash=seed_input["parent_representation_hash"],
    )
    equal_metrics = _metrics(
        dev_labels, equal_predictions, equal_probabilities, classes, teacher_dev
    )
    weighted_metrics = _metrics(
        dev_labels,
        weighted_predictions,
        weighted_probabilities,
        classes,
        teacher_dev,
    )

    replay_student = _fit(
        fields=cohort_fields,
        candidate_labels=candidate_labels,
        train_labels=train_labels[cohort_indices],
        classes=classes,
        parent_student=parent_student,
        parent_student_sha256=sha256_file(parent_path),
        config=config,
        weighted=True,
    )
    replay_predictions, replay_probabilities = predict_weighted_student(
        replay_student,
        dev_features,
        parent_representation_hash=seed_input["parent_representation_hash"],
    )
    exact_replay = (
        canonical_json(weighted_student) == canonical_json(replay_student)
        and np.array_equal(weighted_predictions, replay_predictions)
        and np.array_equal(weighted_probabilities, replay_probabilities)
    )
    latency = config["latency"]
    equal_inference = _median_inference_seconds(
        equal_student,
        dev_features,
        parent_representation_hash=seed_input["parent_representation_hash"],
        warmup_runs=int(latency["warmup_runs"]),
        measured_runs=int(latency["measured_runs"]),
    )
    weighted_inference = _median_inference_seconds(
        weighted_student,
        dev_features,
        parent_representation_hash=seed_input["parent_representation_hash"],
        warmup_runs=int(latency["warmup_runs"]),
        measured_runs=int(latency["measured_runs"]),
    )
    latency_ratio = weighted_inference / equal_inference
    lifecycle = weighted_local_edit_rollback_evidence(
        weighted_student,
        dev_features,
        parent_representation_hash=seed_input["parent_representation_hash"],
    )
    collapse = readout_collapse_summary(
        weighted_student,
        threshold=float(config["advancement_gate"]["collapse_weight_threshold"]),
    )
    parameter_count = weighted_student_parameter_count(weighted_student)
    gate = config["advancement_gate"]
    accuracy_improvement = (
        float(weighted_metrics["balanced_accuracy"])
        - float(equal_metrics["balanced_accuracy"])
    )
    nll_reduction = (
        float(equal_metrics["nll"]) - float(weighted_metrics["nll"])
    ) / float(equal_metrics["nll"])
    accuracy_loss = (
        float(equal_metrics["balanced_accuracy"])
        - float(weighted_metrics["balanced_accuracy"])
    )
    accuracy_path = accuracy_improvement >= float(
        gate["minimum_accuracy_improvement"]
    )
    nll_path = (
        nll_reduction >= float(gate["minimum_nll_reduction_fraction"])
        and accuracy_loss <= float(gate["maximum_accuracy_loss_for_nll"])
    )
    latency_passed = latency_ratio <= float(latency["maximum_ratio"])
    budget_passed = (
        len(weighted_student["selected_candidates"])
        == int(config["budget"]["component_count"])
        and parameter_count <= int(config["budget"]["parameter_limit"])
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
        and latency_passed
        and budget_passed
        and lifecycle_passed
        and replay_passed
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    write_canonical_json(output_dir / "equal_weight_student.json", equal_student)
    write_canonical_json(output_dir / "weighted_student.json", weighted_student)
    for name, values in (
        ("equal_weight_development_predictions.npy", equal_predictions),
        ("equal_weight_development_probabilities.npy", equal_probabilities),
        ("weighted_development_predictions.npy", weighted_predictions),
        ("weighted_development_probabilities.npy", weighted_probabilities),
    ):
        np.save(output_dir / name, values, allow_pickle=False)
    evidence = {
        "schema_version": 1,
        "amendment": "v6.1",
        "milestone": "A1-W",
        "stage": "S1",
        "configuration_hash": payload_hash(config),
        "parent_hashes": {
            "a0_parent_index": sha256_file(a0_path),
            "m30_evidence": sha256_file(m30_path),
            "m31_evidence": sha256_file(m31_path),
            "m31_selected_student": sha256_file(parent_path),
            "a1_tangent_evidence": sha256_file(tangent_path),
        },
        "seed": 11,
        "split_hashes": parent_seed["split_hashes"],
        "parent_representation_hash": seed_input["parent_representation_hash"],
        "cohort": cohort,
        "component_set_frozen": (
            equal_student["parent_component_hash"]
            == weighted_student["parent_component_hash"]
        ),
        "equal_weight": {
            "development": equal_metrics,
            "fit_seconds": equal_fit_seconds,
            "median_inference_seconds": equal_inference,
            "global_temperature": equal_student["global_temperature"],
            "optimizer": equal_student["optimizer"],
        },
        "weighted": {
            "development": weighted_metrics,
            "fit_seconds": weighted_fit_seconds,
            "median_inference_seconds": weighted_inference,
            "global_temperature": weighted_student["global_temperature"],
            "optimizer": weighted_student["optimizer"],
            "parameter_count": parameter_count,
            "serialized_bytes": serialized_size(weighted_student),
        },
        "comparison": {
            "accuracy_improvement": accuracy_improvement,
            "nll_reduction_fraction": nll_reduction,
            "accuracy_loss": accuracy_loss,
            "inference_time_ratio": latency_ratio,
        },
        "collapse": collapse,
        "lifecycle": lifecycle,
        "exact_replay": exact_replay,
        "gate_operands": {
            "accuracy_path": accuracy_path,
            "nll_path": nll_path,
            "latency": latency_passed,
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
                "equal_weight_balanced_accuracy": evidence["equal_weight"][
                    "development"
                ]["balanced_accuracy"],
                "weighted_balanced_accuracy": evidence["weighted"]["development"][
                    "balanced_accuracy"
                ],
                "nll_reduction_fraction": evidence["comparison"][
                    "nll_reduction_fraction"
                ],
                "inference_time_ratio": evidence["comparison"][
                    "inference_time_ratio"
                ],
                "majority_collapsed": evidence["collapse"]["majority_collapsed"],
                "exact_replay": evidence["exact_replay"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
