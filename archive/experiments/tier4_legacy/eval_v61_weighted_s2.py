"""Run the frozen v6.1 A2 three-seed weighted-readout retention gate."""

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
from experiments.common.v5_statistics import (
    paired_prediction_interval,
    paired_seed_t_interval,
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
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v6_1" / "a2_weighted_s2.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v6_1" / "a2_weighted_s2"

_READOUT_CONTRACT = {
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
}


def _resolve(path: str) -> Path:
    return REPO_ROOT / Path(path)


def _load_locked_json(item: dict[str, Any], name: str) -> tuple[Path, dict[str, Any]]:
    path = _resolve(str(item["path"]))
    actual = sha256_file(path)
    if actual != str(item["sha256"]):
        raise ValueError(
            f"{name} hash mismatch: expected {item['sha256']}, got {actual}."
        )
    return path, json.loads(path.read_text(encoding="utf-8"))


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != 1
        or config.get("amendment") != "v6.1"
        or config.get("milestone") != "A2"
        or config.get("stage") != "S2"
        or config.get("seeds") != [11, 23, 37]
        or config.get("retained_mechanisms") != ["weighted_affine_rank32"]
        or config.get("closed_mechanisms")
        != ["tangent_cap_rank32", "weighted_tangent_cap_rank32"]
        or config.get("test_labels_opened") is not False
    ):
        raise ValueError("Unsupported, scope-open, or test-open A2 configuration.")
    if config.get("readout") != _READOUT_CONTRACT:
        raise ValueError("A2 readout differs from the passing A1-W contract.")
    if config.get("budget") != {
        "component_count": 46,
        "parameter_limit": 584476,
        "maximum_optimizer_iterations": 500,
        "maximum_function_evaluations": 1000,
    }:
        raise ValueError("A2 resource contract mismatch.")
    if set(config.get("parent_students", {})) != {"11", "23", "37"}:
        raise ValueError("A2 parent students do not match the frozen seeds.")
    if (
        config["a1_weighted_evidence"].get("required_advancement_status") is not True
        or config["a1_tangent_evidence"].get("required_advancement_status") is not False
        or config["a1_budget_evidence"].get("required_a2_component_count") != 46
    ):
        raise ValueError("A2 mechanism or component scope differs from A1.")
    if config.get("parity_gate") != {
        "maximum_same_space_gap": 0.02,
        "maximum_m31_accuracy_loss": 0.0025,
        "minimum_unaffected_prediction_preservation": 0.999,
        "require_exact_replay": True,
        "require_exact_rollback": True,
    }:
        raise ValueError("A2 parity gate mismatch.")


def _fit_student(
    fields: np.ndarray,
    candidate_labels: list[int],
    labels: np.ndarray,
    classes: np.ndarray,
    parent: dict[str, Any],
    parent_sha256: str,
    config: dict[str, Any],
    *,
    weighted: bool,
) -> dict[str, Any]:
    readout = config["readout"]
    fitted = fit_weighted_readout(
        fields,
        candidate_labels,
        labels,
        classes,
        regularization=float(readout["regularization"]),
        maximum_iterations=int(readout["maximum_iterations"]),
        gradient_tolerance=float(readout["gradient_tolerance"]),
        minimum_temperature=float(readout["minimum_temperature"]),
        maximum_temperature=float(readout["maximum_temperature"]),
        initial_temperature=float(parent["global_temperature"]),
        fit_component_weights=weighted,
    )
    return serialize_weighted_student(
        parent, fitted, parent_student_sha256=parent_sha256
    )


def _median_inference_seconds(
    student: dict[str, Any],
    features: np.ndarray,
    representation_hash: str,
    latency: dict[str, Any],
) -> float:
    for _ in range(int(latency["warmup_runs"])):
        predict_weighted_student(
            student, features, parent_representation_hash=representation_hash
        )
    timings = []
    for _ in range(int(latency["measured_runs"])):
        started = time.perf_counter()
        predict_weighted_student(
            student, features, parent_representation_hash=representation_hash
        )
        timings.append(time.perf_counter() - started)
    return float(np.median(timings))


def _fit_seed(
    seed: int,
    config: dict[str, Any],
    m30_path: Path,
    m30_config: dict[str, Any],
    m30_evidence: dict[str, Any],
    m31_path: Path,
    m31_evidence: dict[str, Any],
) -> dict[str, Any]:
    seed_key = str(seed)
    seed_input = m30_config["seed_inputs"][seed_key]
    loaded = _load_seed_data(seed_input)
    train_features, train_labels = loaded["datasets"]["train"]
    dev_features, dev_labels = loaded["datasets"]["dev"]
    classes = np.unique(train_labels)
    parent_path, parent = _load_locked_json(
        config["parent_students"][seed_key], f"M31 seed-{seed} student"
    )
    m30_seed = m30_evidence["seed_results"][seed_key]
    m31_seed = m31_evidence["seed_results"][seed_key]
    if (
        parent["parent_representation_hash"]
        != seed_input["parent_representation_hash"]
        or m31_seed["parent_representation_hash"]
        != seed_input["parent_representation_hash"]
        or m31_seed["split_hashes"] != m30_seed["split_hashes"]
    ):
        raise ValueError(f"A2 seed {seed} parent lineage mismatch.")

    teacher_train = _load_array(
        m30_path.parent,
        m30_seed["array_artifacts"]["teacher_train_probabilities"],
    )
    teacher_dev = _load_array(
        m30_path.parent,
        m30_seed["array_artifacts"]["teacher_development_probabilities"],
    )
    frozen_labels = _load_array(
        m30_path.parent, m30_seed["array_artifacts"]["development_labels"]
    )
    if not np.array_equal(dev_labels, frozen_labels):
        raise ValueError(f"A2 seed {seed} development labels changed.")
    cohort = select_boundary_cohort(
        teacher_train,
        fraction=float(config["cohort"]["fraction"]),
        minimum_count=int(config["cohort"]["minimum_count"]),
    )
    if cohort != m31_seed["cohort"]:
        raise ValueError(f"A2 seed {seed} cohort differs from M31.")
    cohort_indices = np.asarray(cohort["selected_indices"], dtype=np.int64)
    candidates = [
        SubspacePrimitive.from_dict(item["payload"])
        for item in parent["selected_candidates"]
    ]
    candidate_labels = [int(item.class_label) for item in candidates]
    fields = primitive_field_matrix(
        candidates,
        train_features[cohort_indices],
        primitive="subspace_r32",
        score="normalized_radial",
    )
    parent_sha256 = sha256_file(parent_path)
    started = time.perf_counter()
    equal_student = _fit_student(
        fields,
        candidate_labels,
        train_labels[cohort_indices],
        classes,
        parent,
        parent_sha256,
        config,
        weighted=False,
    )
    equal_fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    weighted_student = _fit_student(
        fields,
        candidate_labels,
        train_labels[cohort_indices],
        classes,
        parent,
        parent_sha256,
        config,
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
    replay_student = _fit_student(
        fields,
        candidate_labels,
        train_labels[cohort_indices],
        classes,
        parent,
        parent_sha256,
        config,
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
    representation_hash = seed_input["parent_representation_hash"]
    equal_inference = _median_inference_seconds(
        equal_student, dev_features, representation_hash, config["latency"]
    )
    weighted_inference = _median_inference_seconds(
        weighted_student, dev_features, representation_hash, config["latency"]
    )
    lifecycle = weighted_local_edit_rollback_evidence(
        weighted_student,
        dev_features,
        parent_representation_hash=representation_hash,
    )
    parameter_count = weighted_student_parameter_count(weighted_student)
    optimizer = weighted_student["optimizer"]
    resource_passed = (
        len(weighted_student["selected_candidates"])
        == int(config["budget"]["component_count"])
        and parameter_count <= int(config["budget"]["parameter_limit"])
        and int(optimizer["iterations"])
        <= int(config["budget"]["maximum_optimizer_iterations"])
        and int(optimizer["function_evaluations"])
        <= int(config["budget"]["maximum_function_evaluations"])
        and weighted_inference / equal_inference
        <= float(config["latency"]["maximum_ratio"])
    )
    m31_prediction = _load_array(
        m31_path.parent,
        m31_seed["selected_prediction_artifacts"][
            "direct_subspace_radial_component"
        ],
    )
    rbf_prediction = classes[np.argmax(teacher_dev, axis=1)]
    return {
        "seed": seed,
        "split_hashes": m30_seed["split_hashes"],
        "parent_representation_hash": representation_hash,
        "cohort": cohort,
        "parent_student_sha256": parent_sha256,
        "students": {"equal_weight": equal_student, "weighted": weighted_student},
        "development_labels": dev_labels,
        "predictions": {
            "equal_weight": equal_predictions,
            "weighted": weighted_predictions,
            "m31": m31_prediction,
            "rbf": rbf_prediction,
        },
        "probabilities": {
            "equal_weight": equal_probabilities,
            "weighted": weighted_probabilities,
        },
        "equal_weight": {
            "development": _metrics(
                dev_labels, equal_predictions, equal_probabilities, classes, teacher_dev
            ),
            "fit_seconds": equal_fit_seconds,
            "median_inference_seconds": equal_inference,
        },
        "weighted": {
            "development": _metrics(
                dev_labels,
                weighted_predictions,
                weighted_probabilities,
                classes,
                teacher_dev,
            ),
            "fit_seconds": weighted_fit_seconds,
            "median_inference_seconds": weighted_inference,
            "inference_time_ratio": weighted_inference / equal_inference,
            "parameter_count": parameter_count,
            "serialized_bytes": serialized_size(weighted_student),
            "optimizer": optimizer,
        },
        "controls": m31_seed["controls"],
        "m31": m31_seed["cells"]["direct_subspace_radial_component"]["development"],
        "collapse": readout_collapse_summary(
            weighted_student,
            threshold=0.9,
        ),
        "lifecycle": lifecycle,
        "exact_replay": exact_replay,
        "resource_passed": resource_passed,
    }


def _paired_intervals(
    seed_results: list[dict[str, Any]],
    first: str,
    second: str,
    statistics: dict[str, Any],
) -> dict[str, Any]:
    def balanced_accuracy(result: dict[str, Any], name: str) -> float:
        if name == "weighted":
            return float(result["weighted"]["development"]["balanced_accuracy"])
        if name in {"rbf", "weighted_knn"}:
            return float(result["controls"][name]["balanced_accuracy"])
        return float(result[name]["balanced_accuracy"])

    first_values = np.asarray(
        [balanced_accuracy(result, first) for result in seed_results]
    )
    second_values = np.asarray(
        [balanced_accuracy(result, second) for result in seed_results]
    )
    pooled_labels = np.concatenate(
        [result["development_labels"] for result in seed_results]
    )
    pooled_first = np.concatenate(
        [result["predictions"][first] for result in seed_results]
    )
    pooled_second = np.concatenate(
        [result["predictions"][second] for result in seed_results]
    )
    return {
        "seed_paired_t": paired_seed_t_interval(
            first_values,
            second_values,
            confidence=float(statistics["confidence"]),
        ),
        "pooled_per_example_bootstrap": paired_prediction_interval(
            pooled_labels,
            pooled_first,
            pooled_second,
            metric="balanced_accuracy",
            confidence=float(statistics["confidence"]),
            n_resamples=int(statistics["bootstrap_resamples"]),
            seed=int(statistics["bootstrap_seed"]),
        ),
    }


def _evaluate_gate(
    *,
    weighted_mean: float,
    strongest_same_space_mean: float,
    m31_mean: float,
    seed_results: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    tolerance = 1e-12
    parity = strongest_same_space_mean - weighted_mean <= float(
        gate["maximum_same_space_gap"]
    ) + tolerance
    m31_non_regression = m31_mean - weighted_mean <= float(
        gate["maximum_m31_accuracy_loss"]
    ) + tolerance
    replay = all(result["exact_replay"] for result in seed_results)
    rollback = all(
        result["lifecycle"]["exact_json_rollback"]
        and result["lifecycle"]["rollback_restored_predictions"]
        for result in seed_results
    )
    locality = all(
        result["lifecycle"]["unaffected_prediction_preservation"]
        >= float(gate["minimum_unaffected_prediction_preservation"])
        for result in seed_results
    )
    resources = all(result["resource_passed"] for result in seed_results)
    passed = (
        parity
        and m31_non_regression
        and (replay if gate["require_exact_replay"] else True)
        and (rollback if gate["require_exact_rollback"] else True)
        and locality
        and resources
    )
    return {
        "same_space_parity": parity,
        "m31_non_regression": m31_non_regression,
        "exact_replay_every_seed": replay,
        "exact_rollback_every_seed": rollback,
        "outside_region_preservation_every_seed": locality,
        "resources_every_seed": resources,
        "passed": passed,
    }


def run_evaluation(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    m30_config_path, m30_config = _load_locked_json(
        config["m30_config"], "M30 configuration"
    )
    m30_path, m30_evidence = _load_locked_json(
        config["m30_evidence"], "M30 evidence"
    )
    m31_path, m31_evidence = _load_locked_json(
        config["m31_evidence"], "M31 evidence"
    )
    weighted_path, weighted_evidence = _load_locked_json(
        config["a1_weighted_evidence"], "A1-W evidence"
    )
    tangent_path, tangent_evidence = _load_locked_json(
        config["a1_tangent_evidence"], "A1-T evidence"
    )
    budget_path, budget_evidence = _load_locked_json(
        config["a1_budget_evidence"], "A1-B evidence"
    )
    if (
        weighted_evidence["advancement_passed"] is not True
        or tangent_evidence["advancement_passed"] is not False
        or budget_evidence["a2_component_count_unchanged"] != 46
        or m31_evidence["selected_cell"] != "direct_subspace_radial_component"
    ):
        raise ValueError("A2 parent decisions do not authorize this scope.")
    seeds = [int(seed) for seed in config["seeds"]]
    seed_results = [
        _fit_seed(
            seed,
            config,
            m30_path,
            m30_config,
            m30_evidence,
            m31_path,
            m31_evidence,
        )
        for seed in seeds
    ]
    weighted_mean = float(
        np.mean(
            [
                result["weighted"]["development"]["balanced_accuracy"]
                for result in seed_results
            ]
        )
    )
    m31_mean = float(
        np.mean([result["m31"]["balanced_accuracy"] for result in seed_results])
    )
    controls = {
        name: float(
            np.mean(
                [
                    result["controls"][name]["balanced_accuracy"]
                    for result in seed_results
                ]
            )
        )
        for name in ("rbf", "weighted_knn")
    }
    strongest_name = max(controls, key=controls.__getitem__)
    strongest_mean = controls[strongest_name]
    gate_operands = _evaluate_gate(
        weighted_mean=weighted_mean,
        strongest_same_space_mean=strongest_mean,
        m31_mean=m31_mean,
        seed_results=seed_results,
        gate=config["parity_gate"],
    )
    intervals = {
        "weighted_vs_rbf": _paired_intervals(
            seed_results, "weighted", "rbf", config["statistics"]
        ),
        "weighted_vs_m31": _paired_intervals(
            seed_results, "weighted", "m31", config["statistics"]
        ),
    }

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    serializable_results: dict[str, Any] = {}
    deterministic_paths: list[str] = []
    for result in seed_results:
        seed = result["seed"]
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir()
        for name, student in result["students"].items():
            path = seed_dir / f"{name}_student.json"
            write_canonical_json(path, student)
            deterministic_paths.append(path.relative_to(output_dir).as_posix())
        artifacts = {}
        arrays = {
            "development_labels": result["development_labels"],
            **{
                f"{name}_development_predictions": values
                for name, values in result["predictions"].items()
            },
            **{
                f"{name}_development_probabilities": values
                for name, values in result["probabilities"].items()
            },
        }
        for name, values in arrays.items():
            path = seed_dir / f"{name}.npy"
            np.save(path, values, allow_pickle=False)
            relative = path.relative_to(output_dir).as_posix()
            deterministic_paths.append(relative)
            artifacts[name] = {"path": relative, "sha256": sha256_file(path)}
        serializable_results[str(seed)] = {
            key: value
            for key, value in result.items()
            if key
            not in {
                "students",
                "development_labels",
                "predictions",
                "probabilities",
            }
        }
        serializable_results[str(seed)]["array_artifacts"] = artifacts
    evidence = {
        "schema_version": 1,
        "amendment": "v6.1",
        "milestone": "A2",
        "stage": "S2",
        "configuration_hash": payload_hash(config),
        "parent_hashes": {
            "m30_config": sha256_file(m30_config_path),
            "m30_evidence": sha256_file(m30_path),
            "m31_evidence": sha256_file(m31_path),
            "a1_weighted_evidence": sha256_file(weighted_path),
            "a1_tangent_evidence": sha256_file(tangent_path),
            "a1_budget_evidence": sha256_file(budget_path),
        },
        "seeds": seeds,
        "retained_mechanism": "weighted_affine_rank32",
        "closed_mechanisms": config["closed_mechanisms"],
        "seed_results": serializable_results,
        "mean_metrics": {
            "weighted_balanced_accuracy": weighted_mean,
            "weighted_nll": float(
                np.mean(
                    [
                        result["weighted"]["development"]["nll"]
                        for result in seed_results
                    ]
                )
            ),
            "m31_balanced_accuracy": m31_mean,
            "rbf_balanced_accuracy": controls["rbf"],
            "weighted_knn_balanced_accuracy": controls["weighted_knn"],
        },
        "strongest_same_space_control": strongest_name,
        "strongest_same_space_balanced_accuracy": strongest_mean,
        "same_space_gap": strongest_mean - weighted_mean,
        "m31_accuracy_difference": weighted_mean - m31_mean,
        "paired_intervals": intervals,
        "gate_operands": gate_operands,
        "predictive_amendment_passed": gate_operands["passed"],
        "final_outcome": "retained" if gate_operands["passed"] else "Outcome D",
        "test_labels_opened": False,
        "deterministic_paths": deterministic_paths,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    index = build_artifact_index(output_dir)
    return {
        "weighted_balanced_accuracy": weighted_mean,
        "strongest_same_space_balanced_accuracy": strongest_mean,
        "same_space_gap": strongest_mean - weighted_mean,
        "m31_accuracy_difference": weighted_mean - m31_mean,
        "predictive_amendment_passed": gate_operands["passed"],
        "final_outcome": evidence["final_outcome"],
        "artifact_count": len(index["artifacts"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run_evaluation(args.config, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
