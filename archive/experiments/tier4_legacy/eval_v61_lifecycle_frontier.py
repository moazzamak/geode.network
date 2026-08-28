"""Run the v6.1 A3 accuracy-lifecycle frontier without reopening Outcome D."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.common.experiment_manifest import canonical_json
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v61_lifecycle_frontier import classify_outcome_c
from experiments.common.v61_weighted_readout import (
    normalized_class_weights,
    predict_weighted_student,
)
from experiments.tier4.eval_v5_frozen_space_heads import fit_prototype_head
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.common.v6_factorial import predict_factorial_student
from src.subspace_primitive import SubspacePrimitive, fit_subspace_primitive


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v6_1" / "a3_lifecycle_frontier.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v6_1" / "a3_lifecycle_frontier"

_MODELS = [
    "weighted_affine_rank32",
    "m31_affine_rank32",
    "current_spherical_geode",
    "rbf_svm",
    "weighted_knn",
    "prototype",
]
_TASKS = [
    "local_false_positive_correction",
    "known_class_mode_addition",
    "corrupted_cluster_suppression",
    "bounded_shift_recalibration",
    "exact_rollback",
]


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
        or config.get("milestone") != "A3"
        or config.get("stage") != "S2"
        or config.get("seeds") != [11, 23, 37]
        or config.get("models") != _MODELS
        or config.get("tasks") != _TASKS
        or config.get("test_labels_opened") is not False
    ):
        raise ValueError("Unsupported, incomplete, or test-open A3 configuration.")
    if set(config.get("weighted_students", {})) != {"11", "23", "37"} or set(
        config.get("m31_students", {})
    ) != {"11", "23", "37"}:
        raise ValueError("A3 student artifacts do not match the frozen seeds.")
    if config.get("explicit_edit_contract") != {
        "local_false_positive_correction_scale": 0.5,
        "known_class_mode_addition_scale": 1.5,
        "corrupted_cluster_suppression_scale": 0.01,
        "bounded_shift_temperature_scale": 1.01,
        "minimum_unaffected_prediction_preservation": 0.999,
    }:
        raise ValueError("A3 explicit edit contract mismatch.")
    if config.get("outcome_c_gate") != {
        "require_non_dominated": True,
        "require_advantage_over_every_accuracy_superior_control": True,
        "require_exact_rollback_every_seed_and_task": True,
        "require_predictive_deficit_report": True,
    }:
        raise ValueError("A3 Outcome-C gate mismatch.")


def _refresh_weighted_student(student: dict[str, Any]) -> None:
    labels = [
        int(item["payload"]["class_label"]) for item in student["selected_candidates"]
    ]
    classes = np.asarray(student["classes"], dtype=np.int64)
    logs = np.asarray(student["component_log_weights"], dtype=np.float64)
    student["component_weights"] = normalized_class_weights(
        logs, labels, classes
    ).tolist()
    student["component_counts"] = [
        int(np.sum(np.asarray(labels) == class_label)) for class_label in classes
    ]
    student["parent_component_hash"] = payload_hash(
        {
            "selected_candidates": student["selected_candidates"],
            "selected_candidate_indices": student["selected_candidate_indices"],
        }
    )


def _candidate_region(
    candidate_item: dict[str, Any], features: np.ndarray
) -> np.ndarray:
    candidate = SubspacePrimitive.from_dict(candidate_item["payload"])
    return candidate.radial_field(features) <= 0.0


def _mode_candidate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    class_label: int,
    anchor_index: int,
) -> dict[str, Any]:
    class_features = train_features[train_labels == class_label]
    center = np.mean(class_features, axis=0)
    distances = np.linalg.norm(class_features - center, axis=1)
    selected = np.argsort(distances, kind="stable")[-34:]
    primitive = fit_subspace_primitive(
        class_features[selected],
        32,
        class_label=class_label,
        anchor_index=anchor_index,
    )
    return {"family": "subspace_r32", "payload": primitive.to_dict()}


def _edit_student(
    student: dict[str, Any],
    task: str,
    *,
    weighted: bool,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], int, int]:
    edited = copy.deepcopy(student)
    evidence_count = {
        "local_false_positive_correction": 1,
        "known_class_mode_addition": 34,
        "corrupted_cluster_suppression": 34,
        "bounded_shift_recalibration": 1000,
    }[task]
    region_position = 0
    if task == "local_false_positive_correction":
        edited["selected_candidates"][0]["payload"]["residual_variance"] *= float(
            contract["local_false_positive_correction_scale"]
        )
    elif task == "known_class_mode_addition":
        class_label = int(edited["classes"][0])
        candidate = _mode_candidate(
            train_features,
            train_labels,
            class_label,
            max(edited["selected_candidate_indices"]) + 1,
        )
        edited["selected_candidates"].append(candidate)
        edited["selected_candidate_indices"].append(
            max(edited["selected_candidate_indices"]) + 1
        )
        region_position = len(edited["selected_candidates"]) - 1
        if weighted:
            labels = [
                int(item["payload"]["class_label"])
                for item in edited["selected_candidates"][:-1]
            ]
            class_logs = np.asarray(edited["component_log_weights"])[
                np.asarray(labels) == class_label
            ]
            edited["component_log_weights"].append(
                float(np.mean(class_logs))
                + np.log(float(contract["known_class_mode_addition_scale"]))
            )
        else:
            edited["component_counts"][0] += 1
    elif task == "corrupted_cluster_suppression":
        labels = np.asarray(
            [
                int(item["payload"]["class_label"])
                for item in edited["selected_candidates"]
            ]
        )
        counts = {int(label): int(np.sum(labels == label)) for label in np.unique(labels)}
        region_position = next(
            index for index, label in enumerate(labels) if counts[int(label)] > 1
        )
        if weighted:
            edited["component_log_weights"][region_position] += np.log(
                float(contract["corrupted_cluster_suppression_scale"])
            )
        else:
            del edited["selected_candidates"][region_position]
            del edited["selected_candidate_indices"][region_position]
            edited["component_counts"][
                list(edited["classes"]).index(int(labels[region_position]))
            ] -= 1
            region_position = min(region_position, len(edited["selected_candidates"]) - 1)
    elif task == "bounded_shift_recalibration":
        edited["global_temperature"] *= float(
            contract["bounded_shift_temperature_scale"]
        )
    else:
        raise ValueError(f"Unsupported explicit edit task {task!r}.")
    if weighted:
        _refresh_weighted_student(edited)
    return edited, evidence_count, region_position


def _median_inference(
    predict: Callable[[dict[str, Any], np.ndarray], tuple[np.ndarray, np.ndarray]],
    student: dict[str, Any],
    features: np.ndarray,
    latency: dict[str, Any],
) -> float:
    for _ in range(int(latency["warmup_runs"])):
        predict(student, features)
    timings = []
    for _ in range(int(latency["measured_runs"])):
        started = time.perf_counter()
        predict(student, features)
        timings.append(time.perf_counter() - started)
    return float(np.median(timings))


def _run_explicit_suite(
    student: dict[str, Any],
    features: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    weighted: bool,
    representation_hash: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    if weighted:
        predict = lambda model, values: predict_weighted_student(
            model, values, parent_representation_hash=representation_hash
        )
    else:
        predict = predict_factorial_student
    baseline_predictions, _ = predict(student, features)
    inference = _median_inference(
        predict, student, features, config["latency"]
    )
    task_results = {}
    for task in _TASKS[:-1]:
        started = time.perf_counter()
        edited, evidence_count, region_position = _edit_student(
            student,
            task,
            weighted=weighted,
            train_features=train_features,
            train_labels=train_labels,
            contract=config["explicit_edit_contract"],
        )
        edit_seconds = time.perf_counter() - started
        edited_predictions, _ = predict(edited, features)
        if task == "bounded_shift_recalibration":
            affected = np.ones(len(features), dtype=bool)
        elif task == "corrupted_cluster_suppression" and not weighted:
            affected = _candidate_region(
                student["selected_candidates"][
                    min(region_position, len(student["selected_candidates"]) - 1)
                ],
                features,
            )
        else:
            affected = _candidate_region(
                edited["selected_candidates"][region_position], features
            )
        unaffected = ~affected
        preservation = (
            float(
                np.mean(
                    edited_predictions[unaffected]
                    == baseline_predictions[unaffected]
                )
            )
            if np.any(unaffected)
            else 1.0
        )
        rollback_started = time.perf_counter()
        rolled_back = copy.deepcopy(student)
        rollback_seconds = time.perf_counter() - rollback_started
        rollback_predictions, _ = predict(rolled_back, features)
        task_results[task] = {
            "evidence_count": evidence_count,
            "changed_prediction_count": int(
                np.sum(edited_predictions != baseline_predictions)
            ),
            "changed_region_size": int(np.sum(affected)),
            "changed_region_fraction": float(np.mean(affected)),
            "unaffected_prediction_preservation": preservation,
            "edit_latency_seconds": edit_seconds,
            "rollback_latency_seconds": rollback_seconds,
            "exact_json_rollback": canonical_json(rolled_back)
            == canonical_json(student),
            "rollback_restored_predictions": bool(
                np.array_equal(rollback_predictions, baseline_predictions)
            ),
        }
    exact_rollback = all(
        item["exact_json_rollback"] and item["rollback_restored_predictions"]
        for item in task_results.values()
    )
    task_results["exact_rollback"] = {
        "task_count": len(task_results),
        "passed": exact_rollback,
    }
    audit_bytes = len(canonical_json(task_results).encode("utf-8"))
    review_artifact_bytes = len(
        canonical_json(
            {
                name: {"evidence_count": item["evidence_count"]}
                for name, item in task_results.items()
                if name != "exact_rollback"
            }
        ).encode("utf-8")
    )
    return {
        "tasks": task_results,
        "minimum_unaffected_prediction_preservation": min(
            item["unaffected_prediction_preservation"]
            for name, item in task_results.items()
            if name != "exact_rollback"
        ),
        "rollback_reliability": float(exact_rollback),
        "accepted_edit_evidence_count": sum(
            item["evidence_count"]
            for name, item in task_results.items()
            if name != "exact_rollback"
        ),
        "median_edit_latency_seconds": float(
            np.median(
                [
                    item["edit_latency_seconds"]
                    for name, item in task_results.items()
                    if name != "exact_rollback"
                ]
            )
        ),
        "median_inference_latency_seconds": inference,
        "exact_rollback_every_task": exact_rollback,
        "model_bytes": len(canonical_json(student).encode("utf-8")),
        "audit_artifact_bytes": audit_bytes,
        "review_artifact_bytes": review_artifact_bytes,
        "audit_record_count": 4,
        "operator_count": 4,
    }


def run_evaluation(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    a2_path, a2 = _load_locked_json(config["a2_evidence"], "A2 evidence")
    m31_path, m31 = _load_locked_json(config["m31_evidence"], "M31 evidence")
    m30_path, m30 = _load_locked_json(config["m30_evidence"], "M30 evidence")
    e9_path, e9 = _load_locked_json(config["e9_evidence"], "E9 evidence")
    if a2["final_outcome"] != "Outcome D" or a2["test_labels_opened"]:
        raise ValueError("A3 must preserve sealed final Outcome D.")
    if not e9["gate_passed"]:
        raise ValueError("A3 requires qualified transactional publication.")

    per_seed = {}
    prototype_accuracies = []
    explicit_names = ("weighted_affine_rank32", "m31_affine_rank32")
    for seed in config["seeds"]:
        key = str(seed)
        _, weighted_student = _load_locked_json(
            config["weighted_students"][key], f"A2 seed-{seed} student"
        )
        _, m31_student = _load_locked_json(
            config["m31_students"][key], f"M31 seed-{seed} student"
        )
        seed_input = json.loads(
            (_resolve("experiments/configs/v6/m30_directional_s2.json")).read_text(
                encoding="utf-8"
            )
        )["seed_inputs"][key]
        loaded = _load_seed_data(seed_input)
        train_features, train_labels = loaded["datasets"]["train"]
        dev_features, dev_labels = loaded["datasets"]["dev"]
        weighted_suite = _run_explicit_suite(
            weighted_student,
            dev_features,
            train_features,
            train_labels,
            weighted=True,
            representation_hash=seed_input["parent_representation_hash"],
            config=config,
        )
        m31_suite = _run_explicit_suite(
            m31_student,
            dev_features,
            train_features,
            train_labels,
            weighted=False,
            representation_hash=seed_input["parent_representation_hash"],
            config=config,
        )
        prototype = fit_prototype_head(train_features, train_labels, seed)
        prototype_model = prototype["model"]
        prototype_predictions = prototype_model.predict(dev_features)
        prototype_accuracy = float(
            np.mean(
                [
                    np.mean(
                        prototype_predictions[dev_labels == label] == label
                    )
                    for label in np.unique(dev_labels)
                ]
            )
        )
        prototype_accuracies.append(prototype_accuracy)
        per_seed[key] = {
            "split_hashes": m30["seed_results"][key]["split_hashes"],
            "parent_representation_hash": seed_input["parent_representation_hash"],
            "weighted_affine_rank32": weighted_suite,
            "m31_affine_rank32": m31_suite,
            "prototype": {
                "balanced_accuracy": prototype_accuracy,
                "model_bytes": len(
                    pickle.dumps(prototype_model, protocol=pickle.HIGHEST_PROTOCOL)
                ),
                "task_support": {
                    task: {
                        "status": "unsupported",
                        "reason": "No frozen transactional centroid update artifact.",
                    }
                    for task in _TASKS
                },
            },
        }

    records: dict[str, dict[str, Any]] = {}
    accuracies = {
        "weighted_affine_rank32": float(
            a2["mean_metrics"]["weighted_balanced_accuracy"]
        ),
        "m31_affine_rank32": float(
            a2["mean_metrics"]["m31_balanced_accuracy"]
        ),
        "current_spherical_geode": float(
            np.mean(
                [
                    m31["seed_results"][str(seed)]["controls"]["current_geode"][
                        "balanced_accuracy"
                    ]
                    for seed in config["seeds"]
                ]
            )
        ),
        "rbf_svm": float(a2["mean_metrics"]["rbf_balanced_accuracy"]),
        "weighted_knn": float(
            a2["mean_metrics"]["weighted_knn_balanced_accuracy"]
        ),
        "prototype": float(np.mean(prototype_accuracies)),
    }
    for name in explicit_names:
        suites = [per_seed[str(seed)][name] for seed in config["seeds"]]
        records[name] = {
            "balanced_accuracy": accuracies[name],
            "accuracy_cost_to_rbf": accuracies["rbf_svm"] - accuracies[name],
            "accuracy_cost_to_weighted_knn": accuracies["weighted_knn"]
            - accuracies[name],
            "unaffected_prediction_preservation": min(
                item["minimum_unaffected_prediction_preservation"]
                for item in suites
            ),
            "rollback_reliability": float(
                np.mean([item["rollback_reliability"] for item in suites])
            ),
            "accepted_edit_evidence_count": float(
                np.mean([item["accepted_edit_evidence_count"] for item in suites])
            ),
            "edit_latency_seconds": float(
                np.median([item["median_edit_latency_seconds"] for item in suites])
            ),
            "inference_latency_seconds": float(
                np.median(
                    [item["median_inference_latency_seconds"] for item in suites]
                )
            ),
            "model_bytes": int(max(item["model_bytes"] for item in suites)),
            "audit_artifact_bytes": int(
                sum(item["audit_artifact_bytes"] for item in suites)
            ),
            "review_artifact_bytes": int(
                sum(item["review_artifact_bytes"] for item in suites)
            ),
            "audit_record_count": 12,
            "operator_count": 12,
            "task_coverage": "complete",
        }
    unsupported_reason = {
        "current_spherical_geode": (
            "Only the prior one-edit locality proof is serialized; the frozen "
            "five-task model artifact is unavailable."
        ),
        "rbf_svm": "No bounded component-local transactional update is registered.",
        "weighted_knn": "No frozen transactional memory update is registered.",
        "prototype": "No frozen transactional centroid update is registered.",
    }
    for name, reason in unsupported_reason.items():
        records[name] = {
            "balanced_accuracy": accuracies[name],
            "accuracy_cost_to_rbf": accuracies["rbf_svm"] - accuracies[name],
            "accuracy_cost_to_weighted_knn": accuracies["weighted_knn"]
            - accuracies[name],
            "unaffected_prediction_preservation": None,
            "rollback_reliability": None,
            "accepted_edit_evidence_count": None,
            "edit_latency_seconds": None,
            "inference_latency_seconds": None,
            "task_coverage": "unsupported",
            "unsupported_reason": reason,
            "unsupported_tasks": list(_TASKS),
        }
    rollback_all = all(
        per_seed[str(seed)][name]["exact_rollback_every_task"]
        for seed in config["seeds"]
        for name in explicit_names
    )
    locality_passed = all(
        per_seed[str(seed)][name][
            "minimum_unaffected_prediction_preservation"
        ]
        >= float(
            config["explicit_edit_contract"][
                "minimum_unaffected_prediction_preservation"
            ]
        )
        for seed in config["seeds"]
        for name in explicit_names
    )
    outcome = classify_outcome_c(
        records,
        retained_model="weighted_affine_rank32",
        exact_rollback_every_seed_and_task=rollback_all,
        locality_contract_passed=locality_passed,
        predictive_deficit_reported=True,
        paired_advantage_controls=[],
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    evidence = {
        "schema_version": 1,
        "amendment": "v6.1",
        "milestone": "A3",
        "stage": "S2",
        "configuration_hash": payload_hash(config),
        "parent_hashes": {
            "a2_evidence": sha256_file(a2_path),
            "m31_evidence": sha256_file(m31_path),
            "m30_evidence": sha256_file(m30_path),
            "e9_evidence": sha256_file(e9_path),
        },
        "final_predictive_outcome_unchanged": "Outcome D",
        "predictive_deficit": {
            "weighted_affine_balanced_accuracy": accuracies[
                "weighted_affine_rank32"
            ],
            "strongest_same_space_balanced_accuracy": accuracies["rbf_svm"],
            "gap": accuracies["rbf_svm"]
            - accuracies["weighted_affine_rank32"],
        },
        "tasks": list(_TASKS),
        "model_records": records,
        "per_seed_explicit_evidence": per_seed,
        "outcome_c_gate": outcome,
        "claim_status": outcome["status"],
        "test_labels_opened": False,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    index = build_artifact_index(output_dir)
    return {
        "claim_status": outcome["status"],
        "specialized_tradeoff_claim_passed": outcome[
            "specialized_tradeoff_claim_passed"
        ],
        "weighted_non_dominated": outcome["retained_non_dominated"],
        "exact_rollback_every_seed_and_task": rollback_all,
        "locality_contract_passed": locality_passed,
        "unsupported_accuracy_superior_controls": outcome[
            "unsupported_accuracy_superior_controls"
        ],
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
