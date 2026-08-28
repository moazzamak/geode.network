"""Run the M28 S1 boundary-distillation falsification study."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.classification_metrics import (
    accuracy,
    balanced_accuracy,
    negative_log_likelihood,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v5_frozen_representations import RepresentationManifest
from experiments.common.v6_boundary_distillation import (
    fit_boundary_distilled_student,
    generate_margin_sphere_candidates,
    predict_boundary_student,
)
from experiments.common.v6_protocol import select_boundary_cohort
from experiments.tier4.eval_v5_native_dinov2_sphere import _load_bound_cache


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "configs"
    / "v6"
    / "m28_boundary_distillation_s1.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "logs" / "results" / "v6" / "m28_boundary_distillation_s1"
)


def _resolve(path: str) -> Path:
    return REPO_ROOT / Path(path)


def _load_hashed_array(path: Path, expected_hash: str) -> np.ndarray:
    if sha256_file(path) != expected_hash:
        raise ValueError(f"Array hash mismatch for {path}.")
    return np.load(path, allow_pickle=False)


def _metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    teacher_probabilities: np.ndarray,
) -> dict[str, float]:
    log_student = np.log(
        np.maximum(probabilities, np.finfo(np.float64).tiny)
    )
    teacher_kl = float(
        np.mean(
            np.sum(
                teacher_probabilities
                * (
                    np.log(
                        np.maximum(
                            teacher_probabilities,
                            np.finfo(np.float64).tiny,
                        )
                    )
                    - log_student
                ),
                axis=1,
            )
        )
    )
    return {
        "accuracy": accuracy(labels, predictions),
        "balanced_accuracy": balanced_accuracy(labels, predictions),
        "nll": negative_log_likelihood(labels, probabilities, classes),
        "teacher_agreement": float(
            np.mean(predictions == classes[np.argmax(teacher_probabilities, axis=1)])
        ),
        "teacher_probability_kl": teacher_kl,
    }


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1 or config.get("milestone") != "M28":
        raise ValueError("Unsupported M28 configuration.")
    if config.get("stage") != "S1" or config.get("seed") != 11:
        raise ValueError("M28 S1 is locked to seed 11.")
    if config.get("test_used_for_selection") is not False:
        raise PermissionError("M28 S1 must not use test labels for selection.")
    if config["budget"]["mode"] != "component_matched":
        raise ValueError("M28 S1 requires a component-matched budget.")


def run_evaluation(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    feature_dir = _resolve(config["feature_dir"])
    baseline_path = _resolve(config["m27_prediction_baseline"])
    teacher_manifest_path = _resolve(config["teacher_manifest"])
    extraction = json.loads(
        (feature_dir / "extraction_summary.json").read_text(encoding="utf-8")
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    teacher_manifest = json.loads(
        teacher_manifest_path.read_text(encoding="utf-8")
    )
    backbone_id = "dinov2-small"
    extracted = extraction["representations"][backbone_id]
    manifest = RepresentationManifest.from_dict(extracted["manifest"])
    if (
        manifest.representation_hash != config["representation_hash"]
        or baseline["representation_hash"] != manifest.representation_hash
        or teacher_manifest["representation_hash"] != manifest.representation_hash
    ):
        raise ValueError("M28 representation lineage mismatch.")
    if teacher_manifest["training_split_hash"] != extraction["split_hashes"]["train"]:
        raise ValueError("M28 teacher training split mismatch.")

    datasets = {
        split: _load_bound_cache(
            feature_dir,
            backbone_id,
            manifest,
            extracted["cache_metadata"][split],
            split,
        )
        for split in ("train", "dev", "test")
    }
    train_features, train_labels = datasets["train"]
    classes = np.asarray(teacher_manifest["classes"], dtype=np.int64)
    teacher_train_labels = _load_hashed_array(
        _resolve(teacher_manifest["labels_path"]),
        teacher_manifest["labels_sha256"],
    )
    teacher_train_probabilities = _load_hashed_array(
        _resolve(teacher_manifest["probabilities_path"]),
        teacher_manifest["probabilities_sha256"],
    )
    if not np.array_equal(train_labels, teacher_train_labels):
        raise ValueError("Teacher and feature-cache training labels differ.")

    teacher_probabilities = {"train": teacher_train_probabilities}
    for source_split, output_split in (("dev", "development"), ("test", "test")):
        item = baseline["heads"]["rbf_svm"][output_split]
        teacher_probabilities[source_split] = _load_hashed_array(
            _resolve(item["probabilities_path"]),
            item["probabilities_sha256"],
        )

    objective = config["objective"]
    cohort = select_boundary_cohort(
        teacher_train_probabilities,
        fraction=float(objective["cohort_fraction"]),
        minimum_count=int(objective["cohort_minimum_count"]),
    )
    candidate_config = config["candidate_generation"]
    started = time.perf_counter()
    candidates = generate_margin_sphere_candidates(
        train_features,
        train_labels,
        teacher_train_probabilities,
        classes,
        candidates_per_class=int(candidate_config["candidates_per_class"]),
        seed_size=int(candidate_config["seed_size"]),
        anchor_fraction=float(candidate_config["anchor_fraction"]),
    )
    candidate_seconds = time.perf_counter() - started

    started = time.perf_counter()
    student = fit_boundary_distilled_student(
        train_features,
        train_labels,
        teacher_train_probabilities,
        classes,
        candidates,
        np.asarray(cohort["selected_indices"], dtype=np.int64),
        component_limit=int(config["budget"]["component_limit"]),
        teacher_weight=float(objective["teacher_weight"]),
        ground_truth_weight=float(objective["ground_truth_weight"]),
        complexity_penalty=float(objective["complexity_penalty"]),
        minimum_improvement=float(objective["minimum_improvement"]),
    )
    fit_seconds = time.perf_counter() - started

    split_results = {}
    prediction_payloads = {}
    for source_split, output_split in (("dev", "development"), ("test", "test")):
        features, labels = datasets[source_split]
        predictions, probabilities = predict_boundary_student(student, features)
        split_results[output_split] = _metrics(
            labels,
            predictions,
            probabilities,
            classes,
            teacher_probabilities[source_split],
        )
        prediction_payloads[output_split] = (predictions, probabilities)

    dev_accuracy = split_results["development"]["balanced_accuracy"]
    teacher_dev = baseline["heads"]["rbf_svm"]["development"]["balanced_accuracy"]
    baseline_dev = baseline["heads"]["current_geode"]["development"][
        "balanced_accuracy"
    ]
    gate = config["s1_gate"]
    teacher_gap = float(teacher_dev - dev_accuracy)
    baseline_improvement = float(dev_accuracy - baseline_dev)
    monotone_objective = all(
        next_value < value
        for value, next_value in zip(
            student["objective_trajectory"],
            student["objective_trajectory"][1:],
        )
    )
    gate_operands = {
        "teacher_gap": {
            "value": teacher_gap,
            "operator": "le",
            "threshold": float(gate["maximum_teacher_gap"]),
            "passed": teacher_gap <= float(gate["maximum_teacher_gap"]),
        },
        "baseline_improvement": {
            "value": baseline_improvement,
            "operator": "ge",
            "threshold": float(gate["minimum_baseline_improvement"]),
            "passed": baseline_improvement
            >= float(gate["minimum_baseline_improvement"]),
        },
        "component_budget": {
            "value": len(student["selected_candidates"]),
            "operator": "le",
            "threshold": int(config["budget"]["component_limit"]),
            "passed": len(student["selected_candidates"])
            <= int(config["budget"]["component_limit"]),
        },
        "monotone_objective": {
            "value": monotone_objective,
            "operator": "eq",
            "threshold": True,
            "passed": monotone_objective,
        },
    }
    advancement_passed = all(item["passed"] for item in gate_operands.values())

    if output_dir.exists():
        shutil.rmtree(output_dir)
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True)
    prediction_artifacts = {}
    for split, (predictions, probabilities) in prediction_payloads.items():
        predictions_path = arrays_dir / f"{split}_predictions.npy"
        probabilities_path = arrays_dir / f"{split}_probabilities.npy"
        np.save(predictions_path, predictions, allow_pickle=False)
        np.save(probabilities_path, probabilities, allow_pickle=False)
        prediction_artifacts[split] = {
            "predictions": {
                "path": predictions_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(predictions_path),
            },
            "probabilities": {
                "path": probabilities_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(probabilities_path),
            },
        }

    write_canonical_json(output_dir / "student.json", student)
    evidence = {
        "schema_version": 1,
        "milestone": "M28",
        "stage": "S1",
        "seed": config["seed"],
        "representation_hash": manifest.representation_hash,
        "configuration_hash": payload_hash(config),
        "teacher_manifest_hash": sha256_file(teacher_manifest_path),
        "prediction_baseline_hash": sha256_file(baseline_path),
        "split_hashes": {
            "train": extraction["split_hashes"]["train"],
            "development": extraction["split_hashes"]["dev"],
            "test": extraction["split_hashes"]["test"],
        },
        "feature_hashes": {
            "train": extracted["cache_metadata"]["train"]["feature_file_hash"],
            "development": extracted["cache_metadata"]["dev"]["feature_file_hash"],
            "test": extracted["cache_metadata"]["test"]["feature_file_hash"],
        },
        "candidate_count": len(candidates),
        "selected_component_count": len(student["selected_candidates"]),
        "component_counts": student["component_counts"],
        "cohort": cohort,
        "objective_initial": student["objective_trajectory"][0],
        "objective_final": student["objective_trajectory"][-1],
        "objective_steps": len(student["objective_trajectory"]) - 1,
        "timing": {
            "candidate_generation_seconds": candidate_seconds,
            "student_fit_seconds": fit_seconds,
        },
        "metrics": split_results,
        "locked_controls": {
            "current_geode_development_balanced_accuracy": baseline_dev,
            "rbf_teacher_development_balanced_accuracy": teacher_dev,
            "current_geode_test_balanced_accuracy": baseline["heads"][
                "current_geode"
            ]["test"]["balanced_accuracy"],
            "rbf_teacher_test_balanced_accuracy": baseline["heads"]["rbf_svm"][
                "test"
            ]["balanced_accuracy"],
        },
        "gate_operands": gate_operands,
        "advancement_passed": advancement_passed,
        "test_used_for_selection": False,
        "prediction_artifacts": prediction_artifacts,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    index = build_artifact_index(output_dir)
    return {
        "advancement_passed": advancement_passed,
        "development_balanced_accuracy": dev_accuracy,
        "test_balanced_accuracy": split_results["test"]["balanced_accuracy"],
        "teacher_gap": teacher_gap,
        "baseline_improvement": baseline_improvement,
        "selected_component_count": len(student["selected_candidates"]),
        "artifact_count": len(index["artifacts"]),
        "evidence_hash": sha256_file(output_dir / "evidence.json"),
    }


def verify_replay(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        first = Path(temporary) / "first"
        second = Path(temporary) / "second"
        first_summary = run_evaluation(config_path, first)
        run_evaluation(config_path, second)
        excluded = {"evidence.json", "artifact_index.json"}
        first_files = {
            path.relative_to(first).as_posix(): path.read_bytes()
            for path in first.rglob("*")
            if path.is_file() and path.name not in excluded
        }
        second_files = {
            path.relative_to(second).as_posix(): path.read_bytes()
            for path in second.rglob("*")
            if path.is_file() and path.name not in excluded
        }
        if first_files != second_files:
            raise RuntimeError("M28 model and prediction replay was not byte-identical.")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(first, output_dir)
    evidence_path = output_dir / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["byte_identical_model_prediction_replay"] = True
    write_canonical_json(evidence_path, evidence)
    build_artifact_index(output_dir)
    return {
        **first_summary,
        "byte_identical_model_prediction_replay": True,
        "evidence_hash": sha256_file(evidence_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-replay", action="store_true")
    args = parser.parse_args()
    runner = verify_replay if args.verify_replay else run_evaluation
    print(json.dumps(runner(args.config, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
