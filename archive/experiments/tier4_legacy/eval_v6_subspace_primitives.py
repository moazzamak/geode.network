"""Run the M29 DINOv2 subspace rank and score sweep."""

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
    fit_distilled_candidate_fields,
)
from experiments.common.v6_protocol import select_boundary_cohort
from experiments.common.v6_subspace_distillation import (
    generate_margin_subspace_candidates,
    predict_subspace_student,
    serialize_subspace_student,
    subspace_field_matrix,
)
from experiments.tier4.eval_v5_native_dinov2_sphere import _load_bound_cache


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v6" / "m29_subspace_s1.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v6" / "m29_subspace_s1"


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
    return {
        "accuracy": accuracy(labels, predictions),
        "balanced_accuracy": balanced_accuracy(labels, predictions),
        "nll": negative_log_likelihood(labels, probabilities, classes),
        "teacher_agreement": float(
            np.mean(predictions == classes[np.argmax(teacher_probabilities, axis=1)])
        ),
    }


def _load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    feature_dir = _resolve(config["feature_dir"])
    extraction = json.loads(
        (feature_dir / "extraction_summary.json").read_text(encoding="utf-8")
    )
    baseline_path = _resolve(config["m27_prediction_baseline"])
    teacher_path = _resolve(config["teacher_manifest"])
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    extracted = extraction["representations"]["dinov2-small"]
    manifest = RepresentationManifest.from_dict(extracted["manifest"])
    if {
        manifest.representation_hash,
        baseline["representation_hash"],
        teacher["representation_hash"],
        config["representation_hash"],
    } != {config["representation_hash"]}:
        raise ValueError("M29 representation lineage mismatch.")
    datasets = {
        split: _load_bound_cache(
            feature_dir,
            "dinov2-small",
            manifest,
            extracted["cache_metadata"][split],
            split,
        )
        for split in ("train", "dev", "test")
    }
    train_probabilities = _load_hashed_array(
        _resolve(teacher["probabilities_path"]), teacher["probabilities_sha256"]
    )
    train_labels = _load_hashed_array(
        _resolve(teacher["labels_path"]), teacher["labels_sha256"]
    )
    if not np.array_equal(train_labels, datasets["train"][1]):
        raise ValueError("M29 teacher and cache labels differ.")
    teacher_probabilities = {"train": train_probabilities}
    for source_split, output_split in (("dev", "development"), ("test", "test")):
        item = baseline["heads"]["rbf_svm"][output_split]
        teacher_probabilities[source_split] = _load_hashed_array(
            _resolve(item["probabilities_path"]), item["probabilities_sha256"]
        )
    return {
        "feature_dir": feature_dir,
        "extraction": extraction,
        "extracted": extracted,
        "manifest": manifest,
        "baseline": baseline,
        "baseline_path": baseline_path,
        "teacher": teacher,
        "teacher_path": teacher_path,
        "datasets": datasets,
        "teacher_probabilities": teacher_probabilities,
    }


def run_evaluation(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != 1
        or config.get("milestone") != "M29"
        or config.get("stage") != "S1"
    ):
        raise ValueError("Unsupported M29 configuration.")
    if config.get("test_used_for_selection") is not False:
        raise PermissionError("M29 test labels cannot be used for selection.")
    inputs = _load_inputs(config)
    datasets = inputs["datasets"]
    teacher_probabilities = inputs["teacher_probabilities"]
    classes = np.asarray(inputs["teacher"]["classes"], dtype=np.int64)
    train_features, train_labels = datasets["train"]
    objective = config["objective"]
    cohort = select_boundary_cohort(
        teacher_probabilities["train"],
        fraction=float(objective["cohort_fraction"]),
        minimum_count=int(objective["cohort_minimum_count"]),
    )
    cohort_indices = np.asarray(cohort["selected_indices"], dtype=np.int64)
    candidate_config = config["candidate_generation"]
    variants = {}
    students = {}
    for rank in config["ranks"]:
        started = time.perf_counter()
        candidates = generate_margin_subspace_candidates(
            train_features,
            train_labels,
            teacher_probabilities["train"],
            classes,
            rank=int(rank),
            candidates_per_class=int(candidate_config["candidates_per_class"]),
            anchor_fraction=float(candidate_config["anchor_fraction"]),
            variance_floor_fraction=float(
                candidate_config["variance_floor_fraction"]
            ),
            residual_floor_fraction=float(
                candidate_config["residual_floor_fraction"]
            ),
        )
        candidate_seconds = time.perf_counter() - started
        candidate_labels = [
            int(candidate.class_label)
            for candidate in candidates
            if candidate.class_label is not None
        ]
        if len(candidate_labels) != len(candidates):
            raise ValueError("All M29 candidates require class labels.")
        for semantics in config["score_semantics"]:
            variant_id = f"rank_{rank}_{semantics}"
            started = time.perf_counter()
            fields = subspace_field_matrix(
                candidates, train_features[cohort_indices], semantics
            )
            selection = fit_distilled_candidate_fields(
                fields,
                candidate_labels,
                teacher_probabilities["train"][cohort_indices],
                train_labels[cohort_indices],
                classes,
                component_limit=int(config["budget"]["component_limit"]),
                teacher_weight=float(objective["teacher_weight"]),
                ground_truth_weight=float(objective["ground_truth_weight"]),
                complexity_penalty=float(objective["complexity_penalty"]),
                minimum_improvement=float(objective["minimum_improvement"]),
                initial_components_per_class=int(
                    objective["initial_components_per_class"]
                ),
            )
            student = serialize_subspace_student(
                classes=classes,
                candidates=candidates,
                selection=selection,
                rank=int(rank),
                score_semantics=semantics,
                cohort_indices=cohort_indices,
                configuration={
                    "candidate_generation": candidate_config,
                    "objective": objective,
                    "budget": config["budget"],
                },
            )
            fit_seconds = time.perf_counter() - started
            split_metrics = {}
            for source_split, output_split in (("dev", "development"), ("test", "test")):
                features, labels = datasets[source_split]
                predictions, probabilities = predict_subspace_student(
                    student, features
                )
                split_metrics[output_split] = _metrics(
                    labels,
                    predictions,
                    probabilities,
                    classes,
                    teacher_probabilities[source_split],
                )
            selected_candidates = [
                candidates[index]
                for index in selection["selected_candidate_indices"]
            ]
            variants[variant_id] = {
                "rank": int(rank),
                "score_semantics": semantics,
                "candidate_count": len(candidates),
                "selected_component_count": len(selected_candidates),
                "component_counts": selection["component_counts"],
                "parameter_count": int(
                    sum(candidate.parameter_count for candidate in selected_candidates)
                ),
                "array_bytes": int(
                    sum(candidate.array_bytes for candidate in selected_candidates)
                ),
                "objective_initial": selection["objective_trajectory"][0],
                "objective_final": selection["objective_trajectory"][-1],
                "objective_steps": len(selection["objective_trajectory"]) - 1,
                "candidate_generation_seconds": candidate_seconds,
                "student_fit_seconds": fit_seconds,
                "metrics": split_metrics,
            }
            students[variant_id] = student

    best_variant_id = max(
        variants,
        key=lambda name: (
            variants[name]["metrics"]["development"]["balanced_accuracy"],
            -variants[name]["metrics"]["development"]["nll"],
            -variants[name]["parameter_count"],
        ),
    )
    best = variants[best_variant_id]
    baseline = inputs["baseline"]
    spherical_control = baseline["heads"]["current_geode"]["development"][
        "balanced_accuracy"
    ]
    teacher_control = baseline["heads"]["rbf_svm"]["development"][
        "balanced_accuracy"
    ]
    minimum_components = min(best["component_counts"])
    accuracy_improvement = (
        best["metrics"]["development"]["balanced_accuracy"] - spherical_control
    )
    teacher_gap = teacher_control - best["metrics"]["development"][
        "balanced_accuracy"
    ]
    gate = config["advancement_gate"]
    flowers = json.loads(
        _resolve(config["flowers_extraction"]).read_text(encoding="utf-8")
    )
    flowers_support = 5
    flowers_feasibility = {
        str(rank): {
            "available_per_class": flowers_support,
            "required_per_class": int(rank) + 2,
            "status": "feasible" if flowers_support >= int(rank) + 2 else "blocked",
        }
        for rank in config["ranks"]
    }
    gate_operands = {
        "minimum_components_per_class": {
            "value": minimum_components,
            "operator": "ge",
            "threshold": int(gate["minimum_components_per_class"]),
            "passed": minimum_components
            >= int(gate["minimum_components_per_class"]),
        },
        "accuracy_or_teacher_gap": {
            "value": bool(
                accuracy_improvement >= float(gate["minimum_accuracy_improvement"])
                or teacher_gap <= float(gate["maximum_teacher_gap"])
            ),
            "operator": "eq",
            "threshold": True,
            "passed": bool(
                accuracy_improvement >= float(gate["minimum_accuracy_improvement"])
                or teacher_gap <= float(gate["maximum_teacher_gap"])
            ),
        },
        "flowers_support_contract_reported": {
            "value": True,
            "operator": "eq",
            "threshold": True,
            "passed": True,
        },
    }
    advancement_passed = all(item["passed"] for item in gate_operands.values())

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    selected_student = students[best_variant_id]
    write_canonical_json(output_dir / "student.json", selected_student)
    prediction_artifacts = {}
    for source_split, output_split in (("dev", "development"), ("test", "test")):
        features, _ = datasets[source_split]
        predictions, probabilities = predict_subspace_student(
            selected_student, features
        )
        predictions_path = output_dir / "arrays" / f"{output_split}_predictions.npy"
        probabilities_path = output_dir / "arrays" / f"{output_split}_probabilities.npy"
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(predictions_path, predictions, allow_pickle=False)
        np.save(probabilities_path, probabilities, allow_pickle=False)
        prediction_artifacts[output_split] = {
            "predictions": {
                "path": predictions_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(predictions_path),
            },
            "probabilities": {
                "path": probabilities_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(probabilities_path),
            },
        }
    evidence = {
        "schema_version": 1,
        "milestone": "M29",
        "stage": "S1",
        "seed": config["seed"],
        "configuration_hash": payload_hash(config),
        "representation_hash": inputs["manifest"].representation_hash,
        "teacher_manifest_hash": sha256_file(inputs["teacher_path"]),
        "prediction_baseline_hash": sha256_file(inputs["baseline_path"]),
        "cohort": cohort,
        "variants": variants,
        "selected_variant": best_variant_id,
        "locked_controls": {
            "current_geode_development_balanced_accuracy": spherical_control,
            "rbf_teacher_development_balanced_accuracy": teacher_control,
        },
        "accuracy_improvement": accuracy_improvement,
        "teacher_gap": teacher_gap,
        "flowers_representation_hash": flowers["representations"]["dinov2-small"][
            "representation_hash"
        ],
        "flowers_support_feasibility": flowers_feasibility,
        "flowers_maximum_feasible_rank": flowers_support - 2,
        "prediction_artifacts": prediction_artifacts,
        "gate_operands": gate_operands,
        "advancement_passed": advancement_passed,
        "test_used_for_selection": False,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    index = build_artifact_index(output_dir)
    return {
        "selected_variant": best_variant_id,
        "development_balanced_accuracy": best["metrics"]["development"][
            "balanced_accuracy"
        ],
        "test_balanced_accuracy": best["metrics"]["test"]["balanced_accuracy"],
        "minimum_components_per_class": minimum_components,
        "accuracy_improvement": accuracy_improvement,
        "teacher_gap": teacher_gap,
        "advancement_passed": advancement_passed,
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
        deterministic_paths = (
            "student.json",
            "arrays/development_predictions.npy",
            "arrays/development_probabilities.npy",
            "arrays/test_predictions.npy",
            "arrays/test_probabilities.npy",
        )
        if any(
            (first / relative).read_bytes() != (second / relative).read_bytes()
            for relative in deterministic_paths
        ):
            raise RuntimeError("M29 student or prediction replay was not byte-identical.")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(first, output_dir)
    evidence_path = output_dir / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["byte_identical_student_prediction_replay"] = True
    write_canonical_json(evidence_path, evidence)
    build_artifact_index(output_dir)
    return {
        **first_summary,
        "byte_identical_student_prediction_replay": True,
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
