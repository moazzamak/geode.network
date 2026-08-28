"""Run the M30 matched Euclidean-sphere versus cosine-cap ablation."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v6_boundary_distillation import (
    fit_distilled_candidate_fields,
)
from experiments.common.v6_directional_distillation import (
    directional_field_matrix,
    generate_paired_directional_candidates,
    normalized_representation_hash,
    predict_directional_student,
    serialize_directional_student,
)
from experiments.common.v6_protocol import select_boundary_cohort
from experiments.tier4.eval_v6_subspace_primitives import _load_inputs, _metrics
from src.directional_primitive import l2_normalize


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v6" / "m30_directional_s1.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v6" / "m30_directional_s1"


def _candidate_labels(candidates: list[Any]) -> list[int]:
    labels = [candidate.class_label for candidate in candidates]
    if any(label is None for label in labels):
        raise ValueError("Every M30 candidate requires a class label.")
    return [int(label) for label in labels]


def run_evaluation(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != 1
        or config.get("milestone") != "M30"
        or config.get("stage") != "S1"
        or config.get("geometries") != ["euclidean_sphere", "cosine_cap"]
    ):
        raise ValueError("Unsupported M30 configuration.")
    if config.get("test_used_for_selection") is not False:
        raise PermissionError("M30 test labels cannot be used for selection.")
    expected_directional_hash = normalized_representation_hash(
        config["parent_representation_hash"]
    )
    if config["directional_representation_hash"] != expected_directional_hash:
        raise ValueError("M30 directional representation hash mismatch.")

    compatibility_config = {
        **config,
        "representation_hash": config["parent_representation_hash"],
    }
    inputs = _load_inputs(compatibility_config)
    datasets = inputs["datasets"]
    teacher_probabilities = inputs["teacher_probabilities"]
    classes = np.asarray(inputs["teacher"]["classes"], dtype=np.int64)
    train_features, train_labels = datasets["train"]
    normalized_train = l2_normalize(train_features)
    objective = config["objective"]
    cohort = select_boundary_cohort(
        teacher_probabilities["train"],
        fraction=float(objective["cohort_fraction"]),
        minimum_count=int(objective["cohort_minimum_count"]),
    )
    cohort_indices = np.asarray(cohort["selected_indices"], dtype=np.int64)
    candidate_config = config["candidate_generation"]

    started = time.perf_counter()
    spheres, caps = generate_paired_directional_candidates(
        normalized_train,
        train_labels,
        teacher_probabilities["train"],
        classes,
        candidates_per_class=int(candidate_config["candidates_per_class"]),
        seed_size=int(candidate_config["seed_size"]),
        anchor_fraction=float(candidate_config["anchor_fraction"]),
    )
    candidate_generation_seconds = time.perf_counter() - started
    if [candidate.anchor_index for candidate in spheres] != [
        candidate.anchor_index for candidate in caps
    ]:
        raise RuntimeError("Paired M30 candidate anchors differ.")
    if any(
        sphere.support_size != len(cap.support_indices)
        for sphere, cap in zip(spheres, caps)
    ):
        raise RuntimeError("Paired M30 candidate support sizes differ.")

    candidates_by_geometry = {
        "euclidean_sphere": spheres,
        "cosine_cap": caps,
    }
    variants: dict[str, Any] = {}
    students: dict[str, dict[str, Any]] = {}
    component_count = int(config["budget"]["component_count"])
    for geometry in config["geometries"]:
        candidates = candidates_by_geometry[geometry]
        started = time.perf_counter()
        fields = directional_field_matrix(
            candidates, normalized_train[cohort_indices], geometry
        )
        selection = fit_distilled_candidate_fields(
            fields,
            _candidate_labels(candidates),
            teacher_probabilities["train"][cohort_indices],
            train_labels[cohort_indices],
            classes,
            component_limit=component_count,
            teacher_weight=float(objective["teacher_weight"]),
            ground_truth_weight=float(objective["ground_truth_weight"]),
            complexity_penalty=float(objective["complexity_penalty"]),
            minimum_improvement=float(objective["minimum_improvement"]),
            initial_components_per_class=int(
                objective["initial_components_per_class"]
            ),
            exact_component_count=True,
        )
        student = serialize_directional_student(
            geometry=geometry,
            classes=classes,
            candidates=candidates,
            selection=selection,
            parent_representation_hash=config["parent_representation_hash"],
            directional_representation_hash=config[
                "directional_representation_hash"
            ],
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
            predictions, probabilities = predict_directional_student(
                student,
                features,
                parent_representation_hash=config["parent_representation_hash"],
            )
            split_metrics[output_split] = _metrics(
                labels,
                predictions,
                probabilities,
                classes,
                teacher_probabilities[source_split],
            )
        selected = [
            candidates[index] for index in selection["selected_candidate_indices"]
        ]
        variants[geometry] = {
            "candidate_count": len(candidates),
            "selected_component_count": len(selected),
            "component_counts": selection["component_counts"],
            "parameter_count": sum(candidate.parameter_count for candidate in selected),
            "array_bytes": sum(candidate.array_bytes for candidate in selected),
            "objective_initial": selection["objective_trajectory"][0],
            "objective_final": selection["objective_trajectory"][-1],
            "objective_monotone": all(
                right <= left
                for left, right in zip(
                    selection["objective_trajectory"],
                    selection["objective_trajectory"][1:],
                )
            ),
            "student_fit_seconds": fit_seconds,
            "metrics": split_metrics,
        }
        students[geometry] = student

    euclidean = variants["euclidean_sphere"]["metrics"]["development"]
    cosine = variants["cosine_cap"]["metrics"]["development"]
    accuracy_improvement = (
        cosine["balanced_accuracy"] - euclidean["balanced_accuracy"]
    )
    knn_accuracy = inputs["baseline"]["heads"]["weighted_knn"]["development"][
        "balanced_accuracy"
    ]
    euclidean_gap = knn_accuracy - euclidean["balanced_accuracy"]
    gap_closure_fraction = (
        accuracy_improvement / euclidean_gap if euclidean_gap > 0.0 else 0.0
    )
    nll_non_regression = cosine["nll"] <= euclidean["nll"]
    parameter_matched = (
        variants["cosine_cap"]["parameter_count"]
        == variants["euclidean_sphere"]["parameter_count"]
    )
    gate = config["s1_promising_gate"]
    direct_improvement = accuracy_improvement >= float(
        gate["minimum_accuracy_improvement"]
    )
    gap_closure = gap_closure_fraction >= float(
        gate["minimum_gap_closure_fraction"]
    ) and (
        nll_non_regression
        or not bool(gate["require_nll_non_regression_for_gap_closure"])
    )
    s1_promising = parameter_matched and (direct_improvement or gap_closure)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    prediction_artifacts: dict[str, Any] = {}
    for geometry, student in students.items():
        write_canonical_json(output_dir / f"{geometry}_student.json", student)
        prediction_artifacts[geometry] = {}
        for source_split, output_split in (("dev", "development"), ("test", "test")):
            features, _ = datasets[source_split]
            predictions, probabilities = predict_directional_student(
                student,
                features,
                parent_representation_hash=config["parent_representation_hash"],
            )
            geometry_dir = output_dir / "arrays" / geometry
            geometry_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = geometry_dir / f"{output_split}_predictions.npy"
            probabilities_path = geometry_dir / f"{output_split}_probabilities.npy"
            np.save(predictions_path, predictions, allow_pickle=False)
            np.save(probabilities_path, probabilities, allow_pickle=False)
            prediction_artifacts[geometry][output_split] = {
                "predictions": {
                    "path": predictions_path.relative_to(output_dir).as_posix(),
                    "sha256": sha256_file(predictions_path),
                },
                "probabilities": {
                    "path": probabilities_path.relative_to(output_dir).as_posix(),
                    "sha256": sha256_file(probabilities_path),
                },
            }

    support_assignment_hash = payload_hash(
        [list(candidate.support_indices) for candidate in caps]
    )
    evidence = {
        "schema_version": 1,
        "milestone": "M30",
        "stage": "S1",
        "seed": config["seed"],
        "configuration_hash": payload_hash(config),
        "parent_representation_hash": config["parent_representation_hash"],
        "directional_representation_hash": config[
            "directional_representation_hash"
        ],
        "teacher_manifest_hash": sha256_file(inputs["teacher_path"]),
        "prediction_baseline_hash": sha256_file(inputs["baseline_path"]),
        "cohort": cohort,
        "candidate_generation_seconds": candidate_generation_seconds,
        "paired_candidate_contract": {
            "same_examples": True,
            "same_anchors": True,
            "same_support_indices": True,
            "same_component_count": True,
            "same_supervision": True,
            "same_score_family": "normalized_radial",
            "support_assignment_hash": support_assignment_hash,
        },
        "variants": variants,
        "locked_controls": {
            "weighted_knn_development_balanced_accuracy": knn_accuracy,
            "current_geode_development_balanced_accuracy": inputs["baseline"]["heads"][
                "current_geode"
            ]["development"]["balanced_accuracy"],
        },
        "accuracy_improvement": accuracy_improvement,
        "euclidean_to_knn_gap": euclidean_gap,
        "gap_closure_fraction": gap_closure_fraction,
        "nll_non_regression": nll_non_regression,
        "parameter_matched": parameter_matched,
        "s1_promising_operands": {
            "direct_accuracy_improvement": direct_improvement,
            "gap_closure_with_nll_non_regression": gap_closure,
        },
        "s1_promising": s1_promising,
        "s2_advancement_gate_status": "not_evaluated_single_seed_s1",
        "prediction_artifacts": prediction_artifacts,
        "test_used_for_selection": False,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    index = build_artifact_index(output_dir)
    return {
        "euclidean_development_balanced_accuracy": euclidean["balanced_accuracy"],
        "cosine_development_balanced_accuracy": cosine["balanced_accuracy"],
        "accuracy_improvement": accuracy_improvement,
        "gap_closure_fraction": gap_closure_fraction,
        "s1_promising": s1_promising,
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
        deterministic_paths = [
            "euclidean_sphere_student.json",
            "cosine_cap_student.json",
            *[
                f"arrays/{geometry}/{split}_{kind}.npy"
                for geometry in ("euclidean_sphere", "cosine_cap")
                for split in ("development", "test")
                for kind in ("predictions", "probabilities")
            ],
        ]
        if any(
            (first / relative).read_bytes() != (second / relative).read_bytes()
            for relative in deterministic_paths
        ):
            raise RuntimeError("M30 student or prediction replay was not byte-identical.")
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
