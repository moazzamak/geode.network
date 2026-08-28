"""Run the M31 objective-primitive-score fractional factorial on S2."""

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
from experiments.common.v5_protocol import DataStage, seeds_for_stage
from experiments.common.v5_statistics import paired_seed_t_interval
from experiments.common.v6_directional_distillation import (
    generate_paired_directional_candidates,
)
from experiments.common.v6_factorial import (
    candidate_array_bytes,
    candidate_class_label,
    candidate_parameter_count,
    fit_global_temperature,
    local_edit_rollback_evidence,
    predict_factorial_student,
    primitive_field_matrix,
    select_coverage_candidates,
    select_predictive_candidates,
    serialize_factorial_student,
    validate_fractional_factorial,
)
from experiments.common.v6_protocol import select_boundary_cohort
from experiments.common.v6_subspace_distillation import (
    generate_margin_subspace_candidates,
)
from experiments.tier4.eval_v5_frozen_space_heads import (
    fit_geode_head,
    fit_weighted_knn_head,
    predict_geode,
)
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.tier4.eval_v6_subspace_primitives import _metrics
from src.directional_primitive import l2_normalize


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v6" / "m31_factorial_s2.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v6" / "m31_factorial_s2"


def _resolve(path: str) -> Path:
    return REPO_ROOT / Path(path)


def _load_array(root: Path, artifact: dict[str, str]) -> np.ndarray:
    path = root / artifact["path"]
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"M31 parent array hash mismatch: {path}.")
    return np.load(path, allow_pickle=False)


def _candidate_evaluations(candidate_count: int, component_limit: int, initial: int) -> int:
    remaining = candidate_count - initial
    additions = component_limit - initial
    return int(sum(remaining - step for step in range(additions)))


def _fit_seed(
    seed: int,
    seed_input: dict[str, Any],
    config: dict[str, Any],
    m30_evidence: dict[str, Any],
    m30_root: Path,
) -> dict[str, Any]:
    loaded = _load_seed_data(seed_input)
    train_features, train_labels = loaded["datasets"]["train"]
    dev_features, dev_labels = loaded["datasets"]["dev"]
    classes, class_counts = np.unique(train_labels, return_counts=True)
    class_priors = class_counts / class_counts.sum()
    parent_seed = m30_evidence["seed_results"][str(seed)]
    teacher_train = _load_array(
        m30_root, parent_seed["array_artifacts"]["teacher_train_probabilities"]
    )
    teacher_dev = _load_array(
        m30_root, parent_seed["array_artifacts"]["teacher_development_probabilities"]
    )
    parent_dev_labels = _load_array(
        m30_root, parent_seed["array_artifacts"]["development_labels"]
    )
    if not np.array_equal(parent_dev_labels, dev_labels):
        raise ValueError("M31 development labels differ from the M30 parent.")

    normalized_train = l2_normalize(train_features)
    normalized_dev = l2_normalize(dev_features)
    cohort = select_boundary_cohort(
        teacher_train,
        fraction=float(config["selection"]["cohort_fraction"]),
        minimum_count=int(config["selection"]["cohort_minimum_count"]),
    )
    cohort_indices = np.asarray(cohort["selected_indices"], dtype=np.int64)
    generation = config["candidate_generation"]
    started = time.perf_counter()
    spheres, caps = generate_paired_directional_candidates(
        normalized_train,
        train_labels,
        teacher_train,
        classes,
        candidates_per_class=int(generation["candidates_per_class"]),
        seed_size=int(generation["sphere_seed_size"]),
        anchor_fraction=float(generation["anchor_fraction"]),
    )
    sphere_directional_generation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    subspaces = generate_margin_subspace_candidates(
        train_features,
        train_labels,
        teacher_train,
        classes,
        rank=int(generation["subspace_rank"]),
        candidates_per_class=int(generation["candidates_per_class"]),
        anchor_fraction=float(generation["anchor_fraction"]),
        variance_floor_fraction=float(generation["variance_floor_fraction"]),
        residual_floor_fraction=float(generation["residual_floor_fraction"]),
    )
    subspace_generation_seconds = time.perf_counter() - started
    candidates_by_primitive = {
        "sphere": spheres,
        "subspace_r32": subspaces,
        "directional": caps,
    }
    train_by_primitive = {
        "sphere": normalized_train,
        "subspace_r32": train_features,
        "directional": normalized_train,
    }
    dev_by_primitive = {
        "sphere": normalized_dev,
        "subspace_r32": dev_features,
        "directional": normalized_dev,
    }
    generation_seconds = {
        "sphere": sphere_directional_generation_seconds / 2.0,
        "directional": sphere_directional_generation_seconds / 2.0,
        "subspace_r32": subspace_generation_seconds,
    }
    primitive_cell_counts = {
        primitive: sum(
            cell["primitive"] == primitive for cell in config["cells"]
        )
        for primitive in candidates_by_primitive
    }

    knn_config = json.loads(
        _resolve(config["m30_config"]).read_text(encoding="utf-8")
    )["weighted_knn"]
    knn = fit_weighted_knn_head(
        train_features,
        train_labels,
        n_neighbors=int(knn_config["n_neighbors"]),
        temperature=float(knn_config["temperature"]),
        query_batch_size=int(knn_config["query_batch_size"]),
    )["model"]
    knn_probabilities = knn.predict_proba(dev_features)
    knn_predictions = knn.predict(dev_features)
    controls = {
        "rbf": _metrics(
            dev_labels,
            classes[np.argmax(teacher_dev, axis=1)],
            teacher_dev,
            classes,
            teacher_dev,
        ),
        "weighted_knn": _metrics(
            dev_labels,
            knn_predictions,
            knn_probabilities,
            classes,
            teacher_dev,
        ),
    }
    geode = fit_geode_head(
        train_features,
        train_labels,
        seed,
        max_iterations=1,
        consensus_threshold=0.05,
        dimension_limit=train_features.shape[1],
    )
    geode_predictions, geode_probabilities = predict_geode(
        geode["model"], dev_features, classes
    )
    controls["current_geode"] = _metrics(
        dev_labels,
        geode_predictions,
        geode_probabilities,
        classes,
        teacher_dev,
    )

    field_cache: dict[tuple[str, str, str], np.ndarray] = {}

    def fields(primitive: str, score: str, split: str) -> np.ndarray:
        field_score = (
            "normalized_radial" if score == "teacher_softmin" else score
        )
        key = (primitive, field_score, split)
        if key not in field_cache:
            source = train_by_primitive[primitive]
            if split == "cohort":
                source = source[cohort_indices]
            elif split == "development":
                source = dev_by_primitive[primitive]
            elif split != "train":
                raise ValueError(f"Unsupported M31 field split {split}.")
            field_cache[key] = primitive_field_matrix(
                candidates_by_primitive[primitive],
                source,
                primitive=primitive,
                score=field_score,
            )
        return field_cache[key]

    students: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    cells: dict[str, Any] = {}
    selection_config = config["selection"]
    for cell in config["cells"]:
        cell_id = cell["id"]
        primitive = cell["primitive"]
        score = cell["score"]
        candidates = candidates_by_primitive[primitive]
        if cell["budget"] == "component":
            component_limit = int(selection_config["component_count"])
        else:
            component_limit = min(
                len(candidates),
                int(selection_config["parameter_limit"])
                // candidate_parameter_count(candidates[0]),
            )
        initial_per_class = int(selection_config["initial_components_per_class"])
        initial_count = initial_per_class * len(classes)
        candidate_evaluations = _candidate_evaluations(
            len(candidates), component_limit, initial_count
        )
        if candidate_evaluations > int(
            selection_config["maximum_candidate_evaluations"]
        ):
            raise ValueError(f"M31 cell {cell_id} exceeds its fit-work budget.")
        labels = [candidate_class_label(candidate) for candidate in candidates]
        started = time.perf_counter()
        if cell["objective"] == "coverage":
            selection = select_coverage_candidates(
                fields(primitive, "normalized_radial", "train"),
                labels,
                train_labels,
                classes,
                component_limit=component_limit,
                initial_components_per_class=initial_per_class,
            )
        else:
            selection = select_predictive_candidates(
                fields(primitive, score, "cohort"),
                labels,
                teacher_train[cohort_indices],
                train_labels[cohort_indices],
                classes,
                objective=cell["objective"],
                score=score,
                component_limit=component_limit,
                initial_components_per_class=initial_per_class,
                minimum_improvement=float(
                    selection_config["minimum_improvement"]
                ),
            )
        student = serialize_factorial_student(
            cell=cell,
            classes=classes,
            candidates=candidates,
            selection=selection,
            parent_representation_hash=seed_input["parent_representation_hash"],
            directional_representation_hash=(
                seed_input["directional_representation_hash"]
                if primitive == "directional"
                else None
            ),
            class_priors=class_priors,
        )
        fit_global_temperature(
            student,
            train_by_primitive[primitive][cohort_indices],
            train_labels[cohort_indices],
            minimum=float(config["temperature"]["minimum"]),
            maximum=float(config["temperature"]["maximum"]),
        )
        selection_seconds = time.perf_counter() - started
        cell_predictions, cell_probabilities = predict_factorial_student(
            student, dev_by_primitive[primitive]
        )
        selected_candidates = [
            candidates[index]
            for index in selection["selected_candidate_indices"]
        ]
        parameter_count = int(
            sum(candidate_parameter_count(candidate) for candidate in selected_candidates)
        )
        budget_passed = (
            len(selected_candidates) == int(selection_config["component_count"])
            if cell["budget"] == "component"
            else parameter_count <= int(selection_config["parameter_limit"])
        )
        cells[cell_id] = {
            "cell": cell,
            "selected_component_count": len(selected_candidates),
            "component_counts": selection["component_counts"],
            "parameter_count": parameter_count,
            "array_bytes": int(
                sum(candidate_array_bytes(candidate) for candidate in selected_candidates)
            ),
            "global_temperature": student["global_temperature"],
            "objective_initial": selection["objective_trajectory"][0],
            "objective_final": selection["objective_trajectory"][-1],
            "candidate_evaluations": candidate_evaluations,
            "selection_seconds": selection_seconds,
            "allocated_generation_seconds": generation_seconds[primitive]
            / primitive_cell_counts[primitive],
            "budget_passed": budget_passed,
            "development": _metrics(
                dev_labels,
                cell_predictions,
                cell_probabilities,
                classes,
                teacher_dev,
            ),
        }
        students[cell_id] = student
        predictions[cell_id] = cell_predictions
        probabilities[cell_id] = cell_probabilities
    return {
        "seed": seed,
        "split_hashes": loaded["extraction"]["split_hashes"],
        "parent_representation_hash": seed_input["parent_representation_hash"],
        "directional_representation_hash": seed_input[
            "directional_representation_hash"
        ],
        "cohort": cohort,
        "controls": controls,
        "cells": cells,
        "students": students,
        "predictions": predictions,
        "probabilities": probabilities,
        "development_labels": dev_labels,
        "dev_features": dev_features,
        "normalized_dev_features": normalized_dev,
    }


def _mean_cell_summaries(
    seed_results: list[dict[str, Any]], cells: list[dict[str, Any]]
) -> dict[str, Any]:
    summaries = {}
    for cell in cells:
        cell_id = cell["id"]
        records = [result["cells"][cell_id] for result in seed_results]
        summaries[cell_id] = {
            "cell": cell,
            "development_balanced_accuracy": float(
                np.mean(
                    [
                        record["development"]["balanced_accuracy"]
                        for record in records
                    ]
                )
            ),
            "development_nll": float(
                np.mean([record["development"]["nll"] for record in records])
            ),
            "parameter_count": float(
                np.mean([record["parameter_count"] for record in records])
            ),
            "array_bytes": float(
                np.mean([record["array_bytes"] for record in records])
            ),
            "fit_seconds": float(
                np.mean(
                    [
                        record["selection_seconds"]
                        + record["allocated_generation_seconds"]
                        for record in records
                    ]
                )
            ),
            "all_budgets_passed": all(record["budget_passed"] for record in records),
        }
    return summaries


def _main_effects(summaries: dict[str, Any]) -> dict[str, Any]:
    factorial = [
        summary
        for summary in summaries.values()
        if not summary["cell"].get("mandatory_control", False)
    ]
    matrix = np.asarray(
        [
            [
                1,
                item["cell"]["objective"] == "direct",
                item["cell"]["objective"] == "teacher",
                item["cell"]["primitive"] == "subspace_r32",
                item["cell"]["primitive"] == "directional",
                item["cell"]["score"] == "proper_likelihood",
                item["cell"]["score"] == "teacher_softmin",
                item["cell"]["budget"] == "parameter",
            ]
            for item in factorial
        ],
        dtype=np.float64,
    )
    outcomes = np.asarray(
        [item["development_balanced_accuracy"] for item in factorial]
    )
    coefficients = np.linalg.solve(matrix, outcomes)
    names = (
        "intercept_coverage_sphere_radial_component",
        "objective_direct",
        "objective_teacher",
        "primitive_subspace_r32",
        "primitive_directional",
        "score_proper_likelihood",
        "score_teacher_softmin",
        "budget_parameter",
    )
    return {
        "coding": "treatment_coded_main_effects",
        "baseline": {
            "objective": "coverage",
            "primitive": "sphere",
            "score": "normalized_radial",
            "budget": "component",
        },
        "coefficients": {
            name: float(value) for name, value in zip(names, coefficients)
        },
        "design_rank": int(np.linalg.matrix_rank(matrix)),
        "residual_max_abs": float(
            np.max(np.abs(matrix @ coefficients - outcomes))
        ),
    }


def run_evaluation(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != 1
        or config.get("milestone") != "M31"
        or config.get("stage") != "S2"
        or config.get("test_labels_opened") is not False
    ):
        raise ValueError("Unsupported or test-open M31 configuration.")
    seeds = seeds_for_stage(
        DataStage.S2, tuple(int(seed) for seed in config["seeds"])
    )
    design = validate_fractional_factorial(
        config["cells"], config["factorial_baselines"]
    )
    m30_config_path = _resolve(config["m30_config"])
    m30_evidence_path = _resolve(config["m30_evidence"])
    m30_config = json.loads(m30_config_path.read_text(encoding="utf-8"))
    m30_evidence = json.loads(m30_evidence_path.read_text(encoding="utf-8"))
    if (
        not m30_evidence["advancement_passed"]
        or m30_evidence["seeds"] != list(seeds)
        or set(m30_config["seed_inputs"]) != {str(seed) for seed in seeds}
    ):
        raise ValueError("M31 requires the passed, seed-matched M30 parent.")
    seed_results = [
        _fit_seed(
            seed,
            m30_config["seed_inputs"][str(seed)],
            config,
            m30_evidence,
            m30_evidence_path.parent,
        )
        for seed in seeds
    ]
    summaries = _mean_cell_summaries(seed_results, config["cells"])
    selected_id = max(
        summaries,
        key=lambda cell_id: (
            summaries[cell_id]["development_balanced_accuracy"],
            -summaries[cell_id]["development_nll"],
            -summaries[cell_id]["array_bytes"],
            -summaries[cell_id]["fit_seconds"],
        ),
    )
    selected_summary = summaries[selected_id]
    retained_control_ids = (
        "teacher_subspace_softmin_component_control",
        "teacher_directional_softmin_component",
    )
    non_topology_control_id = max(
        retained_control_ids,
        key=lambda cell_id: summaries[cell_id]["development_balanced_accuracy"],
    )
    non_topology_summary = summaries[non_topology_control_id]
    selected_values = np.asarray(
        [
            result["cells"][selected_id]["development"]["balanced_accuracy"]
            for result in seed_results
        ]
    )
    non_topology_values = np.asarray(
        [
            result["cells"][non_topology_control_id]["development"][
                "balanced_accuracy"
            ]
            for result in seed_results
        ]
    )
    comparison_interval = paired_seed_t_interval(
        selected_values,
        non_topology_values,
        confidence=float(config["statistics"]["confidence"]),
    )
    controls = {
        name: {
            "development_balanced_accuracy": float(
                np.mean(
                    [
                        result["controls"][name]["balanced_accuracy"]
                        for result in seed_results
                    ]
                )
            ),
            "development_nll": float(
                np.mean([result["controls"][name]["nll"] for result in seed_results])
            ),
        }
        for name in ("rbf", "weighted_knn", "current_geode")
    }
    strongest_same_space = max(
        controls["rbf"]["development_balanced_accuracy"],
        controls["weighted_knn"]["development_balanced_accuracy"],
    )
    same_space_gap = (
        strongest_same_space - selected_summary["development_balanced_accuracy"]
    )
    accuracy_improvement = (
        selected_summary["development_balanced_accuracy"]
        - non_topology_summary["development_balanced_accuracy"]
    )
    nll_reduction_fraction = (
        non_topology_summary["development_nll"]
        - selected_summary["development_nll"]
    ) / non_topology_summary["development_nll"]
    accuracy_loss = (
        non_topology_summary["development_balanced_accuracy"]
        - selected_summary["development_balanced_accuracy"]
    )
    lifecycle = {}
    for result in seed_results:
        primitive = summaries[selected_id]["cell"]["primitive"]
        edit_features = (
            result["normalized_dev_features"]
            if primitive in {"sphere", "directional"}
            else result["dev_features"]
        )
        lifecycle[str(result["seed"])] = local_edit_rollback_evidence(
            result["students"][selected_id], edit_features
        )
    minimum_preservation = min(
        item["unaffected_prediction_preservation"] for item in lifecycle.values()
    )
    exact_rollback = all(
        item["exact_json_rollback"] and item["rollback_restored_predictions"]
        for item in lifecycle.values()
    )
    gate = config["advancement_gate"]
    parity_passed = same_space_gap <= float(gate["maximum_same_space_gap"])
    accuracy_path = (
        accuracy_improvement >= float(gate["minimum_accuracy_improvement"])
        and comparison_interval["lower"] > 0.0
    )
    nll_path = (
        nll_reduction_fraction >= float(gate["minimum_nll_reduction_fraction"])
        and accuracy_loss <= float(gate["maximum_accuracy_loss_for_nll"])
    )
    lifecycle_passed = (
        minimum_preservation
        >= float(gate["minimum_unaffected_prediction_preservation"])
        and (exact_rollback if gate["require_exact_rollback"] else True)
    )
    budget_passed = bool(selected_summary["all_budgets_passed"])
    predictive_gate_passed = parity_passed and (accuracy_path or nll_path)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    deterministic_paths = []
    serializable_seed_results = {}
    for result in seed_results:
        seed = result["seed"]
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        student_path = seed_dir / "selected_student.json"
        write_canonical_json(student_path, result["students"][selected_id])
        deterministic_paths.append(student_path.relative_to(output_dir).as_posix())
        prediction_artifacts = {}
        for cell_id in summaries:
            path = seed_dir / f"{cell_id}_development_predictions.npy"
            np.save(path, result["predictions"][cell_id], allow_pickle=False)
            relative = path.relative_to(output_dir).as_posix()
            deterministic_paths.append(relative)
            prediction_artifacts[cell_id] = {
                "path": relative,
                "sha256": sha256_file(path),
            }
        serializable_seed_results[str(seed)] = {
            "split_hashes": result["split_hashes"],
            "parent_representation_hash": result["parent_representation_hash"],
            "directional_representation_hash": result[
                "directional_representation_hash"
            ],
            "cohort": result["cohort"],
            "controls": result["controls"],
            "cells": result["cells"],
            "selected_prediction_artifacts": prediction_artifacts,
            "lifecycle": lifecycle[str(seed)],
        }
    evidence = {
        "schema_version": 1,
        "milestone": "M31",
        "stage": "S2",
        "configuration_hash": payload_hash(config),
        "m30_config_hash": sha256_file(m30_config_path),
        "m30_evidence_hash": sha256_file(m30_evidence_path),
        "design": design,
        "infeasible_cells": config["infeasible_cells"],
        "seed_results": serializable_seed_results,
        "cell_summaries": summaries,
        "main_effect_estimates": _main_effects(summaries),
        "selected_cell": selected_id,
        "selection_order": [
            "development_balanced_accuracy",
            "development_nll",
            "array_bytes",
            "fit_seconds",
        ],
        "mandatory_controls": controls,
        "strongest_same_space_balanced_accuracy": strongest_same_space,
        "same_space_gap": same_space_gap,
        "non_topology_control_cell": non_topology_control_id,
        "accuracy_improvement_over_non_topology": accuracy_improvement,
        "paired_seed_interval": comparison_interval,
        "nll_reduction_fraction": nll_reduction_fraction,
        "accuracy_loss_for_nll_path": accuracy_loss,
        "lifecycle": {
            "minimum_unaffected_prediction_preservation": minimum_preservation,
            "exact_rollback": exact_rollback,
            "per_seed": lifecycle,
        },
        "gate_operands": {
            "same_space_parity": parity_passed,
            "accuracy_improvement_path": accuracy_path,
            "nll_reduction_path": nll_path,
            "lifecycle": lifecycle_passed,
            "budget": budget_passed,
        },
        "predictive_gate_passed": predictive_gate_passed,
        "exact_replay_verified": False,
        "advancement_passed": False,
        "test_labels_opened": False,
        "deterministic_paths": deterministic_paths,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    index = build_artifact_index(output_dir)
    return {
        "selected_cell": selected_id,
        "development_balanced_accuracy": selected_summary[
            "development_balanced_accuracy"
        ],
        "same_space_gap": same_space_gap,
        "predictive_gate_passed": predictive_gate_passed,
        "advancement_passed": False,
        "artifact_count": len(index["artifacts"]),
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
        evidence = json.loads((first / "evidence.json").read_text(encoding="utf-8"))
        if any(
            (first / relative).read_bytes() != (second / relative).read_bytes()
            for relative in evidence["deterministic_paths"]
        ):
            raise RuntimeError("M31 selected student or prediction replay failed.")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(first, output_dir)
    evidence_path = output_dir / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["exact_replay_verified"] = True
    config = json.loads(config_path.read_text(encoding="utf-8"))
    replay_passed = (
        evidence["exact_replay_verified"]
        if config["advancement_gate"]["require_exact_replay"]
        else True
    )
    evidence["advancement_passed"] = bool(
        evidence["predictive_gate_passed"]
        and evidence["gate_operands"]["lifecycle"]
        and evidence["gate_operands"]["budget"]
        and replay_passed
    )
    write_canonical_json(evidence_path, evidence)
    build_artifact_index(output_dir)
    return {
        **first_summary,
        "exact_replay_verified": True,
        "advancement_passed": evidence["advancement_passed"],
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
