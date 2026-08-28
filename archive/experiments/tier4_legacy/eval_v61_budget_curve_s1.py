"""Run the v6.1 A1-B seed-11 component-budget scaling diagnostic."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    serialized_size,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v6_factorial import (
    candidate_array_bytes,
    candidate_parameter_count,
    fit_global_temperature,
    predict_factorial_student,
    select_predictive_candidates,
    serialize_factorial_student,
)
from experiments.common.v6_protocol import select_boundary_cohort
from experiments.common.v6_subspace_distillation import (
    generate_margin_subspace_candidates,
    subspace_field_matrix,
)
from experiments.common.v61_budget_curve import (
    REGISTERED_COMPONENT_COUNTS,
    classify_capacity_curve,
    marginal_accuracy_per_ten,
    probability_margin_error,
)
from experiments.tier4.eval_v61_weighted_s1 import _load_locked_json
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.tier4.eval_v6_factorial_s2 import (
    _candidate_evaluations,
    _load_array,
)
from experiments.tier4.eval_v6_subspace_primitives import _metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v6_1" / "a1_budget_curve_s1.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v6_1" / "a1_budget_curve_s1"


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != 1
        or config.get("amendment") != "v6.1"
        or config.get("milestone") != "A1-B"
        or config.get("stage") != "S1"
        or config.get("seed") != 11
        or config.get("diagnostic_only") is not True
        or config.get("a2_component_count_immutable") != 46
        or config.get("test_labels_opened") is not False
    ):
        raise ValueError("Unsupported, selective, or test-open A1-B configuration.")
    if config["family"] != {
        "primitive": "subspace_r32",
        "rank": 32,
        "objective": "direct",
        "score": "normalized_radial",
        "readout": "hard_class_minimum",
    }:
        raise ValueError("A1-B family contract mismatch.")
    if tuple(config["selection"]["exact_component_counts"]) != (
        REGISTERED_COMPONENT_COUNTS
    ):
        raise ValueError("A1-B component counts differ from the registered curve.")
    if config["temperature"]["policy"] != "one_global":
        raise ValueError("A1-B permits only one global temperature.")
    if (
        config["a1_weighted_evidence"].get("required_advancement_status")
        is not True
    ):
        raise ValueError("A1-B must preserve the passed A1-W result.")


def _timed_predictions(
    student: dict[str, Any],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    predict_factorial_student(student, features)
    timings = []
    predictions = np.empty(len(features), dtype=np.int64)
    probabilities = np.empty((len(features), len(student["classes"])))
    for _ in range(5):
        started = time.perf_counter()
        predictions, probabilities = predict_factorial_student(student, features)
        timings.append(time.perf_counter() - started)
    return predictions, probabilities, float(np.median(timings))


def run_evaluation(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    _, m30_config = _load_locked_json(config["m30_config"], "M30 configuration")
    m30_path, m30_evidence = _load_locked_json(
        config["m30_evidence"], "M30 evidence"
    )
    m31_config_path, m31_config = _load_locked_json(
        config["m31_config"], "M31 configuration"
    )
    weighted_path, weighted_evidence = _load_locked_json(
        config["a1_weighted_evidence"], "A1-W evidence"
    )
    if weighted_evidence["advancement_passed"] is not True:
        raise ValueError("A1-B requires the passed A1-W parent result.")
    parent_generation = m31_config["candidate_generation"]
    if config["candidate_generation"] != {
        "candidates_per_class": parent_generation["candidates_per_class"],
        "anchor_fraction": parent_generation["anchor_fraction"],
        "variance_floor_fraction": parent_generation[
            "variance_floor_fraction"
        ],
        "residual_floor_fraction": parent_generation[
            "residual_floor_fraction"
        ],
    }:
        raise ValueError("A1-B candidate bank differs from the M31 parent contract.")

    seed_input = m30_config["seed_inputs"]["11"]
    loaded = _load_seed_data(seed_input)
    train_features, train_labels = loaded["datasets"]["train"]
    dev_features, dev_labels = loaded["datasets"]["dev"]
    classes, class_counts = np.unique(train_labels, return_counts=True)
    class_priors = class_counts / class_counts.sum()
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
        raise ValueError("A1-B development labels differ from the frozen parent.")

    selection_config = config["selection"]
    cohort = select_boundary_cohort(
        teacher_train,
        fraction=float(selection_config["cohort_fraction"]),
        minimum_count=int(selection_config["cohort_minimum_count"]),
    )
    if cohort != weighted_evidence["cohort"]:
        raise ValueError("A1-B cohort differs from the frozen A1-W cohort.")
    cohort_indices = np.asarray(cohort["selected_indices"], dtype=np.int64)
    generation = config["candidate_generation"]
    started = time.perf_counter()
    candidates = generate_margin_subspace_candidates(
        train_features,
        train_labels,
        teacher_train,
        classes,
        rank=int(config["family"]["rank"]),
        candidates_per_class=int(generation["candidates_per_class"]),
        anchor_fraction=float(generation["anchor_fraction"]),
        variance_floor_fraction=float(generation["variance_floor_fraction"]),
        residual_floor_fraction=float(generation["residual_floor_fraction"]),
    )
    candidate_generation_seconds = time.perf_counter() - started
    if len(candidates) != 120:
        raise ValueError("A1-B requires one exact 120-component candidate bank.")
    candidate_labels = [int(candidate.class_label) for candidate in candidates]
    cohort_fields = subspace_field_matrix(
        candidates,
        train_features[cohort_indices],
        "normalized_radial",
    )
    cell = {
        "id": "a1b_direct_subspace_radial",
        "objective": "direct",
        "primitive": "subspace_r32",
        "score": "normalized_radial",
        "budget": "diagnostic_component_curve",
    }
    rows = []
    students: dict[int, dict[str, Any]] = {}
    predictions_by_count: dict[int, np.ndarray] = {}
    full_selection: dict[str, Any] | None = None
    for component_count in REGISTERED_COMPONENT_COUNTS:
        started = time.perf_counter()
        selection = select_predictive_candidates(
            cohort_fields,
            candidate_labels,
            teacher_train[cohort_indices],
            train_labels[cohort_indices],
            classes,
            objective="direct",
            score="normalized_radial",
            component_limit=component_count,
            initial_components_per_class=int(
                selection_config["initial_components_per_class"]
            ),
            minimum_improvement=float(selection_config["minimum_improvement"]),
        )
        selection_seconds = time.perf_counter() - started
        if full_selection is not None:
            raise RuntimeError("Full selection must be assigned only at 120 components.")
        student = serialize_factorial_student(
            cell=cell,
            classes=classes,
            candidates=candidates,
            selection=selection,
            parent_representation_hash=seed_input["parent_representation_hash"],
            directional_representation_hash=None,
            class_priors=class_priors,
        )
        temperature = config["temperature"]
        started = time.perf_counter()
        fit_global_temperature(
            student,
            train_features[cohort_indices],
            train_labels[cohort_indices],
            minimum=float(temperature["minimum"]),
            maximum=float(temperature["maximum"]),
        )
        temperature_seconds = time.perf_counter() - started
        predictions, probabilities, inference_seconds = _timed_predictions(
            student, dev_features
        )
        metrics = _metrics(
            dev_labels, predictions, probabilities, classes, teacher_dev
        )
        selected = [
            candidates[index]
            for index in selection["selected_candidate_indices"]
        ]
        rows.append(
            {
                "component_count": component_count,
                "development_balanced_accuracy": float(
                    metrics["balanced_accuracy"]
                ),
                "development_nll": float(metrics["nll"]),
                "teacher_agreement": float(metrics["teacher_agreement"]),
                "probability_margin_error": probability_margin_error(
                    probabilities, teacher_dev
                ),
                "components_per_class": selection["component_counts"],
                "parameter_count": int(
                    sum(candidate_parameter_count(item) for item in selected)
                ),
                "array_bytes": int(
                    sum(candidate_array_bytes(item) for item in selected)
                ),
                "serialized_bytes": serialized_size(student),
                "candidate_evaluations": _candidate_evaluations(
                    len(candidates),
                    component_count,
                    int(selection_config["initial_components_per_class"])
                    * len(classes),
                ),
                "candidate_generation_seconds_shared": candidate_generation_seconds,
                "selection_seconds": selection_seconds,
                "temperature_fit_seconds": temperature_seconds,
                "total_fit_seconds": (
                    candidate_generation_seconds
                    + selection_seconds
                    + temperature_seconds
                ),
                "median_inference_seconds": inference_seconds,
                "global_temperature": float(student["global_temperature"]),
                "selected_candidate_indices": selection[
                    "selected_candidate_indices"
                ],
                "student_hash": payload_hash(student),
            }
        )
        students[component_count] = student
        predictions_by_count[component_count] = predictions
        if component_count == REGISTERED_COMPONENT_COUNTS[-1]:
            full_selection = selection

    for previous_count, current_count in zip(
        REGISTERED_COMPONENT_COUNTS, REGISTERED_COMPONENT_COUNTS[1:]
    ):
        previous = students[previous_count]["selected_candidate_indices"]
        current = students[current_count]["selected_candidate_indices"]
        if current[: len(previous)] != previous:
            raise ValueError("A1-B selections do not form one nested trajectory.")
    classification_config = config["classification"]
    classification = classify_capacity_curve(
        rows,
        minimum_high_budget_slope_per_ten=float(
            classification_config["minimum_high_budget_slope_per_ten"]
        ),
        material_accuracy_reversal=float(
            classification_config["material_accuracy_reversal"]
        ),
        material_nll_reversal_fraction=float(
            classification_config["material_nll_reversal_fraction"]
        ),
    )
    marginal = marginal_accuracy_per_ten(rows)
    for row, item in zip(rows, marginal):
        row["marginal_accuracy_per_ten"] = item[
            "marginal_accuracy_per_ten"
        ]

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    write_canonical_json(
        output_dir / "candidate_bank.json",
        {
            "schema_version": 1,
            "candidate_bank_hash": payload_hash(
                [candidate.to_dict() for candidate in candidates]
            ),
            "candidates": [candidate.to_dict() for candidate in candidates],
        },
    )
    write_canonical_json(
        output_dir / "student_120.json",
        students[REGISTERED_COMPONENT_COUNTS[-1]],
    )
    for count, predictions in predictions_by_count.items():
        np.save(
            output_dir / f"development_predictions_{count}.npy",
            predictions,
            allow_pickle=False,
        )
    evidence = {
        "schema_version": 1,
        "amendment": "v6.1",
        "milestone": "A1-B",
        "stage": "S1",
        "configuration_hash": payload_hash(config),
        "parent_hashes": {
            "m30_evidence": sha256_file(m30_path),
            "m31_config": sha256_file(m31_config_path),
            "a1_weighted_evidence": sha256_file(weighted_path),
        },
        "seed": 11,
        "split_hashes": parent_seed["split_hashes"],
        "parent_representation_hash": seed_input["parent_representation_hash"],
        "cohort": cohort,
        "candidate_bank_hash": payload_hash(
            [candidate.to_dict() for candidate in candidates]
        ),
        "candidate_bank_count": len(candidates),
        "nested_selection_trajectory": True,
        "rows": rows,
        "classification": classification,
        "diagnostic_only": True,
        "a2_component_count_unchanged": 46,
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
                "classification": evidence["classification"]["classification"],
                "endpoint_slope_per_ten": evidence["classification"][
                    "endpoint_slope_per_ten"
                ],
                "accuracies": {
                    str(row["component_count"]): row[
                        "development_balanced_accuracy"
                    ]
                    for row in evidence["rows"]
                },
                "a2_component_count_unchanged": evidence[
                    "a2_component_count_unchanged"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
