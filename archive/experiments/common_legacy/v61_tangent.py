from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from experiments.common.v6_boundary_distillation import probabilities_from_logits
from experiments.common.v6_directional_distillation import (
    NORMALIZATION_POLICY,
    normalized_representation_hash,
    require_unit_features,
)
from src.directional_primitive import l2_normalize
from src.tangent_cap_primitive import TangentCapPrimitive, fit_tangent_cap


def generate_tangent_cap_candidates(
    normalized_features: np.ndarray,
    labels: np.ndarray,
    teacher_probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    rank: int,
    candidates_per_class: int,
    anchor_fraction: float,
    variance_floor_fraction: float = 1e-3,
    residual_floor_fraction: float = 0.05,
) -> list[TangentCapPrimitive]:
    points = require_unit_features(normalized_features)
    targets = np.asarray(labels)
    probabilities = np.asarray(teacher_probabilities, dtype=np.float64)
    class_values = np.asarray(classes)
    if targets.shape != (len(points),):
        raise ValueError("labels must match normalized_features.")
    if probabilities.shape != (len(points), len(class_values)):
        raise ValueError("teacher_probabilities have an incompatible shape.")
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)
    ):
        raise ValueError("teacher_probabilities must be finite and normalized.")
    if candidates_per_class < 1 or not 0.0 < anchor_fraction <= 1.0:
        raise ValueError("Invalid candidate count or anchor fraction.")
    support_size = rank + 2
    ordered = np.sort(probabilities, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    candidates = []
    for class_label in class_values:
        class_indices = np.flatnonzero(targets == class_label)
        if len(class_indices) < support_size:
            raise ValueError(
                f"Class {class_label} has {len(class_indices)} samples; "
                f"rank {rank} requires {support_size}."
            )
        order = np.lexsort((class_indices, margins[class_indices]))
        boundary_count = max(
            candidates_per_class,
            int(np.ceil(anchor_fraction * len(class_indices))),
        )
        anchors = class_indices[order[: min(boundary_count, len(order))]]
        positions = np.linspace(
            0, len(anchors) - 1, candidates_per_class, dtype=np.int64
        )
        class_points = points[class_indices]
        for position in positions:
            anchor_index = int(anchors[position])
            angular_proxy = 1.0 - np.clip(
                class_points @ points[anchor_index], -1.0, 1.0
            )
            support_order = np.lexsort((class_indices, angular_proxy))
            support_indices = class_indices[support_order[:support_size]]
            candidates.append(
                fit_tangent_cap(
                    points[support_indices],
                    rank,
                    variance_floor_fraction=variance_floor_fraction,
                    residual_floor_fraction=residual_floor_fraction,
                    class_label=int(class_label),
                    anchor_index=anchor_index,
                    support_indices=tuple(int(index) for index in support_indices),
                )
            )
    return candidates


def tangent_field_matrix(
    candidates: Sequence[TangentCapPrimitive],
    normalized_features: np.ndarray,
    score: str,
) -> np.ndarray:
    points = require_unit_features(normalized_features)
    if not candidates:
        raise ValueError("At least one tangent-cap candidate is required.")
    if score == "normalized_tangent_radial":
        return np.column_stack(
            [candidate.radial_field(points) for candidate in candidates]
        )
    if score == "tangent_gaussian_log_likelihood":
        return np.column_stack(
            [-candidate.log_likelihood(points) for candidate in candidates]
        )
    raise ValueError(f"Unsupported tangent-cap score {score!r}.")


def serialize_tangent_student(
    *,
    classes: np.ndarray,
    candidates: Sequence[TangentCapPrimitive],
    selection: Mapping[str, Any],
    parent_representation_hash: str,
    directional_representation_hash: str,
    cohort_indices: np.ndarray,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    if directional_representation_hash != normalized_representation_hash(
        parent_representation_hash
    ):
        raise ValueError("Tangent representation hash does not match its contract.")
    selected = [
        candidates[int(index)] for index in selection["selected_candidate_indices"]
    ]
    return {
        "schema_version": 1,
        "geometry": "tangent_cap",
        "score": "normalized_tangent_radial",
        "classes": np.asarray(classes, dtype=np.int64).tolist(),
        "selected_candidates": [candidate.to_dict() for candidate in selected],
        "selected_candidate_indices": list(selection["selected_candidate_indices"]),
        "component_counts": list(selection["component_counts"]),
        "objective_trajectory": list(selection["objective_trajectory"]),
        "parent_representation_hash": parent_representation_hash,
        "directional_representation_hash": directional_representation_hash,
        "normalization_policy": NORMALIZATION_POLICY,
        "cohort_indices": np.asarray(cohort_indices, dtype=np.int64).tolist(),
        "configuration": dict(configuration),
        "global_temperature": 1.0,
    }


def tangent_student_logits(
    student: Mapping[str, Any],
    normalized_features: np.ndarray,
) -> np.ndarray:
    required = {
        "schema_version",
        "geometry",
        "score",
        "classes",
        "selected_candidates",
        "selected_candidate_indices",
        "component_counts",
        "objective_trajectory",
        "parent_representation_hash",
        "directional_representation_hash",
        "normalization_policy",
        "cohort_indices",
        "configuration",
        "global_temperature",
    }
    if (
        set(student) != required
        or student["schema_version"] != 1
        or student["geometry"] != "tangent_cap"
        or student["score"] != "normalized_tangent_radial"
    ):
        raise ValueError("Unsupported tangent-student schema.")
    candidates = [
        TangentCapPrimitive.from_dict(payload)
        for payload in student["selected_candidates"]
    ]
    fields = tangent_field_matrix(
        candidates, normalized_features, "normalized_tangent_radial"
    )
    classes = np.asarray(student["classes"], dtype=np.int64)
    class_to_column = {int(value): index for index, value in enumerate(classes)}
    logits = np.full((len(normalized_features), len(classes)), -np.inf)
    counts = np.zeros(len(classes), dtype=np.int64)
    for index, candidate in enumerate(candidates):
        if candidate.class_label is None:
            raise ValueError("Tangent candidates require class labels.")
        column = class_to_column[int(candidate.class_label)]
        logits[:, column] = np.maximum(logits[:, column], -fields[:, index])
        counts[column] += 1
    if counts.tolist() != student["component_counts"]:
        raise ValueError("Serialized component counts are inconsistent.")
    return logits / float(student["global_temperature"])


def fit_tangent_global_temperature(
    student: dict[str, Any],
    normalized_features: np.ndarray,
    labels: np.ndarray,
    *,
    minimum: float,
    maximum: float,
) -> float:
    logits = tangent_student_logits(student, normalized_features)
    classes = np.asarray(student["classes"], dtype=np.int64)
    class_to_column = {int(value): index for index, value in enumerate(classes)}
    target_columns = np.asarray(
        [class_to_column[int(label)] for label in labels], dtype=np.int64
    )

    def objective(log_temperature: float) -> float:
        scaled = logits / np.exp(log_temperature)
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        log_probabilities = shifted - np.log(
            np.exp(shifted).sum(axis=1, keepdims=True)
        )
        return -float(
            np.mean(log_probabilities[np.arange(len(labels)), target_columns])
        )

    result = minimize_scalar(
        objective,
        bounds=(np.log(minimum), np.log(maximum)),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Global temperature fit failed: {result.message}")
    temperature = float(np.exp(result.x))
    student["global_temperature"] = temperature
    return temperature


def predict_tangent_student(
    student: Mapping[str, Any],
    features: np.ndarray,
    *,
    parent_representation_hash: str,
) -> tuple[np.ndarray, np.ndarray]:
    if parent_representation_hash != student["parent_representation_hash"]:
        raise ValueError("Parent representation hash mismatch.")
    expected = normalized_representation_hash(parent_representation_hash)
    if (
        student["directional_representation_hash"] != expected
        or student["normalization_policy"] != NORMALIZATION_POLICY
    ):
        raise ValueError("Tangent normalization lineage mismatch.")
    normalized = l2_normalize(features)
    logits = tangent_student_logits(student, normalized)
    probabilities = probabilities_from_logits(logits)
    classes = np.asarray(student["classes"], dtype=np.int64)
    return classes[np.argmax(probabilities, axis=1)], probabilities


def tangent_local_edit_rollback_evidence(
    student: dict[str, Any],
    features: np.ndarray,
    *,
    parent_representation_hash: str,
) -> dict[str, Any]:
    baseline = copy.deepcopy(student)
    baseline_predictions, _ = predict_tangent_student(
        baseline, features, parent_representation_hash=parent_representation_hash
    )
    edited = copy.deepcopy(student)
    edited["selected_candidates"][0]["angular_radius"] *= 1.01
    edited_predictions, _ = predict_tangent_student(
        edited, features, parent_representation_hash=parent_representation_hash
    )
    normalized = l2_normalize(features)
    parent_candidate = TangentCapPrimitive.from_dict(
        baseline["selected_candidates"][0]
    )
    edited_candidate = TangentCapPrimitive.from_dict(
        edited["selected_candidates"][0]
    )
    affected = (parent_candidate.radial_field(normalized) <= 0.0) | (
        edited_candidate.radial_field(normalized) <= 0.0
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
    rollback = copy.deepcopy(baseline)
    rollback_predictions, _ = predict_tangent_student(
        rollback, features, parent_representation_hash=parent_representation_hash
    )
    return {
        "edit": {
            "selected_candidate_position": 0,
            "field": "angular_radius",
            "relative_scale": 1.01,
        },
        "affected_count": int(np.sum(affected)),
        "unaffected_count": int(np.sum(unaffected)),
        "changed_prediction_count": int(
            np.sum(edited_predictions != baseline_predictions)
        ),
        "unaffected_prediction_preservation": preservation,
        "exact_json_rollback": rollback == baseline,
        "rollback_restored_predictions": bool(
            np.array_equal(rollback_predictions, baseline_predictions)
        ),
    }
