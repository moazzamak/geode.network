from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.subspace_primitive import SubspacePrimitive, fit_subspace_primitive


def generate_margin_subspace_candidates(
    features: np.ndarray,
    labels: np.ndarray,
    teacher_probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    rank: int,
    candidates_per_class: int,
    anchor_fraction: float,
    variance_floor_fraction: float = 1e-3,
    residual_floor_fraction: float = 0.05,
) -> list[SubspacePrimitive]:
    points = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels)
    probabilities = np.asarray(teacher_probabilities, dtype=np.float64)
    class_values = np.asarray(classes)
    if points.ndim != 2 or targets.shape != (len(points),):
        raise ValueError("Features and labels have incompatible shapes.")
    if probabilities.shape != (len(points), len(class_values)):
        raise ValueError("Teacher probabilities have the wrong shape.")
    if (
        np.any(probabilities < 0.0)
        or not np.all(np.isfinite(probabilities))
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)
    ):
        raise ValueError("Teacher probabilities must be finite and normalized.")
    if candidates_per_class < 1 or not 0.0 < anchor_fraction <= 1.0:
        raise ValueError("Invalid candidate count or anchor fraction.")
    support_size = rank + 2
    ordered_probabilities = np.sort(probabilities, axis=1)
    margins = ordered_probabilities[:, -1] - ordered_probabilities[:, -2]
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
        anchor_positions = np.linspace(
            0, len(anchors) - 1, candidates_per_class, dtype=np.int64
        )
        class_points = points[class_indices]
        class_norms = np.sum(class_points * class_points, axis=1)
        for position in anchor_positions:
            anchor_index = int(anchors[position])
            anchor = points[anchor_index]
            distances = np.maximum(
                class_norms + float(anchor @ anchor) - 2.0 * (class_points @ anchor),
                0.0,
            )
            local_indices = np.argpartition(
                distances, support_size - 1
            )[:support_size]
            candidates.append(
                fit_subspace_primitive(
                    class_points[local_indices],
                    rank,
                    variance_floor_fraction=variance_floor_fraction,
                    residual_floor_fraction=residual_floor_fraction,
                    class_label=int(class_label),
                    anchor_index=anchor_index,
                )
            )
    return candidates


def subspace_field_matrix(
    candidates: Sequence[SubspacePrimitive],
    features: np.ndarray,
    score_semantics: str,
) -> np.ndarray:
    if not candidates:
        raise ValueError("At least one subspace candidate is required.")
    if score_semantics == "normalized_radial":
        return np.column_stack(
            [candidate.radial_field(features) for candidate in candidates]
        )
    if score_semantics == "gaussian_log_likelihood":
        return np.column_stack(
            [-candidate.log_likelihood(features) for candidate in candidates]
        )
    raise ValueError(f"Unsupported score semantics {score_semantics!r}.")


def serialize_subspace_student(
    *,
    classes: np.ndarray,
    candidates: Sequence[SubspacePrimitive],
    selection: dict[str, Any],
    rank: int,
    score_semantics: str,
    cohort_indices: np.ndarray,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    selected_indices = selection["selected_candidate_indices"]
    return {
        "schema_version": 1,
        "classes": np.asarray(classes, dtype=np.int64).tolist(),
        "rank": rank,
        "score_semantics": score_semantics,
        "selected_candidates": [
            candidates[index].to_dict() for index in selected_indices
        ],
        "selected_candidate_indices": selected_indices,
        "component_counts": selection["component_counts"],
        "objective_trajectory": selection["objective_trajectory"],
        "cohort_indices": np.asarray(cohort_indices, dtype=np.int64).tolist(),
        "configuration": configuration,
    }


def subspace_student_logits(
    student: dict[str, Any], features: np.ndarray
) -> np.ndarray:
    candidates = [
        SubspacePrimitive.from_dict(payload)
        for payload in student["selected_candidates"]
    ]
    classes = np.asarray(student["classes"], dtype=np.int64)
    fields = subspace_field_matrix(
        candidates, features, student["score_semantics"]
    )
    fields = fields - fields.min(axis=1, keepdims=True)
    sums = np.zeros((len(features), len(classes)), dtype=np.float64)
    counts = np.zeros(len(classes), dtype=np.int64)
    class_to_column = {int(value): index for index, value in enumerate(classes)}
    for index, candidate in enumerate(candidates):
        if candidate.class_label is None:
            raise ValueError("Student candidates require class labels.")
        column = class_to_column[candidate.class_label]
        sums[:, column] += np.exp(-np.clip(fields[:, index], 0.0, 500.0))
        counts[column] += 1
    if counts.tolist() != student["component_counts"]:
        raise ValueError("Serialized component counts are inconsistent.")
    return np.log(np.maximum(sums, np.finfo(np.float64).tiny)) - np.log(
        counts
    )[None, :]


def predict_subspace_student(
    student: dict[str, Any], features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    logits = subspace_student_logits(student, features)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / (
        exponentials.sum(axis=1, keepdims=True) + 1e-12
    )
    classes = np.asarray(student["classes"], dtype=np.int64)
    return classes[np.argmax(probabilities, axis=1)], probabilities
