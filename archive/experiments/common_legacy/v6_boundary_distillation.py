from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.sdf_engine import EllipsoidExpert, Expert


@dataclass(frozen=True)
class SphereCandidate:
    class_label: int
    center: np.ndarray
    radius: float
    anchor_index: int
    support_size: int

    @property
    def parameter_count(self) -> int:
        return len(self.center) + 1

    @property
    def array_bytes(self) -> int:
        return int(
            np.asarray(self.center, dtype=np.float64).nbytes
            + np.dtype(np.float64).itemsize
        )

    def validate(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        if center.ndim != 1 or not np.all(np.isfinite(center)):
            raise ValueError("Sphere center must be a finite vector.")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("Sphere radius must be finite and positive.")
        if self.anchor_index < 0 or self.support_size < 1:
            raise ValueError("Sphere anchor and support size must be nonnegative.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "class_label": self.class_label,
            "center": np.asarray(self.center, dtype=np.float64).tolist(),
            "radius": float(self.radius),
            "anchor_index": self.anchor_index,
            "support_size": self.support_size,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SphereCandidate":
        if set(payload) != {
            "class_label",
            "center",
            "radius",
            "anchor_index",
            "support_size",
        }:
            raise ValueError("Unsupported sphere-candidate schema.")
        candidate = cls(
            class_label=int(payload["class_label"]),
            center=np.asarray(payload["center"], dtype=np.float64),
            radius=float(payload["radius"]),
            anchor_index=int(payload["anchor_index"]),
            support_size=int(payload["support_size"]),
        )
        candidate.validate()
        return candidate


def _validate_probabilities(
    probabilities: np.ndarray,
    *,
    sample_count: int,
    class_count: int,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (sample_count, class_count):
        raise ValueError(
            f"Teacher probabilities must have shape {(sample_count, class_count)}."
        )
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-8)
    ):
        raise ValueError("Teacher probabilities must be finite and normalized.")
    return values


def teacher_margins(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Teacher probabilities require at least two classes.")
    ordered = np.sort(values, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def generate_margin_sphere_candidates(
    features: np.ndarray,
    labels: np.ndarray,
    teacher_probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    candidates_per_class: int,
    seed_size: int,
    anchor_fraction: float,
) -> list[SphereCandidate]:
    points = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels)
    class_values = np.asarray(classes)
    if points.ndim != 2 or targets.shape != (len(points),):
        raise ValueError("Features and labels have incompatible shapes.")
    if len(class_values) < 2 or len(np.unique(class_values)) != len(class_values):
        raise ValueError("Classes must contain unique identifiers.")
    probabilities = _validate_probabilities(
        teacher_probabilities,
        sample_count=len(points),
        class_count=len(class_values),
    )
    if candidates_per_class < 1 or seed_size < 2:
        raise ValueError("Candidate count and seed size must be positive.")
    if not 0.0 < anchor_fraction <= 1.0:
        raise ValueError("anchor_fraction must lie in (0, 1].")

    margins = teacher_margins(probabilities)
    candidates: list[SphereCandidate] = []
    for class_label in class_values:
        class_indices = np.flatnonzero(targets == class_label)
        if len(class_indices) < seed_size:
            raise ValueError(
                f"Class {class_label} has {len(class_indices)} samples; "
                f"{seed_size} are required."
            )
        order = np.lexsort((class_indices, margins[class_indices]))
        boundary_count = max(
            candidates_per_class,
            int(np.ceil(anchor_fraction * len(class_indices))),
        )
        boundary_indices = class_indices[order[: min(boundary_count, len(order))]]
        positions = np.linspace(
            0,
            len(boundary_indices) - 1,
            num=candidates_per_class,
            dtype=np.int64,
        )
        class_points = points[class_indices]
        class_norms = np.sum(class_points * class_points, axis=1)
        for position in positions:
            anchor_index = int(boundary_indices[position])
            anchor = points[anchor_index]
            squared_distances = (
                class_norms
                + float(anchor @ anchor)
                - 2.0 * (class_points @ anchor)
            )
            squared_distances = np.maximum(squared_distances, 0.0)
            support_local = np.argpartition(
                squared_distances, seed_size - 1
            )[:seed_size]
            support = class_points[support_local]
            center = support.mean(axis=0)
            centered = support - center
            radius = float(
                np.sqrt(np.sum(centered * centered) / (len(support) - 1))
            )
            candidate = SphereCandidate(
                class_label=int(class_label),
                center=center,
                radius=radius,
                anchor_index=anchor_index,
                support_size=len(support),
            )
            candidate.validate()
            candidates.append(candidate)
    return candidates


def candidate_sdf_matrix(
    candidates: Sequence[SphereCandidate],
    features: np.ndarray,
) -> np.ndarray:
    if not candidates:
        raise ValueError("At least one sphere candidate is required.")
    points = np.asarray(features, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("Features must be a two-dimensional array.")
    centers = np.stack(
        [np.asarray(candidate.center, dtype=np.float64) for candidate in candidates]
    )
    radii = np.asarray([candidate.radius for candidate in candidates])
    if centers.shape[1] != points.shape[1]:
        raise ValueError("Candidate and feature dimensions do not match.")
    point_norms = np.sum(points * points, axis=1)[:, None]
    center_norms = np.sum(centers * centers, axis=1)[None, :]
    squared_distances = point_norms + center_norms - 2.0 * (points @ centers.T)
    return np.sqrt(np.maximum(squared_distances, 0.0)) / radii[None, :] - 1.0


def _logits_from_state(
    exp_sums: np.ndarray,
    component_counts: np.ndarray,
) -> np.ndarray:
    if np.any(component_counts < 1):
        raise ValueError("Every class requires at least one component.")
    return np.log(np.maximum(exp_sums, np.finfo(np.float64).tiny)) - np.log(
        component_counts
    )[None, :]


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / (exponentials.sum(axis=1, keepdims=True) + 1e-12)


def distillation_objective(
    logits: np.ndarray,
    teacher_probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    *,
    teacher_weight: float,
    ground_truth_weight: float,
    component_count: int,
    complexity_penalty: float,
) -> float:
    if teacher_weight < 0.0 or ground_truth_weight < 0.0:
        raise ValueError("Loss weights must be nonnegative.")
    if teacher_weight + ground_truth_weight <= 0.0:
        raise ValueError("At least one predictive loss weight must be positive.")
    if complexity_penalty < 0.0:
        raise ValueError("complexity_penalty must be nonnegative.")
    probabilities = probabilities_from_logits(logits)
    teacher = _validate_probabilities(
        teacher_probabilities,
        sample_count=len(probabilities),
        class_count=probabilities.shape[1],
    )
    log_probabilities = np.log(
        np.maximum(probabilities, np.finfo(np.float64).tiny)
    )
    teacher_cross_entropy = -float(np.mean(np.sum(teacher * log_probabilities, axis=1)))
    class_to_column = {int(value): index for index, value in enumerate(classes)}
    try:
        target_columns = np.asarray(
            [class_to_column[int(value)] for value in labels], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("Labels contain a class absent from classes.") from error
    ground_truth_cross_entropy = -float(
        np.mean(log_probabilities[np.arange(len(labels)), target_columns])
    )
    return (
        teacher_weight * teacher_cross_entropy
        + ground_truth_weight * ground_truth_cross_entropy
        + complexity_penalty * component_count
    )


def fit_boundary_distilled_student(
    features: np.ndarray,
    labels: np.ndarray,
    teacher_probabilities: np.ndarray,
    classes: np.ndarray,
    candidates: Sequence[SphereCandidate],
    cohort_indices: np.ndarray,
    *,
    component_limit: int,
    teacher_weight: float = 1.0,
    ground_truth_weight: float = 0.0,
    complexity_penalty: float = 0.0,
    minimum_improvement: float = 1e-8,
) -> dict[str, Any]:
    points = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels)
    class_values = np.asarray(classes)
    cohort = np.asarray(cohort_indices, dtype=np.int64)
    if points.ndim != 2 or targets.shape != (len(points),):
        raise ValueError("Features and labels have incompatible shapes.")
    probabilities = _validate_probabilities(
        teacher_probabilities,
        sample_count=len(points),
        class_count=len(class_values),
    )
    if cohort.ndim != 1 or len(cohort) < 1:
        raise ValueError("cohort_indices must be a non-empty vector.")
    if np.any(cohort < 0) or np.any(cohort >= len(points)):
        raise ValueError("cohort_indices contain an out-of-range sample.")
    if len(np.unique(cohort)) != len(cohort):
        raise ValueError("cohort_indices must be unique.")
    if component_limit < len(class_values):
        raise ValueError("component_limit must permit one sphere per class.")
    if minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be nonnegative.")

    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("At least one candidate is required.")
    class_to_column = {int(value): index for index, value in enumerate(class_values)}
    first_by_class: dict[int, int] = {}
    for index, candidate in enumerate(candidate_list):
        candidate.validate()
        try:
            class_to_column[candidate.class_label]
        except KeyError as error:
            raise ValueError(
                f"Candidate class {candidate.class_label} is not registered."
            ) from error
        first_by_class.setdefault(candidate.class_label, index)
    if set(first_by_class) != set(class_to_column):
        raise ValueError("Every class requires at least one sphere candidate.")

    selection = fit_distilled_candidate_fields(
        candidate_sdf_matrix(candidate_list, points[cohort]),
        [candidate.class_label for candidate in candidate_list],
        probabilities[cohort],
        targets[cohort],
        class_values,
        component_limit=component_limit,
        teacher_weight=teacher_weight,
        ground_truth_weight=ground_truth_weight,
        complexity_penalty=complexity_penalty,
        minimum_improvement=minimum_improvement,
        initial_components_per_class=1,
    )
    selected = selection["selected_candidate_indices"]
    selected_candidates = [candidate_list[index] for index in selected]
    return {
        "schema_version": 1,
        "classes": class_values.astype(int).tolist(),
        "selected_candidates": [
            candidate.to_dict() for candidate in selected_candidates
        ],
        "selected_candidate_indices": selected,
        "component_counts": selection["component_counts"],
        "objective_trajectory": selection["objective_trajectory"],
        "cohort_indices": cohort.tolist(),
        "configuration": {
            "component_limit": component_limit,
            "teacher_weight": teacher_weight,
            "ground_truth_weight": ground_truth_weight,
            "complexity_penalty": complexity_penalty,
            "minimum_improvement": minimum_improvement,
        },
    }


def fit_distilled_candidate_fields(
    candidate_fields: np.ndarray,
    candidate_class_labels: Sequence[int],
    teacher_probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    *,
    component_limit: int,
    teacher_weight: float = 1.0,
    ground_truth_weight: float = 0.0,
    complexity_penalty: float = 0.0,
    minimum_improvement: float = 1e-8,
    initial_components_per_class: int = 1,
    exact_component_count: bool = False,
) -> dict[str, Any]:
    fields = np.asarray(candidate_fields, dtype=np.float64)
    targets = np.asarray(labels)
    class_values = np.asarray(classes)
    if fields.ndim != 2 or fields.shape[1] != len(candidate_class_labels):
        raise ValueError("candidate_fields and labels have incompatible shapes.")
    if not np.all(np.isfinite(fields)):
        raise ValueError("candidate_fields must be finite.")
    probabilities = _validate_probabilities(
        teacher_probabilities,
        sample_count=len(fields),
        class_count=len(class_values),
    )
    if targets.shape != (len(fields),):
        raise ValueError("labels must match the candidate field rows.")
    if initial_components_per_class < 1:
        raise ValueError("initial_components_per_class must be positive.")
    if component_limit < len(class_values) * initial_components_per_class:
        raise ValueError("component_limit is below the required class initialization.")
    if minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be nonnegative.")

    class_to_column = {int(value): index for index, value in enumerate(class_values)}
    candidate_columns = []
    by_class: dict[int, list[int]] = {int(value): [] for value in class_values}
    for index, class_label in enumerate(candidate_class_labels):
        try:
            column = class_to_column[int(class_label)]
        except KeyError as error:
            raise ValueError(f"Candidate class {class_label} is not registered.") from error
        candidate_columns.append(column)
        by_class[int(class_label)].append(index)
    if any(len(indices) < initial_components_per_class for indices in by_class.values()):
        raise ValueError("Every class requires enough initialization candidates.")

    stabilized_fields = fields - fields.min(axis=1, keepdims=True)
    candidate_exp = np.exp(-np.clip(stabilized_fields, 0.0, 500.0))
    selected = [
        index
        for class_label in class_values
        for index in by_class[int(class_label)][:initial_components_per_class]
    ]
    selected_set = set(selected)
    component_counts = np.full(
        len(class_values), initial_components_per_class, dtype=np.int64
    )
    exp_sums = np.zeros((len(fields), len(class_values)), dtype=np.float64)
    for candidate_index in selected:
        exp_sums[:, candidate_columns[candidate_index]] += candidate_exp[:, candidate_index]

    objective = distillation_objective(
        _logits_from_state(exp_sums, component_counts),
        probabilities,
        targets,
        class_values,
        teacher_weight=teacher_weight,
        ground_truth_weight=ground_truth_weight,
        component_count=len(selected),
        complexity_penalty=complexity_penalty,
    )
    trajectory = [objective]
    while len(selected) < min(component_limit, len(candidate_class_labels)):
        best_index: int | None = None
        best_objective = np.inf if exact_component_count else objective
        for candidate_index, class_column in enumerate(candidate_columns):
            if candidate_index in selected_set:
                continue
            trial_sums = exp_sums.copy()
            trial_sums[:, class_column] += candidate_exp[:, candidate_index]
            trial_counts = component_counts.copy()
            trial_counts[class_column] += 1
            trial_objective = distillation_objective(
                _logits_from_state(trial_sums, trial_counts),
                probabilities,
                targets,
                class_values,
                teacher_weight=teacher_weight,
                ground_truth_weight=ground_truth_weight,
                component_count=len(selected) + 1,
                complexity_penalty=complexity_penalty,
            )
            if (
                trial_objective < best_objective
                - (0.0 if exact_component_count else minimum_improvement)
            ):
                best_objective = trial_objective
                best_index = candidate_index
        if best_index is None:
            break
        class_column = candidate_columns[best_index]
        exp_sums[:, class_column] += candidate_exp[:, best_index]
        component_counts[class_column] += 1
        selected.append(best_index)
        selected_set.add(best_index)
        objective = best_objective
        trajectory.append(objective)

    if exact_component_count and len(selected) != component_limit:
        raise ValueError("Not enough candidates to satisfy the exact component count.")

    return {
        "selected_candidate_indices": selected,
        "component_counts": component_counts.tolist(),
        "objective_trajectory": trajectory,
    }


def student_logits(
    student: dict[str, Any],
    features: np.ndarray,
) -> np.ndarray:
    required = {
        "schema_version",
        "classes",
        "selected_candidates",
        "selected_candidate_indices",
        "component_counts",
        "objective_trajectory",
        "cohort_indices",
        "configuration",
    }
    if set(student) != required or student["schema_version"] != 1:
        raise ValueError("Unsupported boundary-student schema.")
    classes = np.asarray(student["classes"], dtype=np.int64)
    candidates = [
        SphereCandidate.from_dict(payload)
        for payload in student["selected_candidates"]
    ]
    fields = candidate_sdf_matrix(candidates, features)
    exp_sums = np.zeros((len(features), len(classes)), dtype=np.float64)
    counts = np.zeros(len(classes), dtype=np.int64)
    class_to_column = {int(value): index for index, value in enumerate(classes)}
    for index, candidate in enumerate(candidates):
        column = class_to_column[candidate.class_label]
        exp_sums[:, column] += np.exp(-fields[:, index])
        counts[column] += 1
    if counts.tolist() != student["component_counts"]:
        raise ValueError("Serialized component counts are inconsistent.")
    return _logits_from_state(exp_sums, counts)


def predict_boundary_student(
    student: dict[str, Any],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    classes = np.asarray(student["classes"], dtype=np.int64)
    probabilities = probabilities_from_logits(student_logits(student, features))
    return classes[np.argmax(probabilities, axis=1)], probabilities


def student_to_geode_models(student: dict[str, Any]) -> list[dict[str, Any]]:
    classes = np.asarray(student["classes"], dtype=np.int64)
    candidates = [
        SphereCandidate.from_dict(payload)
        for payload in student["selected_candidates"]
    ]
    models = []
    for class_label in classes:
        expert = Expert(alpha=1.0)
        for candidate in candidates:
            if candidate.class_label != int(class_label):
                continue
            dimension = len(candidate.center)
            expert.add_ellipsoid(
                EllipsoidExpert(
                    candidate.center,
                    np.full(dimension, candidate.radius, dtype=np.float64),
                    np.eye(dimension, dtype=np.float64),
                )
            )
        models.append(
            {
                "class": int(class_label),
                "model": [expert],
                "constructor": "v6_boundary_distillation",
            }
        )
    return models
