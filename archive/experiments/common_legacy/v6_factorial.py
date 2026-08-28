from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from experiments.common.v6_boundary_distillation import SphereCandidate
from src.directional_primitive import SphericalCapPrimitive
from src.subspace_primitive import SubspacePrimitive


OBJECTIVES = ("coverage", "direct", "teacher")
PRIMITIVES = ("sphere", "subspace_r32", "directional")
SCORES = ("normalized_radial", "proper_likelihood", "teacher_softmin")
BUDGETS = ("component", "parameter")


def validate_fractional_factorial(
    cells: Sequence[Mapping[str, Any]],
    baselines: Mapping[str, str],
) -> dict[str, Any]:
    if len(cells) < 8:
        raise ValueError("M31 main-effect design requires at least eight cells.")
    expected_baselines = {
        "objective": "coverage",
        "primitive": "sphere",
        "score": "normalized_radial",
        "budget": "component",
    }
    if dict(baselines) != expected_baselines:
        raise ValueError("Unsupported M31 factorial baselines.")
    ids = [str(cell["id"]) for cell in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("M31 cell identifiers must be unique.")

    rows = []
    for cell in cells:
        objective = str(cell["objective"])
        primitive = str(cell["primitive"])
        score = str(cell["score"])
        budget = str(cell["budget"])
        if (
            objective not in OBJECTIVES
            or primitive not in PRIMITIVES
            or score not in SCORES
            or budget not in BUDGETS
        ):
            raise ValueError(f"Unsupported M31 cell {cell['id']}.")
        if primitive == "subspace_r32" and budget == "parameter":
            raise ValueError("Rank-32 subspaces are infeasible under the parameter budget.")
        if primitive == "directional" and score == "proper_likelihood":
            raise ValueError("Directional likelihood was not retained by M30.")
        rows.append(
            [
                1,
                objective == "direct",
                objective == "teacher",
                primitive == "subspace_r32",
                primitive == "directional",
                score == "proper_likelihood",
                score == "teacher_softmin",
                budget == "parameter",
            ]
        )
    matrix = np.asarray(rows, dtype=np.float64)
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != 8:
        raise ValueError(f"M31 main-effect design rank is {rank}; expected 8.")
    return {
        "cell_count": len(cells),
        "design_columns": 8,
        "design_rank": rank,
        "main_effects_identifiable": True,
    }


def candidate_parameter_count(candidate: Any) -> int:
    return int(candidate.parameter_count)


def candidate_array_bytes(candidate: Any) -> int:
    return int(candidate.array_bytes)


def candidate_class_label(candidate: Any) -> int:
    if candidate.class_label is None:
        raise ValueError("M31 candidates require class labels.")
    return int(candidate.class_label)


def primitive_field_matrix(
    candidates: Sequence[Any],
    features: np.ndarray,
    *,
    primitive: str,
    score: str,
) -> np.ndarray:
    if not candidates:
        raise ValueError("At least one candidate is required.")
    points = np.asarray(features, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("features must be two-dimensional.")
    if primitive == "sphere":
        sphere_candidates = [
            candidate
            for candidate in candidates
            if isinstance(candidate, SphereCandidate)
        ]
        if len(sphere_candidates) != len(candidates):
            raise ValueError("Sphere field received another primitive type.")
        differences = (
            points[:, None, :]
            - np.stack([candidate.center for candidate in sphere_candidates])[None, :, :]
        )
        squared_distances = np.sum(differences * differences, axis=2)
        radii = np.asarray([candidate.radius for candidate in sphere_candidates])
        if score == "proper_likelihood":
            dimension = points.shape[1]
            variances = np.maximum(
                radii * radii / dimension, np.finfo(np.float64).tiny
            )
            return 0.5 * (
                squared_distances / variances[None, :]
                + dimension * np.log(2.0 * np.pi * variances)[None, :]
            )
        return np.sqrt(np.maximum(squared_distances, 0.0)) / radii[None, :] - 1.0
    if primitive == "subspace_r32":
        subspaces = [
            candidate
            for candidate in candidates
            if isinstance(candidate, SubspacePrimitive)
        ]
        if len(subspaces) != len(candidates):
            raise ValueError("Subspace field received another primitive type.")
        if score == "proper_likelihood":
            return np.column_stack(
                [-candidate.log_likelihood(points) for candidate in subspaces]
            )
        return np.column_stack(
            [candidate.radial_field(points) for candidate in subspaces]
        )
    if primitive == "directional":
        caps = [
            candidate
            for candidate in candidates
            if isinstance(candidate, SphericalCapPrimitive)
        ]
        if len(caps) != len(candidates):
            raise ValueError("Directional field received another primitive type.")
        if score == "proper_likelihood":
            raise ValueError("Directional likelihood is not registered for M31.")
        return np.column_stack([candidate.angular_field(points) for candidate in caps])
    raise ValueError(f"Unsupported primitive {primitive!r}.")


def _target_probabilities(
    objective: str,
    teacher_probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    teacher = np.asarray(teacher_probabilities, dtype=np.float64)
    if teacher.shape != (len(labels), len(classes)):
        raise ValueError("Teacher probabilities have the wrong shape.")
    if objective == "teacher":
        return teacher
    if objective != "direct":
        raise ValueError("Predictive selection requires direct or teacher objective.")
    class_to_column = {int(value): index for index, value in enumerate(classes)}
    targets = np.zeros_like(teacher)
    for row, label in enumerate(labels):
        try:
            targets[row, class_to_column[int(label)]] = 1.0
        except KeyError as error:
            raise ValueError("Label is absent from classes.") from error
    return targets


def _cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return -float(np.mean(np.sum(targets * log_probabilities, axis=1)))


def select_predictive_candidates(
    fields: np.ndarray,
    candidate_labels: Sequence[int],
    teacher_probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    *,
    objective: str,
    score: str,
    component_limit: int,
    initial_components_per_class: int,
    minimum_improvement: float,
) -> dict[str, Any]:
    values = np.asarray(fields, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != len(candidate_labels):
        raise ValueError("Candidate fields and labels have incompatible shapes.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Candidate fields must be finite.")
    if component_limit > len(candidate_labels):
        raise ValueError("Component budget exceeds available candidates.")
    targets = _target_probabilities(
        objective, teacher_probabilities, np.asarray(labels), class_values
    )
    class_to_column = {int(value): index for index, value in enumerate(class_values)}
    candidate_columns = [class_to_column[int(label)] for label in candidate_labels]
    by_class = {
        int(value): [
            index
            for index, label in enumerate(candidate_labels)
            if int(label) == int(value)
        ]
        for value in class_values
    }
    if any(
        len(indices) < initial_components_per_class for indices in by_class.values()
    ):
        raise ValueError("Every class requires enough initialization candidates.")
    selected = [
        index
        for class_label in class_values
        for index in by_class[int(class_label)][:initial_components_per_class]
    ]
    selected_set = set(selected)
    counts = np.full(
        len(class_values), initial_components_per_class, dtype=np.int64
    )
    hard_min = score == "normalized_radial"
    if hard_min:
        state = np.full((len(values), len(class_values)), np.inf)
        for index in selected:
            column = candidate_columns[index]
            state[:, column] = np.minimum(state[:, column], values[:, index])
        logits = -state
    else:
        stabilized = values - values.min(axis=1, keepdims=True)
        exponentials = np.exp(-np.clip(stabilized, 0.0, 500.0))
        state = np.zeros((len(values), len(class_values)))
        for index in selected:
            state[:, candidate_columns[index]] += exponentials[:, index]
        logits = np.log(np.maximum(state, np.finfo(np.float64).tiny)) - np.log(
            counts
        )[None, :]
    objective_value = _cross_entropy(logits, targets)
    trajectory = [objective_value]
    while len(selected) < component_limit:
        best_index: int | None = None
        best_objective = np.inf
        best_state: np.ndarray | None = None
        for index, column in enumerate(candidate_columns):
            if index in selected_set:
                continue
            trial_state = state.copy()
            trial_counts = counts.copy()
            trial_counts[column] += 1
            if hard_min:
                trial_state[:, column] = np.minimum(
                    trial_state[:, column], values[:, index]
                )
                trial_logits = -trial_state
            else:
                trial_state[:, column] += exponentials[:, index]
                trial_logits = np.log(
                    np.maximum(trial_state, np.finfo(np.float64).tiny)
                ) - np.log(trial_counts)[None, :]
            trial_objective = _cross_entropy(trial_logits, targets)
            if trial_objective < best_objective:
                best_objective = trial_objective
                best_index = index
                best_state = trial_state
        if best_index is None or best_state is None:
            raise RuntimeError("Predictive selection could not fill the budget.")
        if (
            best_objective >= objective_value - minimum_improvement
            and len(selected) >= component_limit
        ):
            break
        column = candidate_columns[best_index]
        counts[column] += 1
        state = best_state
        selected.append(best_index)
        selected_set.add(best_index)
        objective_value = best_objective
        trajectory.append(objective_value)
    return {
        "selected_candidate_indices": selected,
        "component_counts": counts.tolist(),
        "objective_trajectory": trajectory,
    }


def select_coverage_candidates(
    radial_fields: np.ndarray,
    candidate_labels: Sequence[int],
    labels: np.ndarray,
    classes: np.ndarray,
    *,
    component_limit: int,
    initial_components_per_class: int,
) -> dict[str, Any]:
    fields = np.asarray(radial_fields, dtype=np.float64)
    targets = np.asarray(labels)
    class_values = np.asarray(classes, dtype=np.int64)
    if fields.shape != (len(targets), len(candidate_labels)):
        raise ValueError("Coverage fields have incompatible shapes.")
    by_class = {
        int(value): [
            index
            for index, label in enumerate(candidate_labels)
            if int(label) == int(value)
        ]
        for value in class_values
    }
    selected = [
        index
        for class_label in class_values
        for index in by_class[int(class_label)][:initial_components_per_class]
    ]
    selected_set = set(selected)
    counts = np.full(
        len(class_values), initial_components_per_class, dtype=np.int64
    )
    covered = np.zeros(len(targets), dtype=bool)
    for index in selected:
        covered |= (targets == candidate_labels[index]) & (fields[:, index] <= 0.0)
    trajectory = [float(np.mean(covered))]
    class_to_column = {int(value): index for index, value in enumerate(class_values)}
    while len(selected) < component_limit:
        best_index: int | None = None
        best_key: tuple[int, float, int] | None = None
        for index, class_label in enumerate(candidate_labels):
            if index in selected_set:
                continue
            class_mask = targets == class_label
            newly_covered = class_mask & ~covered & (fields[:, index] <= 0.0)
            positive_residual = np.maximum(fields[class_mask, index], 0.0)
            key = (
                int(np.sum(newly_covered)),
                -float(np.mean(positive_residual)),
                -index,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        if best_index is None:
            raise RuntimeError("Coverage selection could not fill the budget.")
        selected.append(best_index)
        selected_set.add(best_index)
        counts[class_to_column[int(candidate_labels[best_index])]] += 1
        covered |= (targets == candidate_labels[best_index]) & (
            fields[:, best_index] <= 0.0
        )
        trajectory.append(float(np.mean(covered)))
    return {
        "selected_candidate_indices": selected,
        "component_counts": counts.tolist(),
        "objective_trajectory": trajectory,
    }


def _serialize_candidate(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, SphereCandidate):
        family = "sphere"
    elif isinstance(candidate, SubspacePrimitive):
        family = "subspace_r32"
    elif isinstance(candidate, SphericalCapPrimitive):
        family = "directional"
    else:
        raise ValueError("Unsupported M31 candidate type.")
    return {"family": family, "payload": candidate.to_dict()}


def _deserialize_candidate(item: Mapping[str, Any]) -> Any:
    if set(item) != {"family", "payload"}:
        raise ValueError("Unsupported M31 candidate schema.")
    family = item["family"]
    if family == "sphere":
        return SphereCandidate.from_dict(item["payload"])
    if family == "subspace_r32":
        return SubspacePrimitive.from_dict(item["payload"])
    if family == "directional":
        return SphericalCapPrimitive.from_dict(item["payload"])
    raise ValueError(f"Unsupported candidate family {family!r}.")


def serialize_factorial_student(
    *,
    cell: Mapping[str, Any],
    classes: np.ndarray,
    candidates: Sequence[Any],
    selection: Mapping[str, Any],
    parent_representation_hash: str,
    directional_representation_hash: str | None,
    class_priors: np.ndarray,
) -> dict[str, Any]:
    selected = [
        candidates[index] for index in selection["selected_candidate_indices"]
    ]
    return {
        "schema_version": 1,
        "cell": dict(cell),
        "classes": np.asarray(classes, dtype=np.int64).tolist(),
        "selected_candidates": [_serialize_candidate(candidate) for candidate in selected],
        "selected_candidate_indices": list(
            selection["selected_candidate_indices"]
        ),
        "component_counts": list(selection["component_counts"]),
        "objective_trajectory": list(selection["objective_trajectory"]),
        "parent_representation_hash": parent_representation_hash,
        "directional_representation_hash": directional_representation_hash,
        "class_priors": np.asarray(class_priors, dtype=np.float64).tolist(),
        "global_temperature": 1.0,
    }


def factorial_student_logits(
    student: Mapping[str, Any],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = [
        _deserialize_candidate(item) for item in student["selected_candidates"]
    ]
    cell = student["cell"]
    score = str(cell["score"])
    fields = primitive_field_matrix(
        candidates,
        features,
        primitive=str(cell["primitive"]),
        score=score,
    )
    classes = np.asarray(student["classes"], dtype=np.int64)
    labels = [candidate_class_label(candidate) for candidate in candidates]
    class_to_column = {int(value): index for index, value in enumerate(classes)}
    if score == "normalized_radial":
        logits = np.full((len(features), len(classes)), -np.inf)
        for index, label in enumerate(labels):
            column = class_to_column[label]
            logits[:, column] = np.maximum(logits[:, column], -fields[:, index])
    else:
        stabilized = fields - fields.min(axis=1, keepdims=True)
        sums = np.zeros((len(features), len(classes)), dtype=np.float64)
        counts = np.zeros(len(classes), dtype=np.int64)
        for index, label in enumerate(labels):
            column = class_to_column[label]
            sums[:, column] += np.exp(-np.clip(stabilized[:, index], 0.0, 500.0))
            counts[column] += 1
        logits = np.log(np.maximum(sums, np.finfo(np.float64).tiny)) - np.log(
            counts
        )[None, :]
        if score == "proper_likelihood":
            priors = np.asarray(student["class_priors"], dtype=np.float64)
            logits += np.log(priors)[None, :]
    return logits / float(student["global_temperature"]), fields


def fit_global_temperature(
    student: dict[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
    *,
    minimum: float,
    maximum: float,
) -> float:
    logits, _ = factorial_student_logits(student, features)
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


def predict_factorial_student(
    student: Mapping[str, Any],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    logits, _ = factorial_student_logits(student, features)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    classes = np.asarray(student["classes"], dtype=np.int64)
    return classes[np.argmax(probabilities, axis=1)], probabilities


def local_edit_rollback_evidence(
    student: dict[str, Any],
    features: np.ndarray,
) -> dict[str, Any]:
    baseline = copy.deepcopy(student)
    baseline_predictions, _ = predict_factorial_student(baseline, features)
    edited = copy.deepcopy(student)
    first = edited["selected_candidates"][0]
    family = first["family"]
    payload = first["payload"]
    if family == "sphere":
        payload["radius"] *= 1.01
    elif family == "subspace_r32":
        payload["residual_variance"] *= 1.01
    elif family == "directional":
        payload["angular_radius"] *= 1.01
    else:
        raise ValueError("Unsupported local edit family.")
    edited_predictions, _ = predict_factorial_student(edited, features)
    candidate = _deserialize_candidate(edited["selected_candidates"][0])
    region_score = (
        "normalized_radial"
        if str(edited["cell"]["score"]) == "proper_likelihood"
        else str(edited["cell"]["score"])
    )
    baseline_region_field = primitive_field_matrix(
        [_deserialize_candidate(baseline["selected_candidates"][0])],
        features,
        primitive=family,
        score=region_score,
    )[:, 0]
    edited_field = primitive_field_matrix(
        [candidate],
        features,
        primitive=family,
        score=region_score,
    )[:, 0]
    affected = (baseline_region_field <= 0.0) | (edited_field <= 0.0)
    unaffected = ~affected
    preservation = (
        float(np.mean(edited_predictions[unaffected] == baseline_predictions[unaffected]))
        if np.any(unaffected)
        else 1.0
    )
    rollback = copy.deepcopy(baseline)
    rollback_predictions, _ = predict_factorial_student(rollback, features)
    return {
        "edit": {
            "selected_candidate_position": 0,
            "family": family,
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
