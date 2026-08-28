from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from experiments.common.v5_artifacts import payload_hash
from experiments.common.v5_protocol import require_sha256
from experiments.common.v6_factorial import (
    candidate_class_label,
    primitive_field_matrix,
)


def _class_component_indices(
    candidate_labels: Sequence[int],
    classes: np.ndarray,
) -> list[np.ndarray]:
    labels = np.asarray(candidate_labels, dtype=np.int64)
    class_values = np.asarray(classes, dtype=np.int64)
    if labels.ndim != 1 or class_values.ndim != 1 or len(np.unique(class_values)) != len(
        class_values
    ):
        raise ValueError("Candidate labels and unique classes must be one-dimensional.")
    indices = [np.flatnonzero(labels == value) for value in class_values]
    if any(len(item) == 0 for item in indices):
        raise ValueError("Every class requires at least one component.")
    if sum(len(item) for item in indices) != len(labels):
        raise ValueError("Every candidate label must occur in classes.")
    return indices


def normalized_class_weights(
    log_weights: np.ndarray,
    candidate_labels: Sequence[int],
    classes: np.ndarray,
) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64)
    if values.shape != (len(candidate_labels),) or not np.all(np.isfinite(values)):
        raise ValueError("log_weights must be one finite value per component.")
    weights = np.empty_like(values)
    for indices in _class_component_indices(candidate_labels, classes):
        shifted = values[indices] - np.max(values[indices])
        exponentials = np.exp(shifted)
        weights[indices] = exponentials / np.sum(exponentials)
    return weights


def weighted_class_logits(
    fields: np.ndarray,
    candidate_labels: Sequence[int],
    classes: np.ndarray,
    component_weights: np.ndarray,
    *,
    global_temperature: float,
) -> np.ndarray:
    values = np.asarray(fields, dtype=np.float64)
    weights = np.asarray(component_weights, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != len(candidate_labels):
        raise ValueError("fields and candidate_labels have incompatible shapes.")
    if not np.all(np.isfinite(values)):
        raise ValueError("fields must be finite.")
    if (
        weights.shape != (len(candidate_labels),)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError("component_weights must be finite and nonnegative.")
    if not np.isfinite(global_temperature) or global_temperature <= 0.0:
        raise ValueError("global_temperature must be finite and positive.")
    logits = np.empty((len(values), len(class_values)), dtype=np.float64)
    for column, indices in enumerate(
        _class_component_indices(candidate_labels, class_values)
    ):
        class_weights = weights[indices]
        if not np.isclose(
            np.sum(class_weights), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("Component weights must sum to one within each class.")
        positive = class_weights > 0.0
        if not np.any(positive):
            raise ValueError("Every class requires at least one positive weight.")
        logits[:, column] = logsumexp(
            np.log(class_weights[positive])[None, :]
            - values[:, indices[positive]],
            axis=1,
        )
    return logits / global_temperature


def _objective_and_gradient(
    parameters: np.ndarray,
    *,
    fields: np.ndarray,
    candidate_labels: np.ndarray,
    classes: np.ndarray,
    target_columns: np.ndarray,
    regularization: float,
) -> tuple[float, np.ndarray]:
    log_weights = parameters[:-1]
    log_temperature = float(parameters[-1])
    temperature = float(np.exp(log_temperature))
    class_indices = _class_component_indices(candidate_labels, classes)
    weights = normalized_class_weights(log_weights, candidate_labels, classes)
    class_scores = np.empty((len(fields), len(classes)), dtype=np.float64)
    responsibilities: list[np.ndarray] = []
    for column, indices in enumerate(class_indices):
        terms = log_weights[indices][None, :] - fields[:, indices]
        numerator_logsum = logsumexp(terms, axis=1)
        denominator_logsum = logsumexp(log_weights[indices])
        class_scores[:, column] = numerator_logsum - denominator_logsum
        responsibilities.append(np.exp(terms - numerator_logsum[:, None]))
    logits = class_scores / temperature
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    cross_entropy = -float(
        np.mean(
            np.log(
                np.maximum(
                    probabilities[np.arange(len(fields)), target_columns],
                    np.finfo(np.float64).tiny,
                )
            )
        )
    )
    centered = np.zeros_like(log_weights)
    for indices in class_indices:
        centered[indices] = log_weights[indices] - np.mean(log_weights[indices])
    penalty = regularization * float(np.sum(centered * centered))
    derivative_logits = probabilities
    derivative_logits[np.arange(len(fields)), target_columns] -= 1.0
    derivative_logits /= len(fields)
    derivative_scores = derivative_logits / temperature
    gradient_weights = np.zeros_like(log_weights)
    for column, indices in enumerate(class_indices):
        gradient_weights[indices] = np.sum(
            derivative_scores[:, column, None]
            * (responsibilities[column] - weights[indices][None, :]),
            axis=0,
        )
    gradient_weights += 2.0 * regularization * centered
    gradient_temperature = -float(np.sum(derivative_logits * logits))
    return cross_entropy + penalty, np.concatenate(
        [gradient_weights, [gradient_temperature]]
    )


def fit_weighted_readout(
    fields: np.ndarray,
    candidate_labels: Sequence[int],
    labels: np.ndarray,
    classes: np.ndarray,
    *,
    regularization: float,
    maximum_iterations: int,
    gradient_tolerance: float,
    minimum_temperature: float,
    maximum_temperature: float,
    initial_temperature: float = 1.0,
    fit_component_weights: bool = True,
) -> dict[str, Any]:
    values = np.asarray(fields, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    class_values = np.asarray(classes, dtype=np.int64)
    candidate_values = np.asarray(candidate_labels, dtype=np.int64)
    if values.shape != (len(targets), len(candidate_values)):
        raise ValueError("Training fields and labels have incompatible shapes.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Training fields must be finite.")
    if regularization < 0.0 or maximum_iterations < 1 or gradient_tolerance <= 0.0:
        raise ValueError("Optimizer controls are invalid.")
    if (
        minimum_temperature <= 0.0
        or maximum_temperature <= minimum_temperature
        or not minimum_temperature <= initial_temperature <= maximum_temperature
    ):
        raise ValueError("Temperature bounds or initialization are invalid.")
    class_to_column = {int(value): index for index, value in enumerate(class_values)}
    try:
        target_columns = np.asarray(
            [class_to_column[int(label)] for label in targets], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("A training label is absent from classes.") from error
    _class_component_indices(candidate_values, class_values)
    component_count = len(candidate_values)
    initial = np.concatenate(
        [
            np.zeros(component_count, dtype=np.float64),
            [np.log(initial_temperature)],
        ]
    )
    if fit_component_weights:
        bounds = [(None, None)] * component_count + [
            (np.log(minimum_temperature), np.log(maximum_temperature))
        ]
    else:
        bounds = [(0.0, 0.0)] * component_count + [
            (np.log(minimum_temperature), np.log(maximum_temperature))
        ]

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        return _objective_and_gradient(
            parameters,
            fields=values,
            candidate_labels=candidate_values,
            classes=class_values,
            target_columns=target_columns,
            regularization=regularization if fit_component_weights else 0.0,
        )

    initial_objective, initial_gradient = objective(initial)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": int(maximum_iterations),
            "gtol": float(gradient_tolerance),
            "ftol": 1e-15,
            "maxls": 50,
        },
    )
    if not result.success:
        raise RuntimeError(f"Weighted readout fit failed: {result.message}")
    final_log_weights = np.asarray(result.x[:-1], dtype=np.float64)
    weights = normalized_class_weights(
        final_log_weights, candidate_values, class_values
    )
    return {
        "log_weights": final_log_weights.tolist(),
        "component_weights": weights.tolist(),
        "global_temperature": float(np.exp(result.x[-1])),
        "optimizer": {
            "method": "L-BFGS-B",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "initial_objective": float(initial_objective),
            "final_objective": float(result.fun),
            "initial_gradient_max_abs": float(np.max(np.abs(initial_gradient))),
            "final_gradient_max_abs": float(np.max(np.abs(result.jac))),
            "maximum_iterations": int(maximum_iterations),
            "gradient_tolerance": float(gradient_tolerance),
            "regularization": float(regularization if fit_component_weights else 0.0),
            "initialization": "zero_equal_weights",
        },
    }


def serialize_weighted_student(
    parent_student: Mapping[str, Any],
    fitted_readout: Mapping[str, Any],
    *,
    parent_student_sha256: str,
) -> dict[str, Any]:
    required_parent = {
        "schema_version",
        "cell",
        "classes",
        "selected_candidates",
        "selected_candidate_indices",
        "component_counts",
        "objective_trajectory",
        "parent_representation_hash",
        "directional_representation_hash",
        "class_priors",
        "global_temperature",
    }
    if set(parent_student) != required_parent or parent_student["schema_version"] != 1:
        raise ValueError("Unsupported M31 parent-student schema.")
    if parent_student["cell"] != {
        "id": "direct_subspace_radial_component",
        "objective": "direct",
        "primitive": "subspace_r32",
        "score": "normalized_radial",
        "budget": "component",
    }:
        raise ValueError("A1-W requires the retained direct rank-32 affine parent.")
    candidates = parent_student["selected_candidates"]
    labels = [
        int(item["payload"]["class_label"])
        for item in candidates
        if item.get("family") == "subspace_r32"
        and len(item["payload"]["tangent_variances"]) == 32
    ]
    if len(labels) != len(candidates):
        raise ValueError("A1-W parent components must all be rank-32 subspaces.")
    classes = np.asarray(parent_student["classes"], dtype=np.int64)
    log_weights = np.asarray(fitted_readout["log_weights"], dtype=np.float64)
    weights = np.asarray(fitted_readout["component_weights"], dtype=np.float64)
    expected_weights = normalized_class_weights(log_weights, labels, classes)
    if not np.allclose(weights, expected_weights, rtol=0.0, atol=1e-15):
        raise ValueError("Serialized weights do not match their log-weight parameters.")
    return {
        "schema_version": 1,
        "family": "weighted_affine_rank32",
        "classes": list(parent_student["classes"]),
        "selected_candidates": copy.deepcopy(candidates),
        "selected_candidate_indices": list(
            parent_student["selected_candidate_indices"]
        ),
        "component_counts": list(parent_student["component_counts"]),
        "component_log_weights": log_weights.tolist(),
        "component_weights": weights.tolist(),
        "global_temperature": float(fitted_readout["global_temperature"]),
        "optimizer": copy.deepcopy(fitted_readout["optimizer"]),
        "parent_student_sha256": parent_student_sha256,
        "parent_component_hash": payload_hash(
            {
                "selected_candidates": candidates,
                "selected_candidate_indices": parent_student[
                    "selected_candidate_indices"
                ],
            }
        ),
        "parent_representation_hash": parent_student[
            "parent_representation_hash"
        ],
        "readout_contract": {
            "constraint": "per_class_simplex",
            "parameterization": "softmax_log_weights",
            "temperature_policy": "one_global",
        },
    }


def validate_weighted_student(student: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "family",
        "classes",
        "selected_candidates",
        "selected_candidate_indices",
        "component_counts",
        "component_log_weights",
        "component_weights",
        "global_temperature",
        "optimizer",
        "parent_student_sha256",
        "parent_component_hash",
        "parent_representation_hash",
        "readout_contract",
    }
    if (
        set(student) != required
        or student["schema_version"] != 1
        or student["family"] != "weighted_affine_rank32"
        or student["readout_contract"]
        != {
            "constraint": "per_class_simplex",
            "parameterization": "softmax_log_weights",
            "temperature_policy": "one_global",
        }
    ):
        raise ValueError("Unsupported weighted-student schema.")
    require_sha256(str(student["parent_student_sha256"]), "parent_student_sha256")
    require_sha256(
        str(student["parent_representation_hash"]), "parent_representation_hash"
    )
    candidates = student["selected_candidates"]
    if len(candidates) != len(student["selected_candidate_indices"]):
        raise ValueError("Weighted student component ordering is inconsistent.")
    labels = [
        int(item["payload"]["class_label"])
        for item in candidates
        if item.get("family") == "subspace_r32"
        and len(item["payload"]["tangent_variances"]) == 32
    ]
    if len(labels) != len(candidates):
        raise ValueError("Weighted student requires rank-32 affine components.")
    classes = np.asarray(student["classes"], dtype=np.int64)
    expected_counts = [
        int(np.sum(np.asarray(labels) == value)) for value in classes
    ]
    if expected_counts != student["component_counts"]:
        raise ValueError("Weighted student component counts are inconsistent.")
    log_weights = np.asarray(student["component_log_weights"], dtype=np.float64)
    weights = np.asarray(student["component_weights"], dtype=np.float64)
    expected = normalized_class_weights(log_weights, labels, classes)
    if not np.allclose(weights, expected, rtol=0.0, atol=1e-15):
        raise ValueError("Weighted student weights are inconsistent.")
    if payload_hash(
        {
            "selected_candidates": candidates,
            "selected_candidate_indices": student["selected_candidate_indices"],
        }
    ) != student["parent_component_hash"]:
        raise ValueError("Weighted student parent components were modified.")
    if (
        not np.isfinite(student["global_temperature"])
        or float(student["global_temperature"]) <= 0.0
    ):
        raise ValueError("Weighted student global temperature is invalid.")


def weighted_student_parameter_count(student: Mapping[str, Any]) -> int:
    validate_weighted_student(student)
    from src.subspace_primitive import SubspacePrimitive

    candidate_parameters = sum(
        SubspacePrimitive.from_dict(item["payload"]).parameter_count
        for item in student["selected_candidates"]
    )
    return int(candidate_parameters + len(student["component_weights"]))


def readout_collapse_summary(
    student: Mapping[str, Any],
    *,
    threshold: float = 0.9,
) -> dict[str, Any]:
    validate_weighted_student(student)
    if not 0.0 < threshold < 1.0:
        raise ValueError("Collapse threshold must lie in (0, 1).")
    labels = np.asarray(
        [
            int(item["payload"]["class_label"])
            for item in student["selected_candidates"]
        ],
        dtype=np.int64,
    )
    classes = np.asarray(student["classes"], dtype=np.int64)
    weights = np.asarray(student["component_weights"], dtype=np.float64)
    maxima = [
        float(np.max(weights[labels == class_label])) for class_label in classes
    ]
    collapsed_count = sum(value > threshold for value in maxima)
    return {
        "threshold": threshold,
        "per_class_maximum_weight": maxima,
        "collapsed_class_count": collapsed_count,
        "class_count": len(classes),
        "majority_collapsed": collapsed_count > len(classes) / 2,
    }


def weighted_student_logits(
    student: Mapping[str, Any],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    validate_weighted_student(student)
    from src.subspace_primitive import SubspacePrimitive

    candidates = [
        SubspacePrimitive.from_dict(item["payload"])
        for item in student["selected_candidates"]
    ]
    fields = primitive_field_matrix(
        candidates,
        features,
        primitive="subspace_r32",
        score="normalized_radial",
    )
    labels = [candidate_class_label(candidate) for candidate in candidates]
    logits = weighted_class_logits(
        fields,
        labels,
        np.asarray(student["classes"], dtype=np.int64),
        np.asarray(student["component_weights"], dtype=np.float64),
        global_temperature=float(student["global_temperature"]),
    )
    return logits, fields


def predict_weighted_student(
    student: Mapping[str, Any],
    features: np.ndarray,
    *,
    parent_representation_hash: str,
) -> tuple[np.ndarray, np.ndarray]:
    if parent_representation_hash != student["parent_representation_hash"]:
        raise ValueError("Weighted student representation hash mismatch.")
    logits, _ = weighted_student_logits(student, features)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    classes = np.asarray(student["classes"], dtype=np.int64)
    return classes[np.argmax(probabilities, axis=1)], probabilities


def weighted_local_edit_rollback_evidence(
    student: dict[str, Any],
    features: np.ndarray,
    *,
    parent_representation_hash: str,
) -> dict[str, Any]:
    baseline = copy.deepcopy(student)
    baseline_predictions, _ = predict_weighted_student(
        baseline, features, parent_representation_hash=parent_representation_hash
    )
    edited = copy.deepcopy(student)
    labels = [
        int(item["payload"]["class_label"])
        for item in edited["selected_candidates"]
    ]
    target_class = labels[0]
    class_indices = np.flatnonzero(np.asarray(labels) == target_class)
    edited_logs = np.asarray(edited["component_log_weights"], dtype=np.float64)
    edited_logs[0] += np.log(1.01)
    edited_weights = normalized_class_weights(
        edited_logs,
        labels,
        np.asarray(edited["classes"], dtype=np.int64),
    )
    edited["component_log_weights"] = edited_logs.tolist()
    edited["component_weights"] = edited_weights.tolist()
    edited_predictions, _ = predict_weighted_student(
        edited, features, parent_representation_hash=parent_representation_hash
    )
    from src.subspace_primitive import SubspacePrimitive

    candidate = SubspacePrimitive.from_dict(
        edited["selected_candidates"][0]["payload"]
    )
    affected = candidate.radial_field(features) <= 0.0
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
    rollback_predictions, _ = predict_weighted_student(
        rollback, features, parent_representation_hash=parent_representation_hash
    )
    return {
        "edit": {
            "selected_candidate_position": 0,
            "class_label": int(target_class),
            "class_component_count": int(len(class_indices)),
            "relative_logit_scale": 1.01,
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
