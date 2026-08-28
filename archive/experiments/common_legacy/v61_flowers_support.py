from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from src.directional_primitive import l2_normalize
from src.subspace_primitive import SubspacePrimitive, fit_subspace_primitive
from src.tangent_cap_primitive import TangentCapPrimitive, fit_tangent_cap


def minimum_class_support(labels: np.ndarray) -> int:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("labels must be a non-empty vector.")
    _, counts = np.unique(values, return_counts=True)
    return int(np.min(counts))


def support_tier_status(
    labels: np.ndarray,
    *,
    rank: int,
    allowed_fit_splits: Sequence[str],
) -> dict[str, Any]:
    if list(allowed_fit_splits) != ["train"]:
        raise ValueError("Support status may use the training partition only.")
    available = minimum_class_support(labels)
    required = rank + 2
    return {
        "rank": rank,
        "minimum_support_rule": "r_plus_2",
        "required_per_class": required,
        "available_per_class": available,
        "status": "feasible" if available >= required else "blocked",
    }


def fit_rank3_primitives(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    tangent: bool,
) -> list[SubspacePrimitive] | list[TangentCapPrimitive]:
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or targets.shape != (len(values),):
        raise ValueError("features and labels have incompatible shapes.")
    if minimum_class_support(targets) != 5:
        raise ValueError("A4-F5 requires exactly five training examples per class.")
    candidates = []
    for class_label in np.unique(targets):
        indices = np.flatnonzero(targets == class_label)
        if tangent:
            candidates.append(
                fit_tangent_cap(
                    l2_normalize(values[indices]),
                    3,
                    class_label=int(class_label),
                    anchor_index=int(indices[0]),
                    support_indices=tuple(int(index) for index in indices),
                )
            )
        else:
            candidates.append(
                fit_subspace_primitive(
                    values[indices],
                    3,
                    class_label=int(class_label),
                    anchor_index=int(indices[0]),
                )
            )
    return candidates


def primitive_logits(
    candidates: Sequence[SubspacePrimitive] | Sequence[TangentCapPrimitive],
    features: np.ndarray,
    *,
    tangent: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not candidates:
        raise ValueError("At least one primitive is required.")
    values = l2_normalize(features) if tangent else np.asarray(features, dtype=np.float64)
    fields = np.column_stack(
        [candidate.radial_field(values) for candidate in candidates]
    )
    labels = [candidate.class_label for candidate in candidates]
    if any(label is None for label in labels):
        raise ValueError("A4 primitives require class labels.")
    classes = np.asarray(labels, dtype=np.int64)
    if len(np.unique(classes)) != len(classes):
        raise ValueError("A4 requires exactly one labeled primitive per class.")
    return -fields, classes


def fit_global_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    *,
    minimum: float,
    maximum: float,
) -> float:
    class_to_column = {int(value): index for index, value in enumerate(classes)}
    targets = np.asarray(
        [class_to_column[int(label)] for label in labels], dtype=np.int64
    )

    def objective(log_temperature: float) -> float:
        scaled = logits / np.exp(log_temperature)
        shifted = scaled - np.max(scaled, axis=1, keepdims=True)
        log_probabilities = shifted - np.log(
            np.sum(np.exp(shifted), axis=1, keepdims=True)
        )
        return -float(
            np.mean(log_probabilities[np.arange(len(targets)), targets])
        )

    result = minimize_scalar(
        objective,
        bounds=(np.log(minimum), np.log(maximum)),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Temperature fit failed: {result.message}")
    return float(np.exp(result.x))


def predict_primitives(
    candidates: Sequence[SubspacePrimitive] | Sequence[TangentCapPrimitive],
    features: np.ndarray,
    *,
    tangent: bool,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    logits, classes = primitive_logits(candidates, features, tangent=tangent)
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return classes[np.argmax(probabilities, axis=1)], probabilities


def serialize_primitive_head(
    candidates: Sequence[SubspacePrimitive] | Sequence[TangentCapPrimitive],
    *,
    tangent: bool,
    temperature: float,
    representation_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "family": "rank3_tangent_cap" if tangent else "rank3_affine_subspace",
        "rank": 3,
        "minimum_support_rule": "r_plus_2",
        "components_per_class": 1,
        "representation_hash": representation_hash,
        "global_temperature": temperature,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }


def deserialize_primitive_head(
    payload: Mapping[str, Any],
) -> tuple[list[SubspacePrimitive] | list[TangentCapPrimitive], bool, float]:
    required = {
        "schema_version",
        "family",
        "rank",
        "minimum_support_rule",
        "components_per_class",
        "representation_hash",
        "global_temperature",
        "candidates",
    }
    if (
        set(payload) != required
        or payload["schema_version"] != 1
        or payload["rank"] != 3
        or payload["minimum_support_rule"] != "r_plus_2"
        or payload["components_per_class"] != 1
    ):
        raise ValueError("Unsupported A4 primitive-head schema.")
    tangent = payload["family"] == "rank3_tangent_cap"
    if not tangent and payload["family"] != "rank3_affine_subspace":
        raise ValueError("Unsupported A4 primitive family.")
    constructor = TangentCapPrimitive.from_dict if tangent else SubspacePrimitive.from_dict
    candidates = [constructor(item) for item in payload["candidates"]]
    return candidates, tangent, float(payload["global_temperature"])
