from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from experiments.common.v5_artifacts import payload_hash
from experiments.common.v6_boundary_distillation import (
    SphereCandidate,
    candidate_sdf_matrix,
    probabilities_from_logits,
    teacher_margins,
)
from src.directional_primitive import (
    SphericalCapPrimitive,
    fit_spherical_cap,
    l2_normalize,
)


NORMALIZATION_POLICY = {
    "name": "l2_normalize",
    "dtype": "float64",
    "zero_vector_policy": "reject",
}


def normalized_representation_hash(parent_representation_hash: str) -> str:
    return payload_hash(
        {
            "parent_representation_hash": parent_representation_hash,
            "transform": NORMALIZATION_POLICY,
        }
    )


def require_unit_features(features: np.ndarray) -> np.ndarray:
    points = np.asarray(features, dtype=np.float64)
    if points.ndim != 2 or not np.all(np.isfinite(points)):
        raise ValueError("features must be a finite two-dimensional array.")
    if not np.allclose(
        np.linalg.norm(points, axis=1), 1.0, rtol=0.0, atol=1e-10
    ):
        raise ValueError("Directional candidates require explicitly normalized features.")
    return points


def generate_paired_directional_candidates(
    normalized_features: np.ndarray,
    labels: np.ndarray,
    teacher_probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    candidates_per_class: int,
    seed_size: int,
    anchor_fraction: float,
) -> tuple[list[SphereCandidate], list[SphericalCapPrimitive]]:
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
    if candidates_per_class < 1 or seed_size < 2:
        raise ValueError("Candidate count and seed size must be positive.")
    if not 0.0 < anchor_fraction <= 1.0:
        raise ValueError("anchor_fraction must lie in (0, 1].")

    margins = teacher_margins(probabilities)
    spheres: list[SphereCandidate] = []
    caps: list[SphericalCapPrimitive] = []
    for class_label in class_values:
        class_indices = np.flatnonzero(targets == class_label)
        if len(class_indices) < seed_size:
            raise ValueError(
                f"Class {class_label} has {len(class_indices)} samples; "
                f"{seed_size} are required."
            )
        boundary_order = np.lexsort((class_indices, margins[class_indices]))
        boundary_count = max(
            candidates_per_class,
            int(np.ceil(anchor_fraction * len(class_indices))),
        )
        boundary_indices = class_indices[
            boundary_order[: min(boundary_count, len(boundary_order))]
        ]
        positions = np.linspace(
            0,
            len(boundary_indices) - 1,
            num=candidates_per_class,
            dtype=np.int64,
        )
        class_points = points[class_indices]
        for position in positions:
            anchor_index = int(boundary_indices[position])
            angular_proxy = 1.0 - np.clip(
                class_points @ points[anchor_index], -1.0, 1.0
            )
            support_order = np.lexsort((class_indices, angular_proxy))
            support_indices = class_indices[support_order[:seed_size]]
            support = points[support_indices]
            center = support.mean(axis=0)
            centered = support - center
            radius = float(
                np.sqrt(np.sum(centered * centered) / (len(support) - 1))
            )
            radius = max(radius, 1e-8)
            spheres.append(
                SphereCandidate(
                    class_label=int(class_label),
                    center=center,
                    radius=radius,
                    anchor_index=anchor_index,
                    support_size=len(support),
                )
            )
            caps.append(
                fit_spherical_cap(
                    support,
                    class_label=int(class_label),
                    anchor_index=anchor_index,
                    support_indices=tuple(int(index) for index in support_indices),
                )
            )
    return spheres, caps


def directional_field_matrix(
    candidates: Sequence[SphereCandidate] | Sequence[SphericalCapPrimitive],
    normalized_features: np.ndarray,
    geometry: str,
) -> np.ndarray:
    points = require_unit_features(normalized_features)
    if geometry == "euclidean_sphere":
        return candidate_sdf_matrix(candidates, points)  # type: ignore[arg-type]
    if geometry == "cosine_cap":
        cap_list = list(candidates)
        if not cap_list or not all(
            isinstance(candidate, SphericalCapPrimitive) for candidate in cap_list
        ):
            raise ValueError("cosine_cap requires spherical-cap candidates.")
        return np.column_stack(
            [candidate.angular_field(points) for candidate in cap_list]
        )
    raise ValueError(f"Unsupported directional geometry: {geometry}.")


def serialize_directional_student(
    *,
    geometry: str,
    classes: np.ndarray,
    candidates: Sequence[SphereCandidate] | Sequence[SphericalCapPrimitive],
    selection: dict[str, Any],
    parent_representation_hash: str,
    directional_representation_hash: str,
    cohort_indices: np.ndarray,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    expected_hash = normalized_representation_hash(parent_representation_hash)
    if directional_representation_hash != expected_hash:
        raise ValueError("Directional representation hash does not match its contract.")
    selected_indices = selection["selected_candidate_indices"]
    selected = [candidates[index] for index in selected_indices]
    if geometry == "euclidean_sphere":
        payloads = [candidate.to_dict() for candidate in selected]
    elif geometry == "cosine_cap":
        payloads = [candidate.to_dict() for candidate in selected]
    else:
        raise ValueError(f"Unsupported directional geometry: {geometry}.")
    return {
        "schema_version": 1,
        "geometry": geometry,
        "classes": np.asarray(classes, dtype=np.int64).tolist(),
        "selected_candidates": payloads,
        "selected_candidate_indices": selected_indices,
        "component_counts": selection["component_counts"],
        "objective_trajectory": selection["objective_trajectory"],
        "parent_representation_hash": parent_representation_hash,
        "directional_representation_hash": directional_representation_hash,
        "normalization_policy": NORMALIZATION_POLICY,
        "cohort_indices": np.asarray(cohort_indices, dtype=np.int64).tolist(),
        "configuration": configuration,
    }


def predict_directional_student(
    student: dict[str, Any],
    features: np.ndarray,
    *,
    parent_representation_hash: str,
) -> tuple[np.ndarray, np.ndarray]:
    required = {
        "schema_version",
        "geometry",
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
    }
    if set(student) != required or student["schema_version"] != 1:
        raise ValueError("Unsupported directional-student schema.")
    if parent_representation_hash != student["parent_representation_hash"]:
        raise ValueError("Parent representation hash mismatch.")
    expected_hash = normalized_representation_hash(parent_representation_hash)
    if (
        student["directional_representation_hash"] != expected_hash
        or student["normalization_policy"] != NORMALIZATION_POLICY
    ):
        raise ValueError("Directional normalization lineage mismatch.")

    geometry = student["geometry"]
    if geometry == "euclidean_sphere":
        candidates = [
            SphereCandidate.from_dict(payload)
            for payload in student["selected_candidates"]
        ]
    elif geometry == "cosine_cap":
        candidates = [
            SphericalCapPrimitive.from_dict(payload)
            for payload in student["selected_candidates"]
        ]
    else:
        raise ValueError(f"Unsupported directional geometry: {geometry}.")
    normalized = l2_normalize(features)
    fields = directional_field_matrix(candidates, normalized, geometry)
    classes = np.asarray(student["classes"], dtype=np.int64)
    class_to_column = {int(value): index for index, value in enumerate(classes)}
    exp_sums = np.zeros((len(normalized), len(classes)), dtype=np.float64)
    counts = np.zeros(len(classes), dtype=np.int64)
    for index, candidate in enumerate(candidates):
        class_label = candidate.class_label
        if class_label is None or int(class_label) not in class_to_column:
            raise ValueError("Candidate class is absent from the student classes.")
        column = class_to_column[int(class_label)]
        exp_sums[:, column] += np.exp(-np.clip(fields[:, index], -500.0, 500.0))
        counts[column] += 1
    if counts.tolist() != student["component_counts"]:
        raise ValueError("Serialized component counts are inconsistent.")
    logits = np.log(np.maximum(exp_sums, np.finfo(np.float64).tiny)) - np.log(
        counts
    )[None, :]
    probabilities = probabilities_from_logits(logits)
    return classes[np.argmax(probabilities, axis=1)], probabilities
