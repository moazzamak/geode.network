from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from src.inference_engine import InferenceEngine
from src.sdf_engine import Expert


@dataclass(frozen=True)
class CompressionResult:
    models: dict[int, list[Expert]]
    initial_primitive_count: int
    final_primitive_count: int
    removals: int
    rollbacks: int
    prediction_agreement: float
    maximum_score_drift: float
    confirmation_prediction_agreement: float | None
    confirmation_maximum_score_drift: float | None


def _score_matrix(
    models: dict[int, list[Expert]],
    points: np.ndarray,
    class_ids: np.ndarray,
) -> np.ndarray:
    return np.column_stack([
        InferenceEngine(models[int(class_id)], alpha=2.0).get_fused_sdf(points)
        for class_id in class_ids
    ])


def compress_primitive_budget(
    models: dict[int, list[Expert]],
    calibration_points: np.ndarray,
    primitive_budget_per_class: int,
    *,
    minimum_prediction_agreement: float = 0.99,
    maximum_score_drift: float = 0.25,
    confirmation_points: np.ndarray | None = None,
) -> CompressionResult:
    if primitive_budget_per_class < 1:
        raise ValueError("primitive_budget_per_class must be positive.")
    points = np.asarray(calibration_points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("calibration_points must have shape (samples, dimensions).")
    compressed = copy.deepcopy(models)
    class_ids = np.asarray(sorted(compressed))
    baseline_scores = _score_matrix(compressed, points, class_ids)
    baseline_predictions = np.argmin(baseline_scores, axis=1)
    confirmation = None
    confirmation_baseline_scores = None
    confirmation_baseline_predictions = None
    if confirmation_points is not None:
        confirmation = np.asarray(confirmation_points, dtype=np.float64)
        if confirmation.ndim != 2:
            raise ValueError(
                "confirmation_points must have shape (samples, dimensions)."
            )
        confirmation_baseline_scores = _score_matrix(
            compressed, confirmation, class_ids,
        )
        confirmation_baseline_predictions = np.argmin(
            confirmation_baseline_scores, axis=1,
        )
    initial_count = sum(
        len(expert.ellipsoids)
        for experts in compressed.values()
        for expert in experts
    )
    removals = 0
    rollbacks = 0

    for class_index, class_id in enumerate(class_ids):
        experts = compressed[int(class_id)]
        while sum(len(expert.ellipsoids) for expert in experts) > primitive_budget_per_class:
            current_scores = InferenceEngine(
                experts, alpha=2.0,
            ).get_fused_sdf(points)
            candidates = []
            for expert_index, expert in enumerate(experts):
                positive_count = sum(
                    ellipsoid.polarity > 0 for ellipsoid in expert.ellipsoids
                )
                for primitive_index, primitive in enumerate(expert.ellipsoids):
                    if primitive.polarity > 0 and positive_count <= 1:
                        continue
                    removed = expert.ellipsoids.pop(primitive_index)
                    expert._bs_cache = None
                    candidate_scores = InferenceEngine(
                        experts, alpha=2.0,
                    ).get_fused_sdf(points)
                    contribution = float(np.max(np.abs(
                        candidate_scores - current_scores
                    )))
                    expert.ellipsoids.insert(primitive_index, removed)
                    expert._bs_cache = None
                    candidates.append((
                        contribution, expert_index, primitive_index,
                        candidate_scores,
                    ))
            if not candidates:
                break

            accepted = False
            for _, expert_index, primitive_index, candidate_scores in sorted(
                candidates, key=lambda item: item[0]
            ):
                expert = experts[expert_index]
                removed = expert.ellipsoids.pop(primitive_index)
                expert._bs_cache = None
                trial_scores = baseline_scores.copy()
                trial_scores[:, class_index] = candidate_scores
                agreement = float(np.mean(
                    np.argmin(trial_scores, axis=1) == baseline_predictions
                ))
                score_drift = float(np.max(np.abs(
                    trial_scores - baseline_scores
                )))
                if (
                    agreement >= minimum_prediction_agreement
                    and score_drift <= maximum_score_drift
                ):
                    confirmation_passed = True
                    if confirmation is not None:
                        confirmation_trial = confirmation_baseline_scores.copy()
                        confirmation_trial[:, class_index] = InferenceEngine(
                            experts, alpha=2.0,
                        ).get_fused_sdf(confirmation)
                        confirmation_agreement = float(np.mean(
                            np.argmin(confirmation_trial, axis=1)
                            == confirmation_baseline_predictions
                        ))
                        confirmation_drift = float(np.max(np.abs(
                            confirmation_trial - confirmation_baseline_scores
                        )))
                        confirmation_passed = (
                            confirmation_agreement >= minimum_prediction_agreement
                            and confirmation_drift <= maximum_score_drift
                        )
                    if confirmation_passed:
                        removals += 1
                        accepted = True
                        break
                expert.ellipsoids.insert(primitive_index, removed)
                expert._bs_cache = None
                rollbacks += 1
            if not accepted:
                break

    final_scores = _score_matrix(compressed, points, class_ids)
    confirmation_final_scores = None
    if confirmation is not None:
        confirmation_final_scores = _score_matrix(
            compressed, confirmation, class_ids,
        )
    final_count = sum(
        len(expert.ellipsoids)
        for experts in compressed.values()
        for expert in experts
    )
    return CompressionResult(
        models=compressed,
        initial_primitive_count=initial_count,
        final_primitive_count=final_count,
        removals=removals,
        rollbacks=rollbacks,
        prediction_agreement=float(np.mean(
            np.argmin(final_scores, axis=1) == baseline_predictions
        )),
        maximum_score_drift=float(np.max(np.abs(
            final_scores - baseline_scores
        ))),
        confirmation_prediction_agreement=(
            float(np.mean(
                np.argmin(confirmation_final_scores, axis=1)
                == confirmation_baseline_predictions
            ))
            if confirmation is not None else None
        ),
        confirmation_maximum_score_drift=(
            float(np.max(np.abs(
                confirmation_final_scores - confirmation_baseline_scores
            )))
            if confirmation is not None else None
        ),
    )