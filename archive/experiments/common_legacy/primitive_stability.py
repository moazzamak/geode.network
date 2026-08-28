import numpy as np
from scipy.optimize import linear_sum_assignment

from src.inference_engine import InferenceEngine


def _primitives(experts, polarity: int) -> list:
    return [
        ellipsoid
        for expert in experts
        for ellipsoid in expert.ellipsoids
        if ellipsoid.polarity == polarity
    ]


def _precision(ellipsoid) -> np.ndarray:
    return (
        ellipsoid.orientation
        @ np.diag(1.0 / np.maximum(ellipsoid.radii, 1e-12) ** 2)
        @ ellipsoid.orientation.T
    )


def match_primitives(reference: list, candidate: list) -> dict[str, list[float]]:
    """Match ellipsoids by joint center and relative precision distance."""
    if not reference or not candidate:
        return {"center_drift": [], "precision_drift": []}
    center_cost = np.array([
        [np.linalg.norm(first.center - second.center) for second in candidate]
        for first in reference
    ])
    precision_cost = np.array([
        [
            np.linalg.norm(_precision(first) - _precision(second), ord="fro")
            / max(np.linalg.norm(_precision(first), ord="fro"), 1e-12)
            for second in candidate
        ]
        for first in reference
    ])
    rows, columns = linear_sum_assignment(center_cost + precision_cost)
    return {
        "center_drift": center_cost[rows, columns].tolist(),
        "precision_drift": precision_cost[rows, columns].tolist(),
    }


def _carve_mask(experts, points: np.ndarray) -> np.ndarray | None:
    subtractive = _primitives(experts, polarity=-1)
    if not subtractive:
        return None
    return np.any(np.column_stack([
        ellipsoid.compute_sdf(points) < 0.0 for ellipsoid in subtractive
    ]), axis=1)


def evaluate_primitive_stability(
    models_by_seed: dict[int, dict],
    evaluation_points: np.ndarray,
    *,
    alpha: float = 2.0,
) -> dict:
    seeds = sorted(models_by_seed)
    if len(seeds) < 2:
        raise ValueError("Primitive stability requires at least two fitted seeds.")
    reference = models_by_seed[seeds[0]]
    class_ids = sorted(reference)
    counts = {
        str(seed): {
            str(class_id): len(_primitives(models_by_seed[seed][class_id], polarity=1))
            for class_id in class_ids
        }
        for seed in seeds
    }
    center_drift = []
    precision_drift = []
    carve_overlaps = []
    prediction_agreements = []
    reference_scores = np.column_stack([
        InferenceEngine(reference[class_id], alpha).get_fused_sdf(evaluation_points)
        for class_id in class_ids
    ])
    reference_predictions = np.argmin(reference_scores, axis=1)

    for seed in seeds[1:]:
        candidate = models_by_seed[seed]
        for class_id in class_ids:
            matched = match_primitives(
                _primitives(reference[class_id], polarity=1),
                _primitives(candidate[class_id], polarity=1),
            )
            center_drift.extend(matched["center_drift"])
            precision_drift.extend(matched["precision_drift"])
            reference_carve = _carve_mask(reference[class_id], evaluation_points)
            candidate_carve = _carve_mask(candidate[class_id], evaluation_points)
            if reference_carve is not None or candidate_carve is not None:
                first = np.zeros(len(evaluation_points), dtype=bool) if reference_carve is None else reference_carve
                second = np.zeros(len(evaluation_points), dtype=bool) if candidate_carve is None else candidate_carve
                union = np.count_nonzero(first | second)
                carve_overlaps.append(
                    float(np.count_nonzero(first & second) / union) if union else 1.0,
                )
        candidate_scores = np.column_stack([
            InferenceEngine(candidate[class_id], alpha).get_fused_sdf(evaluation_points)
            for class_id in class_ids
        ])
        prediction_agreements.append(float(np.mean(
            np.argmin(candidate_scores, axis=1) == reference_predictions,
        )))

    total_counts = np.array([
        sum(class_counts.values()) for class_counts in counts.values()
    ], dtype=np.float64)
    return {
        "reference_seed": seeds[0],
        "component_counts": counts,
        "total_component_count_mean": float(np.mean(total_counts)),
        "total_component_count_variance": float(np.var(total_counts)),
        "matched_center_drift_mean": float(np.mean(center_drift)) if center_drift else None,
        "matched_precision_drift_mean": (
            float(np.mean(precision_drift)) if precision_drift else None
        ),
        "prediction_agreement_mean": float(np.mean(prediction_agreements)),
        "carve_region_overlap_mean": (
            float(np.mean(carve_overlaps)) if carve_overlaps else None
        ),
    }