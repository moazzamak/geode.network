"""Fast synthetic validation for Tier 6 geometry and score calibration."""

from __future__ import annotations

import numpy as np

from experiments.tier4.eval_complex_classification import (
    compute_raw_scores,
    compute_score_scales,
    fit_class_models,
)
from experiments.tier6.eval_temporal_text_prediction import (
    fit_score_calibrator,
    predict_calibrated_labels,
    probability_perplexity,
)


def _make_split(seed: int, counts: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array([[-2.0, -0.5], [1.8, -0.8], [0.2, 2.0]])
    features = []
    labels = []
    for class_id, count in enumerate(counts):
        covariance = np.diag([0.35 + 0.1 * class_id, 0.25])
        features.append(rng.multivariate_normal(centers[class_id], covariance, count))
        labels.extend([class_id] * count)
    return np.vstack(features), np.asarray(labels, dtype=np.int32)


def run_toy_benchmark() -> dict[str, float]:
    X_geometry, y_geometry = _make_split(1, (160, 80, 40))
    X_calibration, y_calibration = _make_split(2, (60, 30, 15))
    X_test, y_test = _make_split(3, (120, 60, 30))
    class_ids = np.unique(y_geometry)

    models = fit_class_models(
        X_geometry,
        y_geometry,
        class_ids,
        consensus_threshold=0.1,
        capture_threshold=0.1,
        alpha=2.0,
        max_iterations=30,
        nudge_iterations=2,
        nudge_learning_rate=0.02,
    )
    scales = compute_score_scales(
        models,
        X_geometry,
        alpha=2.0,
        class_labels=y_geometry,
    )
    calibration_scores = compute_raw_scores(models, X_calibration, 2.0, scales)
    calibrator = fit_score_calibrator(calibration_scores, y_calibration)

    test_scores = compute_raw_scores(models, X_test, 2.0, scales)
    raw_predictions = class_ids[np.argmin(test_scores, axis=1)]
    calibrated_predictions, probabilities = predict_calibrated_labels(
        calibrator, test_scores,
    )
    result = {
        "raw_accuracy": float(np.mean(raw_predictions == y_test)),
        "calibrated_accuracy": float(np.mean(calibrated_predictions == y_test)),
        "calibrated_perplexity": probability_perplexity(
            y_test, probabilities, calibrator.classes_,
        ),
    }
    print(result)
    return result


if __name__ == "__main__":
    metrics = run_toy_benchmark()
    if metrics["calibrated_accuracy"] < 0.8:
        raise SystemExit("Toy benchmark failed: calibrated accuracy is below 80%.")