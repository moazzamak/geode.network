import argparse
import json
from pathlib import Path

import numpy as np

from experiments.common.classification_metrics import classification_metrics
from experiments.common.ood_metrics import (
    ood_detection_metrics,
    ood_operating_point,
    risk_coverage_curve,
    select_ood_threshold,
)
from experiments.common.ood_scores import (
    fit_feature_ood_scorers,
    maximum_probability_score,
    minimum_sdf_score,
    sdf_energy_score,
)
from experiments.common.score_readouts import fit_score_readout
from experiments.tier4.eval_complex_classification import (
    compute_raw_scores,
    compute_score_scales,
    fit_class_models,
)
from src.inference_engine import InferenceEngine


def _make_id_split(seed: int, samples_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array([[-2.0, -0.5], [1.8, -0.8], [0.2, 2.0]])
    features = [
        rng.multivariate_normal(
            center,
            np.diag([0.35 + 0.1 * class_id, 0.25]),
            samples_per_class,
        )
        for class_id, center in enumerate(centers)
    ]
    labels = np.repeat(np.arange(len(centers)), samples_per_class)
    return np.vstack(features), labels.astype(np.int32)


def _make_radial_ood(seed: int, sample_count: int, distance: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(sample_count, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions * distance + rng.normal(0.0, 0.15, size=(sample_count, 2))


def _score_all(
    class_sdfs: np.ndarray,
    metric_class_sdfs: np.ndarray,
    probabilities: np.ndarray,
    feature_scores: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "minimum_raw_sdf": minimum_sdf_score(class_sdfs),
        "minimum_metric_sdf": minimum_sdf_score(metric_class_sdfs),
        "sdf_energy": sdf_energy_score(class_sdfs),
        "maximum_probability": maximum_probability_score(probabilities),
        **feature_scores,
    }


def _metric_class_sdfs(models: dict, features: np.ndarray, alpha: float) -> np.ndarray:
    return np.column_stack([
        InferenceEngine(models[class_id], alpha=alpha).get_metric_corrected_sdf(features)
        if models[class_id] else np.full(len(features), 10.0)
        for class_id in sorted(models)
    ])


def run_controlled_ood_experiment(
    *,
    seed: int = 42,
    distances: tuple[float, ...] = (4.0, 6.0, 8.0),
    geometry_per_class: int = 100,
    evaluation_per_class: int = 40,
    max_iterations: int = 30,
) -> dict:
    """Evaluate controlled shifts without using final OOD test data for fitting."""
    X_geometry, y_geometry = _make_id_split(seed, geometry_per_class)
    X_calibration, y_calibration = _make_id_split(seed + 1, evaluation_per_class)
    X_id_validation, _ = _make_id_split(seed + 2, evaluation_per_class)
    X_id_test, y_id_test = _make_id_split(seed + 3, evaluation_per_class)
    class_ids = np.unique(y_geometry)

    models = fit_class_models(
        X_geometry,
        y_geometry,
        class_ids,
        consensus_threshold=0.1,
        capture_threshold=0.1,
        alpha=2.0,
        max_iterations=max_iterations,
        nudge_iterations=2,
        nudge_learning_rate=0.02,
        seed=seed,
    )
    scales = compute_score_scales(
        models, X_geometry, alpha=2.0, class_labels=y_geometry,
    )
    calibration_sdfs = compute_raw_scores(models, X_calibration, 2.0, scales)
    readout = fit_score_readout(
        "multinomial", calibration_sdfs, y_calibration, class_ids, seed=seed,
    )
    density_scorers = fit_feature_ood_scorers(
        X_geometry, gmm_components=len(class_ids), knn_k=5, seed=seed,
    )

    id_validation_sdfs = compute_raw_scores(models, X_id_validation, 2.0, scales)
    id_validation_scores = _score_all(
        id_validation_sdfs,
        _metric_class_sdfs(models, X_id_validation, 2.0),
        readout.predict_proba(id_validation_sdfs),
        density_scorers.score(X_id_validation),
    )
    id_test_sdfs = compute_raw_scores(models, X_id_test, 2.0, scales)
    id_test_probabilities = readout.predict_proba(id_test_sdfs)
    id_test_scores = _score_all(
        id_test_sdfs,
        _metric_class_sdfs(models, X_id_test, 2.0),
        id_test_probabilities,
        density_scorers.score(X_id_test),
    )

    validation = {}
    test = {}
    for index, distance in enumerate(distances):
        X_ood_validation = _make_radial_ood(
            seed + 100 + index, len(X_id_validation), distance,
        )
        X_ood_test = _make_radial_ood(
            seed + 200 + index, len(X_id_test), distance,
        )
        validation_sdfs = compute_raw_scores(models, X_ood_validation, 2.0, scales)
        validation_scores = _score_all(
            validation_sdfs,
            _metric_class_sdfs(models, X_ood_validation, 2.0),
            readout.predict_proba(validation_sdfs),
            density_scorers.score(X_ood_validation),
        )
        test_sdfs = compute_raw_scores(models, X_ood_test, 2.0, scales)
        test_scores = _score_all(
            test_sdfs,
            _metric_class_sdfs(models, X_ood_test, 2.0),
            readout.predict_proba(test_sdfs),
            density_scorers.score(X_ood_test),
        )
        key = str(distance)
        validation[key] = {}
        test[key] = {}
        for name, validation_values in validation_scores.items():
            threshold = select_ood_threshold(validation_values)
            validation[key][name] = {
                "detection": ood_detection_metrics(
                    id_validation_scores[name], validation_values,
                ),
                "selected_threshold": threshold,
            }
            test[key][name] = {
                "detection": ood_detection_metrics(
                    id_test_scores[name], test_scores[name],
                ),
                "operating_point": ood_operating_point(
                    id_test_scores[name], test_scores[name], threshold,
                ),
            }

    predictions = class_ids[id_test_probabilities.argmax(axis=1)]
    return {
        "protocol": {
            "seed": seed,
            "distances": list(distances),
            "ood_validation_used_for_selection": True,
            "ood_test_used_for_selection": False,
            "first_order_metric_sdf_status": "evaluated",
        },
        "in_distribution_test": {
            "classification": classification_metrics(
                y_id_test, id_test_probabilities, class_ids,
            ),
            "selective_prediction": risk_coverage_curve(
                y_id_test, predictions, id_test_probabilities.max(axis=1),
            ),
        },
        "ood_validation": validation,
        "ood_test": test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled feature-space OOD study")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("logs/results/tier4_controlled_ood.json"))
    args = parser.parse_args()
    result = run_controlled_ood_experiment(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["ood_test"], indent=2))


if __name__ == "__main__":
    main()