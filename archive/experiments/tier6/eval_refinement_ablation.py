import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from experiments.common.classification_metrics import classification_metrics
from experiments.common.model_stats import model_structure_stats
from experiments.tier4.eval_complex_classification import add_subtractive_ellipsoids
from experiments.tier6.eval_temporal_text_prediction import (
    apply_transform_pipeline,
    compute_score_scales,
    fit_adaptive_class_models,
    fit_score_calibrator,
    fit_transform_pipeline,
    predict_calibrated_labels,
    predict_char_labels,
    prepare_text_corpus,
    sample_context_pairs,
    supervised_refinement,
)


def ordered_ablation_split(
    sample_count: int,
    carve_fraction: float = 0.10,
    calibration_fraction: float = 0.15,
    validation_fraction: float = 0.15,
    gap: int = 5,
) -> dict[str, np.ndarray]:
    fractions = (carve_fraction, calibration_fraction, validation_fraction)
    if sample_count < 8 or any(fraction <= 0.0 for fraction in fractions):
        raise ValueError("Ablation splits require positive fractions and eight samples.")
    if sum(fractions) >= 1.0 or gap < 0:
        raise ValueError("Split fractions and gaps leave no geometry region.")

    sizes = [max(1, int(round(sample_count * fraction))) for fraction in fractions]
    validation_start = sample_count - sizes[2]
    calibration_end = validation_start - gap
    calibration_start = calibration_end - sizes[1]
    carve_end = calibration_start - gap
    carve_start = carve_end - sizes[0]
    geometry_end = carve_start - gap
    if geometry_end <= 0:
        raise ValueError("Split fractions and gaps leave no geometry samples.")
    return {
        "geometry": np.arange(0, geometry_end),
        "carve": np.arange(carve_start, carve_end),
        "calibration": np.arange(calibration_start, calibration_end),
        "validation": np.arange(validation_start, sample_count),
    }


def _split_hash(splits: dict[str, np.ndarray], test_count: int) -> str:
    digest = hashlib.sha256()
    for name, indices in splits.items():
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(indices, dtype=np.int64).tobytes())
    digest.update(np.arange(test_count, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _align_probabilities(
    probabilities: np.ndarray,
    source_classes: np.ndarray,
    evaluation_classes: np.ndarray,
) -> np.ndarray:
    aligned = np.zeros(
        (len(probabilities), len(evaluation_classes)), dtype=np.float64,
    )
    target_columns = {
        int(class_id): column
        for column, class_id in enumerate(evaluation_classes)
    }
    for source_column, class_id in enumerate(source_classes):
        aligned[:, target_columns[int(class_id)]] = probabilities[:, source_column]
    return aligned


def _evaluate_variant(
    name: str,
    models: dict,
    X_geometry: np.ndarray,
    y_geometry: np.ndarray,
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    use_gpu: bool,
    fit_seconds: float,
) -> dict:
    score_scales = compute_score_scales(
        models, X_geometry, alpha=alpha, use_gpu=use_gpu,
        class_labels=y_geometry,
    )
    calibration_mask = np.isin(y_calibration, np.fromiter(models, dtype=np.int32))
    _, calibration_scores = predict_char_labels(
        models, X_calibration[calibration_mask], alpha, score_scales, use_gpu,
    )
    calibrator = fit_score_calibrator(
        calibration_scores, y_calibration[calibration_mask],
    )

    split_metrics = {}
    predictions = {}
    evaluation_classes = np.unique(np.concatenate((y_validation, y_test)))
    evaluation_classes = np.union1d(evaluation_classes, calibrator.classes_)
    for split_name, features, labels in (
        ("validation", X_validation, y_validation),
        ("test", X_test, y_test),
    ):
        start = time.perf_counter()
        _, scores = predict_char_labels(
            models, features, alpha, score_scales, use_gpu,
        )
        predicted, probabilities = predict_calibrated_labels(calibrator, scores)
        inference_seconds = time.perf_counter() - start
        aligned_probabilities = _align_probabilities(
            probabilities, calibrator.classes_, evaluation_classes,
        )
        split_metrics[split_name] = classification_metrics(
            labels, aligned_probabilities, evaluation_classes,
        )
        split_metrics[split_name]["inference_seconds"] = inference_seconds
        split_metrics[split_name]["modeled_class_coverage"] = float(
            np.isin(labels, calibrator.classes_).mean()
        )
        predictions[split_name] = predicted.tolist()
    return {
        "variant": name,
        "metrics": split_metrics,
        "predictions": predictions,
        "fit_seconds": fit_seconds,
        "model_stats": model_structure_stats(models),
        "score_scales": {str(key): float(value) for key, value in score_scales.items()},
    }


def run_refinement_ablation_from_samples(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    *,
    window: int = 5,
    pca_components: int = 24,
    alpha: float = 2.0,
    consensus_threshold: float = 0.06,
    capture_threshold: float = 0.08,
    max_iterations: int | None = 300,
    nudge_iterations: int = 0,
    nudge_lr: float = 0.02,
    refinement_epochs: int = 5,
    refinement_lr: float = 0.005,
    refinement_batch_size: int = 256,
    refinement_max_batches_per_epoch: int = 8,
    carve_fraction: float = 0.10,
    calibration_fraction: float = 0.15,
    validation_fraction: float = 0.15,
    mdl_penalty_weight: float = 0.0,
    min_penalized_gain: float = 0.0,
    seed: int = 42,
    use_gpu: bool = False,
) -> dict:
    X_train_raw = np.asarray(X_train_raw, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test_raw = np.asarray(X_test_raw, dtype=np.float64)
    y_test = np.asarray(y_test)
    if len(X_train_raw) != len(y_train) or len(X_test_raw) != len(y_test):
        raise ValueError("Feature and label arrays must have matching lengths.")

    splits = ordered_ablation_split(
        len(X_train_raw), carve_fraction, calibration_fraction,
        validation_fraction, gap=window,
    )
    geometry_idx = splits["geometry"]
    pca, lda, scaler = fit_transform_pipeline(
        X_train_raw[geometry_idx], y_train[geometry_idx], pca_components, seed,
    )
    X_train = apply_transform_pipeline(X_train_raw, pca, lda, scaler)
    X_test = apply_transform_pipeline(X_test_raw, pca, lda, scaler)
    X_geometry = X_train[geometry_idx]
    y_geometry = y_train[geometry_idx]
    X_carve = X_train[splits["carve"]]
    y_carve = y_train[splits["carve"]]
    X_calibration = X_train[splits["calibration"]]
    y_calibration = y_train[splits["calibration"]]
    X_validation = X_train[splits["validation"]]
    y_validation = y_train[splits["validation"]]
    class_ids = np.unique(y_geometry)

    fit_start = time.perf_counter()
    initial_models, _ = fit_adaptive_class_models(
        X_geometry, y_geometry, class_ids,
        consensus_threshold=consensus_threshold,
        capture_threshold=capture_threshold,
        alpha=alpha,
        max_iterations=max_iterations,
        nudge_iterations=nudge_iterations,
        nudge_learning_rate=nudge_lr,
        use_gpu=use_gpu,
        seed=seed + 100_000,
    )
    initial_fit_seconds = time.perf_counter() - fit_start
    records = [_evaluate_variant(
        "R0_additive", initial_models, X_geometry, y_geometry,
        X_calibration, y_calibration, X_validation, y_validation, X_test,
        y_test, alpha, use_gpu, initial_fit_seconds,
    )]

    refined_models = copy.deepcopy(initial_models)
    refinement_history = []
    refinement_seconds = initial_fit_seconds
    for step in (1, 2):
        score_scales = compute_score_scales(
            refined_models, X_geometry, alpha=alpha, use_gpu=use_gpu,
            class_labels=y_geometry,
        )
        step_start = time.perf_counter()
        refined_models, _ = supervised_refinement(
            models=refined_models,
            train_ids=np.array([], dtype=np.int32),
            pca=None,
            lda=None,
            scaler=None,
            window=window,
            alpha=alpha,
            score_scales=score_scales,
            n_iters=1,
            n_epochs=refinement_epochs,
            learning_rate=refinement_lr,
            max_samples=len(X_geometry),
            seed=seed + step * 1_000,
            batch_size=refinement_batch_size,
            max_batches_per_epoch=refinement_max_batches_per_epoch,
            use_gpu=use_gpu,
            monitor_X=X_validation,
            monitor_y=y_validation,
            epoch_history=refinement_history,
            refinement_X=X_geometry,
            refinement_y=y_geometry,
        )
        refinement_seconds += time.perf_counter() - step_start
        records.append(_evaluate_variant(
            f"R{step}_refinement", refined_models, X_geometry, y_geometry,
            X_calibration, y_calibration, X_validation, y_validation, X_test,
            y_test, alpha, use_gpu, refinement_seconds,
        ))

    subtractive_models = copy.deepcopy(initial_models)
    carve_audit = []
    subtractive_start = time.perf_counter()
    add_subtractive_ellipsoids(
        subtractive_models, X_geometry, y_geometry, class_ids,
        capture_threshold=capture_threshold,
        alpha=alpha,
        max_iterations=max_iterations,
        use_gpu=use_gpu,
        seed=seed + 200_000,
        acceptance_X=X_carve,
        acceptance_y=y_carve,
        audit_trail=carve_audit,
        mdl_penalty_weight=mdl_penalty_weight,
        min_penalized_gain=min_penalized_gain,
    )
    records.append(_evaluate_variant(
        "S1_subtractive", subtractive_models, X_geometry, y_geometry,
        X_calibration, y_calibration, X_validation, y_validation, X_test,
        y_test, alpha, use_gpu,
        initial_fit_seconds + time.perf_counter() - subtractive_start,
    ))

    baseline_validation = records[0]["metrics"]["validation"]["accuracy"]
    baseline_test = records[0]["metrics"]["test"]["accuracy"]
    for record in records:
        validation_gain = (
            record["metrics"]["validation"]["accuracy"] - baseline_validation
        )
        test_gain = record["metrics"]["test"]["accuracy"] - baseline_test
        record["validation_gain"] = validation_gain
        record["advance_on_validation"] = validation_gain > 0.0
        record["observational_test_gain"] = test_gain
        record["observational_test_confirms"] = test_gain > 0.0

    return {
        "protocol": {
            "selection_split": "forward_validation",
            "test_policy": "observational_only",
            "fixed_samples_across_variants": True,
            "refinement_source": "geometry_only",
            "subtractive_source": "R0_additive",
            "calibrator_refit_per_variant": True,
            "split_hash": _split_hash(splits, len(X_test)),
            "split_counts": {
                name: len(indices) for name, indices in splits.items()
            } | {"test": len(X_test)},
        },
        "records": records,
        "refinement_history": refinement_history,
        "carve_audit": carve_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 6 refinement/CSG ablation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    train_ids, test_ids = prepare_text_corpus(
        dataset=config.pop("dataset", "wikitext103"),
        max_chars=config.pop("max_chars", None),
        seed=config["seed"],
    )
    max_train_samples = config.pop("max_train_samples")
    max_test_samples = config.pop("max_test_samples")
    window = config["window"]
    X_train, y_train = sample_context_pairs(
        train_ids, window=window, max_samples=max_train_samples,
        seed=config["seed"],
    )
    X_test, y_test = sample_context_pairs(
        test_ids, window=window, max_samples=max_test_samples,
        seed=config["seed"] + 99,
    )
    result = run_refinement_ablation_from_samples(
        X_train, y_train, X_test, y_test, **config,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()