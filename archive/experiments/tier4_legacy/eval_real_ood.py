import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.common.classification_metrics import classification_metrics
from experiments.common.ood_metrics import (
    conformal_prediction_sets,
    conformal_probability_threshold,
    conformal_set_metrics,
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
    _apply_transform,
    _build_transform,
    compute_raw_scores,
    compute_score_scales,
    fit_class_models,
    stratified_geometry_calibration_split,
)
from src.inference_engine import InferenceEngine


def _load_cifar10_cached_batch(
    dataset_path: Path, max_samples: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(dataset_path) as data:
        images = data["images"]
        labels = data["labels"].astype(np.int32)
        rng = np.random.default_rng(seed)
        source_indices = np.arange(len(images))
        rng.shuffle(source_indices)
        source_indices = source_indices[:max_samples]
        sampled_images = images[source_indices]
    probe = sampled_images[:1000].tobytes() + str(sampled_images.shape).encode()
    cache_key = hashlib.md5(probe).hexdigest()[:16]
    cache_path = Path(__file__).parent / ".feat_cache" / f"cnn_{cache_key}_{max_samples}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Required locked CIFAR-10 feature cache is missing: {cache_path}",
        )
    with np.load(cache_path) as cached:
        return cached["feats"], labels[source_indices], source_indices


def _metric_class_sdfs(models: dict, features: np.ndarray, alpha: float) -> np.ndarray:
    return np.column_stack([
        InferenceEngine(models[class_id], alpha=alpha).get_metric_corrected_sdf(features)
        if models[class_id] else np.full(len(features), 10.0)
        for class_id in sorted(models)
    ])


def _all_scores(
    models: dict,
    features: np.ndarray,
    raw_scores: np.ndarray,
    probabilities: np.ndarray,
    density_scorers,
    alpha: float,
) -> dict[str, np.ndarray]:
    return {
        "minimum_raw_sdf": minimum_sdf_score(raw_scores),
        "minimum_metric_sdf": minimum_sdf_score(
            _metric_class_sdfs(models, features, alpha),
        ),
        "sdf_energy": sdf_energy_score(raw_scores),
        "maximum_probability": maximum_probability_score(probabilities),
        **density_scorers.score(features),
    }


def run_feature_ood_experiment(
    id_features: np.ndarray,
    id_labels: np.ndarray,
    id_source_indices: np.ndarray,
    ood_features: np.ndarray | dict[str, np.ndarray],
    *,
    seed: int = 42,
    original_train_size: int = 50_000,
    pca_components: int = 128,
    max_iterations: int = 10,
    alpha: float = 2.0,
) -> dict:
    train_pool = np.flatnonzero(id_source_indices < original_train_size)
    id_test_idx = np.flatnonzero(id_source_indices >= original_train_size)
    geometry_validation_idx, calibration_idx = stratified_geometry_calibration_split(
        train_pool, id_labels[train_pool], calibration_fraction=0.2, seed=seed,
    )
    geometry_idx, id_validation_idx = stratified_geometry_calibration_split(
        geometry_validation_idx,
        id_labels[geometry_validation_idx],
        calibration_fraction=0.2,
        seed=seed + 1,
    )
    if not len(id_test_idx):
        raise ValueError("The selected ID batch contains no original test samples.")

    is_multi_family = isinstance(ood_features, dict)
    families = ood_features if is_multi_family else {"ood": ood_features}
    ood_validation_count = len(id_validation_idx)
    ood_test_count = len(id_test_idx)
    ood_splits = {}
    for family_position, (name, family_features) in enumerate(families.items()):
        ood_order = np.random.default_rng(seed + family_position).permutation(
            len(family_features),
        )
        if len(ood_order) < ood_validation_count + ood_test_count:
            raise ValueError(
                f"OOD family {name!r} cannot match the ID validation and test counts.",
            )
        ood_splits[name] = (
            family_features[ood_order[:ood_validation_count]],
            family_features[
                ood_order[ood_validation_count:ood_validation_count + ood_test_count]
            ],
        )

    pca, lda, scaler = _build_transform(
        id_features[geometry_idx], id_labels[geometry_idx], pca_components, seed,
    )
    X_geometry = _apply_transform(id_features[geometry_idx], pca, lda, scaler)
    X_calibration = _apply_transform(id_features[calibration_idx], pca, lda, scaler)
    X_id_validation = _apply_transform(id_features[id_validation_idx], pca, lda, scaler)
    X_id_test = _apply_transform(id_features[id_test_idx], pca, lda, scaler)
    transformed_ood = {
        name: (
            _apply_transform(validation_raw, pca, lda, scaler),
            _apply_transform(test_raw, pca, lda, scaler),
        )
        for name, (validation_raw, test_raw) in ood_splits.items()
    }
    class_ids = np.unique(id_labels[geometry_idx])

    models = fit_class_models(
        X_geometry,
        id_labels[geometry_idx],
        class_ids,
        consensus_threshold=0.1,
        capture_threshold=0.08,
        alpha=alpha,
        max_iterations=max_iterations,
        nudge_iterations=0,
        nudge_learning_rate=0.02,
        seed=seed,
    )
    scales = compute_score_scales(
        models, X_geometry, alpha, class_labels=id_labels[geometry_idx],
    )
    calibration_raw_scores = compute_raw_scores(models, X_calibration, alpha, scales)
    readout = fit_score_readout(
        "multinomial",
        calibration_raw_scores,
        id_labels[calibration_idx],
        class_ids,
        seed=seed,
    )
    density_scorers = fit_feature_ood_scorers(
        X_geometry, gmm_components=len(class_ids), knn_k=5, seed=seed,
    )

    def evaluate(features: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        raw_scores = compute_raw_scores(models, features, alpha, scales)
        probabilities = readout.predict_proba(raw_scores)
        return probabilities, _all_scores(
            models, features, raw_scores, probabilities, density_scorers, alpha,
        )

    _, id_validation_scores = evaluate(X_id_validation)
    id_test_probabilities, id_test_scores = evaluate(X_id_test)
    family_detection = {}
    for family_name, (X_ood_validation, X_ood_test) in transformed_ood.items():
        _, ood_validation_scores = evaluate(X_ood_validation)
        _, ood_test_scores = evaluate(X_ood_test)
        detection = {}
        for score_name, validation_scores in ood_validation_scores.items():
            threshold = select_ood_threshold(validation_scores)
            detection[score_name] = {
                "validation": ood_detection_metrics(
                    id_validation_scores[score_name], validation_scores,
                ),
                "test": ood_detection_metrics(
                    id_test_scores[score_name], ood_test_scores[score_name],
                ),
                "test_operating_point": ood_operating_point(
                    id_test_scores[score_name],
                    ood_test_scores[score_name],
                    threshold,
                ),
            }
        family_detection[family_name] = detection

    test_labels = id_labels[id_test_idx]
    predictions = class_ids[id_test_probabilities.argmax(axis=1)]
    conformal_threshold = conformal_probability_threshold(
        id_labels[calibration_idx],
        readout.predict_proba(calibration_raw_scores),
        class_ids,
    )
    prediction_sets = conformal_prediction_sets(
        id_test_probabilities, conformal_threshold,
    )
    return {
        "protocol": {
            "seed": seed,
            "geometry_count": len(geometry_idx),
            "calibration_count": len(calibration_idx),
            "id_validation_count": len(id_validation_idx),
            "id_test_count": len(id_test_idx),
            "ood_validation_count": (
                {name: ood_validation_count for name in families}
                if is_multi_family else ood_validation_count
            ),
            "ood_test_count": (
                {name: ood_test_count for name in families}
                if is_multi_family else ood_test_count
            ),
            "ood_test_used_for_selection": False,
        },
        "in_distribution_test": {
            "classification": classification_metrics(
                test_labels, id_test_probabilities, class_ids,
            ),
            "selective_prediction": risk_coverage_curve(
                test_labels, predictions, id_test_probabilities.max(axis=1),
            ),
            "conformal": conformal_set_metrics(
                test_labels, prediction_sets, class_ids,
            ),
        },
        "ood_detection": (
            family_detection if is_multi_family else family_detection["ood"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cached CIFAR-10/CIFAR-100 OOD study")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    id_features, id_labels, source_indices = _load_cifar10_cached_batch(
        Path(config["id_dataset_path"]), config["id_max_samples"], config["seed"],
    )
    if "ood_feature_cache_paths" in config:
        ood_features = {}
        for name, path in config["ood_feature_cache_paths"].items():
            with np.load(path) as cached:
                ood_features[name] = cached["features"]
    else:
        with np.load(config["ood_feature_cache_path"]) as cached:
            ood_features = cached["features"]
    result = run_feature_ood_experiment(
        id_features,
        id_labels,
        source_indices,
        ood_features,
        seed=config["seed"],
        pca_components=config["pca_components"],
        max_iterations=config["max_iterations"],
        alpha=config["alpha"],
    )
    result["protocol"]["id_dataset"] = "cifar10"
    result["protocol"]["ood_dataset"] = (
        list(config["ood_feature_cache_paths"])
        if "ood_feature_cache_paths" in config else "cifar100"
    )
    output_path = Path(config["artifact_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["ood_detection"], indent=2))


if __name__ == "__main__":
    main()