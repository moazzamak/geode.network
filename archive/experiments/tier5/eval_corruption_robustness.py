import argparse
import json
from pathlib import Path

import numpy as np

from experiments.common.classification_baselines import fit_classification_baselines
from experiments.common.classification_metrics import accuracy, balanced_accuracy
from experiments.common.model_stats import model_structure_stats
from experiments.common.robustness_corruptions import (
    apply_covariance_shift,
    class_conditional_label_noise,
    inject_feature_outliers,
    mask_feature_dimensions,
    symmetric_label_noise,
)
from experiments.common.score_readouts import fit_score_readout
from experiments.tier4.eval_complex_classification import (
    compute_raw_scores,
    compute_score_scales,
    fit_class_models,
)


BASELINE_NAMES = {
    "shrinkage_gaussian",
    "matched_gmm",
    "knn",
    "linear_svm",
    "histogram_gradient_boosting",
}


def generate_multiclass_problem(
    *,
    seed: int,
    dimensions: int,
    class_count: int,
    geometry_per_class: int,
    calibration_per_class: int,
    test_per_class: int,
    center_radius: float = 4.0,
    mode_offset: float = 0.9,
    noise_scale: float = 0.65,
) -> dict[str, np.ndarray]:
    """Generate fixed train/calibration/test slices from a multimodal problem."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(class_count, dimensions))
    centers *= center_radius / np.maximum(
        np.linalg.norm(centers, axis=1, keepdims=True), 1e-12,
    )
    mode_axes = rng.normal(size=(class_count, dimensions))
    mode_axes /= np.maximum(np.linalg.norm(mode_axes, axis=1, keepdims=True), 1e-12)

    def sample(count: int) -> tuple[np.ndarray, np.ndarray]:
        features = []
        labels = []
        for class_id in range(class_count):
            signs = rng.choice((-1.0, 1.0), size=(count, 1))
            noise = rng.normal(0.0, noise_scale, size=(count, dimensions))
            features.append(
                centers[class_id] + signs * mode_axes[class_id] * mode_offset + noise,
            )
            labels.extend([class_id] * count)
        return np.vstack(features), np.asarray(labels, dtype=np.int32)

    X_geometry, y_geometry = sample(geometry_per_class)
    X_calibration, y_calibration = sample(calibration_per_class)
    X_test, y_test = sample(test_per_class)
    return {
        "X_geometry": X_geometry,
        "y_geometry": y_geometry,
        "X_calibration": X_calibration,
        "y_calibration": y_calibration,
        "X_test": X_test,
        "y_test": y_test,
    }


def apply_training_corruption(
    features: np.ndarray,
    labels: np.ndarray,
    scenario: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    kind = scenario["kind"]
    if kind == "clean":
        return features.copy(), labels.copy(), {"kind": "clean"}
    if kind == "symmetric_label_noise":
        corrupted_labels, audit = symmetric_label_noise(
            labels, scenario["rate"], seed,
        )
        return features.copy(), corrupted_labels, audit
    if kind == "class_conditional_label_noise":
        corrupted_labels, audit = class_conditional_label_noise(
            labels,
            scenario["source_class"],
            scenario["target_class"],
            scenario["rate"],
            seed,
        )
        return features.copy(), corrupted_labels, audit
    if kind == "feature_outliers":
        corrupted_features, audit = inject_feature_outliers(
            features, scenario["rate"], scenario["distance"], seed,
        )
        return corrupted_features, labels.copy(), audit
    if kind == "missing_dimensions":
        corrupted_features, audit = mask_feature_dimensions(
            features, scenario["fraction"], seed,
        )
        return corrupted_features, labels.copy(), audit
    if kind == "covariance_shift":
        corrupted_features, audit = apply_covariance_shift(
            features, scenario["strength"], seed,
        )
        return corrupted_features, labels.copy(), audit
    raise ValueError(f"Unknown corruption kind: {kind}")


def evaluate_condition(
    problem: dict[str, np.ndarray],
    scenario: dict,
    *,
    seed: int,
    max_iterations: int,
) -> dict:
    X_geometry, y_geometry, audit = apply_training_corruption(
        problem["X_geometry"], problem["y_geometry"], scenario, seed + 10_000,
    )
    class_ids = np.unique(problem["y_geometry"])
    models = fit_class_models(
        X_geometry,
        y_geometry,
        class_ids,
        consensus_threshold=0.1,
        capture_threshold=0.1,
        alpha=2.0,
        max_iterations=max_iterations,
        nudge_iterations=0,
        nudge_learning_rate=0.02,
        seed=seed,
    )
    scales = compute_score_scales(
        models, X_geometry, alpha=2.0, class_labels=y_geometry,
    )
    calibration_scores = compute_raw_scores(
        models, problem["X_calibration"], 2.0, scales,
    )
    readout = fit_score_readout(
        "multinomial",
        calibration_scores,
        problem["y_calibration"],
        class_ids,
        seed=seed,
    )
    test_scores = compute_raw_scores(models, problem["X_test"], 2.0, scales)
    geode_predictions = readout.predict(test_scores)
    methods = {
        "geode": {
            "accuracy": accuracy(problem["y_test"], geode_predictions),
            "balanced_accuracy": balanced_accuracy(
                problem["y_test"], geode_predictions,
            ),
        },
    }
    components_by_class = {
        int(class_id): max(1, sum(
            ellipsoid.polarity > 0
            for expert in models.get(int(class_id), [])
            for ellipsoid in expert.ellipsoids
        ))
        for class_id in class_ids
    }
    baselines = fit_classification_baselines(
        X_geometry,
        y_geometry,
        components_by_class,
        seed=seed,
        rbf_sample_limit=0,
        include_names=BASELINE_NAMES,
    )
    for name, baseline in baselines.items():
        predictions = baseline.predict(problem["X_test"])
        methods[name] = {
            "accuracy": accuracy(problem["y_test"], predictions),
            "balanced_accuracy": balanced_accuracy(
                problem["y_test"], predictions,
            ),
        }
    return {
        "scenario": scenario,
        "audit": audit,
        "methods": methods,
        "geode_structure": model_structure_stats(models),
        "test_used_for_fitting": False,
    }


def run_benchmark(config: dict) -> dict:
    records = []
    for seed in config["seeds"]:
        problem = generate_multiclass_problem(
            seed=seed,
            dimensions=config["dimensions"],
            class_count=config["class_count"],
            geometry_per_class=config["geometry_per_class"],
            calibration_per_class=config["calibration_per_class"],
            test_per_class=config["test_per_class"],
            center_radius=config.get("center_radius", 4.0),
            mode_offset=config.get("mode_offset", 0.9),
            noise_scale=config.get("noise_scale", 0.65),
        )
        for scenario_index, scenario in enumerate(config["scenarios"]):
            record = evaluate_condition(
                problem,
                scenario,
                seed=seed + scenario_index * 1_000,
                max_iterations=config["max_iterations"],
            )
            record["seed"] = seed
            records.append(record)

    clean_by_seed = {
        record["seed"]: record
        for record in records if record["scenario"]["kind"] == "clean"
    }
    for record in records:
        clean = clean_by_seed[record["seed"]]
        for name, metrics in record["methods"].items():
            metrics["accuracy_change_from_clean"] = (
                metrics["accuracy"] - clean["methods"][name]["accuracy"]
            )
    summary = {}
    for scenario in config["scenarios"]:
        scenario_records = [
            record for record in records
            if record["scenario"]["name"] == scenario["name"]
        ]
        summary[scenario["name"]] = {}
        for method in scenario_records[0]["methods"]:
            accuracies = np.array([
                record["methods"][method]["accuracy"]
                for record in scenario_records
            ])
            changes = np.array([
                record["methods"][method]["accuracy_change_from_clean"]
                for record in scenario_records
            ])
            summary[scenario["name"]][method] = {
                "mean_accuracy": float(np.mean(accuracies)),
                "std_accuracy": float(np.std(accuracies)),
                "mean_accuracy_change_from_clean": float(np.mean(changes)),
                "std_accuracy_change_from_clean": float(np.std(changes)),
            }
    return {"config": config, "records": records, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched corruption robustness study")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_benchmark(config)
    output_path = Path(config["artifact_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Artifact: {output_path} ({len(result['records'])} records)")


if __name__ == "__main__":
    main()