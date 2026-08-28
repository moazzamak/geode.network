from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.special import softmax
from scipy.stats import binomtest
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v5_statistics import (
    paired_prediction_interval,
    paired_seed_t_interval,
)
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.tier4.eval_v9_m51_surface_diagnostics import _partition_seed
from src.subspace_primitive import SubspacePrimitive, fit_subspace_primitive


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v12" / "m71_gaussian_classifier.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v12" / "m71_gaussian_classifier"


def _resolve(path: str) -> Path:
    resolved = (REPO_ROOT / Path(path)).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError("M71 paths must remain inside the repository")
    return resolved


def _verify(specification: dict[str, str]) -> Path:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"M71 immutable artifact hash mismatch: {path}")
    return path


def _fit_gaussian(
    features: np.ndarray, labels: np.ndarray, *, rank: int
) -> tuple[np.ndarray, list[SubspacePrimitive]]:
    classes = np.unique(labels)
    primitives = [
        fit_subspace_primitive(
            features[labels == class_label],
            min(rank, features.shape[1] - 1, int(np.sum(labels == class_label)) - 2),
            class_label=int(class_label),
        )
        for class_label in classes
    ]
    return classes, primitives


def _gaussian_outputs(
    classes: np.ndarray,
    primitives: list[SubspacePrimitive],
    query: np.ndarray,
) -> dict[str, np.ndarray]:
    log_likelihoods = np.column_stack(
        [primitive.log_likelihood(query) for primitive in primitives]
    )
    probabilities = softmax(log_likelihoods, axis=1)
    return {
        "predictions": classes[np.argmax(log_likelihoods, axis=1)],
        "probabilities": probabilities,
        "novelty": -np.max(log_likelihoods, axis=1),
        "log_likelihoods": log_likelihoods,
    }


def _threshold(values: np.ndarray, coverage: float) -> float:
    return float(np.quantile(values, coverage, method="higher"))


def _head_metrics(
    calibration_novelty: np.ndarray,
    evaluation_predictions: np.ndarray,
    evaluation_novelty: np.ndarray,
    evaluation_labels: np.ndarray,
    *,
    known_classes: np.ndarray,
    coverage: float,
) -> dict[str, float]:
    known = np.isin(evaluation_labels, known_classes)
    threshold = _threshold(calibration_novelty, coverage)
    accepted = evaluation_novelty <= threshold
    targets = (~known).astype(np.int64)
    return {
        "known_balanced_accuracy": float(
            balanced_accuracy_score(
                evaluation_labels[known], evaluation_predictions[known]
            )
        ),
        "known_coverage": float(np.mean(accepted[known])),
        "unknown_recall": float(np.mean(~accepted[~known])),
        "unknown_count": int(np.sum(~known)),
        "unknown_rejected_count": int(np.sum(~accepted[~known])),
        "auroc": float(roc_auc_score(targets, evaluation_novelty)),
        "novelty_threshold": threshold,
    }


def _fit_controls(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    query: np.ndarray,
    *,
    seed: int,
    neighbors: int,
) -> dict[str, dict[str, np.ndarray | int]]:
    logistic = LogisticRegression(
        max_iter=1000, random_state=seed, solver="lbfgs"
    ).fit(fit_x, fit_y)
    logistic_probabilities = logistic.predict_proba(query)
    logistic_calibration_probabilities = logistic.predict_proba(calibration_x)

    rbf = SVC(
        C=1.0, gamma="scale", kernel="rbf", random_state=seed
    ).fit(fit_x, fit_y)
    support_before = rbf.support_vectors_.copy()
    calibrated_rbf = CalibratedClassifierCV(
        FrozenEstimator(rbf), method="sigmoid"
    ).fit(calibration_x, calibration_y)
    if not np.array_equal(support_before, rbf.support_vectors_):
        raise RuntimeError("M71 RBF calibration retrained the frozen estimator")
    rbf_probabilities = calibrated_rbf.predict_proba(query)
    rbf_calibration_probabilities = calibrated_rbf.predict_proba(calibration_x)

    knn = KNeighborsClassifier(
        n_neighbors=neighbors, weights="distance", algorithm="brute"
    ).fit(fit_x, fit_y)
    knn_probabilities = knn.predict_proba(query)
    knn_calibration_distances = knn.kneighbors(
        calibration_x, n_neighbors=1, return_distance=True
    )[0][:, 0]
    knn_query_distances = knn.kneighbors(
        query, n_neighbors=1, return_distance=True
    )[0][:, 0]
    return {
        "logistic": {
            "predictions": logistic.classes_[np.argmax(logistic_probabilities, axis=1)],
            "novelty": 1.0 - np.max(logistic_probabilities, axis=1),
            "calibration_novelty": 1.0
            - np.max(logistic_calibration_probabilities, axis=1),
            "size_bytes": int(logistic.coef_.nbytes + logistic.intercept_.nbytes),
        },
        "rbf": {
            "predictions": calibrated_rbf.classes_[
                np.argmax(rbf_probabilities, axis=1)
            ],
            "novelty": 1.0 - np.max(rbf_probabilities, axis=1),
            "calibration_novelty": 1.0
            - np.max(rbf_calibration_probabilities, axis=1),
            "size_bytes": int(
                rbf.support_vectors_.nbytes
                + rbf.dual_coef_.nbytes
                + rbf.intercept_.nbytes
            ),
        },
        "knn": {
            "predictions": knn.classes_[np.argmax(knn_probabilities, axis=1)],
            "novelty": knn_query_distances,
            "calibration_novelty": knn_calibration_distances,
            "size_bytes": int(fit_x.nbytes + fit_y.nbytes),
        },
    }


def _decomposition_audit(
    primitives: list[SubspacePrimitive], query: np.ndarray
) -> dict[str, float | bool]:
    residuals = []
    for primitive in primitives:
        deltas = query - primitive.center
        tangent = deltas @ primitive.basis
        tangent_terms = tangent * tangent / primitive.tangent_variances[None, :]
        residual = deltas - tangent @ primitive.basis.T
        residual_term = (
            np.sum(residual * residual, axis=1) / primitive.residual_variance
        )
        reconstructed = np.sum(tangent_terms, axis=1) + residual_term
        residuals.extend((reconstructed - primitive.quadratic_form(query)).tolist())
    absolute = np.abs(np.asarray(residuals, dtype=np.float64))
    return {
        "intrinsic_parameter_semantics": True,
        "exact_score_decomposition": bool(np.max(absolute) <= 1e-9),
        "mean_absolute_residual": float(np.mean(absolute)),
        "maximum_absolute_residual": float(np.max(absolute)),
    }


def _directional_deletion_audit(
    classes: np.ndarray,
    primitives: list[SubspacePrimitive],
    query: np.ndarray,
    *,
    seed: int,
    k_values: tuple[int, ...] = (1, 4, 8),
) -> dict[str, Any]:
    baseline = _gaussian_outputs(classes, primitives, query)
    predictions = baseline["predictions"]
    rng = np.random.default_rng(seed + 71_000)
    rows = np.arange(len(query))
    results = {}
    for k in k_values:
        top_changed = []
        random_changed = []
        bottom_changed = []
        for row, class_label in zip(rows, predictions, strict=True):
            primitive_index = int(np.flatnonzero(classes == class_label)[0])
            primitive = primitives[primitive_index]
            delta = query[row] - primitive.center
            coordinates = delta @ primitive.basis
            contributions = coordinates * coordinates / primitive.tangent_variances
            order = np.argsort(contributions)
            count = min(k, primitive.rank)
            choices = {
                "top": order[-count:],
                "bottom": order[:count],
                "random": rng.choice(primitive.rank, size=count, replace=False),
            }
            changed = {}
            for name, selected in choices.items():
                ablated = query[row].copy()
                ablated -= primitive.basis[:, selected] @ coordinates[selected]
                changed[name] = bool(
                    _gaussian_outputs(classes, primitives, ablated[None, :])[
                        "predictions"
                    ][0]
                    != class_label
                )
            top_changed.append(changed["top"])
            random_changed.append(changed["random"])
            bottom_changed.append(changed["bottom"])
        results[str(k)] = {
            "top_k_flip_rate": float(np.mean(top_changed)),
            "random_k_flip_rate": float(np.mean(random_changed)),
            "bottom_k_flip_rate": float(np.mean(bottom_changed)),
        }
    return {
        "protocol": "deletion_comprehensiveness_without_retraining",
        "k_results": results,
    }


def _nested_subset(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 71_000)
    selected = []
    for class_label in np.unique(labels):
        indices = np.flatnonzero(labels == class_label)
        if len(indices) < per_class:
            raise ValueError("M71 sample-efficiency subset exceeds class support")
        selected.extend(rng.permutation(indices)[:per_class].tolist())
    selected_array = np.asarray(sorted(selected), dtype=np.int64)
    return features[selected_array], labels[selected_array]


def _sample_efficiency(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    *,
    known_classes: np.ndarray,
    per_class_values: list[int],
    rank: int,
    seed: int,
    neighbors: int,
) -> dict[str, dict[str, float]]:
    known_train = np.isin(train_y, known_classes)
    known_dev = np.isin(dev_y, known_classes)
    fitters: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
        "logistic": lambda x, y, q: LogisticRegression(
            max_iter=1000, random_state=seed, solver="lbfgs"
        ).fit(x, y).predict(q),
        "rbf": lambda x, y, q: SVC(
            C=1.0, gamma="scale", kernel="rbf", random_state=seed
        ).fit(x, y).predict(q),
        "knn": lambda x, y, q: KNeighborsClassifier(
            n_neighbors=min(neighbors, len(x)), weights="distance", algorithm="brute"
        ).fit(x, y).predict(q),
    }
    result = {}
    for per_class in per_class_values:
        subset_x, subset_y = _nested_subset(
            train_x[known_train],
            train_y[known_train],
            per_class=per_class,
            seed=seed,
        )
        classes, primitives = _fit_gaussian(subset_x, subset_y, rank=rank)
        predictions = {
            "gaussian": _gaussian_outputs(
                classes, primitives, dev_x[known_dev]
            )["predictions"]
        }
        predictions.update(
            {
                name: fitter(subset_x, subset_y, dev_x[known_dev])
                for name, fitter in fitters.items()
            }
        )
        result[str(per_class)] = {
            name: float(balanced_accuracy_score(dev_y[known_dev], values))
            for name, values in predictions.items()
        }
    return result


def _evaluate_seed(
    seed: int, source: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    loaded = _load_seed_data(source["seed_inputs"][str(seed)])
    train_x, train_y = loaded["datasets"]["train"]
    dev_x, dev_y = loaded["datasets"]["dev"]
    known_classes = np.asarray(config["known_classes"], dtype=np.int64)
    unknown_classes = np.asarray(config["proxy_unknown_classes"], dtype=np.int64)
    partitions = _partition_seed(
        train_y,
        dev_y,
        seed=seed,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        geometry_fraction=float(config["geometry_fraction"]),
    )
    fit_x = train_x[partitions["geometry_fit"]]
    fit_y = train_y[partitions["geometry_fit"]]
    calibration_x = train_x[partitions["score_calibration"]]
    calibration_y = train_y[partitions["score_calibration"]]
    evaluation_indices = np.concatenate(
        [partitions["development_eval"], partitions["unknown_eval"]]
    )
    evaluation_x = dev_x[evaluation_indices]
    evaluation_y = dev_y[evaluation_indices]
    query = np.vstack([calibration_x, evaluation_x])

    classes, primitives = _fit_gaussian(
        fit_x, fit_y, rank=int(config["gaussian_rank"])
    )
    gaussian = _gaussian_outputs(classes, primitives, query)
    split = len(calibration_x)
    head_outputs: dict[str, dict[str, Any]] = {
        "gaussian": {
            "predictions": gaussian["predictions"][split:],
            "novelty": gaussian["novelty"][split:],
            "calibration_novelty": gaussian["novelty"][:split],
            "size_bytes": int(sum(item.array_bytes for item in primitives)),
        }
    }
    head_outputs.update(
        _fit_controls(
            fit_x,
            fit_y,
            calibration_x,
            calibration_y,
            evaluation_x,
            seed=seed,
            neighbors=int(config["knn_neighbors"]),
        )
    )
    results = {}
    prediction_arrays = {}
    for name, output in head_outputs.items():
        calibration_novelty = np.asarray(output["calibration_novelty"])
        predictions = np.asarray(output["predictions"], dtype=np.int64)
        prediction_arrays[name] = predictions
        results[name] = {
            **_head_metrics(
                calibration_novelty,
                predictions,
                np.asarray(output["novelty"], dtype=np.float64),
                evaluation_y,
                known_classes=known_classes,
                coverage=float(config["known_coverage_target"]),
            ),
            "size_bytes": int(output["size_bytes"]),
            "serialized_megabytes": float(int(output["size_bytes"]) / 1_000_000),
        }
    known_evaluation = np.isin(evaluation_y, known_classes)
    record = {
        "seed": seed,
        "partition_hashes": {
            name: payload_hash(indices.tolist()) for name, indices in partitions.items()
        },
        "heads": results,
        "sample_efficiency": _sample_efficiency(
            train_x,
            train_y,
            dev_x,
            dev_y,
            known_classes=known_classes,
            per_class_values=[int(value) for value in config["sample_efficiency_per_class"]],
            rank=int(config["gaussian_rank"]),
            seed=seed,
            neighbors=int(config["knn_neighbors"]),
        ),
        "inspectability": {
            **_decomposition_audit(primitives, evaluation_x[:256]),
            "deletion_comprehensiveness": _directional_deletion_audit(
                classes, primitives, evaluation_x[known_evaluation][:256], seed=seed
            ),
            "minimum_counterfactual_distance": {
                "closed_form_available": False,
                "validity_evaluated": False,
                "reason": (
                    "pairwise boundaries between class-specific low-rank Gaussian "
                    "quadratic forms are general quadrics; no registered closed-form "
                    "minimum Euclidean displacement exists"
                ),
            },
        },
        "exact_replay_hash": payload_hash(
            {
                "classes": classes.tolist(),
                "primitives": [item.to_dict() for item in primitives],
                "gaussian_predictions": prediction_arrays["gaussian"].tolist(),
            }
        ),
    }
    return record, evaluation_y[known_evaluation], {
        name: values[known_evaluation] for name, values in prediction_arrays.items()
    }


def run_evaluation(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _verify(config["source_config"])
    _verify(config["v11_parent_index"])
    source = json.loads(_resolve(config["source_config"]["path"]).read_text())
    seed_results = []
    truths = []
    predictions: dict[str, list[np.ndarray]] = {
        "gaussian": [], "logistic": [], "rbf": [], "knn": []
    }
    for seed in config["seeds"]:
        record, truth, seed_predictions = _evaluate_seed(
            int(seed), source, config
        )
        seed_results.append(record)
        truths.append(truth)
        for name in predictions:
            predictions[name].append(seed_predictions[name])
    pooled_truth = np.concatenate(truths)
    pooled_predictions = {
        name: np.concatenate(values) for name, values in predictions.items()
    }
    control_means = {
        name: float(
            np.mean(
                [record["heads"][name]["known_balanced_accuracy"] for record in seed_results]
            )
        )
        for name in ("logistic", "rbf", "knn")
    }
    strongest = max(control_means, key=control_means.get)
    gaussian_accuracies = np.asarray(
        [record["heads"]["gaussian"]["known_balanced_accuracy"] for record in seed_results]
    )
    strongest_accuracies = np.asarray(
        [record["heads"][strongest]["known_balanced_accuracy"] for record in seed_results]
    )
    paired = paired_prediction_interval(
        pooled_truth,
        pooled_predictions["gaussian"],
        pooled_predictions[strongest],
        metric="balanced_accuracy",
        confidence=0.95,
        n_resamples=int(config["bootstrap_resamples"]),
        seed=int(config["bootstrap_seed"]),
    )
    seed_interval = paired_seed_t_interval(
        gaussian_accuracies, strongest_accuracies, confidence=0.95
    )
    gaussian_unknown_recall = float(
        np.mean(
            [record["heads"]["gaussian"]["unknown_recall"] for record in seed_results]
        )
    )
    unknown_count = sum(
        record["heads"]["gaussian"]["unknown_count"] for record in seed_results
    )
    unknown_rejected = sum(
        record["heads"]["gaussian"]["unknown_rejected_count"]
        for record in seed_results
    )
    unknown_interval = binomtest(
        unknown_rejected, unknown_count
    ).proportion_ci(confidence_level=0.95, method="exact")
    open_set_bar = float(config["gaussian_unknown_recall_bar"])
    gaussian_size = max(
        record["heads"]["gaussian"]["serialized_megabytes"]
        for record in seed_results
    )
    gate = {
        "strongest_control": strongest,
        "l1_mean_accuracy_difference": float(
            np.mean(gaussian_accuracies - strongest_accuracies)
        ),
        "l1_paired_prediction_interval": paired,
        "l1_paired_seed_interval": seed_interval,
        "l1_accuracy_parity": bool(
            np.mean(gaussian_accuracies - strongest_accuracies)
            >= -float(config["parity_tolerance"])
        ),
        "l2_mean_unknown_recall": gaussian_unknown_recall,
        "l2_pooled_unknown_recall_interval": {
            "estimate": float(unknown_rejected / unknown_count),
            "lower": float(unknown_interval.low),
            "upper": float(unknown_interval.high),
            "confidence": 0.95,
            "method": "clopper_pearson",
            "unknown_count": int(unknown_count),
        },
        "l2_open_set_competence": bool(
            gaussian_unknown_recall >= open_set_bar
        ),
        "l2_interpretation": (
            "formal_near_miss_statistically_indistinguishable_from_bar"
            if gaussian_unknown_recall < open_set_bar
            and unknown_interval.low <= open_set_bar <= unknown_interval.high
            else (
                "passed"
                if gaussian_unknown_recall >= open_set_bar
                else "failed_below_bar"
            )
        ),
        "l4_gaussian_megabytes": float(gaussian_size),
        "l4_below_knn_size": bool(
            gaussian_size < float(config["knn_size_megabytes"])
        ),
        "i1_parameter_semantics": all(
            record["inspectability"]["intrinsic_parameter_semantics"]
            for record in seed_results
        ),
        "i2_exact_decomposition": all(
            record["inspectability"]["exact_score_decomposition"]
            for record in seed_results
        ),
        "i4_closed_form_counterfactual": all(
            record["inspectability"]["minimum_counterfactual_distance"][
                "closed_form_available"
            ]
            for record in seed_results
        ),
        "final_labels_opened": False,
    }
    evidence = {
        "schema_version": 1,
        "milestone": "M71",
        "configuration_hash": sha256_file(config_path),
        "seeds": config["seeds"],
        "seed_results": seed_results,
        "control_mean_known_balanced_accuracy": control_means,
        "gate": gate,
        "final_labels_opened": False,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = run_evaluation(arguments.config, arguments.output)
    print(json.dumps(result["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
