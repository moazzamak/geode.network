from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
from scipy.special import softmax
from scipy.stats import weibull_min
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

from experiments.common.v5_artifacts import payload_hash
from experiments.common.v61_weighted_readout import (
    fit_weighted_readout,
    weighted_class_logits,
)
from src.subspace_primitive import SubspacePrimitive, fit_subspace_primitive


def _array_hash(*arrays: np.ndarray) -> str:
    return payload_hash(
        {
            "arrays": [
                {
                    "dtype": str(np.asarray(array).dtype),
                    "shape": list(np.asarray(array).shape),
                    "bytes": np.asarray(array).tobytes().hex(),
                }
                for array in arrays
            ]
        }
    )


def _stratified_fit_calibration(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit_indices: list[np.ndarray] = []
    calibration_indices: list[np.ndarray] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        shuffled = rng.permutation(indices)
        count = max(1, int(round(calibration_fraction * len(indices))))
        calibration_indices.append(shuffled[:count])
        fit_indices.append(shuffled[count:])
    fit = np.concatenate(fit_indices)
    calibration = np.concatenate(calibration_indices)
    return features[fit], labels[fit], features[calibration], labels[calibration]


def _subsample_by_class(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        selected.append(rng.choice(indices, min(per_class, len(indices)), replace=False))
    result = np.concatenate(selected)
    return features[result], labels[result]


@dataclass(frozen=True)
class AcceptanceOutput:
    predictions: np.ndarray
    novelty: np.ndarray
    state_hash: str
    size_bytes: int


def _posterior_head(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    query: np.ndarray,
    *,
    seed: int,
) -> AcceptanceOutput:
    model = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    ).fit(fit_x, fit_y)
    probabilities = model.predict_proba(query)
    return AcceptanceOutput(
        predictions=model.classes_[np.argmax(probabilities, axis=1)],
        novelty=1.0 - np.max(probabilities, axis=1),
        state_hash=_array_hash(model.classes_, model.coef_, model.intercept_),
        size_bytes=int(model.coef_.nbytes + model.intercept_.nbytes),
    )


def _knn_head(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    query: np.ndarray,
    *,
    per_class: int,
    seed: int,
) -> AcceptanceOutput:
    support_x, support_y = _subsample_by_class(
        fit_x, fit_y, per_class=per_class, seed=seed
    )
    model = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(support_x)
    distances, indices = model.kneighbors(query)
    return AcceptanceOutput(
        predictions=support_y[indices[:, 0]],
        novelty=distances[:, 0],
        state_hash=_array_hash(support_x, support_y),
        size_bytes=int(support_x.nbytes + support_y.nbytes),
    )


def _low_rank_gaussian_head(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    query: np.ndarray,
    *,
    rank: int,
) -> AcceptanceOutput:
    classes = np.unique(fit_y)
    primitives = [
        fit_subspace_primitive(
            fit_x[fit_y == label],
            min(rank, fit_x.shape[1] - 1, np.sum(fit_y == label) - 2),
            class_label=int(label),
        )
        for label in classes
    ]
    log_likelihoods = np.column_stack(
        [primitive.log_likelihood(query) for primitive in primitives]
    )
    state = np.concatenate(
        [
            np.concatenate(
                [
                    primitive.center,
                    primitive.basis.ravel(),
                    primitive.tangent_variances,
                    [primitive.residual_variance],
                ]
            )
            for primitive in primitives
        ]
    )
    return AcceptanceOutput(
        predictions=classes[np.argmax(log_likelihoods, axis=1)],
        novelty=-np.max(log_likelihoods, axis=1),
        state_hash=_array_hash(classes, state),
        size_bytes=int(state.nbytes),
    )


def _evm_style_head(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    query: np.ndarray,
    *,
    per_class: int,
    seed: int,
) -> AcceptanceOutput:
    support_x, support_y = _subsample_by_class(
        fit_x, fit_y, per_class=per_class, seed=seed
    )
    classes = np.unique(support_y)
    scales = []
    shapes = []
    for label in classes:
        own = support_x[support_y == label]
        other = support_x[support_y != label]
        distances = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(
            other
        ).kneighbors(own, return_distance=True)[0][:, 0]
        tail = np.sort(0.5 * distances)[-max(16, len(distances) // 2) :]
        shape, _, scale = weibull_min.fit(tail, floc=0.0)
        shapes.append(float(shape))
        scales.append(float(scale))
    class_distances = []
    for label in classes:
        own = support_x[support_y == label]
        class_distances.append(
            NearestNeighbors(n_neighbors=1, algorithm="brute")
            .fit(own)
            .kneighbors(query, return_distance=True)[0][:, 0]
        )
    distances = np.column_stack(class_distances)
    inclusion = np.column_stack(
        [
            weibull_min.sf(distances[:, column], shapes[column], scale=scales[column])
            for column in range(len(classes))
        ]
    )
    state = np.asarray([value for pair in zip(shapes, scales) for value in pair])
    return AcceptanceOutput(
        predictions=classes[np.argmax(inclusion, axis=1)],
        novelty=1.0 - np.max(inclusion, axis=1),
        state_hash=_array_hash(support_x, support_y, state),
        size_bytes=int(support_x.nbytes + support_y.nbytes + state.nbytes),
    )


def _primitive_fields(
    primitives: list[SubspacePrimitive], features: np.ndarray
) -> np.ndarray:
    return np.column_stack([primitive.radial_field(features) for primitive in primitives])


def _weighted_sdf_head(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    query: np.ndarray,
    *,
    rank: int,
    components_per_class: int,
    seed: int,
) -> AcceptanceOutput:
    classes = np.unique(fit_y)
    primitives: list[SubspacePrimitive] = []
    candidate_labels: list[int] = []
    for label in classes:
        points = fit_x[fit_y == label]
        assignments = KMeans(
            n_clusters=components_per_class,
            random_state=seed + int(label),
            n_init=10,
        ).fit_predict(points)
        for component in range(components_per_class):
            support = points[assignments == component]
            primitives.append(
                fit_subspace_primitive(
                    support,
                    min(rank, support.shape[1] - 1, len(support) - 2),
                    class_label=int(label),
                )
            )
            candidate_labels.append(int(label))
    calibration_fields = _primitive_fields(primitives, calibration_x)
    readout = fit_weighted_readout(
        calibration_fields,
        candidate_labels,
        calibration_y,
        classes,
        regularization=1e-4,
        maximum_iterations=500,
        gradient_tolerance=1e-8,
        minimum_temperature=0.05,
        maximum_temperature=20.0,
    )
    component_weights = np.asarray(readout["component_weights"], dtype=np.float64)
    temperature = float(readout["global_temperature"])
    probabilities = softmax(
        weighted_class_logits(
            _primitive_fields(primitives, query),
            candidate_labels,
            classes,
            component_weights,
            global_temperature=temperature,
        ),
        axis=1,
    )
    state = np.concatenate(
        [
            np.concatenate(
                [
                    primitive.center,
                    primitive.basis.ravel(),
                    primitive.tangent_variances,
                    [primitive.residual_variance],
                ]
            )
            for primitive in primitives
        ]
        + [component_weights, np.asarray([temperature])]
    )
    return AcceptanceOutput(
        predictions=classes[np.argmax(probabilities, axis=1)],
        novelty=1.0 - np.max(probabilities, axis=1),
        state_hash=_array_hash(np.asarray(candidate_labels), state),
        size_bytes=int(state.nbytes),
    )


def _rbf_head(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    query: np.ndarray,
    *,
    seed: int,
    c_value: float,
    gamma: str,
) -> AcceptanceOutput:
    model = CalibratedClassifierCV(
        SVC(
            C=c_value,
            gamma=gamma,
            kernel="rbf",
            random_state=seed,
        ),
        method="sigmoid",
        cv=3,
        ensemble=False,
    ).fit(fit_x, fit_y)
    probabilities = model.predict_proba(query)
    estimator = model.calibrated_classifiers_[0].estimator
    state = np.concatenate(
        [
            model.classes_.astype(np.float64),
            estimator.dual_coef_.ravel(),
            estimator.intercept_,
        ]
    )
    return AcceptanceOutput(
        predictions=model.classes_[np.argmax(probabilities, axis=1)],
        novelty=1.0 - np.max(probabilities, axis=1),
        state_hash=_array_hash(state, estimator.support_vectors_),
        size_bytes=int(state.nbytes + estimator.support_vectors_.nbytes),
    )


def evaluate_acceptance_heads(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    evaluation_features: np.ndarray,
    evaluation_labels: np.ndarray,
    config: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    unknown_classes = np.asarray(config["proxy_unknown_classes"], dtype=np.int64)
    train_known = ~np.isin(train_labels, unknown_classes)
    evaluation_unknown = np.isin(evaluation_labels, unknown_classes)
    fit_x, fit_y, calibration_x, calibration_y = _stratified_fit_calibration(
        train_features[train_known],
        train_labels[train_known],
        calibration_fraction=float(config["calibration_fraction"]),
        seed=seed,
    )
    known_evaluation = ~evaluation_unknown
    rng = np.random.default_rng(seed + 39_000)
    corruption_scale = (
        np.std(fit_x, axis=0)
        * float(config["corruption_noise_scale_fraction"])
    )
    corrupted_known = evaluation_features[known_evaluation] + rng.normal(
        size=evaluation_features[known_evaluation].shape
    ) * corruption_scale
    combined_query = np.concatenate(
        [calibration_x, evaluation_features, corrupted_known]
    )
    calibration_count = len(calibration_x)

    factories: dict[str, Callable[[np.ndarray], AcceptanceOutput]] = {
        "maximum_posterior": lambda query: _posterior_head(
            fit_x, fit_y, query, seed=seed
        ),
        "knn_support": lambda query: _knn_head(
            fit_x,
            fit_y,
            query,
            per_class=int(config["knn_support_per_class"]),
            seed=seed,
        ),
        "low_rank_gaussian": lambda query: _low_rank_gaussian_head(
            fit_x, fit_y, query, rank=int(config["gaussian_rank"])
        ),
        "evm_style_weibull_margin": lambda query: _evm_style_head(
            fit_x,
            fit_y,
            query,
            per_class=int(config["evm_support_per_class"]),
            seed=seed,
        ),
        "weighted_affine_sdf": lambda query: _weighted_sdf_head(
            fit_x,
            fit_y,
            calibration_x,
            calibration_y,
            query,
            rank=int(config["sdf_rank"]),
            components_per_class=int(config["sdf_components_per_class"]),
            seed=seed,
        ),
        "rbf_svm_evidence": lambda query: _rbf_head(
            fit_x,
            fit_y,
            query,
            seed=seed,
            c_value=float(config["rbf_svm"]["C"]),
            gamma=str(config["rbf_svm"]["gamma"]),
        ),
    }

    results: dict[str, Any] = {}
    novelty_targets = evaluation_unknown.astype(np.int64)
    budget = max(
        1,
        int(round(len(evaluation_labels) * config["review_budget_per_1000"] / 1000)),
    )
    for name, factory in factories.items():
        started = perf_counter()
        output = factory(combined_query)
        elapsed = perf_counter() - started
        calibration_novelty = output.novelty[:calibration_count]
        evaluation_novelty = output.novelty[
            calibration_count : calibration_count + len(evaluation_features)
        ]
        corruption_novelty = output.novelty[-len(corrupted_known) :]
        predictions = output.predictions[
            calibration_count : calibration_count + len(evaluation_features)
        ]
        threshold = float(
            np.quantile(
                calibration_novelty,
                float(config["calibration_known_coverage_target"]),
                method="higher",
            )
        )
        accepted = evaluation_novelty <= threshold
        corruption_false_reject_rate = float(np.mean(corruption_novelty > threshold))
        known_false_reject_rate = float(np.mean(~accepted[known_evaluation]))
        selected = np.argsort(evaluation_novelty)[-budget:]
        closed_accuracy = balanced_accuracy_score(
            evaluation_labels[known_evaluation], predictions[known_evaluation]
        )
        accepted_known = known_evaluation & accepted
        accepted_accuracy = balanced_accuracy_score(
            evaluation_labels[accepted_known], predictions[accepted_known]
        )
        replay = factory(combined_query)
        replay_exact = (
            output.state_hash == replay.state_hash
            and np.array_equal(output.predictions, replay.predictions)
            and np.array_equal(output.novelty, replay.novelty)
        )
        results[name] = {
            "threshold": threshold,
            "known_coverage": float(np.mean(accepted[known_evaluation])),
            "known_extension_false_reject_rate": known_false_reject_rate,
            "corruption_false_reject_rate": corruption_false_reject_rate,
            "corruption_false_reject_increase": float(
                corruption_false_reject_rate - known_false_reject_rate
            ),
            "unknown_recall": float(np.mean(~accepted[evaluation_unknown])),
            "known_closed_balanced_accuracy": float(closed_accuracy),
            "accepted_known_balanced_accuracy": float(accepted_accuracy),
            "accepted_known_accuracy_loss": float(closed_accuracy - accepted_accuracy),
            "auroc": float(roc_auc_score(novelty_targets, evaluation_novelty)),
            "review_precision": float(np.mean(evaluation_unknown[selected])),
            "review_recall": float(np.sum(evaluation_unknown[selected]) / np.sum(evaluation_unknown)),
            "fit_and_score_seconds": float(elapsed),
            "serialized_megabytes": float(output.size_bytes / (1024 * 1024)),
            "state_hash": output.state_hash,
            "exact_replay": replay_exact,
            "incremental_update_contract": {
                "mode": (
                    "local_support"
                    if name
                    in {
                        "knn_support",
                        "evm_style_weibull_margin",
                        "weighted_affine_sdf",
                    }
                    else "deterministic_refit"
                ),
                "exact_rollback": replay_exact,
                "add_support": True,
                "remove_support": True,
                "split_merge": name
                in {
                    "knn_support",
                    "evm_style_weibull_margin",
                    "weighted_affine_sdf",
                },
                "delete_class_local_object": True,
            },
        }
    return {
        "seed": seed,
        "proxy_unknown_classes": unknown_classes.tolist(),
        "fit_count": int(len(fit_x)),
        "calibration_count": int(calibration_count),
        "evaluation_count": int(len(evaluation_labels)),
        "unknown_evaluation_count": int(np.sum(evaluation_unknown)),
        "heads": results,
    }
