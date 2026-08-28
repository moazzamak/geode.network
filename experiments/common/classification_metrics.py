from collections.abc import Callable

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    classes = np.unique(truth)
    recalls = [
        np.mean(predictions[truth == class_id] == class_id)
        for class_id in classes
    ]
    return float(np.mean(recalls))


def _class_columns(y_true: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {class_id: index for index, class_id in enumerate(classes.tolist())}
    try:
        return np.array([lookup[label] for label in y_true.tolist()], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Target class {error.args[0]!r} is absent from classes.") from error


def negative_log_likelihood(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    truth = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    columns = _class_columns(truth, np.asarray(classes))
    selected = probabilities[np.arange(len(truth)), columns]
    return float(-np.mean(np.log(np.clip(selected, epsilon, 1.0))))


def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> float:
    truth = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    columns = _class_columns(truth, np.asarray(classes))
    targets = np.zeros_like(probabilities)
    targets[np.arange(len(truth)), columns] = 1.0
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    n_bins: int = 10,
) -> float:
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    classes = np.asarray(classes)
    confidence = probabilities.max(axis=1)
    predictions = classes[probabilities.argmax(axis=1)]
    correctness = predictions == np.asarray(y_true)
    bin_indices = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        if np.any(mask):
            error += float(mask.mean()) * abs(
                float(correctness[mask].mean()) - float(confidence[mask].mean())
            )
    return error


def top_k_accuracy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    k: int = 5,
) -> float:
    probabilities = np.asarray(probabilities)
    classes = np.asarray(classes)
    if not 1 <= k <= probabilities.shape[1]:
        raise ValueError("k must be between 1 and the number of classes.")
    top_columns = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    top_classes = classes[top_columns]
    return float(np.mean(np.any(top_classes == np.asarray(y_true)[:, None], axis=1)))


def paired_bootstrap_interval(
    y_true: np.ndarray,
    first_predictions: np.ndarray,
    second_predictions: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = accuracy,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    truth = np.asarray(y_true)
    first = np.asarray(first_predictions)
    second = np.asarray(second_predictions)
    if not (len(truth) == len(first) == len(second)) or len(truth) == 0:
        raise ValueError("Paired arrays must have the same positive length.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one.")

    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sample = rng.integers(0, len(truth), size=len(truth))
        differences[index] = (
            metric(truth[sample], first[sample])
            - metric(truth[sample], second[sample])
        )
    tail = (1.0 - confidence) / 2.0
    return {
        "difference": metric(truth, first) - metric(truth, second),
        "lower": float(np.quantile(differences, tail)),
        "upper": float(np.quantile(differences, 1.0 - tail)),
        "confidence": float(confidence),
    }


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    ece_bins: int = 10,
    top_k: int = 5,
) -> dict[str, float]:
    classes = np.asarray(classes)
    predictions = classes[np.asarray(probabilities).argmax(axis=1)]
    return {
        "accuracy": accuracy(y_true, predictions),
        "balanced_accuracy": balanced_accuracy(y_true, predictions),
        "negative_log_likelihood": negative_log_likelihood(
            y_true, probabilities, classes,
        ),
        "brier_score": multiclass_brier_score(y_true, probabilities, classes),
        "expected_calibration_error": expected_calibration_error(
            y_true, probabilities, classes, n_bins=ece_bins,
        ),
        f"top_{top_k}_accuracy": top_k_accuracy(
            y_true, probabilities, classes, k=min(top_k, len(classes)),
        ),
    }


def bootstrap_metric_interval(
    y_true: np.ndarray,
    predictions: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = accuracy,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    truth = np.asarray(y_true)
    predictions = np.asarray(predictions)
    if len(truth) != len(predictions) or len(truth) == 0:
        raise ValueError("Arrays must have the same positive length.")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sample = rng.integers(0, len(truth), size=len(truth))
        estimates[index] = metric(truth[sample], predictions[sample])
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": metric(truth, predictions),
        "lower": float(np.quantile(estimates, tail)),
        "upper": float(np.quantile(estimates, 1.0 - tail)),
        "confidence": float(confidence),
    }