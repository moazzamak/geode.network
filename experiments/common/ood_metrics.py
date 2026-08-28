import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def ood_detection_metrics(
    in_distribution_scores: np.ndarray,
    out_distribution_scores: np.ndarray,
) -> dict[str, float]:
    """Compute OOD metrics assuming larger scores indicate more OOD-like input."""
    in_scores = np.asarray(in_distribution_scores, dtype=np.float64)
    out_scores = np.asarray(out_distribution_scores, dtype=np.float64)
    if in_scores.ndim != 1 or out_scores.ndim != 1 or not len(in_scores) or not len(out_scores):
        raise ValueError("In- and out-distribution scores must be non-empty vectors.")
    labels_out = np.concatenate([
        np.zeros(len(in_scores), dtype=np.int64),
        np.ones(len(out_scores), dtype=np.int64),
    ])
    scores = np.concatenate([in_scores, out_scores])
    false_positive_rate, true_positive_rate, _ = roc_curve(labels_out, scores)
    eligible = np.flatnonzero(true_positive_rate >= 0.95)
    fpr95 = float(false_positive_rate[eligible[0]]) if len(eligible) else 1.0
    labels_in = 1 - labels_out
    return {
        "auroc": float(roc_auc_score(labels_out, scores)),
        "aupr_out": float(average_precision_score(labels_out, scores)),
        "aupr_in": float(average_precision_score(labels_in, -scores)),
        "fpr95": fpr95,
    }


def select_ood_threshold(
    out_distribution_validation_scores: np.ndarray,
    target_true_positive_rate: float = 0.95,
) -> float:
    """Select a score threshold from OOD validation data at a target TPR."""
    scores = np.asarray(out_distribution_validation_scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("OOD validation scores must be a non-empty vector.")
    if not 0.0 < target_true_positive_rate <= 1.0:
        raise ValueError("target_true_positive_rate must be in (0, 1].")
    return float(np.quantile(
        scores, 1.0 - target_true_positive_rate, method="lower",
    ))


def select_ood_threshold_at_known_coverage(
    in_distribution_validation_scores: np.ndarray,
    minimum_known_coverage: float = 0.9,
) -> float:
    """Select the smallest threshold retaining the requested known coverage."""
    scores = np.asarray(in_distribution_validation_scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("ID validation scores must be a non-empty vector.")
    if not 0.0 < minimum_known_coverage <= 1.0:
        raise ValueError("minimum_known_coverage must be in (0, 1].")
    rank = min(
        len(scores) - 1,
        int(np.ceil(minimum_known_coverage * len(scores))) - 1,
    )
    boundary = np.partition(scores, rank)[rank]
    return float(np.nextafter(boundary, np.inf))


def ood_operating_point(
    in_distribution_scores: np.ndarray,
    out_distribution_scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Evaluate a frozen larger-is-OOD threshold on ID and OOD test scores."""
    in_scores = np.asarray(in_distribution_scores, dtype=np.float64)
    out_scores = np.asarray(out_distribution_scores, dtype=np.float64)
    return {
        "threshold": float(threshold),
        "false_positive_rate": float(np.mean(in_scores >= threshold)),
        "true_positive_rate": float(np.mean(out_scores >= threshold)),
    }


def risk_coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, list[float]]:
    """Return selective risk after retaining examples from most to least confident."""
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=np.float64)
    if not (len(truth) == len(predictions) == len(confidence)) or not len(truth):
        raise ValueError("Selective-prediction arrays must have equal positive length.")
    order = np.argsort(-confidence, kind="stable")
    errors = (predictions[order] != truth[order]).astype(np.float64)
    retained = np.arange(1, len(truth) + 1)
    return {
        "coverage": (retained / len(truth)).tolist(),
        "risk": (np.cumsum(errors) / retained).tolist(),
        "accuracy": (1.0 - np.cumsum(errors) / retained).tolist(),
    }


def conformal_probability_threshold(
    calibration_y: np.ndarray,
    calibration_probabilities: np.ndarray,
    classes: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Fit a finite-sample split-conformal threshold from calibration labels only."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one.")
    classes = np.asarray(classes)
    lookup = {class_id: index for index, class_id in enumerate(classes.tolist())}
    try:
        columns = np.array([lookup[label] for label in np.asarray(calibration_y).tolist()])
    except KeyError as error:
        raise ValueError(f"Calibration class {error.args[0]!r} is absent.") from error
    probabilities = np.asarray(calibration_probabilities, dtype=np.float64)
    nonconformity = 1.0 - probabilities[np.arange(len(columns)), columns]
    level = min(np.ceil((len(nonconformity) + 1) * (1.0 - alpha)) / len(nonconformity), 1.0)
    return float(np.quantile(nonconformity, level, method="higher"))


def conformal_prediction_sets(
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return a boolean class-membership matrix using a calibrated threshold."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (samples, classes).")
    return (1.0 - probabilities) <= threshold


def conformal_set_metrics(
    y_true: np.ndarray,
    prediction_sets: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    classes = np.asarray(classes)
    lookup = {class_id: index for index, class_id in enumerate(classes.tolist())}
    columns = np.array([lookup[label] for label in np.asarray(y_true).tolist()])
    sets = np.asarray(prediction_sets, dtype=bool)
    covered = sets[np.arange(len(columns)), columns]
    return {
        "coverage": float(np.mean(covered)),
        "average_set_size": float(np.mean(np.sum(sets, axis=1))),
    }