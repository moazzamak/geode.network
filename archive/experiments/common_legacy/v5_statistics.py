from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from scipy.stats import t

from experiments.common.classification_metrics import accuracy, balanced_accuracy


_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
}


def paired_prediction_interval(
    y_true: np.ndarray,
    first_predictions: np.ndarray,
    second_predictions: np.ndarray,
    *,
    metric: str = "accuracy",
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    try:
        metric_function = _METRICS[metric]
    except KeyError as error:
        raise ValueError(f"Unsupported paired metric {metric!r}.") from error
    truth = np.asarray(y_true)
    first = np.asarray(first_predictions)
    second = np.asarray(second_predictions)
    if len(truth) == 0 or not (len(truth) == len(first) == len(second)):
        raise ValueError("Paired prediction arrays must have the same positive length.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one.")

    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sample = rng.integers(0, len(truth), size=len(truth))
        differences[index] = (
            metric_function(truth[sample], first[sample])
            - metric_function(truth[sample], second[sample])
        )
    tail = (1.0 - confidence) / 2.0
    return {
        "metric": metric,
        "difference": metric_function(truth, first) - metric_function(truth, second),
        "lower": float(np.quantile(differences, tail)),
        "upper": float(np.quantile(differences, 1.0 - tail)),
        "confidence": float(confidence),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def paired_seed_t_interval(
    first_values: np.ndarray,
    second_values: np.ndarray,
    *,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    first = np.asarray(first_values, dtype=np.float64)
    second = np.asarray(second_values, dtype=np.float64)
    if first.ndim != 1 or second.shape != first.shape or len(first) < 2:
        raise ValueError("Paired seed values must be equal-length vectors of size >= 2.")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Paired seed values must be finite.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one.")
    differences = first - second
    mean = float(np.mean(differences))
    standard_error = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
    half_width = float(
        t.ppf(0.5 + confidence / 2.0, len(differences) - 1) * standard_error
    )
    return {
        "difference": mean,
        "lower": mean - half_width,
        "upper": mean + half_width,
        "confidence": float(confidence),
        "seed_count": len(differences),
    }


def pareto_dominates(
    first: Mapping[str, float],
    second: Mapping[str, float],
    directions: Mapping[str, str],
) -> bool:
    if not directions or set(first) != set(directions) or set(second) != set(directions):
        raise ValueError("Both points must contain exactly the declared Pareto axes.")
    no_worse = True
    strictly_better = False
    for axis, direction in directions.items():
        first_value = float(first[axis])
        second_value = float(second[axis])
        if not np.isfinite(first_value) or not np.isfinite(second_value):
            raise ValueError("Pareto values must be finite.")
        if direction == "higher":
            no_worse &= first_value >= second_value
            strictly_better |= first_value > second_value
        elif direction == "lower":
            no_worse &= first_value <= second_value
            strictly_better |= first_value < second_value
        else:
            raise ValueError(f"Unsupported Pareto direction {direction!r}.")
    return bool(no_worse and strictly_better)
