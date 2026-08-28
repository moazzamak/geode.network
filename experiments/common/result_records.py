from typing import Any

import numpy as np

from experiments.common.classification_metrics import (
    bootstrap_metric_interval,
    classification_metrics,
)


def classification_result_record(
    *,
    dataset: str,
    split: str,
    seed: int,
    method: str,
    representation: str,
    geometry_variant: str,
    readout: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    model_stats: dict[str, Any],
    performance: dict[str, Any],
    adequacy: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    converged: bool = True,
    bootstrap_resamples: int = 2000,
    split_hash: str | None = None,
    feature_hash: str | None = None,
) -> dict[str, Any]:
    classes = np.asarray(classes)
    predictions = classes[np.asarray(probabilities).argmax(axis=1)]
    metrics = classification_metrics(
        y_true, probabilities, classes, top_k=min(5, len(classes)),
    )
    return {
        "dataset": str(dataset),
        "split": str(split),
        "seed": int(seed),
        "method": str(method),
        "representation": str(representation),
        "geometry_variant": str(geometry_variant),
        "readout": str(readout),
        "metrics": metrics,
        "confidence_intervals": {
            "accuracy": bootstrap_metric_interval(
                y_true,
                predictions,
                n_resamples=bootstrap_resamples,
                seed=seed,
            ),
        },
        "model_stats": dict(model_stats),
        "performance": dict(performance),
        "adequacy": dict(adequacy or {}),
        "warnings": list(warnings or []),
        "converged": bool(converged),
        "sample_count": int(len(y_true)),
        "class_count": int(len(classes)),
        "split_hash": split_hash,
        "feature_hash": feature_hash,
    }