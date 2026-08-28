"""Parent-scoped local residual masks and prediction fusion."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax

from experiments.common.v7_adaptation import GaussianBundle


def frozen_affected_region(
    parent: GaussianBundle,
    support: np.ndarray,
    evaluation: np.ndarray,
    *,
    responsibility_threshold: float,
    support_radius_multiplier: float,
) -> np.ndarray:
    support_values = np.asarray(support, dtype=np.float64)
    evaluation_values = np.asarray(evaluation, dtype=np.float64)
    if (
        support_values.ndim != 2
        or evaluation_values.ndim != 2
        or support_values.shape[1] != evaluation_values.shape[1]
        or len(support_values) < 2
    ):
        raise ValueError("local residual scope requires aligned support/evaluation matrices")
    if not 0.0 < responsibility_threshold <= 1.0 or support_radius_multiplier <= 0.0:
        raise ValueError("invalid local residual scope thresholds")
    support_norm = np.sum(support_values**2, axis=1)
    pairwise = np.sqrt(
        np.maximum(
            support_norm[:, None]
            + support_norm[None, :]
            - 2.0 * support_values @ support_values.T,
            0.0,
        )
    )
    pairwise[pairwise == 0.0] = np.inf
    radius = float(
        np.quantile(np.min(pairwise, axis=1), 0.95) * support_radius_multiplier
    )
    evaluation_norm = np.sum(evaluation_values**2, axis=1)
    distances = np.sqrt(
        np.maximum(
            evaluation_norm[:, None]
            + support_norm[None, :]
            - 2.0 * evaluation_values @ support_values.T,
            0.0,
        )
    )
    support_region = np.min(distances, axis=1) <= radius
    support_responsibility = softmax(
        parent.class_log_likelihoods(support_values), axis=1
    )
    activated_count = min(2, support_responsibility.shape[1])
    activated = np.argsort(
        np.mean(support_responsibility, axis=0), kind="stable"
    )[-activated_count:]
    evaluation_responsibility = softmax(
        parent.class_log_likelihoods(evaluation_values), axis=1
    )
    component_region = (
        np.max(evaluation_responsibility[:, activated], axis=1)
        >= responsibility_threshold
    )
    affected = support_region | component_region
    if np.all(affected) or not np.any(affected):
        raise ValueError("parent-only scope must leave affected and unaffected examples")
    return affected


def residual_predictions(
    child: GaussianBundle,
    features: np.ndarray,
    affected: np.ndarray,
    *,
    target_label: int,
    target_temperature: float | None = None,
    target_correction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    region = np.asarray(affected, dtype=bool)
    if region.shape != (len(values),):
        raise ValueError("affected mask must align with features")
    if (target_temperature is None) == (target_correction is None):
        raise ValueError("provide exactly one local residual")
    try:
        target_column = child.class_order.index(target_label)
    except ValueError as error:
        raise ValueError("target class is absent from child bundle") from error
    likelihoods = child.class_log_likelihoods(values)
    if target_temperature is not None:
        if target_temperature <= 0.0:
            raise ValueError("target temperature must be positive")
        likelihoods[region, target_column] /= target_temperature
    else:
        correction = np.asarray(target_correction, dtype=np.float64)
        if correction.shape != (len(values),) or not np.all(np.isfinite(correction)):
            raise ValueError("target correction must be a finite aligned vector")
        likelihoods[region, target_column] += correction[region]
    columns = np.argmax(likelihoods, axis=1)
    predictions = np.asarray(child.class_order, dtype=np.int64)[columns]
    novelty = -np.max(likelihoods, axis=1)
    predictions[novelty > child.threshold] = -1
    return predictions, novelty
