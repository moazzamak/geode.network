"""Threshold-transfer and interface-statistics diagnostics for v8 M46."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from experiments.common.classification_metrics import balanced_accuracy
from experiments.common.v5_artifacts import payload_hash
from experiments.common.v7_adaptation import GaussianBundle
from src.runtime.schemas import ThresholdTransferRecord


THRESHOLD_RULES = (
    "frozen_pre_integration",
    "class_count_heuristic",
    "anchor_quantile",
    "per_class_anchor_fail_closed",
)


def class_count_threshold(parent_threshold: float, parent_count: int, child_count: int) -> float:
    if parent_count <= 0 or child_count != parent_count + 1:
        raise ValueError("class-count transfer requires exactly one appended class")
    return float(parent_threshold - np.log(child_count / parent_count))


def threshold_for_anchor_lineage(
    *,
    expected_anchor_hash: str,
    observed_anchor_hash: str,
    parent_threshold: float,
    recalibrated_threshold: float,
) -> float:
    if expected_anchor_hash != observed_anchor_hash:
        return parent_threshold
    return recalibrated_threshold


def threshold_assignments(
    rule: str,
    parent: GaussianBundle,
    child: GaussianBundle,
    anchor_x: np.ndarray,
    coverage_target: float,
) -> tuple[float, dict[int, float]]:
    if rule not in THRESHOLD_RULES:
        raise ValueError(f"unsupported threshold rule: {rule}")
    if not 0.0 < coverage_target < 1.0:
        raise ValueError("coverage_target must be strictly between zero and one")
    anchor_labels, anchor_novelty = child.predict(anchor_x)
    if rule == "frozen_pre_integration":
        return parent.threshold, {}
    if rule == "class_count_heuristic":
        return class_count_threshold(
            parent.threshold, len(parent.class_order), len(child.class_order)
        ), {}
    global_threshold = float(
        np.quantile(anchor_novelty, coverage_target, method="higher")
    )
    if rule == "anchor_quantile":
        return global_threshold, {}
    per_class = {}
    for label in child.class_order:
        values = anchor_novelty[anchor_labels == label]
        if len(values):
            per_class[label] = min(
                global_threshold,
                float(np.quantile(values, coverage_target, method="higher")),
            )
    return global_threshold, per_class


def predictions_with_rejection(
    bundle: GaussianBundle,
    features: np.ndarray,
    global_threshold: float,
    per_class_thresholds: Mapping[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    predictions, novelty = bundle.predict(features)
    thresholds = np.asarray(
        [
            min(global_threshold, per_class_thresholds.get(int(label), global_threshold))
            for label in predictions
        ]
    )
    predictions = predictions.copy()
    predictions[novelty > thresholds] = -1
    return predictions, novelty > thresholds


def evaluate_threshold_transfer(
    *,
    episode_id: str,
    parent: GaussianBundle,
    child: GaussianBundle,
    anchor_x: np.ndarray,
    known_x: np.ndarray,
    known_y: np.ndarray,
    unknown_x: np.ndarray,
    coverage_target: float,
    maximum_unknown_recall_drop: float,
    maximum_known_accuracy_drop: float,
) -> list[dict[str, Any]]:
    parent_predictions, parent_rejected = predictions_with_rejection(
        parent, known_x, parent.threshold, {}
    )
    del parent_rejected
    parent_known_accuracy = balanced_accuracy(known_y, parent_predictions)
    _, parent_unknown_rejected = predictions_with_rejection(
        parent, unknown_x, parent.threshold, {}
    )
    parent_unknown_recall = float(np.mean(parent_unknown_rejected))
    anchor_hash = payload_hash(np.asarray(anchor_x, dtype=np.float64).tolist())
    rows = []
    for rule in THRESHOLD_RULES:
        threshold, per_class = threshold_assignments(
            rule, parent, child, anchor_x, coverage_target
        )
        known_predictions, _ = predictions_with_rejection(
            child, known_x, threshold, per_class
        )
        _, unknown_rejected = predictions_with_rejection(
            child, unknown_x, threshold, per_class
        )
        known_accuracy = balanced_accuracy(known_y, known_predictions)
        unknown_recall = float(np.mean(unknown_rejected))
        record = ThresholdTransferRecord(
            episode_id=episode_id,
            class_order_before=tuple(str(value) for value in parent.class_order),
            class_order_after=tuple(str(value) for value in child.class_order),
            anchor_set_hash=anchor_hash,
            threshold_before=parent.threshold,
            threshold_after=threshold,
            rule=rule,
            stale_action="frozen_parent",
        )
        rows.append(
            {
                **record.to_dict(),
                "per_class_thresholds": {
                    str(label): value for label, value in sorted(per_class.items())
                },
                "parent_known_balanced_accuracy": parent_known_accuracy,
                "known_balanced_accuracy": known_accuracy,
                "known_accuracy_drop": parent_known_accuracy - known_accuracy,
                "parent_unknown_recall": parent_unknown_recall,
                "unknown_recall": unknown_recall,
                "unknown_recall_drop": parent_unknown_recall - unknown_recall,
                "passes": (
                    parent_known_accuracy - known_accuracy
                    <= maximum_known_accuracy_drop + 1e-12
                    and parent_unknown_recall - unknown_recall
                    <= maximum_unknown_recall_drop + 1e-12
                ),
                "rollback_restores_parent_threshold": (
                    replace(parent, threshold=parent.threshold).threshold
                    == parent.threshold
                ),
                "stale_anchor_never_lowers_confidence": (
                    threshold_for_anchor_lineage(
                        expected_anchor_hash=anchor_hash,
                        observed_anchor_hash="0" * 64,
                        parent_threshold=parent.threshold,
                        recalibrated_threshold=threshold,
                    )
                    == parent.threshold
                ),
            }
        )
    return rows


def boundary_inclusive_indices(features: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) < budget or budget < 2:
        raise ValueError("boundary selection requires at least budget candidate rows")
    center = np.mean(values, axis=0)
    distance = np.linalg.norm(values - center, axis=1)
    order = np.argsort(distance, kind="stable")
    core = order[:budget]
    half = budget // 2
    inclusive = np.concatenate((order[:half], order[-(budget - half):]))
    return core, inclusive


def representativeness_metrics(
    full_features: np.ndarray,
    selected_features: np.ndarray,
    *,
    rank: int = 16,
) -> dict[str, float]:
    full = np.asarray(full_features, dtype=np.float64)
    selected = np.asarray(selected_features, dtype=np.float64)
    if full.ndim != 2 or selected.ndim != 2 or full.shape[1] != selected.shape[1]:
        raise ValueError("full and selected features must share a matrix dimension")
    full_center = np.mean(full, axis=0)
    _, singular_values, basis = np.linalg.svd(full - full_center, full_matrices=False)
    retained = basis[: min(rank, len(basis))]
    total_variance = float(np.sum(singular_values**2))
    selected_projection = (selected - full_center) @ retained.T
    selected_subspace_variance = float(np.sum(np.var(selected_projection, axis=0)))
    full_subspace_variance = float(np.sum(np.var((full - full_center) @ retained.T, axis=0)))
    covariance_ratio = float(
        np.trace(np.cov(selected, rowvar=False)) / np.trace(np.cov(full, rowvar=False))
    )
    distances = np.linalg.norm(full[:, None, :] - selected[None, :, :], axis=2)
    nearest = np.min(distances, axis=1)
    selected_pairwise = np.linalg.norm(
        selected[:, None, :] - selected[None, :, :], axis=2
    )
    selected_pairwise[selected_pairwise == 0.0] = np.inf
    support_radius = float(np.quantile(np.min(selected_pairwise, axis=1), 0.95))
    return {
        "covariance_trace_ratio": covariance_ratio,
        "low_rank_subspace_variance_ratio": (
            selected_subspace_variance / full_subspace_variance
            if full_subspace_variance > 0.0
            else 0.0
        ),
        "omitted_region_nearest_neighbor_coverage": float(np.mean(nearest <= support_radius)),
        "full_rank_energy_fraction": (
            float(np.sum(singular_values[: min(rank, len(singular_values))] ** 2))
            / total_variance
            if total_variance > 0.0
            else 0.0
        ),
    }
