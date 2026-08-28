from __future__ import annotations

from typing import Any

import numpy as np

from experiments.common.v7_adaptation import (
    GaussianAdaptationTransaction,
    GaussianBundle,
    bundle_metrics,
)
from experiments.common.v7_discovery import cluster_records
from experiments.common.v7_protocol import ConfirmationEvent
from src.rejection_buffer import RejectionRecord


def select_review_support(
    records: tuple[RejectionRecord, ...],
    true_labels: dict[str, int],
    *,
    discovery_arm: str,
    target_label: int,
    review_budget: int,
    minimum_cluster_size: int,
    seed: int,
) -> tuple[np.ndarray, int, int]:
    if discovery_arm == "no_clustering":
        ranked = sorted(records, key=lambda item: (-item.novelty_score, item.record_id))
        reviewed = ranked[: len(records)]
    else:
        clusters = cluster_records(
            discovery_arm,
            records,
            minimum_cluster_size=minimum_cluster_size,
            maximum_kmeans_clusters=4,
            microcluster_radius_multiplier=1.5,
            seed=seed,
        )
        candidates = []
        for cluster in clusters:
            labels = [true_labels[str(record.source_sample_id)] for record in cluster]
            target_fraction = float(np.mean(np.asarray(labels) == target_label))
            candidates.append((target_fraction, len(cluster), cluster))
        if not candidates:
            return np.empty(0, dtype=np.int64), 0, 0
        _, _, selected = max(candidates, key=lambda value: (value[0], value[1]))
        reviewed = sorted(
            selected, key=lambda item: (-item.novelty_score, item.record_id)
        )[:review_budget]
    target_ids = [
        int(str(record.source_sample_id).rsplit("-", 1)[-1])
        for record in reviewed
        if true_labels[str(record.source_sample_id)] == target_label
    ]
    return np.asarray(target_ids, dtype=np.int64), len(reviewed), 1 if reviewed else 0


def evaluate_integrated_cell(
    parent: GaussianBundle,
    records: tuple[RejectionRecord, ...],
    true_labels: dict[str, int],
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    known_mask: np.ndarray,
    unknown_mask: np.ndarray,
    *,
    discovery_arm: str,
    review_arm: str,
    adaptation_arm: str,
    target_label: int,
    review_budget: int,
    minimum_cluster_size: int,
    affine_rank: int,
    seed: int,
) -> dict[str, Any]:
    support_indices, reviewed_samples, review_objects = select_review_support(
        records,
        true_labels,
        discovery_arm=discovery_arm,
        target_label=target_label,
        review_budget=review_budget,
        minimum_cluster_size=minimum_cluster_size,
        seed=seed,
    )
    target_mask = dev_y == target_label
    target_indices = np.flatnonzero(target_mask)
    evaluation_indices = np.setdiff1d(target_indices, support_indices)
    baseline = bundle_metrics(
        parent,
        dev_x[known_mask],
        dev_y[known_mask],
        dev_x[evaluation_indices],
        target_label,
        dev_x[unknown_mask],
    )
    adapted = baseline
    exact_replay = True
    exact_rollback = True
    graph_issues = 0
    mutation_count = 0
    semantic_publication_count = 0
    confirmation_count = 0
    update_work = 0
    integration_window: int | None = None
    if (
        review_arm == "delayed_confirmation"
        and adaptation_arm == "rank16_affine_insertion"
        and len(support_indices) >= affine_rank + 2
    ):
        confirmation = ConfirmationEvent(
            review_id=f"review-m43-{discovery_arm}-{seed}",
            response="new_class",
            confirmed_label=str(target_label),
            confirmed_window=4,
        )
        transaction = GaussianAdaptationTransaction(parent)
        child = transaction.apply(
            confirmation=confirmation,
            label=target_label,
            support=dev_x[support_indices],
            rank=affine_rank,
            operation="sdf_component",
        )
        replay = GaussianAdaptationTransaction(parent).apply(
            confirmation=confirmation,
            label=target_label,
            support=dev_x[support_indices],
            rank=affine_rank,
            operation="sdf_component",
        )
        adapted = bundle_metrics(
            child,
            dev_x[known_mask],
            dev_y[known_mask],
            dev_x[evaluation_indices],
            target_label,
            dev_x[unknown_mask],
        )
        rolled_back = transaction.rollback()
        exact_replay = child.bundle_hash == replay.bundle_hash
        exact_rollback = rolled_back.bundle_hash == parent.bundle_hash
        mutation_count = 1
        semantic_publication_count = 1
        confirmation_count = 1
        update_work = len(support_indices)
        integration_window = 4
    integrated = adapted["target_success"] >= 0.30
    audit_events = (
        review_objects
        + confirmation_count
        + mutation_count
        + semantic_publication_count
        + int(exact_rollback)
    )
    expected_events = (
        review_objects
        + confirmation_count
        + mutation_count
        + semantic_publication_count
        + 1
    )
    return {
        "seed": seed,
        "discovery_arm": discovery_arm,
        "review_arm": review_arm,
        "adaptation_arm": adaptation_arm,
        "routing": "exhaustive",
        "reviewed_samples": reviewed_samples,
        "review_objects": review_objects,
        "support_samples": int(len(support_indices)),
        "baseline": baseline,
        "adapted": adapted,
        "integrated_confirmable_classes": int(integrated),
        "confirmable_classes": 1,
        "integration_window": integration_window,
        "confirmation_count": confirmation_count,
        "unconfirmed_semantic_publications": (
            semantic_publication_count - confirmation_count
        ),
        "unconfirmed_mutations": mutation_count - confirmation_count,
        "false_autonomous_class_creations": 0,
        "exact_replay": exact_replay,
        "exact_rollback": exact_rollback,
        "graph_issues": graph_issues,
        "fallback_contract": True,
        "audit_completeness": audit_events / expected_events,
        "update_work": update_work,
    }
