from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from experiments.tier4.eval_real_feature_ood_transfer import run_real_feature_ood_episode
from src.discovery_clustering import (
    dbscan_rejections,
    estimated_kmeans_rejections,
    finch_rejections,
    hdbscan_rejections,
)
from src.discovery_policy import ClusterProposalPolicy, evaluate_cluster_proposal
from src.feedback_constraints import (
    ConstraintMetric,
    PairwiseConstraint,
    build_pairwise_constraints,
    fit_diagonal_constraint_metric,
    refine_rejection_partition,
)
from src.open_set import OpenSetPrediction, OpenSetReason, UNKNOWN_LABEL
from src.rejection_buffer import RejectionBuffer
from src.streaming_discovery import (
    StreamingClusterPolicy,
    StreamingRejectionMemory,
)


def _flag_prediction(score: float) -> OpenSetPrediction:
    return OpenSetPrediction(
        label=UNKNOWN_LABEL,
        accepted=False,
        candidate_model_signature="event-review-model",
        candidate_class_id=None,
        raw_novelty_score=score,
        calibrated_novelty_score=score,
        threshold=score,
        decision_margin=0.0,
        support_profile_version="event-review-v1",
        reason_code=OpenSetReason.OUTSIDE_SUPPORT,
    )


def _split_arrays(
    payload: dict, split: str, score_index: int, embedding_space: str,
) -> tuple[np.ndarray, ...]:
    scores = np.asarray(payload[split], dtype=np.float64)[:, score_index]
    if embedding_space == "geode":
        embeddings = np.asarray(payload[f"{split}_embeddings"], dtype=np.float64)
    elif embedding_space == "representation_l2":
        embeddings = np.asarray(
            payload[f"{split}_representation_embeddings"], dtype=np.float64,
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, np.finfo(float).eps)
    else:
        raise ValueError(f"Unsupported embedding space: {embedding_space}")
    labels = np.asarray(payload[f"{split}_labels"], dtype=np.int64)
    predicted = np.asarray(payload[f"{split}_predicted_classes"], dtype=np.int64)
    return scores, embeddings, labels, predicted


def _cluster_flagged_records(
    records: tuple,
    *,
    method: str,
    dbscan_epsilon: float | None,
    minimum_support: int,
    hdbscan_minimum_cluster_size: int,
    hdbscan_minimum_samples: int | None,
    finch_hierarchy_level: int,
    estimated_kmeans_maximum_cluster_count: int,
) -> tuple:
    if method == "dbscan":
        if dbscan_epsilon is None:
            raise ValueError("DBSCAN requires an epsilon value.")
        return dbscan_rejections(
            records, epsilon=dbscan_epsilon, minimum_samples=minimum_support,
        )
    if method == "hdbscan":
        return hdbscan_rejections(
            records,
            minimum_cluster_size=hdbscan_minimum_cluster_size,
            minimum_samples=hdbscan_minimum_samples,
        )
    if method == "finch":
        return finch_rejections(records, hierarchy_level=finch_hierarchy_level)
    if method == "estimated_kmeans":
        return estimated_kmeans_rejections(
            records,
            maximum_cluster_count=estimated_kmeans_maximum_cluster_count,
        )
    raise ValueError(f"Unsupported clustering method: {method}")


def evaluate_event_review_payload(
    payload: dict,
    *,
    known_split: str,
    unknown_split: str,
    flag_fraction: float,
    embedding_space: str = "geode",
    joint_pca_components: int = 32,
    feedback_metric: ConstraintMetric | None = None,
    partition_constraints: tuple[PairwiseConstraint, ...] = (),
    windows: int = 5,
    clustering_method: str = "dbscan",
    dbscan_epsilon: float = 2.0,
    hdbscan_minimum_cluster_size: int = 5,
    hdbscan_minimum_samples: int | None = None,
    finch_hierarchy_level: int = 0,
    estimated_kmeans_maximum_cluster_count: int = 5,
    streaming_assignment_radius: float = 0.3,
    streaming_fading_rate: float = 0.1,
    streaming_minimum_weight: float = 0.5,
    streaming_promotion_weight: float = 2.0,
    streaming_minimum_known_separation: float = 0.3,
    minimum_support: int = 3,
    minimum_windows: int = 2,
    maximum_rms_radius: float = 4.0,
) -> dict:
    """Evaluate persistent review creation; labels are joined only for final metrics."""
    if not 0.0 < flag_fraction <= 1.0 or windows < 2:
        raise ValueError("flag_fraction must be in (0, 1] and windows must be >= 2.")
    score_index = payload["score_names"].index("maximum_probability")
    source_embedding_space = (
        "representation_l2"
        if embedding_space in {"joint_pca_l2", "feedback_metric_l2"}
        else embedding_space
    )
    known = _split_arrays(payload, known_split, score_index, source_embedding_space)
    unknown = _split_arrays(payload, unknown_split, score_index, source_embedding_space)
    if embedding_space == "joint_pca_l2":
        joint_embeddings = np.vstack((known[1], unknown[1]))
        component_count = min(
            joint_pca_components,
            joint_embeddings.shape[0] - 1,
            joint_embeddings.shape[1],
        )
        if component_count < 2:
            raise ValueError("joint_pca_l2 requires at least two components.")
        projected = PCA(
            n_components=component_count,
            random_state=0,
        ).fit_transform(joint_embeddings)
        projected /= np.maximum(
            np.linalg.norm(projected, axis=1, keepdims=True),
            np.finfo(float).eps,
        )
        known = (known[0], projected[:len(known[0])], known[2], known[3])
        unknown = (unknown[0], projected[len(known[0]):], unknown[2], unknown[3])
    if embedding_space == "feedback_metric_l2":
        if feedback_metric is None:
            raise ValueError("feedback_metric_l2 requires a frozen feedback metric.")
        feature_weights = np.asarray(
            feedback_metric.feature_weights, dtype=np.float64,
        )
        if feature_weights.shape != (known[1].shape[1],):
            raise ValueError("Feedback metric dimension does not match embeddings.")
        scaled = np.vstack((known[1], unknown[1])) * np.sqrt(feature_weights)
        scaled /= np.maximum(
            np.linalg.norm(scaled, axis=1, keepdims=True),
            np.finfo(float).eps,
        )
        known = (known[0], scaled[:len(known[0])], known[2], known[3])
        unknown = (unknown[0], scaled[len(known[0]):], unknown[2], unknown[3])
    all_scores = np.concatenate((known[0], unknown[0]))
    all_embeddings = np.vstack((known[1], unknown[1]))
    all_labels = np.concatenate((known[2], unknown[2]))
    all_predicted = np.concatenate((known[3], unknown[3]))
    is_unknown = np.concatenate((
        np.zeros(len(known[0]), dtype=bool),
        np.ones(len(unknown[0]), dtype=bool),
    ))

    order = np.random.default_rng(0).permutation(len(all_scores))
    window_ids = np.arange(len(order)) % windows
    sample_windows = np.empty(len(order), dtype=np.int64)
    sample_windows[order] = window_ids
    buffer = RejectionBuffer(len(order), all_embeddings.shape[1])
    policy = ClusterProposalPolicy(
        minimum_support=minimum_support,
        minimum_windows=minimum_windows,
        maximum_rms_radius=maximum_rms_radius,
        minimum_known_separation=0.0,
        review_only=True,
    )
    known_centroids = np.vstack([
        known[1][known[2] == label].mean(axis=0) for label in np.unique(known[2])
    ])
    oracle: dict[int, tuple[int, bool, int]] = {}
    reviews: list[dict] = []
    reviewed_record_ids: set[int] = set()
    duplicate_attempts = 0
    streaming_memory = None
    if clustering_method == "streaming":
        streaming_memory = StreamingRejectionMemory(StreamingClusterPolicy(
            assignment_radius=streaming_assignment_radius,
            fading_rate=streaming_fading_rate,
            minimum_weight=streaming_minimum_weight,
            promotion_weight=streaming_promotion_weight,
            minimum_windows=minimum_windows,
            minimum_known_separation=streaming_minimum_known_separation,
            max_clusters=len(order),
            max_records_per_cluster=len(order),
        ))

    def current_clusters(window_id: int) -> tuple:
        if streaming_memory is not None:
            return tuple(
                snapshot.records for snapshot in streaming_memory.snapshots(
                    window_id=window_id,
                ) if snapshot.state == "established"
            )
        clusters = _cluster_flagged_records(
            buffer.snapshot(),
            method=clustering_method,
            dbscan_epsilon=dbscan_epsilon,
            minimum_support=minimum_support,
            hdbscan_minimum_cluster_size=hdbscan_minimum_cluster_size,
            hdbscan_minimum_samples=hdbscan_minimum_samples,
            finch_hierarchy_level=finch_hierarchy_level,
            estimated_kmeans_maximum_cluster_count=(
                estimated_kmeans_maximum_cluster_count
            ),
        )
        return refine_rejection_partition(clusters, partition_constraints)

    for window_id in range(windows):
        members = order[window_ids == window_id]
        budget = max(1, math.ceil(flag_fraction * len(members)))
        flagged = members[np.argsort(all_scores[members])[-budget:]]
        new_records = []
        for sample_index in flagged:
            record = buffer.append_rejection(
                all_embeddings[sample_index],
                timestamp=float(window_id),
                window_id=window_id,
                prediction=_flag_prediction(float(all_scores[sample_index])),
                nearest_candidates=(int(all_predicted[sample_index]),),
                source_sample_id=int(sample_index),
            )
            oracle[record.record_id] = (
                int(all_labels[sample_index]), bool(is_unknown[sample_index]), window_id,
            )
            new_records.append(record)
        if streaming_memory is not None:
            streaming_memory.ingest_window(tuple(new_records), window_id=window_id)

        lifecycle_by_cluster = {
            hypothesis.cluster_id: hypothesis
            for hypothesis in streaming_memory.review_hypotheses(
                known_centroids, window_id=window_id,
            )
        } if streaming_memory is not None else {}
        for cluster in current_clusters(window_id):
            decision = evaluate_cluster_proposal(cluster, known_centroids, policy)
            if not decision.review_required:
                continue
            record_ids = {record.record_id for record in cluster}
            if record_ids & reviewed_record_ids:
                duplicate_attempts += 1
                reviewed_record_ids.update(record_ids)
                continue
            reviewed_record_ids.update(record_ids)
            reviews.append({
                "review_id": decision.review_id,
                "window_id": window_id,
                "record_ids": sorted(record_ids),
                "source_sample_ids": sorted(
                    int(record.source_sample_id) for record in cluster
                ),
                "lifecycle_relation": next((
                    hypothesis.relation
                    for hypothesis in lifecycle_by_cluster.values()
                    if {record.record_id for record in hypothesis.snapshot.records}
                    == record_ids
                ), None),
            })

    unknown_events = set(int(value) for value in unknown[2])
    reviewed_events: set[int] = set()
    useful_reviews = 0
    delays = []
    first_event_window = {
        label: int(np.min(sample_windows[len(known[0]):][unknown[2] == label]))
        for label in unknown_events
    }
    for review in reviews:
        rows = [oracle[record_id] for record_id in review["record_ids"]]
        labels = np.asarray([row[0] for row in rows])
        unknown_mask = np.asarray([row[1] for row in rows])
        if np.mean(unknown_mask) > 0.5:
            useful_reviews += 1
            event_labels = set(int(value) for value in labels[unknown_mask])
            for label in event_labels - reviewed_events:
                delays.append(review["window_id"] - first_event_window[label])
            reviewed_events.update(event_labels)

    final_partition = current_clusters(windows - 1)
    accumulated_groups = []
    for cluster in final_partition:
        if evaluate_cluster_proposal(cluster, known_centroids, policy).review_required:
            accumulated_groups.append(cluster)

    final_lifecycle = []
    if streaming_memory is not None:
        established_ids = {
            frozenset(record.record_id for record in cluster)
            for cluster in accumulated_groups
        }
        final_lifecycle = [
            hypothesis for hypothesis in streaming_memory.review_hypotheses(
                known_centroids, window_id=windows - 1,
            )
            if frozenset(
                record.record_id for record in hypothesis.snapshot.records
            ) in established_ids
        ]
    lifecycle_correct = []
    for hypothesis in final_lifecycle:
        majority_unknown = np.mean([
            oracle[record.record_id][1] for record in hypothesis.snapshot.records
        ]) > 0.5
        lifecycle_correct.append(
            (hypothesis.relation == "emerging_novel") == majority_unknown
        )

    unknown_records = [
        record for record in buffer.snapshot() if oracle[record.record_id][1]
    ]
    unknown_group_true = [oracle[record.record_id][0] for record in unknown_records]
    predicted_by_record = {record.record_id: -1 for record in unknown_records}
    accumulated_purities = []
    accumulated_events: set[int] = set()
    recovered_unknown_groups: set[int] = set()
    accumulated_unknown_group_count = 0
    for group_index, group in enumerate(accumulated_groups):
        rows = [oracle[record.record_id] for record in group]
        labels = np.asarray([row[0] for row in rows])
        unknown_mask = np.asarray([row[1] for row in rows])
        unique_labels, counts = np.unique(labels, return_counts=True)
        purity = float(np.max(counts) / len(labels))
        accumulated_purities.append(purity)
        for record in group:
            if oracle[record.record_id][1]:
                predicted_by_record[record.record_id] = group_index
        if np.mean(unknown_mask) > 0.5:
            accumulated_unknown_group_count += 1
            event_labels = set(int(value) for value in labels[unknown_mask])
            accumulated_events.update(event_labels)
            majority_label = int(unique_labels[np.argmax(counts)])
            if majority_label in unknown_events and purity >= 0.8:
                recovered_unknown_groups.add(majority_label)
    unknown_group_predicted = [
        predicted_by_record[record.record_id] for record in unknown_records
    ]

    total = len(all_scores)
    flagged_count = len(buffer)
    reviewed_flags = len(reviewed_record_ids)
    return {
        "protocol": {
            "score": "maximum_probability",
            "flag_fraction": flag_fraction,
            "windows": windows,
            "clustering_method": clustering_method,
            "embedding_space": embedding_space,
            "feature_model_version": (
                f"joint-pca-l2-{component_count}-v1"
                if embedding_space == "joint_pca_l2"
                else (
                    feedback_metric.version
                    if feedback_metric is not None
                    else "source-v1"
                )
            ),
            "transductive_representation": embedding_space == "joint_pca_l2",
            "delayed_feedback_metric": feedback_metric is not None,
            "must_link_constraint_count": (
                feedback_metric.must_link_count if feedback_metric else 0
            ),
            "cannot_link_constraint_count": (
                feedback_metric.cannot_link_count if feedback_metric else 0
            ),
            "partition_feedback_refinement": bool(partition_constraints),
            "partition_constraint_count": len(partition_constraints),
            "oracle_used_for_flagging_or_grouping": False,
            "review_only": True,
            "temporary_unknown_ids_emitted": 0,
            "mutation_published": False,
            "streaming_memory": streaming_memory is not None,
        },
        "metrics": {
            "event_recall": len(reviewed_events) / len(unknown_events),
            "accumulated_event_recall": len(accumulated_events) / len(unknown_events),
            "distinct_group_recall": len(recovered_unknown_groups) / len(unknown_events),
            "recovered_unknown_group_count": len(recovered_unknown_groups),
            "accumulated_group_count": len(accumulated_groups),
            "partition_group_count": len(final_partition),
            "accumulated_unknown_group_count": accumulated_unknown_group_count,
            "unknown_event_count": len(unknown_events),
            "unknown_group_ari": float(adjusted_rand_score(
                unknown_group_true, unknown_group_predicted,
            )) if len(set(unknown_group_true)) > 1 else 0.0,
            "useful_review_precision": useful_reviews / len(reviews) if reviews else 0.0,
            "mean_time_to_review_windows": float(np.mean(delays)) if delays else None,
            "flags_per_1000": 1000.0 * flagged_count / total,
            "reviews_per_1000": 1000.0 * len(reviews) / total,
            "duplicate_review_rate": duplicate_attempts / (duplicate_attempts + len(reviews)) if reviews or duplicate_attempts else 0.0,
            "mean_cluster_purity": float(np.mean(accumulated_purities)) if accumulated_purities else 0.0,
            "expired_flag_fraction": 1.0 - reviewed_flags / flagged_count if flagged_count else 0.0,
            "review_count": len(reviews),
            "lifecycle_relation_accuracy": (
                float(np.mean(lifecycle_correct)) if lifecycle_correct else None
            ),
            "emerging_novel_group_count": sum(
                hypothesis.relation == "emerging_novel"
                for hypothesis in final_lifecycle
            ),
            "known_extension_group_count": sum(
                hypothesis.relation == "known_extension"
                for hypothesis in final_lifecycle
            ),
        },
        "reviews": reviews,
        "partition_source_sample_ids": [
            sorted(int(record.source_sample_id) for record in group)
            for group in final_partition
        ],
    }


def run_event_review_episode(*, flag_fraction: float, **episode_config) -> dict:
    episode = run_real_feature_ood_episode(
        **episode_config, include_score_payload=True,
    )
    payload = episode["score_payload"]
    return {
        "development": evaluate_event_review_payload(
            payload,
            known_split="id_validation",
            unknown_split="proxy_unknown",
            flag_fraction=flag_fraction,
        ),
        "final": evaluate_event_review_payload(
            payload,
            known_split="id_test",
            unknown_split="final_unknown",
            flag_fraction=flag_fraction,
        ),
    }


def _fit_review_feedback_metric(
    payload: dict,
    review: dict,
    *,
    known_split: str,
    unknown_split: str,
) -> ConstraintMetric:
    score_index = payload["score_names"].index("maximum_probability")
    known = _split_arrays(payload, known_split, score_index, "representation_l2")
    unknown = _split_arrays(payload, unknown_split, score_index, "representation_l2")
    embeddings = np.vstack((known[1], unknown[1]))
    labels = np.concatenate((known[2], unknown[2]))
    reviewed_source_ids = np.asarray(sorted({
        source_id
        for reviewed_group in review["reviews"]
        for source_id in reviewed_group["source_sample_ids"]
    }), dtype=np.int64)
    constraints = build_pairwise_constraints(
        reviewed_source_ids,
        labels[reviewed_source_ids],
    )
    return fit_diagonal_constraint_metric(
        {
            int(source_id): embeddings[source_id]
            for source_id in reviewed_source_ids
        } or {0: embeddings[0]},
        constraints,
    )


def run_event_review_transfer(
    *,
    dataset_path: str,
    episodes: list[dict],
    seeds: list[int],
    flag_fractions: list[float],
    dbscan_epsilons: list[float] | None = None,
    clustering_methods: list[str] | None = None,
    hdbscan_minimum_cluster_sizes: list[int] | None = None,
    hdbscan_minimum_samples: int | None = None,
    finch_hierarchy_levels: list[int] | None = None,
    estimated_kmeans_maximum_cluster_counts: list[int] | None = None,
    streaming_assignment_radii: list[float] | None = None,
    streaming_fading_rates: list[float] | None = None,
    streaming_minimum_weight: float = 0.5,
    streaming_promotion_weight: float = 2.0,
    streaming_minimum_known_separation: float = 0.3,
    embedding_spaces: list[str] | None = None,
    joint_pca_components: int = 32,
    samples_per_slice: int = 100,
    pca_components: int = 8,
    representation: str = "mobilenetv2",
) -> dict:
    epsilon_candidates = dbscan_epsilons or [2.0]
    methods = clustering_methods or ["dbscan"]
    unsupported = set(methods) - {
        "dbscan", "hdbscan", "finch", "streaming", "estimated_kmeans",
    }
    if unsupported:
        raise ValueError(f"Unsupported clustering methods: {sorted(unsupported)}")
    hdbscan_sizes = hdbscan_minimum_cluster_sizes or [5]
    finch_levels = finch_hierarchy_levels or [0]
    estimated_kmeans_counts = estimated_kmeans_maximum_cluster_counts or [5]
    streaming_radii = streaming_assignment_radii or [0.3]
    fading_rates = streaming_fading_rates or [0.1]
    spaces = embedding_spaces or ["geode"]
    unsupported_spaces = set(spaces) - {
        "geode", "representation_l2", "joint_pca_l2", "feedback_metric_l2",
    }
    if unsupported_spaces:
        raise ValueError(f"Unsupported embedding spaces: {sorted(unsupported_spaces)}")
    clustering_candidates = []
    if "dbscan" in methods:
        clustering_candidates.extend({
            "method": "dbscan",
            "dbscan_epsilon": epsilon,
            "hdbscan_minimum_cluster_size": hdbscan_sizes[0],
            "finch_hierarchy_level": finch_levels[0],
            "estimated_kmeans_maximum_cluster_count": estimated_kmeans_counts[0],
            "streaming_assignment_radius": streaming_radii[0],
            "streaming_fading_rate": fading_rates[0],
        } for epsilon in epsilon_candidates)
    if "hdbscan" in methods:
        clustering_candidates.extend({
            "method": "hdbscan",
            "dbscan_epsilon": None,
            "hdbscan_minimum_cluster_size": size,
            "finch_hierarchy_level": finch_levels[0],
            "estimated_kmeans_maximum_cluster_count": estimated_kmeans_counts[0],
            "streaming_assignment_radius": streaming_radii[0],
            "streaming_fading_rate": fading_rates[0],
        } for size in hdbscan_sizes)
    if "finch" in methods:
        clustering_candidates.extend({
            "method": "finch",
            "dbscan_epsilon": None,
            "hdbscan_minimum_cluster_size": hdbscan_sizes[0],
            "finch_hierarchy_level": level,
            "estimated_kmeans_maximum_cluster_count": estimated_kmeans_counts[0],
            "streaming_assignment_radius": streaming_radii[0],
            "streaming_fading_rate": fading_rates[0],
        } for level in finch_levels)
    if "streaming" in methods:
        clustering_candidates.extend({
            "method": "streaming",
            "dbscan_epsilon": None,
            "hdbscan_minimum_cluster_size": hdbscan_sizes[0],
            "finch_hierarchy_level": finch_levels[0],
            "estimated_kmeans_maximum_cluster_count": estimated_kmeans_counts[0],
            "streaming_assignment_radius": radius,
            "streaming_fading_rate": fading_rate,
        } for radius in streaming_radii for fading_rate in fading_rates)
    if "estimated_kmeans" in methods:
        clustering_candidates.extend({
            "method": "estimated_kmeans",
            "dbscan_epsilon": None,
            "hdbscan_minimum_cluster_size": hdbscan_sizes[0],
            "finch_hierarchy_level": finch_levels[0],
            "estimated_kmeans_maximum_cluster_count": maximum_count,
            "streaming_assignment_radius": streaming_radii[0],
            "streaming_fading_rate": fading_rates[0],
        } for maximum_count in estimated_kmeans_counts)
    cells = []
    for seed in seeds:
        for episode_config in episodes:
            episode = run_real_feature_ood_episode(
                dataset_path=dataset_path,
                known_classes=tuple(episode_config["known_classes"]),
                proxy_unknown_classes=tuple(episode_config["proxy_unknown_classes"]),
                final_unknown_classes=tuple(episode_config["final_unknown_classes"]),
                seed=seed,
                samples_per_slice=samples_per_slice,
                pca_components=pca_components,
                representation=representation,
                include_score_payload=True,
            )
            cells.append({
                "seed": seed,
                "known_classes": episode_config["known_classes"],
                "payload": episode["score_payload"],
            })

    def evaluate_cell(
        cell: dict,
        *,
        known_split: str,
        unknown_split: str,
        fraction: float,
        embedding_space: str,
        candidate: dict,
    ) -> dict:
        evaluation_arguments = {
            "flag_fraction": fraction,
            "joint_pca_components": joint_pca_components,
            "clustering_method": candidate["method"],
            "dbscan_epsilon": candidate["dbscan_epsilon"],
            "hdbscan_minimum_cluster_size": candidate[
                "hdbscan_minimum_cluster_size"
            ],
            "hdbscan_minimum_samples": hdbscan_minimum_samples,
            "finch_hierarchy_level": candidate["finch_hierarchy_level"],
            "estimated_kmeans_maximum_cluster_count": candidate[
                "estimated_kmeans_maximum_cluster_count"
            ],
            "streaming_assignment_radius": candidate[
                "streaming_assignment_radius"
            ],
            "streaming_fading_rate": candidate["streaming_fading_rate"],
            "streaming_minimum_weight": streaming_minimum_weight,
            "streaming_promotion_weight": streaming_promotion_weight,
            "streaming_minimum_known_separation": (
                streaming_minimum_known_separation
            ),
        }
        feedback_metric = None
        if embedding_space == "feedback_metric_l2":
            proxy_review = evaluate_event_review_payload(
                cell["payload"],
                known_split="id_validation",
                unknown_split="proxy_unknown",
                embedding_space="representation_l2",
                **evaluation_arguments,
            )
            feedback_metric = _fit_review_feedback_metric(
                cell["payload"],
                proxy_review,
                known_split="id_validation",
                unknown_split="proxy_unknown",
            )
        return evaluate_event_review_payload(
            cell["payload"],
            known_split=known_split,
            unknown_split=unknown_split,
            embedding_space=embedding_space,
            feedback_metric=feedback_metric,
            **evaluation_arguments,
        )

    development = {}
    for fraction in flag_fractions:
        for embedding_space in spaces:
            for candidate in clustering_candidates:
                runs = [
                    evaluate_cell(
                        cell,
                        known_split="id_validation",
                        unknown_split="proxy_unknown",
                        fraction=fraction,
                        embedding_space=embedding_space,
                        candidate=candidate,
                    )
                    for cell in cells
                ]
                if candidate["method"] == "dbscan":
                    candidate_name = f"dbscan:{candidate['dbscan_epsilon']}"
                elif candidate["method"] == "hdbscan":
                    candidate_name = (
                        f"hdbscan:{candidate['hdbscan_minimum_cluster_size']}"
                    )
                elif candidate["method"] == "finch":
                    candidate_name = f"finch:{candidate['finch_hierarchy_level']}"
                elif candidate["method"] == "streaming":
                    candidate_name = (
                        f"streaming:{candidate['streaming_assignment_radius']}:"
                        f"{candidate['streaming_fading_rate']}"
                    )
                else:
                    candidate_name = (
                        "estimated_kmeans:"
                        f"{candidate['estimated_kmeans_maximum_cluster_count']}"
                    )
                development[f"{fraction}|{embedding_space}|{candidate_name}"] = {
                    "flag_fraction": fraction,
                    "embedding_space": embedding_space,
                    "clustering_method": candidate["method"],
                    "dbscan_epsilon": candidate["dbscan_epsilon"],
                    "hdbscan_minimum_cluster_size": candidate[
                        "hdbscan_minimum_cluster_size"
                    ],
                    "finch_hierarchy_level": candidate["finch_hierarchy_level"],
                    "estimated_kmeans_maximum_cluster_count": candidate[
                        "estimated_kmeans_maximum_cluster_count"
                    ],
                    "streaming_assignment_radius": candidate[
                        "streaming_assignment_radius"
                    ],
                    "streaming_fading_rate": candidate["streaming_fading_rate"],
                    "event_recall_mean": float(np.mean([
                        run["metrics"]["event_recall"] for run in runs
                    ])),
                    "distinct_group_recall_mean": float(np.mean([
                        run["metrics"]["distinct_group_recall"] for run in runs
                    ])),
                    "unknown_group_ari_mean": float(np.mean([
                        run["metrics"]["unknown_group_ari"] for run in runs
                    ])),
                    "useful_review_precision_mean": float(np.mean([
                        run["metrics"]["useful_review_precision"] for run in runs
                    ])),
                    "reviews_per_1000_mean": float(np.mean([
                        run["metrics"]["reviews_per_1000"] for run in runs
                    ])),
                    "partition_group_count_mean": float(np.mean([
                        run["metrics"]["partition_group_count"] for run in runs
                    ])),
                }
    selected = max(
        development.values(),
        key=lambda value: (
            value["distinct_group_recall_mean"],
            value["unknown_group_ari_mean"],
            value["event_recall_mean"],
            value["useful_review_precision_mean"],
            -value["reviews_per_1000_mean"],
        ),
    )
    selected_fraction = selected["flag_fraction"]
    selected_embedding_space = selected["embedding_space"]
    selected_method = selected["clustering_method"]
    selected_epsilon = selected["dbscan_epsilon"]
    selected_hdbscan_size = selected["hdbscan_minimum_cluster_size"]
    selected_finch_level = selected["finch_hierarchy_level"]
    selected_estimated_kmeans_count = selected[
        "estimated_kmeans_maximum_cluster_count"
    ]
    selected_streaming_radius = selected["streaming_assignment_radius"]
    selected_streaming_fading_rate = selected["streaming_fading_rate"]
    final_cells = []
    selected_candidate = {
        "method": selected_method,
        "dbscan_epsilon": selected_epsilon,
        "hdbscan_minimum_cluster_size": selected_hdbscan_size,
        "finch_hierarchy_level": selected_finch_level,
        "estimated_kmeans_maximum_cluster_count": selected_estimated_kmeans_count,
        "streaming_assignment_radius": selected_streaming_radius,
        "streaming_fading_rate": selected_streaming_fading_rate,
    }
    for cell in cells:
        review = evaluate_cell(
            cell,
            known_split="id_test",
            unknown_split="final_unknown",
            fraction=selected_fraction,
            embedding_space=selected_embedding_space,
            candidate=selected_candidate,
        )
        final_cells.append({
            "seed": cell["seed"],
            "known_classes": cell["known_classes"],
            "must_link_constraint_count": review.get("protocol", {}).get(
                "must_link_constraint_count", 0,
            ),
            "cannot_link_constraint_count": review.get("protocol", {}).get(
                "cannot_link_constraint_count", 0,
            ),
            **review["metrics"],
        })
    metric_names = tuple(final_cells[0].keys() - {"seed", "known_classes"})
    return {
        "protocol": {
            "representation": representation,
            "samples_per_slice": samples_per_slice,
            "joint_pca_components": joint_pca_components,
            "feature_model_mutation_published": False,
            "feedback_constraints_from_proxy_reviews_only": True,
            "development_unknown_classes": sorted({
                value for episode in episodes
                for value in episode["proxy_unknown_classes"]
            }),
            "final_unknown_classes": sorted({
                value for episode in episodes
                for value in episode["final_unknown_classes"]
            }),
            "final_labels_used_for_selection": False,
            "review_only": True,
            "mutation_published": False,
        },
        "development": development,
        "selected_flag_fraction": selected_fraction,
        "selected_embedding_space": selected_embedding_space,
        "selected_clustering_method": selected_method,
        "selected_dbscan_epsilon": selected_epsilon,
        "selected_hdbscan_minimum_cluster_size": selected_hdbscan_size,
        "selected_finch_hierarchy_level": selected_finch_level,
        "selected_estimated_kmeans_maximum_cluster_count": (
            selected_estimated_kmeans_count
        ),
        "selected_streaming_assignment_radius": selected_streaming_radius,
        "selected_streaming_fading_rate": selected_streaming_fading_rate,
        "final_summary": {
            name: float(np.mean([cell[name] for cell in final_cells]))
            for name in metric_names if cell_values_are_numeric(final_cells, name)
        },
        "final_cells": final_cells,
    }


def cell_values_are_numeric(cells: list[dict], name: str) -> bool:
    return all(isinstance(cell[name], (int, float)) and cell[name] is not None for cell in cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    runner = run_event_review_transfer if "episodes" in config else run_event_review_episode
    result = runner(**config)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
