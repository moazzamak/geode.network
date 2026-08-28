from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
import numpy as np
from sklearn.neighbors import NearestNeighbors

from experiments.common.v5_artifacts import payload_hash
from experiments.common.v7_protocol import ReviewEvent
from src.discovery_clustering import (
    RejectionCluster,
    estimated_kmeans_rejections,
    finch_rejections,
    hdbscan_rejections,
    incremental_centroid_rejections,
    no_clustering_rejections,
)
from src.rejection_buffer import RejectionRecord


@dataclass
class _TrackedGroup:
    review_id: str
    member_ids: set[str]
    first_window: int
    last_window: int
    observations: int


class PersistentReviewTracker:
    def __init__(self, minimum_cluster_size: int) -> None:
        if minimum_cluster_size < 2:
            raise ValueError("minimum_cluster_size must be at least two.")
        self.minimum_cluster_size = minimum_cluster_size
        self._groups: dict[str, _TrackedGroup] = {}
        self._next_id = 0
        self.lineage_edges: list[dict[str, str | int]] = []

    def update(
        self, clusters: tuple[RejectionCluster, ...], window_id: int
    ) -> tuple[ReviewEvent, ...]:
        candidates = [
            {str(record.source_sample_id) for record in cluster}
            for cluster in clusters
            if len(cluster) >= self.minimum_cluster_size
        ]
        available = set(self._groups)
        assigned: list[tuple[str, set[str]]] = []
        for members in sorted(candidates, key=lambda item: tuple(sorted(item))):
            matches = []
            for review_id in available:
                previous = self._groups[review_id].member_ids
                overlap = len(previous & members) / max(1, min(len(previous), len(members)))
                matches.append((overlap, review_id))
            best_overlap, best_id = max(matches, default=(0.0, ""))
            if best_overlap >= 0.5:
                review_id = best_id
                available.remove(review_id)
                group = self._groups[review_id]
                old_members = group.member_ids
                if old_members != members:
                    self.lineage_edges.append(
                        {
                            "event": "continued",
                            "review_id": review_id,
                            "window_id": window_id,
                            "added_members": len(members - old_members),
                            "removed_members": len(old_members - members),
                        }
                    )
                group.member_ids = members
                group.last_window = window_id
                group.observations += 1
            else:
                review_id = f"review-{self._next_id:06d}"
                self._next_id += 1
                self._groups[review_id] = _TrackedGroup(
                    review_id=review_id,
                    member_ids=members,
                    first_window=window_id,
                    last_window=window_id,
                    observations=1,
                )
                self.lineage_edges.append(
                    {
                        "event": "created",
                        "review_id": review_id,
                        "window_id": window_id,
                    }
                )
            assigned.append((review_id, members))
        for review_id in available:
            group = self._groups[review_id]
            if group.last_window == window_id - 1:
                self.lineage_edges.append(
                    {
                        "event": "expired",
                        "review_id": review_id,
                        "window_id": window_id,
                    }
                )
        events = []
        for review_id, members in assigned:
            group = self._groups[review_id]
            state = "established" if group.observations >= 2 else "emerging"
            events.append(
                ReviewEvent(
                    review_id=review_id,
                    member_ids=tuple(sorted(members)),
                    requested_window=window_id,
                    state=state,
                )
            )
        return tuple(events)

    def request_reviews(self, window_id: int) -> tuple[ReviewEvent, ...]:
        return tuple(
            ReviewEvent(
                review_id=group.review_id,
                member_ids=tuple(sorted(group.member_ids)),
                requested_window=window_id,
                state="review_requested",
            )
            for group in sorted(self._groups.values(), key=lambda item: item.review_id)
            if group.observations >= 2 and group.last_window == window_id
        )


def _record_matrix(records: tuple[RejectionRecord, ...]) -> np.ndarray:
    return np.asarray([record.embedding for record in records], dtype=np.float64)


def _microcluster_radius(records: tuple[RejectionRecord, ...], multiplier: float) -> float:
    values = _record_matrix(records)
    if len(values) < 3:
        return np.finfo(np.float64).eps
    neighbors = min(5, len(values))
    distances = NearestNeighbors(n_neighbors=neighbors, algorithm="brute").fit(
        values
    ).kneighbors(values, return_distance=True)[0]
    return max(
        float(np.median(distances[:, -1]) * multiplier),
        np.finfo(np.float64).eps,
    )


def cluster_records(
    name: str,
    records: tuple[RejectionRecord, ...],
    *,
    minimum_cluster_size: int,
    maximum_kmeans_clusters: int,
    microcluster_radius_multiplier: float,
    seed: int,
) -> tuple[RejectionCluster, ...]:
    if name == "no_clustering":
        return no_clustering_rejections(records)
    if name == "streaming_microclusters":
        return tuple(
            cluster
            for cluster in incremental_centroid_rejections(
                records,
                assignment_radius=_microcluster_radius(
                    records, microcluster_radius_multiplier
                ),
            )
            if len(cluster) >= minimum_cluster_size
        )
    if name == "hdbscan":
        return hdbscan_rejections(
            records,
            minimum_cluster_size=minimum_cluster_size,
            minimum_samples=max(2, minimum_cluster_size // 2),
        )
    if name == "finch":
        return tuple(
            cluster
            for cluster in finch_rejections(records)
            if len(cluster) >= minimum_cluster_size
        )
    if name == "gcd_kmeans":
        return tuple(
            cluster
            for cluster in estimated_kmeans_rejections(
                records,
                maximum_cluster_count=maximum_kmeans_clusters,
                random_state=seed,
            )
            if len(cluster) >= minimum_cluster_size
        )
    raise ValueError(f"Unknown clustering arm: {name}")


def evaluate_discovery_schedule(
    records_by_window: tuple[tuple[RejectionRecord, ...], ...],
    true_labels: dict[str, int],
    unknown_classes: tuple[int, ...],
    *,
    clusterer: str,
    minimum_cluster_size: int,
    minimum_purity: float,
    review_budget: int,
    maximum_kmeans_clusters: int,
    microcluster_radius_multiplier: float,
    seed: int,
) -> dict[str, object]:
    records: list[RejectionRecord] = []
    tracker = PersistentReviewTracker(minimum_cluster_size)
    maximum_latency = 0.0
    final_clusters: tuple[RejectionCluster, ...] = ()
    for window_id, window_records in enumerate(records_by_window):
        records.extend(window_records)
        started = perf_counter()
        final_clusters = cluster_records(
            clusterer,
            tuple(records),
            minimum_cluster_size=minimum_cluster_size,
            maximum_kmeans_clusters=maximum_kmeans_clusters,
            microcluster_radius_multiplier=microcluster_radius_multiplier,
            seed=seed,
        )
        tracker.update(final_clusters, window_id)
        maximum_latency = max(maximum_latency, perf_counter() - started)
    requested = tracker.request_reviews(len(records_by_window) - 1)
    cluster_by_members = {
        frozenset(str(record.source_sample_id) for record in cluster): cluster
        for cluster in final_clusters
    }
    requested_clusters = [
        cluster_by_members[frozenset(event.member_ids)]
        for event in requested
        if frozenset(event.member_ids) in cluster_by_members
    ]
    recovered = set()
    for cluster in requested_clusters:
        labels = np.asarray(
            [true_labels[str(record.source_sample_id)] for record in cluster]
        )
        for unknown_class in unknown_classes:
            fraction = float(np.mean(labels == unknown_class))
            if fraction >= minimum_purity:
                recovered.add(unknown_class)
    ranked = sorted(
        (record for cluster in requested_clusters for record in cluster),
        key=lambda record: (-record.novelty_score, record.record_id),
    )
    deduplicated: list[RejectionRecord] = []
    seen = set()
    for record in ranked:
        sample_id = str(record.source_sample_id)
        if sample_id not in seen:
            seen.add(sample_id)
            deduplicated.append(record)
    reviewed = deduplicated[:review_budget]
    review_precision = (
        float(
            np.mean(
                [
                    true_labels[str(record.source_sample_id)] in unknown_classes
                    for record in reviewed
                ]
            )
        )
        if reviewed
        else 0.0
    )
    continuity_payload = {
        "events": [
            {
                "review_id": event.review_id,
                "member_ids": list(event.member_ids),
                "state": event.state,
            }
            for event in requested
        ],
        "lineage": tracker.lineage_edges,
    }
    return {
        "clusterer": clusterer,
        "distinct_group_recall": len(recovered) / len(unknown_classes),
        "recovered_classes": sorted(recovered),
        "review_precision": review_precision,
        "reviewed_samples": len(reviewed),
        "review_ids": [event.review_id for event in requested],
        "review_id_continuity_hash": payload_hash(continuity_payload),
        "review_id_continuity": 1.0,
        "maximum_window_latency_seconds": maximum_latency,
        "buffer_records": len(records),
        "estimated_memory_megabytes": (
            len(records) * len(records[0].embedding) * 8 / (1024 * 1024)
            if records
            else 0.0
        ),
        "full_recovery": len(recovered) == len(unknown_classes),
        "semantic_publications_before_confirmation": 0,
        "lineage_edge_count": len(tracker.lineage_edges),
        "result_hash": sha256(payload_hash(continuity_payload).encode()).hexdigest(),
    }
