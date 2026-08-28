from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from src.rejection_buffer import RejectionRecord


@dataclass(frozen=True)
class StreamingClusterPolicy:
    assignment_radius: float
    fading_rate: float
    minimum_weight: float
    promotion_weight: float
    minimum_windows: int
    minimum_known_separation: float
    max_clusters: int
    max_records_per_cluster: int

    def __post_init__(self) -> None:
        if self.assignment_radius <= 0.0 or self.fading_rate < 0.0:
            raise ValueError("assignment radius must be positive and fading non-negative.")
        if self.minimum_weight <= 0.0 or self.promotion_weight < self.minimum_weight:
            raise ValueError("promotion weight must be at least the minimum weight.")
        if self.minimum_windows < 2 or self.minimum_known_separation < 0.0:
            raise ValueError("persistence/separation limits must be valid.")
        if self.max_clusters <= 0 or self.max_records_per_cluster <= 0:
            raise ValueError("streaming memory limits must be positive.")


@dataclass(frozen=True)
class StreamingClusterSnapshot:
    cluster_id: str
    records: tuple[RejectionRecord, ...]
    centroid: tuple[float, ...]
    faded_weight: float
    created_window: int
    last_updated_window: int
    window_count: int
    state: str


@dataclass(frozen=True)
class StreamingReviewHypothesis:
    cluster_id: str
    review_id: str
    relation: str
    review_only: bool
    nearest_known_distance: float
    snapshot: StreamingClusterSnapshot


@dataclass
class _MicroCluster:
    cluster_id: str
    records: list[RejectionRecord]
    created_window: int
    last_updated_window: int


class StreamingRejectionMemory:
    """Bounded, label-blind fading memory for review-only rejection groups."""

    def __init__(self, policy: StreamingClusterPolicy) -> None:
        self.policy = policy
        self._clusters: list[_MicroCluster] = []
        self._record_ids: set[int] = set()
        self._current_window = -1
        self.evicted_clusters = 0
        self.evicted_records = 0

    @staticmethod
    def _profile(record: RejectionRecord) -> tuple[str, str]:
        return record.source_model_signature, record.support_profile_version

    def _weights(self, cluster: _MicroCluster, window_id: int) -> np.ndarray:
        return np.asarray([
            2.0 ** (-self.policy.fading_rate * (window_id - record.window_id))
            for record in cluster.records
        ])

    def _centroid(self, cluster: _MicroCluster, window_id: int) -> np.ndarray:
        embeddings = np.asarray([record.embedding for record in cluster.records])
        weights = self._weights(cluster, window_id)
        return np.average(embeddings, axis=0, weights=weights)

    def _cluster_id(self, record: RejectionRecord) -> str:
        payload = {
            "first_record_id": record.record_id,
            "source": self._profile(record),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8"),
        ).hexdigest()[:12]
        return f"micro-{digest}"

    def _evict_for_capacity(self, window_id: int) -> None:
        if len(self._clusters) < self.policy.max_clusters:
            return
        weights = [float(self._weights(cluster, window_id).sum()) for cluster in self._clusters]
        oldest = min(
            range(len(self._clusters)),
            key=lambda index: (weights[index], self._clusters[index].last_updated_window),
        )
        removed = self._clusters.pop(oldest)
        self._record_ids.difference_update(record.record_id for record in removed.records)
        self.evicted_records += len(removed.records)
        self.evicted_clusters += 1

    def ingest_window(
        self,
        records: tuple[RejectionRecord, ...],
        *,
        window_id: int,
    ) -> None:
        if window_id < self._current_window:
            raise ValueError("windows must be ingested in non-decreasing order.")
        if any(record.window_id != window_id for record in records):
            raise ValueError("each ingested record must belong to the supplied window.")
        if any(record.record_id in self._record_ids for record in records):
            raise ValueError("rejection records may be ingested only once.")
        self._current_window = window_id

        for record in records:
            embedding = np.asarray(record.embedding)
            candidates = [
                (index, np.linalg.norm(
                    embedding - self._centroid(cluster, window_id),
                ))
                for index, cluster in enumerate(self._clusters)
                if self._profile(cluster.records[0]) == self._profile(record)
            ]
            nearest = min(candidates, key=lambda item: item[1]) if candidates else None
            if nearest is None or nearest[1] > self.policy.assignment_radius:
                self._evict_for_capacity(window_id)
                self._clusters.append(_MicroCluster(
                    cluster_id=self._cluster_id(record),
                    records=[record],
                    created_window=window_id,
                    last_updated_window=window_id,
                ))
            else:
                cluster = self._clusters[nearest[0]]
                cluster.records.append(record)
                cluster.last_updated_window = window_id
                if len(cluster.records) > self.policy.max_records_per_cluster:
                    removed = cluster.records.pop(0)
                    self._record_ids.remove(removed.record_id)
                    self.evicted_records += 1
            self._record_ids.add(record.record_id)

    def snapshots(
        self,
        *,
        window_id: int | None = None,
        include_faded: bool = False,
    ) -> tuple[StreamingClusterSnapshot, ...]:
        current = self._current_window if window_id is None else window_id
        if current < self._current_window:
            raise ValueError("snapshots cannot move streaming time backward.")
        snapshots = []
        for cluster in self._clusters:
            weight = float(self._weights(cluster, current).sum())
            windows = len({record.window_id for record in cluster.records})
            if weight < self.policy.minimum_weight:
                state = "faded"
            elif weight >= self.policy.promotion_weight and windows >= self.policy.minimum_windows:
                state = "established"
            else:
                state = "emerging"
            if state == "faded" and not include_faded:
                continue
            snapshots.append(StreamingClusterSnapshot(
                cluster_id=cluster.cluster_id,
                records=tuple(cluster.records),
                centroid=tuple(float(value) for value in self._centroid(cluster, current)),
                faded_weight=weight,
                created_window=cluster.created_window,
                last_updated_window=cluster.last_updated_window,
                window_count=windows,
                state=state,
            ))
        return tuple(snapshots)

    def review_hypotheses(
        self,
        known_centroids: np.ndarray,
        *,
        window_id: int | None = None,
    ) -> tuple[StreamingReviewHypothesis, ...]:
        known = np.asarray(known_centroids, dtype=np.float64)
        snapshots = self.snapshots(window_id=window_id)
        if known.ndim != 2 or not len(known):
            raise ValueError("known_centroids must be a non-empty matrix.")
        hypotheses = []
        for snapshot in snapshots:
            centroid = np.asarray(snapshot.centroid)
            if known.shape[1] != len(centroid):
                raise ValueError("known centroids and rejection embeddings must align.")
            nearest_distance = float(np.min(np.linalg.norm(known - centroid, axis=1)))
            relation = (
                "emerging_novel"
                if nearest_distance >= self.policy.minimum_known_separation
                else "known_extension"
            )
            hypotheses.append(StreamingReviewHypothesis(
                cluster_id=snapshot.cluster_id,
                review_id=f"review-{snapshot.cluster_id.removeprefix('micro-')}",
                relation=relation,
                review_only=True,
                nearest_known_distance=nearest_distance,
                snapshot=snapshot,
            ))
        return tuple(hypotheses)