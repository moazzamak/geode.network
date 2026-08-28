from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import silhouette_score

from src.rejection_buffer import RejectionRecord


RejectionCluster = tuple[RejectionRecord, ...]
Clusterer = Callable[[tuple[RejectionRecord, ...]], tuple[RejectionCluster, ...]]


def dbscan_rejections(
    records: tuple[RejectionRecord, ...],
    *,
    epsilon: float,
    minimum_samples: int,
) -> tuple[RejectionCluster, ...]:
    if epsilon <= 0.0 or minimum_samples <= 0:
        raise ValueError("DBSCAN parameters must be positive.")
    if not records:
        return ()
    embeddings = np.asarray([record.embedding for record in records])
    labels = DBSCAN(eps=epsilon, min_samples=minimum_samples).fit_predict(embeddings)
    return tuple(
        tuple(record for record, assigned in zip(records, labels) if assigned == label)
        for label in sorted(set(labels) - {-1})
    )


def hdbscan_rejections(
    records: tuple[RejectionRecord, ...],
    *,
    minimum_cluster_size: int,
    minimum_samples: int | None = None,
) -> tuple[RejectionCluster, ...]:
    if minimum_cluster_size < 2:
        raise ValueError("minimum_cluster_size must be at least two.")
    if minimum_samples is not None and minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive when provided.")
    if len(records) < minimum_cluster_size:
        return ()
    embeddings = np.asarray([record.embedding for record in records])
    labels = HDBSCAN(
        min_cluster_size=minimum_cluster_size,
        min_samples=minimum_samples,
        copy=True,
    ).fit_predict(embeddings)
    return tuple(
        tuple(record for record, assigned in zip(records, labels) if assigned == label)
        for label in sorted(set(labels) - {-1})
    )


def finch_rejections(
    records: tuple[RejectionRecord, ...],
    *,
    hierarchy_level: int = 0,
) -> tuple[RejectionCluster, ...]:
    if hierarchy_level < 0:
        raise ValueError("hierarchy_level must be non-negative.")
    if len(records) < 2:
        return ()

    embeddings = np.asarray([record.embedding for record in records], dtype=np.float64)
    original_membership = np.arange(len(records), dtype=np.int64)
    partitions: list[np.ndarray] = []
    current_embeddings = embeddings

    while len(current_embeddings) > 1:
        squared_distances = np.sum(
            (current_embeddings[:, None, :] - current_embeddings[None, :, :]) ** 2,
            axis=2,
        )
        np.fill_diagonal(squared_distances, np.inf)
        first_neighbors = np.argmin(squared_distances, axis=1)
        parents = np.arange(len(current_embeddings), dtype=np.int64)

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = int(parents[index])
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        shared_neighbor_representatives: dict[int, int] = {}
        for index, neighbor in enumerate(first_neighbors):
            union(index, int(neighbor))
            representative = shared_neighbor_representatives.setdefault(
                int(neighbor), index,
            )
            union(index, representative)

        roots = np.asarray([find(index) for index in range(len(parents))])
        _, level_labels = np.unique(roots, return_inverse=True)
        original_membership = level_labels[original_membership]
        partitions.append(original_membership.copy())
        cluster_count = int(np.max(level_labels)) + 1
        if cluster_count == 1 or cluster_count == len(current_embeddings):
            break
        current_embeddings = np.vstack([
            current_embeddings[level_labels == label].mean(axis=0)
            for label in range(cluster_count)
        ])

    if hierarchy_level >= len(partitions):
        return ()
    labels = partitions[hierarchy_level]
    return tuple(
        tuple(record for record, assigned in zip(records, labels) if assigned == label)
        for label in sorted(set(labels))
    )


def estimated_kmeans_rejections(
    records: tuple[RejectionRecord, ...],
    *,
    maximum_cluster_count: int,
    random_state: int = 0,
) -> tuple[RejectionCluster, ...]:
    if maximum_cluster_count < 2:
        raise ValueError("maximum_cluster_count must be at least two.")
    if len(records) < 3:
        return ()

    embeddings = np.asarray([record.embedding for record in records], dtype=np.float64)
    best_labels = None
    best_key = None
    for cluster_count in range(2, min(maximum_cluster_count, len(records) - 1) + 1):
        labels = KMeans(
            n_clusters=cluster_count,
            random_state=random_state,
            n_init=10,
        ).fit_predict(embeddings)
        score = float(silhouette_score(embeddings, labels))
        key = (score, -cluster_count)
        if best_key is None or key > best_key:
            best_key = key
            best_labels = labels

    if best_labels is None:
        return ()
    return tuple(
        tuple(
            record for record, assigned in zip(records, best_labels)
            if assigned == label
        )
        for label in sorted(set(best_labels))
    )


def incremental_centroid_rejections(
    records: tuple[RejectionRecord, ...],
    *,
    assignment_radius: float,
) -> tuple[RejectionCluster, ...]:
    if assignment_radius <= 0.0:
        raise ValueError("assignment_radius must be positive.")
    centroids: list[np.ndarray] = []
    clusters: list[list[RejectionRecord]] = []
    for record in records:
        embedding = np.asarray(record.embedding)
        if not centroids:
            centroids.append(embedding.copy())
            clusters.append([record])
            continue
        distances = np.asarray([
            np.linalg.norm(embedding - centroid) for centroid in centroids
        ])
        nearest = int(np.argmin(distances))
        if distances[nearest] <= assignment_radius:
            clusters[nearest].append(record)
            centroids[nearest] = np.mean(
                [member.embedding for member in clusters[nearest]], axis=0,
            )
        else:
            centroids.append(embedding.copy())
            clusters.append([record])
    return tuple(tuple(cluster) for cluster in clusters)


def no_clustering_rejections(
    records: tuple[RejectionRecord, ...],
) -> tuple[RejectionCluster, ...]:
    del records
    return ()