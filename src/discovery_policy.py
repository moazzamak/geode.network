from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from src.rejection_buffer import RejectionRecord


@dataclass(frozen=True)
class ClusterProposalPolicy:
    minimum_support: int
    minimum_windows: int
    maximum_rms_radius: float
    minimum_known_separation: float
    review_only: bool = False

    def __post_init__(self) -> None:
        if self.minimum_support <= 0 or self.minimum_windows < 2:
            raise ValueError("support must be positive and persistence requires >= 2 windows.")
        if self.maximum_rms_radius <= 0.0 or self.minimum_known_separation < 0.0:
            raise ValueError("compactness/separation limits must be valid.")


@dataclass(frozen=True)
class ClusterProposalDecision:
    eligible: bool
    temporary_unknown_id: str | None
    review_required: bool
    review_id: str | None
    support: int
    windows: int
    rms_radius: float
    nearest_known_distance: float
    failed_criteria: tuple[str, ...]


def evaluate_cluster_proposal(
    records: tuple[RejectionRecord, ...],
    known_centroids: np.ndarray,
    policy: ClusterProposalPolicy,
) -> ClusterProposalDecision:
    """Evaluate evidence for a temporary unknown ID without mutating any model."""
    if not records:
        raise ValueError("cluster records must be non-empty.")
    embeddings = np.asarray([record.embedding for record in records], dtype=np.float64)
    known = np.asarray(known_centroids, dtype=np.float64)
    if known.ndim != 2 or known.shape[1] != embeddings.shape[1] or not len(known):
        raise ValueError("known_centroids must be a non-empty aligned matrix.")
    source_pairs = {
        (record.source_model_signature, record.support_profile_version)
        for record in records
    }
    centroid = embeddings.mean(axis=0)
    rms_radius = float(np.sqrt(np.mean(np.sum((embeddings - centroid) ** 2, axis=1))))
    nearest_known_distance = float(np.min(np.linalg.norm(known - centroid, axis=1)))
    windows = len({record.window_id for record in records})
    failed = []
    if len(records) < policy.minimum_support:
        failed.append("insufficient_support")
    if windows < policy.minimum_windows:
        failed.append("insufficient_persistence")
    if rms_radius > policy.maximum_rms_radius:
        failed.append("insufficient_compactness")
    if nearest_known_distance < policy.minimum_known_separation:
        failed.append("insufficient_separation")
    if len(source_pairs) != 1:
        failed.append("mixed_support_profiles")

    temporary_id = None
    review_required = (
        tuple(failed) == ("insufficient_separation",)
        or (not failed and policy.review_only)
    )
    review_id = None
    if not failed or review_required:
        payload = {
            "record_ids": sorted(record.record_id for record in records),
            "source": next(iter(source_pairs)),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8"),
        ).hexdigest()[:12]
        if review_required:
            review_id = f"review-{digest}"
        else:
            temporary_id = f"unknown-{digest}"
    return ClusterProposalDecision(
        eligible=not failed and not policy.review_only,
        temporary_unknown_id=temporary_id,
        review_required=review_required,
        review_id=review_id,
        support=len(records),
        windows=windows,
        rms_radius=rms_radius,
        nearest_known_distance=nearest_known_distance,
        failed_criteria=tuple(failed),
    )