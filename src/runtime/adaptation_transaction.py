"""Confirmation-gated publication and rollback for adaptation transactions."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from src.adaptation_policy import ConfirmationKind
from src.runtime.model_bundle import (
    BundleNode,
    BundleProvenance,
    LocalModelBundleStore,
)


@dataclass(frozen=True)
class ReviewConfirmation:
    review_id: str
    kind: ConfirmationKind
    confirmed_label: str
    confirmed_at: str

    def __post_init__(self) -> None:
        if not self.review_id.startswith("review-"):
            raise ValueError("confirmation must link to a persistent review ID")
        if not self.confirmed_label or not self.confirmed_at:
            raise ValueError("confirmation label and timestamp are required")

    @property
    def confirmation_id(self) -> str:
        payload = json.dumps({
            "review_id": self.review_id,
            "kind": self.kind.value,
            "confirmed_label": self.confirmed_label,
            "confirmed_at": self.confirmed_at,
        }, sort_keys=True, separators=(",", ":"))
        return "confirmation-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class AdaptationPublicationEvidence:
    replay_verified: bool
    calibration_nll_before: float
    calibration_nll_after: float
    graph_validation_issues: tuple[str, ...]
    final_novel_labels_hidden: bool

    def __post_init__(self) -> None:
        values = (self.calibration_nll_before, self.calibration_nll_after)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("calibration NLL values must be finite and non-negative")


@dataclass(frozen=True)
class AdaptationPublicationPolicy:
    maximum_calibration_nll_increase: float

    def __post_init__(self) -> None:
        if self.maximum_calibration_nll_increase < 0.0:
            raise ValueError("maximum calibration NLL increase must be non-negative")


@dataclass(frozen=True)
class AdaptationTransactionRecord:
    transaction_id: str
    review_id: str
    confirmation_id: str | None
    parent_bundle_id: str
    child_bundle_id: str | None
    status: str
    failed_gates: tuple[str, ...]
    mutation_published: bool
    rollback_bundle_id: str | None = None
    class_order_version: str = ""
    threshold_lineage_hash: str = ""
    routing_profile_hash: str = ""

    def __post_init__(self) -> None:
        lineage = (
            self.class_order_version,
            self.threshold_lineage_hash,
            self.routing_profile_hash,
        )
        if any(lineage) and not all(lineage):
            raise ValueError("adaptation lineage must be supplied as a complete set")
        for name in ("threshold_lineage_hash", "routing_profile_hash"):
            value = getattr(self, name)
            if value and (
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "review_id": self.review_id,
            "confirmation_id": self.confirmation_id,
            "parent_bundle_id": self.parent_bundle_id,
            "child_bundle_id": self.child_bundle_id,
            "status": self.status,
            "failed_gates": list(self.failed_gates),
            "mutation_published": self.mutation_published,
            "rollback_bundle_id": self.rollback_bundle_id,
            "class_order_version": self.class_order_version,
            "threshold_lineage_hash": self.threshold_lineage_hash,
            "routing_profile_hash": self.routing_profile_hash,
        }


def _transaction_id(review_id: str, parent_bundle_id: str) -> str:
    digest = hashlib.sha256(f"{review_id}|{parent_bundle_id}".encode("utf-8")).hexdigest()
    return "adaptation-" + digest[:12]


def publish_confirmed_adaptation(
    store: LocalModelBundleStore,
    *,
    review_id: str,
    confirmation: ReviewConfirmation | None,
    evidence: AdaptationPublicationEvidence,
    policy: AdaptationPublicationPolicy,
    components: Mapping[str, bytes],
    nodes: Sequence[BundleNode],
    provenance: BundleProvenance,
    publish: bool,
) -> AdaptationTransactionRecord:
    """Validate an adaptation and optionally activate a parent-linked child bundle."""
    current = store.current()
    if current is None:
        raise ValueError("adaptation publication requires an active parent bundle")
    failed = []
    if confirmation is None:
        failed.append("confirmation_required")
    elif confirmation.review_id != review_id:
        failed.append("confirmation_review_mismatch")
    if not evidence.replay_verified:
        failed.append("replay_gate_failed")
    if (
        evidence.calibration_nll_after - evidence.calibration_nll_before
        > policy.maximum_calibration_nll_increase
    ):
        failed.append("calibration_gate_failed")
    if evidence.graph_validation_issues:
        failed.append("graph_gate_failed")
    if not evidence.final_novel_labels_hidden:
        failed.append("final_novel_labels_exposed")
    record = AdaptationTransactionRecord(
        transaction_id=_transaction_id(review_id, current.bundle_id),
        review_id=review_id,
        confirmation_id=None if confirmation is None else confirmation.confirmation_id,
        parent_bundle_id=current.bundle_id,
        child_bundle_id=None,
        status="rejected" if failed else "validated_dry_run",
        failed_gates=tuple(failed),
        mutation_published=False,
    )
    if failed or not publish:
        return record
    child = store.publish(
        components,
        nodes,
        provenance=provenance,
        parent_bundle_id=current.bundle_id,
    )
    store.activate(child.bundle_id)
    return replace(
        record,
        child_bundle_id=child.bundle_id,
        status="published",
        mutation_published=True,
    )


def rollback_adaptation(
    store: LocalModelBundleStore,
    record: AdaptationTransactionRecord,
) -> AdaptationTransactionRecord:
    if not record.mutation_published or record.child_bundle_id is None:
        raise ValueError("only a published adaptation can be rolled back")
    current = store.current()
    if current is None or current.bundle_id != record.child_bundle_id:
        raise ValueError("published adaptation is not the active bundle")
    restored = store.rollback()
    if restored.bundle_id != record.parent_bundle_id:
        raise ValueError("rollback did not restore the transaction parent")
    return replace(
        record,
        status="rolled_back",
        mutation_published=False,
        rollback_bundle_id=restored.bundle_id,
    )