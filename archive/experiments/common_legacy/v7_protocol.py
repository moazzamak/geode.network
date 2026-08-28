from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import payload_hash, sha256_file


@dataclass(frozen=True)
class AcceptanceHeadSpec:
    family: str
    representation_hash: str
    class_order: tuple[str, ...]
    score_direction: str
    threshold_policy: str
    incremental_update: str

    def __post_init__(self) -> None:
        if not self.family or len(self.representation_hash) != 64:
            raise ValueError("Acceptance heads require a family and SHA-256 lineage.")
        if not self.class_order or len(set(self.class_order)) != len(self.class_order):
            raise ValueError("Acceptance-head class order must be non-empty and unique.")
        if self.score_direction not in {"higher_is_novel", "lower_is_novel"}:
            raise ValueError("Unsupported novelty-score direction.")
        if not self.threshold_policy or not self.incremental_update:
            raise ValueError("Acceptance-head policy fields must be non-empty.")


@dataclass(frozen=True)
class EmpiricalRoutingProfile:
    model_signature: str
    representation_hash: str
    class_order: tuple[str, ...]
    profile_family: str
    fit_data_hash: str
    calibration_data_hash: str
    score_direction: str
    threshold: float
    dimension: int

    def __post_init__(self) -> None:
        hashes = (
            self.representation_hash,
            self.fit_data_hash,
            self.calibration_data_hash,
        )
        if not self.model_signature or any(len(value) != 64 for value in hashes):
            raise ValueError("Routing profiles require model identity and SHA-256 lineage.")
        if not self.class_order or len(set(self.class_order)) != len(self.class_order):
            raise ValueError("Routing-profile class order must be non-empty and unique.")
        if self.score_direction not in {"higher_is_match", "lower_is_match"}:
            raise ValueError("Unsupported routing-score direction.")
        if not self.profile_family or self.dimension <= 0 or not math.isfinite(self.threshold):
            raise ValueError("Routing-profile geometry must be finite and non-empty.")

    @property
    def profile_id(self) -> str:
        return payload_hash(self.__dict__)


@dataclass(frozen=True)
class ReviewEvent:
    review_id: str
    member_ids: tuple[str, ...]
    requested_window: int
    state: str

    def __post_init__(self) -> None:
        if not self.review_id.startswith("review-"):
            raise ValueError("Review IDs must be stable review-prefixed identifiers.")
        if not self.member_ids or len(set(self.member_ids)) != len(self.member_ids):
            raise ValueError("Review members must be non-empty and unique.")
        if self.requested_window < 0 or self.state not in {
            "emerging",
            "established",
            "review_requested",
            "confirmed",
            "quarantined",
            "expired",
        }:
            raise ValueError("Unsupported review state.")


@dataclass(frozen=True)
class ConfirmationEvent:
    review_id: str
    response: str
    confirmed_label: str | None
    confirmed_window: int

    def __post_init__(self) -> None:
        responses = {
            "existing_class",
            "new_class",
            "corruption_or_irrelevant",
            "unresolved",
        }
        if not self.review_id.startswith("review-") or self.response not in responses:
            raise ValueError("Unsupported review confirmation.")
        needs_label = self.response in {"existing_class", "new_class"}
        if needs_label != bool(self.confirmed_label):
            raise ValueError("Only semantic confirmations may carry a label.")
        if self.confirmed_window < 0:
            raise ValueError("Confirmation windows must be non-negative.")


@dataclass(frozen=True)
class GraphMigrationSpec:
    parent_bundle_hash: str
    parent_class_order: tuple[str, ...]
    child_class_order: tuple[str, ...]
    review_id: str
    confirmation_id: str
    rollback_bundle_hash: str

    def __post_init__(self) -> None:
        hashes = (self.parent_bundle_hash, self.rollback_bundle_hash)
        if any(len(value) != 64 for value in hashes):
            raise ValueError("Graph migrations require SHA-256 bundle lineage.")
        if self.parent_bundle_hash != self.rollback_bundle_hash:
            raise ValueError("Rollback target must be the exact migration parent.")
        if not self.review_id.startswith("review-") or not self.confirmation_id:
            raise ValueError("Graph migrations require linked human confirmation.")
        if (
            not self.parent_class_order
            or len(self.child_class_order) != len(self.parent_class_order) + 1
            or tuple(self.child_class_order[:-1]) != self.parent_class_order
        ):
            raise ValueError("New-class migrations must append one class atomically.")


def validate_parent_locks(
    locks: Sequence[Mapping[str, Any]], repository_root: str | Path
) -> list[dict[str, Any]]:
    root = Path(repository_root).resolve()
    if not locks:
        raise ValueError("M38 requires immutable parent locks.")
    verified = []
    seen = set()
    for lock in locks:
        if set(lock) != {"id", "path", "sha256"}:
            raise ValueError("Unsupported M38 parent-lock schema.")
        identifier = str(lock["id"])
        relative = Path(str(lock["path"]))
        if not identifier or identifier in seen or relative.is_absolute():
            raise ValueError("Parent-lock identities and paths must be safe and unique.")
        seen.add(identifier)
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("Parent locks must resolve to repository files.")
        actual = sha256_file(path)
        if actual != str(lock["sha256"]):
            raise ValueError(f"Parent lock {identifier!r} drifted.")
        verified.append(
            {
                "id": identifier,
                "path": relative.as_posix(),
                "sha256": actual,
                "bytes": path.stat().st_size,
            }
        )
    return verified


def validate_v7_m38_config(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "program",
        "milestone",
        "parent_file_locks",
        "seeds",
        "stages",
        "schedules",
        "review_policy",
        "routing_policy",
        "resource_limits",
        "closed_claims",
        "literature_audit",
        "training_data_loaded",
        "final_labels_opened",
    }
    if set(payload) != required or payload.get("schema_version") != 1:
        raise ValueError("Unsupported v7 M38 configuration schema.")
    if payload["program"] != "v7" or payload["milestone"] != "M38":
        raise ValueError("Protocol must identify v7/M38.")
    if payload["training_data_loaded"] is not False or payload["final_labels_opened"] is not False:
        raise PermissionError("M38 lock cannot load training data or final labels.")
    if payload["seeds"] != [11, 23, 37]:
        raise ValueError("M38 seeds must remain 11, 23, and 37.")
    if set(payload["stages"]) != {"S0", "S1", "S2", "S3", "S4"}:
        raise ValueError("M38 must freeze all five data stages.")
    schedules = payload["schedules"]
    if len(schedules) != 3 or len({item["id"] for item in schedules}) != 3:
        raise ValueError("M38 requires three unique frozen schedules.")
    if any(item.get("final_labels_sealed") is not True for item in schedules):
        raise PermissionError("Every M38 schedule must seal final labels.")
    review = payload["review_policy"]
    if (
        review["semantic_publication_requires_confirmation"] is not True
        or review["mutation_requires_confirmation"] is not True
    ):
        raise PermissionError("Semantic publication and mutation require confirmation.")
    routing = payload["routing_policy"]
    if (
        routing["compatibility_identity"] != "ModelFingerprint"
        or routing["empirical_support"] != "SupportProfile"
        or routing["authoritative_default"] != "exhaustive"
        or routing["stale_profile_action"] != "exhaustive_fallback"
        or routing["semantic_route_can_override_unknown"] is not False
    ):
        raise ValueError("M38 routing must remain compatibility-first and fail closed.")
    audit = payload["literature_audit"]
    if (
        audit["all_seven_stage_system_found"] is not False
        or audit["outcome_e_triggered"] is not False
        or audit["search_date"] != "2026-07-26"
    ):
        raise ValueError("M38 literature conclusion does not match the audit.")
    if len(payload["closed_claims"]) != 4:
        raise ValueError("M38 must retain all four closed claim families.")


def schedule_locks(schedules: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"id": str(schedule["id"]), "sha256": payload_hash(schedule)}
        for schedule in schedules
    ]


def synthetic_contract_fixture() -> dict[str, Any]:
    representation_hash = hashlib.sha256(b"v7-synthetic-representation").hexdigest()
    fit_hash = hashlib.sha256(b"v7-synthetic-fit").hexdigest()
    calibration_hash = hashlib.sha256(b"v7-synthetic-calibration").hexdigest()
    parent_hash = hashlib.sha256(b"v7-parent-bundle").hexdigest()
    head = AcceptanceHeadSpec(
        family="evm_margin_tail",
        representation_hash=representation_hash,
        class_order=("known-a", "known-b"),
        score_direction="higher_is_novel",
        threshold_policy="calibration_quantile",
        incremental_update="native_incremental",
    )
    profile = EmpiricalRoutingProfile(
        model_signature="synthetic-model",
        representation_hash=representation_hash,
        class_order=head.class_order,
        profile_family="low_rank_gaussian",
        fit_data_hash=fit_hash,
        calibration_data_hash=calibration_hash,
        score_direction="higher_is_match",
        threshold=0.5,
        dimension=4,
    )
    review = ReviewEvent(
        review_id="review-synthetic",
        member_ids=("sample-1", "sample-2"),
        requested_window=2,
        state="review_requested",
    )
    confirmation = ConfirmationEvent(
        review_id=review.review_id,
        response="new_class",
        confirmed_label="novel-c",
        confirmed_window=3,
    )
    migration = GraphMigrationSpec(
        parent_bundle_hash=parent_hash,
        parent_class_order=head.class_order,
        child_class_order=(*head.class_order, "novel-c"),
        review_id=review.review_id,
        confirmation_id=payload_hash(confirmation.__dict__),
        rollback_bundle_hash=parent_hash,
    )
    return {
        "acceptance_head": head.__dict__,
        "routing_profile": {**profile.__dict__, "profile_id": profile.profile_id},
        "review": review.__dict__,
        "confirmation": confirmation.__dict__,
        "migration": migration.__dict__,
    }
