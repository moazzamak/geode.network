from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class AdaptationAction(StrEnum):
    QUARANTINE = "quarantine"
    UPDATE_EXISTING = "update_existing"
    CREATE_NEW = "create_new"


class ConfirmationKind(StrEnum):
    EXISTING_CLASS = "existing_class"
    NEW_CLASS = "new_class"


@dataclass(frozen=True)
class AdaptationCandidateEvidence:
    action: AdaptationAction
    proposal_gain: float
    replay_accuracy_before: float
    replay_accuracy_after: float
    ood_unknown_recall_before: float
    ood_unknown_recall_after: float
    transaction_validated: bool
    target_class_id: Any | None = None

    def __post_init__(self) -> None:
        values = (
            self.proposal_gain,
            self.replay_accuracy_before,
            self.replay_accuracy_after,
            self.ood_unknown_recall_before,
            self.ood_unknown_recall_after,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("adaptation evidence must be finite.")
        rates = values[1:]
        if any(value < 0.0 or value > 1.0 for value in rates):
            raise ValueError("accuracy and recall evidence must be in [0, 1].")
        if self.action == AdaptationAction.QUARANTINE:
            raise ValueError("quarantine is a decision, not a mutation candidate.")
        if (
            self.action == AdaptationAction.UPDATE_EXISTING
            and self.target_class_id is None
        ):
            raise ValueError("existing-class updates require a target class.")
        if self.action == AdaptationAction.CREATE_NEW and self.target_class_id is not None:
            raise ValueError("new-class candidates cannot name an existing target.")


@dataclass(frozen=True)
class AdaptationGatePolicy:
    minimum_proposal_gain: float
    maximum_replay_accuracy_drop: float
    maximum_ood_recall_drop: float

    def __post_init__(self) -> None:
        if self.minimum_proposal_gain < 0.0:
            raise ValueError("minimum_proposal_gain must be non-negative.")
        if self.maximum_replay_accuracy_drop < 0.0:
            raise ValueError("maximum_replay_accuracy_drop must be non-negative.")
        if self.maximum_ood_recall_drop < 0.0:
            raise ValueError("maximum_ood_recall_drop must be non-negative.")


@dataclass(frozen=True)
class AdaptationDecision:
    action: AdaptationAction
    target_class_id: Any | None
    confirmation: ConfirmationKind | None
    failed_gates: tuple[str, ...]


def select_adaptation_action(
    candidates: tuple[AdaptationCandidateEvidence, ...],
    *,
    confirmation: ConfirmationKind | None,
    policy: AdaptationGatePolicy,
) -> AdaptationDecision:
    """Select an action from frozen evidence; this function never mutates a model."""
    if confirmation is None:
        return AdaptationDecision(
            AdaptationAction.QUARANTINE,
            None,
            None,
            ("confirmation_required",),
        )
    required_action = (
        AdaptationAction.UPDATE_EXISTING
        if confirmation == ConfirmationKind.EXISTING_CLASS
        else AdaptationAction.CREATE_NEW
    )
    matching = [candidate for candidate in candidates if candidate.action == required_action]
    if len(matching) != 1:
        return AdaptationDecision(
            AdaptationAction.QUARANTINE,
            None,
            confirmation,
            ("candidate_unavailable",),
        )
    candidate = matching[0]
    failed = []
    if candidate.proposal_gain < policy.minimum_proposal_gain:
        failed.append("insufficient_proposal_gain")
    if (
        candidate.replay_accuracy_before - candidate.replay_accuracy_after
        > policy.maximum_replay_accuracy_drop
    ):
        failed.append("replay_regression")
    if (
        candidate.ood_unknown_recall_before - candidate.ood_unknown_recall_after
        > policy.maximum_ood_recall_drop
    ):
        failed.append("ood_regression")
    if not candidate.transaction_validated:
        failed.append("transaction_validation_failed")
    if failed:
        return AdaptationDecision(
            AdaptationAction.QUARANTINE,
            None,
            confirmation,
            tuple(failed),
        )
    return AdaptationDecision(
        candidate.action,
        candidate.target_class_id,
        confirmation,
        (),
    )