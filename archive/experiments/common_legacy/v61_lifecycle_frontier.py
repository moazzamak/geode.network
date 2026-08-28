from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


AXIS_DIRECTIONS = {
    "balanced_accuracy": "higher",
    "unaffected_prediction_preservation": "higher",
    "rollback_reliability": "higher",
    "accepted_edit_evidence_count": "lower",
    "edit_latency_seconds": "lower",
    "inference_latency_seconds": "lower",
}


def complete_frontier_point(record: Mapping[str, Any]) -> bool:
    return all(record.get(axis) is not None for axis in AXIS_DIRECTIONS)


def dominates(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    if not complete_frontier_point(first) or not complete_frontier_point(second):
        raise ValueError("Dominance requires complete frontier points.")
    no_worse = True
    strictly_better = False
    for axis, direction in AXIS_DIRECTIONS.items():
        left = float(first[axis])
        right = float(second[axis])
        if direction == "higher":
            no_worse &= left >= right
            strictly_better |= left > right
        else:
            no_worse &= left <= right
            strictly_better |= left < right
    return bool(no_worse and strictly_better)


def non_dominated_models(records: Mapping[str, Mapping[str, Any]]) -> list[str]:
    complete = {
        name: record for name, record in records.items() if complete_frontier_point(record)
    }
    return sorted(
        name
        for name, record in complete.items()
        if not any(
            other_name != name and dominates(other, record)
            for other_name, other in complete.items()
        )
    )


def classify_outcome_c(
    records: Mapping[str, Mapping[str, Any]],
    *,
    retained_model: str,
    exact_rollback_every_seed_and_task: bool,
    locality_contract_passed: bool,
    predictive_deficit_reported: bool,
    paired_advantage_controls: Sequence[str],
) -> dict[str, Any]:
    retained = records[retained_model]
    frontier = non_dominated_models(records)
    superior = sorted(
        name
        for name, record in records.items()
        if name != retained_model
        and record.get("balanced_accuracy") is not None
        and float(record["balanced_accuracy"])
        > float(retained["balanced_accuracy"])
    )
    complete_superior = sorted(
        name for name in superior if complete_frontier_point(records[name])
    )
    unsupported_superior = sorted(set(superior) - set(complete_superior))
    advantage = set(complete_superior).issubset(paired_advantage_controls)
    comparative_evidence_complete = not unsupported_superior
    passed = (
        retained_model in frontier
        and comparative_evidence_complete
        and advantage
        and exact_rollback_every_seed_and_task
        and locality_contract_passed
        and predictive_deficit_reported
    )
    return {
        "retained_non_dominated": retained_model in frontier,
        "non_dominated_models": frontier,
        "accuracy_superior_controls": superior,
        "complete_accuracy_superior_controls": complete_superior,
        "unsupported_accuracy_superior_controls": unsupported_superior,
        "paired_advantage_over_complete_superior_controls": advantage,
        "comparative_evidence_complete": comparative_evidence_complete,
        "exact_rollback_every_seed_and_task": exact_rollback_every_seed_and_task,
        "locality_contract_passed": locality_contract_passed,
        "predictive_deficit_reported": predictive_deficit_reported,
        "specialized_tradeoff_claim_passed": passed,
        "status": (
            "specialized_lifecycle_tradeoff"
            if passed
            else (
                "lifecycle_safety_qualification_only"
                if locality_contract_passed
                else "rollback_qualification_only"
            )
        ),
    }
