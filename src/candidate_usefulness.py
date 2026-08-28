from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateUsefulnessPolicy:
    minimum_realized_gain: float
    minimum_geometric_coverage: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_realized_gain <= 1.0:
            raise ValueError("minimum_realized_gain must be in [0, 1].")
        if not 0.0 <= self.minimum_geometric_coverage <= 1.0:
            raise ValueError("minimum_geometric_coverage must be in [0, 1].")


@dataclass(frozen=True)
class CandidateUsefulnessDecision:
    eligible: bool
    baseline_success: float
    maximum_possible_gain: float
    geometric_coverage: float
    failed_criteria: tuple[str, ...]


def evaluate_candidate_usefulness(
    *,
    baseline_success: float,
    geometric_coverage: float,
    policy: CandidateUsefulnessPolicy,
) -> CandidateUsefulnessDecision:
    """Reject candidates that cannot meet gain or representation coverage gates."""
    if not math.isfinite(baseline_success) or not 0.0 <= baseline_success <= 1.0:
        raise ValueError("baseline_success must be finite and in [0, 1].")
    if not math.isfinite(geometric_coverage) or not 0.0 <= geometric_coverage <= 1.0:
        raise ValueError("geometric_coverage must be finite and in [0, 1].")
    maximum_possible_gain = 1.0 - baseline_success
    failed = []
    if maximum_possible_gain < policy.minimum_realized_gain:
        failed.append("insufficient_gain_headroom")
    if geometric_coverage < policy.minimum_geometric_coverage:
        failed.append("insufficient_geometric_coverage")
    return CandidateUsefulnessDecision(
        eligible=not failed,
        baseline_success=baseline_success,
        maximum_possible_gain=maximum_possible_gain,
        geometric_coverage=geometric_coverage,
        failed_criteria=tuple(failed),
    )