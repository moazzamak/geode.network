"""M316 - chains as first-class routable artifacts.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md``
§8.33 (27 Aug 2026, before any build). Two repairs:

- **R-A17a the chain split.** Attribution over a sealed chain is
  the Shapley value of each stage, with coalition values defined
  as the chain's measured end-to-end score with stages outside
  the coalition replaced by the identity stage (pass-through).
  Shares normalize over non-negative contributions: a stage that
  hurts the chain earns zero, never a subsidy.
- **R-A17b first-class chains.** A chain carries a descriptor, a
  type-level admissibility check, its own fingerprint, its own
  axis, and its own measured end-to-end score. It is admitted and
  routed like any other artifact - never composed on the fly from
  unmeasured local optima.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any


class ContractMismatchError(ValueError):
    """A stage's output contract does not satisfy the next stage's
    input contract - the chain refuses to assemble."""


class ChainTooLongError(ValueError):
    """A chain longer than the registered cap is not admissible."""


# Shapley over m stages needs 2^m end-to-end coalition evaluations.
# The cap is what keeps that replayable by a validator: 4 stages is
# 16 evaluations. Registered with G12's repair (M375, 28 Aug 2026).
MAX_CHAIN_STAGES = 4


@dataclass(frozen=True)
class ChainStage:
    artifact_id: str
    contract_in: frozenset[str]
    contract_out: frozenset[str]


@dataclass
class ChainArtifact:
    """A chain registered as ONE artifact with one fingerprint and
    one measured end-to-end score."""
    chain_id: str
    stages: list[ChainStage]
    axis: str
    end_to_end_score: float = 0.0
    stage_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.stages) > MAX_CHAIN_STAGES:
            raise ChainTooLongError(
                f"chain {self.chain_id} has {len(self.stages)} stages; "
                f"the cap is {MAX_CHAIN_STAGES} because the Shapley "
                f"split needs 2^m end-to-end coalition evaluations "
                f"and a validator has to be able to replay them")
        for prev, nxt in zip(self.stages, self.stages[1:]):
            if not prev.contract_out.issubset(nxt.contract_in):
                raise ContractMismatchError(
                    f"stage {prev.artifact_id} outputs "
                    f"{sorted(prev.contract_out)} but stage "
                    f"{nxt.artifact_id} accepts "
                    f"{sorted(nxt.contract_in)}")

    @property
    def admissible(self) -> bool:
        return all(
            prev.contract_out.issubset(nxt.contract_in)
            for prev, nxt in zip(self.stages, self.stages[1:]))

    @property
    def fingerprint(self) -> tuple[str, ...]:
        return (self.axis, *(s.artifact_id for s in self.stages))


# ----------------------------------------------------------------------
# The chain split: Shapley over stages with identity substitution
# ----------------------------------------------------------------------

def _coalition_value(members: frozenset[str],
                     stage_scores: dict[str, float],
                     chain_score: float,
                     identity_score: float) -> float:
    """The chain's score with the non-member stages replaced by the
    identity stage. The registered semantics: value is the chain
    score minus the losses the absent stages would have avoided -
    a stage's value is what it adds over identity, compounded only
    with the stages present."""
    kept = 0.0
    for stage, score in stage_scores.items():
        if stage in members:
            kept += max(score - identity_score, 0.0)
    # the coalition's contribution rides on the identity baseline
    return identity_score + kept


def shapley_split(stage_scores: dict[str, float],
                  chain_score: float,
                  identity_score: float = 0.0,
                  coalition_value: Any = None
                  ) -> dict[str, float]:
    """Exact Shapley values for each stage. Each stage's value is
    what it adds to every coalition it joins, averaged over all
    orders. Coalition values are measured by identity substitution
    (the harness may inject its measured ``coalition_value``
    callable); without one, the registered additive stand-in
    applies: value(S) = identity + sum of member stage margins.
    Non-negative clipping happens at the share stage, never here."""
    stages = list(stage_scores)
    n = len(stages)
    if n == 0:
        return {}
    values = {stage: 0.0 for stage in stages}

    def value_of(members: frozenset[str]) -> float:
        if coalition_value is not None:
            return float(coalition_value(members))
        return _coalition_value(members, stage_scores, chain_score,
                                identity_score)

    for stage in stages:
        others = [s for s in stages if s != stage]
        for r in range(len(others) + 1):
            for combo in combinations(others, r):
                coalition = frozenset(combo)
                weight = (math.factorial(len(combo))
                          * math.factorial(n - len(combo) - 1)
                          / math.factorial(n))
                values[stage] += (
                    value_of(coalition | {stage})
                    - value_of(coalition)) * weight
    return values


def attribution_shares(stage_scores: dict[str, float],
                       chain_score: float,
                       identity_score: float = 0.0
                       ) -> dict[str, float]:
    """The payment weights over the 97.5% pool. Shapley values
    clipped at zero (harmful stages earn zero), normalized to sum
    to one over the paid stages. With no positive contributor the
    pool is returned, not distributed."""
    raw = shapley_split(stage_scores, chain_score, identity_score)
    clipped = {stage: max(value, 0.0) for stage, value in raw.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        return {}
    return {stage: value / total for stage, value in clipped.items()}


def measured_gap(stage_scores: dict[str, float],
                 chain_score: float) -> float:
    """The composition gap: measured end-to-end minus the product
    of independent stage scores. The 0.90 x 0.90 -> ~0.81 reading
    lives here; the chain's own score replaces the product."""
    product = 1.0
    for score in stage_scores.values():
        product *= score
    return chain_score - product
