"""M334 - the adaptive composite-campaign harness (the real H26-8).

Registered in ``analysis/FEASIBILITY_THREAT_REVIEW_2026-08-28.md``
(F6, R-F6). The M321 harness is structural: each row names the
attack and the repair module, and the closures are mostly
"module shipped". F6 sharpened it: H26-8 was registered as the
instrument that tests the severity ordering and the composite
attack, and what sealed was an attribution table, not a
simulation of an adaptive adversary chaining the steps.

This module is the adaptive instrument. The adversary runs an
EPISODE against the live module stack:

- the state is the remaining budget and the set of un-attempted
  steps;
- each step has a registered attempt cost and a success payoff
  (the registered campaign economics);
- the step's outcome is decided by the LIVE closure: when the
  repair module reports the step closed, the attack fails and the
  cost is paid with zero payoff - and the failure is attributed
  to the module that produced it;
- the adversary chooses the next step greedily by expected
  payoff/cost ratio from the OBSERVED state (a failed step is
  never re-chosen: the closure is deterministic), and stops when
  the budget is exhausted or no step remains.

The registered economics: axis value 24,000 (demand 1000/epoch,
price 1.0, horizon 24 - the M293 scenario numbers); each step's
payoff is the share of the axis its success would capture; each
attempt cost is the registered attack cost (fees, deposits,
compute at the registered rates).

Gate (registered before the run): the adversary's expected profit
is negative at the registered parameters, and every failed step
is attributed to the module that produced the failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from geode.core.composite_campaign import build_campaign

# the registered campaign economics (the M293 scenario scale)
AXIS_VALUE = 24_000.0


@dataclass(frozen=True)
class AdaptiveStep:
    """One chainable campaign step with its registered economics."""
    name: str
    cost: float
    payoff: float
    closure: Callable[[], bool]
    module: str

    @property
    def ratio(self) -> float:
        return self.payoff / self.cost if self.cost > 0.0 \
            else float("inf")


@dataclass
class EpisodeTrace:
    """One adversary episode: the chosen sequence, each attempt's
    outcome, and the final profit."""
    steps: list[dict[str, Any]] = field(default_factory=list)
    final_budget: float = 0.0

    @property
    def profit(self) -> float:
        """The episode profit: payoffs on successes, costs on
        failures (the initial budget is the working capital, not
        profit)."""
        return sum(s["payoff"] if s["attacker_succeeded"]
                   else -s["cost"] for s in self.steps)

    def attributions(self) -> dict[str, str]:
        return {s["step"]: s["module"]
                for s in self.steps if not s["attacker_succeeded"]}


def _adaptive_steps() -> list[AdaptiveStep]:
    """The registered step set: the M321 rows with registered
    attempt costs and success payoffs at the campaign scale."""
    campaign = build_campaign()
    registered = [
        # (step, cost, payoff share of the axis value)
        ("validator_fleet", 50.0, 0.9),      # capture routing quorum
        ("shard_purchase", 500.0, 0.5),      # read the corpus
        ("pool_infiltration", 25.0, 0.9),    # take the weights
        ("flip_re_register", 10.0, 0.9),     # near-copy re-admit
        ("traffic_or_takedown", 200.0, 0.9), # force the axis
        ("substitute", 30.0, 0.7),           # serve a knockoff
        ("meter_inflation", 20.0, 0.4),      # bill more tokens
        ("abstention_suppression", 15.0, 0.3),  # hide abstention
        ("claim_every_epoch", 5.0, 0.8),     # drain before burns
        ("commit_and_abort", 5.0, 0.6),      # dodge probes
        ("chain_attribution", 5.0, 0.5),     # undefined splits
    ]
    by_name = {row.step: row for row in campaign.rows}
    # the campaign rows are keyed by step number; map names to rows
    row_by_step = {row.step: row for row in campaign.rows}
    steps: list[AdaptiveStep] = []
    for idx, (name, cost, share) in enumerate(registered, start=1):
        row = row_by_step[idx]
        steps.append(AdaptiveStep(
            name=name, cost=cost, payoff=AXIS_VALUE * share,
            closure=row.closure, module=row.module))
    return steps


def run_adaptive_episode(budget: float,
                         seed: int = 20260828,
                         ) -> EpisodeTrace:
    """One adaptive episode against the live stack. The adversary
    re-plans from observed state after every attempt; deterministic
    under the seed (ties broken by seed order)."""
    import random
    rng = random.Random(seed)
    steps = _adaptive_steps()
    trace = EpisodeTrace(final_budget=budget)
    remaining: list[AdaptiveStep] = list(steps)
    while remaining and trace.final_budget > 0.0:
        # observed-state choice: greedy by expected payoff/cost,
        # seeded tie-break among equal ratios
        remaining.sort(key=lambda s: (s.ratio, rng.random()))
        chosen = remaining.pop()   # the best expected ratio
        if chosen.cost > trace.final_budget:
            continue               # cannot afford it; it is skipped
        closed = bool(chosen.closure())
        if closed:
            # the attack fails against the shipped repair: the
            # attempt cost is paid, the failure attributed
            trace.steps.append({
                "step": chosen.name, "cost": chosen.cost,
                "payoff": chosen.payoff,
                "attacker_succeeded": False,
                "module": chosen.module,
            })
            trace.final_budget -= chosen.cost
        else:
            trace.steps.append({
                "step": chosen.name, "cost": chosen.cost,
                "payoff": chosen.payoff,
                "attacker_succeeded": True,
                "module": chosen.module,
            })
            trace.final_budget += chosen.payoff - chosen.cost
    return trace


def adaptive_campaign_gate(trace: EpisodeTrace) -> dict[str, Any]:
    """H26-8 in its adaptive form: the adversary's realized profit
    is negative, and every failed step is attributed to the module
    that produced the failure."""
    succeeded = [s for s in trace.steps if s["attacker_succeeded"]]
    failed = [s for s in trace.steps if not s["attacker_succeeded"]]
    attributable = bool(failed) and all(
        s.get("module") for s in failed)
    return {
        "steps_attempted": len(trace.steps),
        "succeeded": [s["step"] for s in succeeded],
        "failed": [s["step"] for s in failed],
        "profit": trace.profit,
        "profit_negative": bool(trace.profit < 0.0),
        "attributions": trace.attributions(),
        "h26_8_adaptive": bool(trace.profit < 0.0 and attributable),
    }
