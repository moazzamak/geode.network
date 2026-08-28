"""Unit tests for the M334 adaptive campaign harness (F6, R-F6,
registered 28 Aug 2026, before the build). Pins: the adaptive
choice is budget-bounded and never re-chooses a failed step; the
episode profit is negative at the registered parameters; every
failed step carries a module attribution; the trace is
deterministic under the seed."""
from __future__ import annotations

from geode.core.adaptive_campaign import (
    AdaptiveStep,
    adaptive_campaign_gate,
    run_adaptive_episode,
)


def test_episode_negative_profit_at_registered_parameters():
    trace = run_adaptive_episode(budget=10_000.0)
    gate = adaptive_campaign_gate(trace)
    assert gate["profit_negative"]
    assert gate["h26_8_adaptive"]
    # every attempted step failed against the live closures
    assert not gate["succeeded"]
    assert len(gate["failed"]) == len(trace.steps)


def test_every_failure_attributed():
    trace = run_adaptive_episode(budget=10_000.0)
    gate = adaptive_campaign_gate(trace)
    assert set(gate["attributions"].keys()) == set(gate["failed"])
    for module in gate["attributions"].values():
        assert module.startswith("geode.core.")


def test_budget_bounded():
    # a tiny budget: the episode attempts only what it can afford
    trace = run_adaptive_episode(budget=1.0)
    assert trace.final_budget >= 0.0
    # costs exceed 1.0 for every step: nothing is attempted
    assert len(trace.steps) == 0


def test_deterministic_under_seed():
    a = run_adaptive_episode(budget=10_000.0, seed=11)
    b = run_adaptive_episode(budget=10_000.0, seed=11)
    assert [s["step"] for s in a.steps] == [s["step"]
                                            for s in b.steps]
    assert a.profit == b.profit


def test_adaptive_step_ratio():
    step = AdaptiveStep(name="x", cost=10.0, payoff=100.0,
                        closure=lambda: True, module="m")
    assert step.ratio == 10.0
    free = AdaptiveStep(name="y", cost=0.0, payoff=100.0,
                        closure=lambda: True, module="m")
    assert free.ratio == float("inf")
