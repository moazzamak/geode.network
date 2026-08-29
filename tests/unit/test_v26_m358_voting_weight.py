"""M358 — externally-verified voting weight (review-R2 finding G13)."""
from __future__ import annotations

import pytest

from geode.core.voting_weight import (
    LINKAGE_DEPTH,
    CreditBalances,
    credit_edges_from_sessions,
    is_external,
    linkage_closure,
    mutual_trade_cost,
    split_credits,
    wash_ring_gate,
    weight_snapshot_weights,
)
from geode.privacy.vote_machinery import WeightSnapshot, ratifies


def _ring_sessions(n: int, credit: float = 100.0):
    members = [f"ring-{i}" for i in range(n)]
    return members, [{"payer": members[i],
                      "beneficiary": members[(i + 1) % n],
                      "credit": credit} for i in range(n)]


def test_linkage_closure_terminates_on_a_cycle():
    _, sessions = _ring_sessions(3)
    edges = credit_edges_from_sessions(sessions)
    closure = linkage_closure(edges, "ring-0", LINKAGE_DEPTH)
    assert closure == {"ring-0", "ring-1", "ring-2"}


def test_linkage_closure_respects_depth():
    _, sessions = _ring_sessions(4)
    edges = credit_edges_from_sessions(sessions)
    assert linkage_closure(edges, "ring-0", 1) == {"ring-0", "ring-1"}
    assert linkage_closure(edges, "ring-0", 0) == {"ring-0"}


def test_self_payment_is_the_depth_zero_case():
    edges = credit_edges_from_sessions(
        [{"payer": "a", "beneficiary": "a", "credit": 1.0}])
    assert not is_external("a", "a", edges, 0)


def test_three_cycle_earns_money_but_no_weight():
    members, sessions = _ring_sessions(3)
    edges = credit_edges_from_sessions(sessions)
    for member in members:
        balances = split_credits(sessions, member, edges)
        assert balances.total > 0.0
        assert balances.verified == 0.0


def test_external_payer_earns_full_weight():
    sessions = [{"payer": "buyer", "beneficiary": "honest",
                 "credit": 97.5}]
    edges = credit_edges_from_sessions(sessions)
    balances = split_credits(sessions, "honest", edges)
    assert balances == CreditBalances(total=97.5, verified=97.5)


def test_ring_longer_than_the_depth_is_not_caught():
    # honest boundary: a 5-cycle exceeds depth 3, so it keeps weight.
    # Registered, not hidden — each extra hop costs another dock and
    # another behaviourally distinct artifact.
    members, sessions = _ring_sessions(5)
    edges = credit_edges_from_sessions(sessions)
    assert split_credits(sessions, members[0], edges).verified > 0.0
    deep = split_credits(sessions, members[0], edges, depth=5)
    assert deep.verified == 0.0


@pytest.mark.parametrize("n", [2, 3, 4])
def test_wash_ring_gate_passes(n: int):
    result = wash_ring_gate(n, 100.0)
    assert result["passes"]
    assert result["ring_weight_post_repair"] == 0.0
    assert result["ring_weight_pre_repair"] > 0.0
    assert result["honest_weight"] == result["honest_claimable"]
    assert result["haircut"] == pytest.approx(0.025)


def test_wash_ring_gate_rejects_a_degenerate_ring():
    with pytest.raises(ValueError):
        wash_ring_gate(1, 100.0)


def test_snapshot_is_built_from_the_verified_balance():
    members, ring = _ring_sessions(3)
    honest = [{"payer": "buyer", "beneficiary": "honest",
               "credit": 97.5}]
    sessions = ring + honest
    edges = credit_edges_from_sessions(sessions)
    weights = weight_snapshot_weights(
        sessions, [*members, "honest"], edges)
    snapshot = WeightSnapshot(anchor="epoch-1", weights=weights)
    assert snapshot.total(members) == 0.0
    assert snapshot.weight_of("honest") == 97.5


def test_ring_cannot_ratify_a_vote_it_funded():
    # the d>=3 diversity floor is satisfied by the ring's three
    # distinct behavioural identities; only the weight base stops it.
    members, ring = _ring_sessions(3)
    honest = [{"payer": "buyer", "beneficiary": "honest",
               "credit": 97.5}]
    sessions = ring + honest
    edges = credit_edges_from_sessions(sessions)
    weights = weight_snapshot_weights(
        sessions, [*members, "honest"], edges)
    snapshot = WeightSnapshot(anchor="epoch-1", weights=weights)
    verdict, reason = ratifies(
        support_weight=snapshot.total(members),
        total_weight=snapshot.total([*members, "honest"]),
        supporting_identities=members,
        pool_size=10, responders=4)
    assert not verdict
    assert reason == "below_two_thirds"


def test_mutual_trade_residual_is_reported():
    residual = mutual_trade_cost(100.0)
    assert residual["weight_lost_to_mutual_trade"] > 0.0
    assert residual["claimable"] > residual["voting_weight"]
    assert residual["voting_weight"] > 0.0   # the external buyer counts


def test_negative_credit_is_refused():
    edges = {"a": {"b"}}
    with pytest.raises(ValueError):
        split_credits([{"payer": "x", "beneficiary": "a",
                        "credit": -1.0}], "a", edges)
