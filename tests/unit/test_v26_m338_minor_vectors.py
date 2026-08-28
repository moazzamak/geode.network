"""Unit tests for the M338 minor-vector pass (F10, registered 28
Aug 2026). Pins: the verified-only trailing revenue filter (wash
inflation closed); the public-quantity bond estimator and its
conservatism direction; the zakat rule's frozen charter, refusal
until set, and deterministic split."""
from __future__ import annotations

import pytest

from geode.core.economic_repairs import (
    BOND_SAVING_CONSERVATISM,
    bond_from_publics,
    saving_estimate,
)
from geode.core.takedown_containment import (
    proposer_deposit,
    takedown_deposit_verified,
    verified_trailing_revenue,
)
from geode.core.zakat_rule import (
    ZakatRule,
    ZakatRuleError,
    zakat_trigger_ok,
)


def _sessions() -> list[dict]:
    return [
        # a wash-ring fake session: self-sourced, never verified
        {"source": "self_served", "value": 1000.0},
        # a verified session initiated by another party
        {"source": "probe_reference", "value": 250.0},
        {"source": "probe_reference", "value": 150.0},
        {"source": "sampled_challenge", "value": 0.0},
    ]


def test_verified_trailing_revenue_filters_wash():
    rev = verified_trailing_revenue(_sessions())
    assert rev == pytest.approx(400.0)   # only the verified pair


def test_wash_inflation_closed():
    # the wash ring's self-generated volume does not move the
    # deposit; only externally verified revenue does
    sessions = [{"source": "self_served", "value": 1_000_000.0}]
    assert verified_trailing_revenue(sessions) == 0.0
    assert takedown_deposit_verified(sessions) == 0.0
    assert proposer_deposit(400.0) == pytest.approx(200.0)


def test_negative_values_refused():
    with pytest.raises(ValueError):
        verified_trailing_revenue([{"source": "probe_reference",
                                    "value": -1.0}])


def test_saving_estimate_direction():
    # a substitute saves at most price - cost; the conservatism
    # factor over-states (the defense-pessimistic direction)
    assert saving_estimate(10.0, 4.0) == pytest.approx(
        BOND_SAVING_CONSERVATISM * 6.0)
    # price below cost: no saving to arbitrage
    assert saving_estimate(3.0, 4.0) == 0.0
    with pytest.raises(ValueError):
        saving_estimate(-1.0, 1.0)
    with pytest.raises(ValueError):
        saving_estimate(1.0, 1.0, conservatism=0.5)


def test_bond_from_publics():
    # exposure 100 units, saving estimate 12 (2.0 * (10 - 4))
    bond = bond_from_publics(10.0, 4.0, exposure_units=100.0)
    assert bond == pytest.approx(1200.0)
    # no arbitrage -> zero bond
    assert bond_from_publics(3.0, 4.0, exposure_units=100.0) == 0.0


def test_zakat_rule_frozen_charter():
    rule = ZakatRule(recipients=(("org_a", 0.6), ("org_b", 0.4)))
    assert rule.ready()
    plan = rule.disburse(1000.0)
    assert plan[0] == {"recipient": "org_a", "amount": 600.0}
    assert plan[1] == {"recipient": "org_b", "amount": 400.0}
    # frozen: no mutator field on the dataclass
    assert not hasattr(rule, "add_recipient")
    assert isinstance(rule.recipients, tuple)


def test_zakat_rule_validation():
    with pytest.raises(ZakatRuleError):
        ZakatRule(recipients=(("a", 0.6), ("b", 0.3)))   # sum != 1
    with pytest.raises(ZakatRuleError):
        ZakatRule(recipients=(("a", 0.6), ("a", 0.4)))   # duplicate
    with pytest.raises(ZakatRuleError):
        ZakatRule(recipients=(("a", 0.0), ("b", 1.0)))   # zero share
    with pytest.raises(ZakatRuleError):
        ZakatRule(recipients=(("", 1.0),))


def test_zakat_unset_charter_refuses():
    rule = ZakatRule()
    assert not rule.ready()
    with pytest.raises(ZakatRuleError):
        rule.disburse(1000.0)
    ok = zakat_trigger_ok(1000.0, rule)
    assert not ok["ok"]
    assert "no recipients" in ok["reason"]


def test_zakat_trigger_gate():
    rule = ZakatRule(recipients=(("org_a", 1.0),))
    ok = zakat_trigger_ok(500.0, rule)
    assert ok["ok"]
    assert ok["plan"][0]["amount"] == 500.0
    assert not zakat_trigger_ok(0.0, rule)["ok"]
