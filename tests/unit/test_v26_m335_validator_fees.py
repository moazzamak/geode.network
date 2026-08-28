"""Unit tests for the M335 validator fee-schedule machinery
(R-F7, registered 28 Aug 2026, before the build). Pins: the
break-even fee; the recovery-fraction arithmetic and its k-cancel
property; the sybil-safety ceiling; the admissible-window verdict
and its margin; the no-window verdict (stake-like addition
required)."""
from __future__ import annotations

import pytest

from geode.core.validator_fees import (
    break_even_fee,
    fee_schedule_verdict,
    sybil_recovery_fraction,
    sybil_safety_ceiling,
)


def test_break_even_fee():
    assert break_even_fee(0.01) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        break_even_fee(0.0)


def test_recovery_fraction_k_cancels():
    # the fraction is per-identity: fleet size never enters
    assert sybil_recovery_fraction(0.01, 50.0, 8, 10.0) \
        == pytest.approx(0.4)
    # the same fraction for any k
    assert sybil_recovery_fraction(0.025, 50.0, 8, 10.0) \
        == pytest.approx(1.0)


def test_ceiling_is_the_unit_fraction_fee():
    ceiling = sybil_safety_ceiling(50.0, 8, 10.0)
    assert ceiling == pytest.approx(0.025)
    assert sybil_recovery_fraction(ceiling, 50.0, 8, 10.0) \
        == pytest.approx(1.0)


def test_validation():
    with pytest.raises(ValueError):
        sybil_recovery_fraction(-0.1, 50.0, 8, 10.0)
    with pytest.raises(ValueError):
        sybil_recovery_fraction(0.01, 50.0, 0, 10.0)
    with pytest.raises(ValueError):
        sybil_safety_ceiling(0.0, 8, 10.0)


def test_window_exists_verdict():
    verdict = fee_schedule_verdict(
        cost_per_challenge=0.01, challenges_per_epoch=50.0,
        horizon_epochs=8, registration_fee=10.0,
        fee_ladder=[0.005, 0.01, 0.02, 0.025, 0.03, 0.05])
    assert verdict["window_exists"]
    assert verdict["break_even_fee"] == pytest.approx(0.01)
    assert verdict["sybil_safety_ceiling"] == pytest.approx(0.025)
    assert verdict["margin_factor"] == pytest.approx(2.5)
    # 0.02 sits inside the window; 0.01 is the boundary (fee must
    # EXCEED cost to pay the validator), 0.025 is the ceiling
    assert verdict["admissible_ladder_points"] == [0.02]
    assert verdict["recovery_fractions"]["0.01"] == pytest.approx(0.4)
    assert verdict["recovery_fractions"]["0.03"] == pytest.approx(1.2)


def test_no_window_verdict_requires_stake_like_addition():
    # a validator cost ABOVE the ceiling: no fee works
    verdict = fee_schedule_verdict(
        cost_per_challenge=0.1, challenges_per_epoch=50.0,
        horizon_epochs=8, registration_fee=10.0,
        fee_ladder=[0.05, 0.1, 0.2])
    assert not verdict["window_exists"]
    assert verdict["margin_factor"] == 0.0
    assert verdict["admissible_ladder_points"] == []
    assert "stake-like addition" in verdict["verdict"]
