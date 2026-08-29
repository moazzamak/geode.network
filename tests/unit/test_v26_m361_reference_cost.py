"""M361 (G22) — reference hosting cost as a multi-party statistic.
Registered 29 Aug 2026 before the build. Pins: the reference cost is
the median of admitted arms' posted costs with >= 1 epoch verified
traffic; the developer's figure is the floor only below 3 such arms;
the registry publishes which rule is operative."""
from __future__ import annotations

import pytest

from geode.core.economics import (
    REFERENCE_COST_MIN_ARMS,
    reference_hosting_cost,
)


def test_developer_floor_below_three_arms():
    # fewer than 3 verified arms: the developer's figure is the floor
    assert reference_hosting_cost([], 50.0)["reference"] == 50.0
    assert reference_hosting_cost([40.0], 50.0)["reference"] == 50.0
    assert reference_hosting_cost([40.0, 30.0],
                                  50.0)["reference"] == 50.0
    assert reference_hosting_cost(
        [40.0, 30.0], 50.0)["rule"] == "developer floor (<min_arms)"


def test_median_after_three_arms():
    costs = [40.0, 30.0, 60.0]
    out = reference_hosting_cost(costs, 50.0)
    assert out["reference"] == 40.0          # median of the three
    assert out["rule"] == "median of admitted"
    assert out["admitted_arms"] == 3


def test_even_median_is_midpoint():
    out = reference_hosting_cost([10.0, 20.0, 30.0, 40.0], 5.0)
    assert out["reference"] == 25.0


def test_outliers_do_not_move_the_median():
    # one extreme developer-side cost cannot shift a 5-arm median
    out = reference_hosting_cost([10.0, 11.0, 12.0, 13.0, 1000.0],
                                 1000.0)
    assert out["reference"] == 12.0


def test_min_arms_registered():
    assert REFERENCE_COST_MIN_ARMS == 3


def test_invalid_inputs():
    with pytest.raises(ValueError):
        reference_hosting_cost([-1.0], 5.0)
    with pytest.raises(ValueError):
        reference_hosting_cost([1.0], -5.0)
    with pytest.raises(ValueError):
        reference_hosting_cost([1.0], 5.0, min_arms=0)
