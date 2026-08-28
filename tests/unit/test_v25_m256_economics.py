"""Unit tests for M256 cells 1, 3, 4: stake sizing, the
attacker-payoff cap, and the free-rider equilibrium estimate.
"""
from __future__ import annotations

import unittest

from geode.attribution.incentives import free_rider_report
from geode.attribution.payoff_cap import (
    capped_session_value,
    capture_window_value,
    capture_worth_budget,
)
from geode.attribution.stake import (
    MeasurementClass,
    minimum_bond,
    simulate_liar,
    stake_schedule,
)


class TestM256StakeSizing(unittest.TestCase):

    def test_closed_form_bound(self):
        cls = MeasurementClass(name="accuracy", gain_from_lie=10.0,
                               detection_prob=0.5)
        self.assertAlmostEqual(minimum_bond(cls), 20.0)

    def test_safety_margin_scales(self):
        cls = MeasurementClass(name="accuracy", gain_from_lie=10.0,
                               detection_prob=0.5)
        self.assertAlmostEqual(minimum_bond(cls, safety_margin=2.0),
                               40.0)

    def test_schedule_covers_every_class(self):
        classes = [MeasurementClass(name="a", gain_from_lie=5.0,
                                    detection_prob=0.25),
                   MeasurementClass(name="b", gain_from_lie=2.0,
                                    detection_prob=0.5)]
        schedule = stake_schedule(classes)
        self.assertEqual(schedule, {"a": 20.0, "b": 4.0})

    def test_liar_unprofitable_under_the_bond(self):
        cls = MeasurementClass(name="accuracy", gain_from_lie=10.0,
                               detection_prob=0.5)
        bond = minimum_bond(cls)
        out = simulate_liar(cls, bond, rounds=2000, fee=0.1)
        self.assertTrue(out["liar_unprofitable"])
        self.assertGreater(out["honest_cash"], 0.0)

    def test_invalid_probability_raises(self):
        cls = MeasurementClass(name="x", gain_from_lie=1.0,
                               detection_prob=0.0)
        with self.assertRaises(ValueError):
            minimum_bond(cls)


class TestM256PayoffCap(unittest.TestCase):

    def test_cap_bounds_session_value(self):
        self.assertEqual(capped_session_value(100.0, 40.0), 40.0)
        self.assertEqual(capped_session_value(30.0, 40.0), 30.0)

    def test_window_value_scales_with_sessions(self):
        self.assertEqual(capture_window_value(40.0, 5), 200.0)

    def test_capture_unprofitable_gate(self):
        out = capture_worth_budget(200.0, 1000.0)
        self.assertTrue(out["unprofitable"])
        out2 = capture_worth_budget(1200.0, 1000.0)
        self.assertFalse(out2["unprofitable"])

    def test_negative_inputs_raise(self):
        with self.assertRaises(ValueError):
            capped_session_value(-1.0, 10.0)


class TestM256FreeRider(unittest.TestCase):

    def test_free_rider_pays_near_zero_and_costs_near_zero(self):
        report = free_rider_report(coop_count=10, free_count=5,
                                   contribution=2.0, cost=0.5,
                                   rounds=100, demand=10.0, lag=2,
                                   seed=11)
        # the registered reading: the free-rider earns ZERO (its
        # measured contribution is zero, so its share is zero) and
        # pays ZERO — the incentive gap is that riding costs nothing
        self.assertAlmostEqual(report["free_rider_mean_cash"], 0.0,
                               delta=1e-9)
        self.assertAlmostEqual(
            report["free_rider_advantage"],
            -report["coop_mean_cash"], delta=1e-6)

    def test_lost_progress_fraction_matches_counterfactual(self):
        report = free_rider_report(coop_count=10, free_count=10,
                                   contribution=2.0, cost=0.5,
                                   rounds=10, demand=10.0, lag=2,
                                   seed=11)
        self.assertAlmostEqual(report["lost_progress_fraction"], 0.5,
                               delta=1e-6)

    def test_deterministic(self):
        a = free_rider_report(10, 5, 2.0, 0.5, 100, 10.0, 2, 11)
        b = free_rider_report(10, 5, 2.0, 0.5, 100, 10.0, 2, 11)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
