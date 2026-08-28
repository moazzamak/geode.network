"""Unit tests for M293: the H-series economic simulation battery.

Pure stdlib + numpy; deterministic seeds. Covers the registered
scenario gates: A (copycat race: momentum, copying unprofitable,
coverage, incumbent control), B (detection-horizon sweep sets N),
C (bootstrap headroom handover).
"""
from __future__ import annotations

import unittest

from geode.attribution.hseries import (
    bootstrap_run,
    copycat_race_cell,
    copycat_race_sweep,
    detection_horizon_class,
    detection_horizon_sweep,
)


class TestCopycatRace(unittest.TestCase):

    def test_equal_price_keeps_publisher(self):
        cell = copycat_race_cell(demand=100.0, price=1.0, epochs=10,
                                 quality=0.9, copycat_epoch=2,
                                 undercut=0.0, serving_cost=20.0,
                                 seed=1)
        self.assertTrue(cell["gate_a1_share"])
        self.assertEqual(cell["publisher_fee_share"], 1.0)
        # Equal price: the earlier registration serves every epoch.
        self.assertEqual(cell["copycat_traffic_share"], 0.0)
        self.assertEqual(cell["publisher_traffic_share"], 1.0)

    def test_undercut_steals_traffic_but_never_attribution(self):
        cell = copycat_race_cell(demand=100.0, price=1.0, epochs=10,
                                 quality=0.9, copycat_epoch=2,
                                 undercut=0.2, serving_cost=20.0,
                                 seed=1)
        self.assertEqual(cell["copycat_fees"], 0.0)      # marginal = 0
        # Serves every epoch it is present (2..9 = 8 of 10 epochs).
        self.assertAlmostEqual(cell["copycat_traffic_share"], 0.8)
        self.assertTrue(cell["gate_a2_copycat_net"])
        self.assertLess(cell["copycat_net"], 0.0)        # pays serving
        self.assertTrue(cell["gate_a3_served"])

    def test_publisher_absent_copycat_is_incumbent(self):
        cell = copycat_race_cell(demand=100.0, price=1.0, epochs=10,
                                 quality=0.9, copycat_epoch=-1,
                                 undercut=0.0, serving_cost=20.0,
                                 seed=1)
        self.assertTrue(cell["publisher_absent"])
        self.assertTrue(cell["gate_a4_incumbent"])
        self.assertGreater(cell["copycat_net"], 0.0)
        self.assertEqual(cell["copycat_fee_share"], 1.0)

    def test_full_sweep_passes_all_gates(self):
        result = copycat_race_sweep(
            demand=100.0, price=1.0, epochs=12, quality=0.9,
            copycat_epochs=[0, 1, 4, 8, -1], undercuts=[0.0, 0.1, 0.3],
            serving_cost=20.0, seed=5)
        self.assertTrue(result["passes"], result["gates"])
        # The undercut cells dent the publisher's stream but never
        # below the registered sweep depth.
        self.assertLess(result["worst_publisher_fees_vs_no_copycat"], 1.0)

    def test_bad_undercut_raises(self):
        with self.assertRaises(ValueError):
            copycat_race_cell(demand=100.0, price=1.0, epochs=10,
                              quality=0.9, copycat_epoch=2,
                              undercut=1.0, serving_cost=20.0, seed=1)


class TestDetectionHorizon(unittest.TestCase):

    def test_serving_deviation_horizon_is_sub_epoch(self):
        import numpy as np
        rng = np.random.default_rng(1)
        cell = detection_horizon_class(
            "B1_serving_deviation", rng, draws=500, probe_rate=0.05,
            epoch_volume=1000.0, gaming_rate=0.5, ring_rate=0.5,
            health_probes=10, health_hit_rate=0.9)
        self.assertLess(cell["p90"], 1.0)

    def test_gaming_horizon_p90_above_one_epoch(self):
        import numpy as np
        rng = np.random.default_rng(1)
        cell = detection_horizon_class(
            "B2_attribution_gaming", rng, draws=500, probe_rate=0.05,
            epoch_volume=1000.0, gaming_rate=0.5, ring_rate=0.5,
            health_probes=10, health_hit_rate=0.9)
        self.assertGreater(cell["p90"], 1.0)
        self.assertEqual(cell["median"], 1.0)  # geometric(0.5)

    def test_window_four_passes_window_one_fails(self):
        common = dict(draws=500, probe_rate=0.05, epoch_volume=1000.0,
                      gaming_rate=0.5, ring_rate=0.5, health_probes=10,
                      health_hit_rate=0.9)
        passes = detection_horizon_sweep(**common, vesting_window=4,
                                          seed=7)
        self.assertTrue(passes["passes"], passes["per_class"])
        self.assertIn(passes["binding_class"],
                      ("B2_attribution_gaming", "B3_wash_ring"))
        self.assertEqual(passes["binding_p90"],
                         max(v["p90_horizon"]
                             for v in passes["per_class"].values()))
        fails = detection_horizon_sweep(**common, vesting_window=1,
                                         seed=7)
        self.assertFalse(fails["passes"])

    def test_unknown_class_raises(self):
        import numpy as np
        with self.assertRaises(ValueError):
            detection_horizon_class(
                "B9_made_up", np.random.default_rng(1), draws=10,
                probe_rate=0.05, epoch_volume=1000.0, gaming_rate=0.5,
                ring_rate=0.5, health_probes=10, health_hit_rate=0.9)


class TestBootstrapDynamics(unittest.TestCase):

    def _run(self):
        return bootstrap_run(
            demand=100.0, price=1.0, epochs=24, bootstrap_quality=0.8,
            arrivals={3: 0.75, 6: 0.8, 9: 0.85}, fallback_share=0.02,
            vesting_window=4, seed=3)

    def test_handover_by_measurement_alone(self):
        result = self._run()
        self.assertTrue(result["passes"], result["gates"])
        self.assertEqual(result["handover_epoch"], 9)
        self.assertEqual(result["contributor_serving"], "contrib_9")

    def test_equal_quality_keeps_bootstrap(self):
        result = self._run()
        for entry in result["log"]:
            if 6 <= entry["epoch"] <= 8:
                self.assertEqual(entry["winner"], "bootstrap")
                self.assertEqual(entry["bootstrap_traffic_share"], 1.0)

    def test_after_handover_bootstrap_serves_only_fallback(self):
        result = self._run()
        for entry in result["log"]:
            if entry["epoch"] > 9:
                self.assertAlmostEqual(entry["bootstrap_traffic_share"],
                                       0.02, places=12)

    def test_no_handover_when_nobody_beats_the_bar(self):
        result = bootstrap_run(
            demand=100.0, price=1.0, epochs=12, bootstrap_quality=0.8,
            arrivals={3: 0.75, 6: 0.8}, fallback_share=0.02,
            vesting_window=4, seed=3)
        self.assertIsNone(result["handover_epoch"])
        self.assertFalse(result["gates"]["C1_handover"])
        self.assertTrue(result["gates"]["C3_equal_keeps"])

    def test_bad_fallback_share_raises(self):
        with self.assertRaises(ValueError):
            bootstrap_run(demand=100.0, price=1.0, epochs=10,
                          bootstrap_quality=0.8, arrivals={3: 0.9},
                          fallback_share=0.0, vesting_window=4, seed=3)


if __name__ == "__main__":
    unittest.main()
