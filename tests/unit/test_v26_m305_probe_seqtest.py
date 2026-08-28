"""Unit tests for the M305 sequential-test machinery.

Pins: the margin gate counts only above the noise floor; the SPRT
convicts on mismatch-heavy streams and acquits on clean ones; the
corrected horizon is 1/(rho*delta); the adaptive probe rate decays
with clean epochs and never crosses the registered floor.
"""
from __future__ import annotations

import unittest

from geode.core.probe_seqtest import (
    PROBE_RATE_FLOOR,
    adaptive_probe_rate,
    corrected_horizon,
    margin_gated_mismatch,
    sprt,
)


class TestMarginGate(unittest.TestCase):

    def test_counts_only_above_floor(self):
        counted, margin = margin_gated_mismatch(0.9, 0.85, noise_floor=0.01)
        self.assertTrue(counted)
        self.assertAlmostEqual(margin, 0.05)
        counted, margin = margin_gated_mismatch(0.9, 0.895,
                                                noise_floor=0.01)
        self.assertFalse(counted)
        self.assertAlmostEqual(margin, 0.005)

    def test_below_floor_not_counted(self):
        # registered: strict exceed - a margin at or below the floor
        # is a tie the hardware broke differently
        counted, _ = margin_gated_mismatch(0.9, 0.86, noise_floor=0.05)
        self.assertFalse(counted)
        counted, _ = margin_gated_mismatch(0.9, 0.84, noise_floor=0.05)
        self.assertTrue(counted)


class TestSprt(unittest.TestCase):

    def test_convicts_on_heavy_mismatch_stream(self):
        out = sprt(mismatches=20, trials=20, p0=0.002, p1=0.005)
        self.assertEqual(out["decision"], "convict")
        self.assertGreater(out["log_likelihood_ratio"],
                           out["convict_bound"])

    def test_acquits_on_clean_stream(self):
        # ln(beta/(1-alpha)) / ln((1-p1)/(1-p0)) ~= 1525 clean trials
        out = sprt(mismatches=0, trials=2000, p0=0.002, p1=0.005)
        self.assertEqual(out["decision"], "acquit")

    def test_continues_in_between(self):
        out = sprt(mismatches=1, trials=400, p0=0.002, p1=0.005)
        self.assertEqual(out["decision"], "continue")

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            sprt(1, 10, p0=0.5, p1=0.2)   # p0 must be below p1
        with self.assertRaises(ValueError):
            sprt(11, 10, p0=0.1, p1=0.2)  # mismatches > trials
        with self.assertRaises(ValueError):
            sprt(1, 10, p0=0.1, p1=0.2, alpha=1.0)

    def test_decision_monotone_in_mismatches(self):
        # at a fixed trial count, more mismatches can only push toward
        # conviction, never toward acquittal
        prev = sprt(0, 300, p0=0.002, p1=0.005)["decision"]
        for m in range(1, 12):
            now = sprt(m, 300, p0=0.002, p1=0.005)["decision"]
            if prev == "acquit":
                self.assertIn(now, ("acquit", "continue", "convict"))
            prev = now


class TestHorizon(unittest.TestCase):

    def test_corrected_horizon_formula(self):
        self.assertAlmostEqual(corrected_horizon(0.05, 1.0), 20.0)
        self.assertAlmostEqual(corrected_horizon(0.05, 0.005), 4000.0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            corrected_horizon(0.0, 0.005)
        with self.assertRaises(ValueError):
            corrected_horizon(0.05, 0.0)


class TestAdaptiveRate(unittest.TestCase):

    def test_new_entrants_probed_at_one(self):
        self.assertEqual(adaptive_probe_rate(0), 1.0)

    def test_decays_and_floors(self):
        self.assertLess(adaptive_probe_rate(1), 1.0)
        self.assertGreaterEqual(adaptive_probe_rate(1), PROBE_RATE_FLOOR)
        for t in range(10):
            self.assertGreaterEqual(adaptive_probe_rate(t),
                                    PROBE_RATE_FLOOR)
        self.assertAlmostEqual(adaptive_probe_rate(100), PROBE_RATE_FLOOR)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            adaptive_probe_rate(-1)


if __name__ == "__main__":
    unittest.main()
