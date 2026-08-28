"""Unit tests for the M304 expected-units machinery.

Pins: the reference-workload mean is deterministic; empty or
non-positive workloads are rejected; the meter drift is the ratio of
observed mean to the sealed ubar; and the registered drift band is
applied correctly.
"""
from __future__ import annotations

import unittest

from geode.core.economics import (
    DRIFT_BAND,
    drift_in_band,
    meter_drift,
    reference_workload_units,
)


class TestReferenceWorkloadUnits(unittest.TestCase):

    def test_mean_is_deterministic(self):
        units = [12.0, 13.0, 14.0]
        self.assertEqual(reference_workload_units(units), 13.0)
        self.assertEqual(reference_workload_units(units),
                         reference_workload_units(units))

    def test_empty_workload_rejected(self):
        with self.assertRaises(ValueError):
            reference_workload_units([])

    def test_nonpositive_unit_rejected(self):
        with self.assertRaises(ValueError):
            reference_workload_units([1.0, 0.0])
        with self.assertRaises(ValueError):
            reference_workload_units([1.0, -2.0])


class TestMeterDrift(unittest.TestCase):

    def test_drift_is_observed_over_sealed(self):
        self.assertAlmostEqual(meter_drift(20.0, 10.0), 2.0)
        self.assertAlmostEqual(meter_drift(5.0, 10.0), 0.5)

    def test_nonpositive_ubar_rejected(self):
        with self.assertRaises(ValueError):
            meter_drift(10.0, 0.0)

    def test_band_edges_inclusive(self):
        self.assertTrue(drift_in_band(0.5))
        self.assertTrue(drift_in_band(2.0))
        self.assertFalse(drift_in_band(2.01))
        self.assertFalse(drift_in_band(0.49))

    def test_registered_band_is_sane(self):
        lo, hi = DRIFT_BAND
        self.assertGreater(lo, 0.0)
        self.assertGreaterEqual(hi, lo)

    def test_invalid_band_rejected(self):
        with self.assertRaises(ValueError):
            drift_in_band(1.0, band=(0.0, 2.0))
        with self.assertRaises(ValueError):
            drift_in_band(1.0, band=(3.0, 2.0))


if __name__ == "__main__":
    unittest.main()
