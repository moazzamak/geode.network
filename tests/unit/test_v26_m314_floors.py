"""Unit tests for the v26 M314 security floors.

The floors sit outside ordinary governance (like the zakat rule):
every registered floor has a name and a value, the guard admits values
at or above the floor, rejects values below it, and refuses
unregistered parameter names so a new adjustable knob cannot slip past
the floor mechanism unnoticed.
"""
from __future__ import annotations

import unittest

from geode.core.economics import (
    SECURITY_FLOORS,
    assert_at_or_above_floor,
)


class TestSecurityFloors(unittest.TestCase):

    def test_registered_floors_cover_the_named_parameters(self):
        # the whitepaper names five floored parameters; all must be
        # present, or an adjustment path has nowhere to call the guard.
        for name in ("shadow_probe_rate", "vesting_epochs",
                     "admission_validator_sample",
                     "reference_executor_sample", "audit_fraction"):
            self.assertIn(name, SECURITY_FLOORS)
            self.assertGreater(float(SECURITY_FLOORS[name]), 0.0)

    def test_guard_admits_at_and_above_floor(self):
        self.assertEqual(
            assert_at_or_above_floor("shadow_probe_rate", 0.05), 0.05)
        self.assertEqual(
            assert_at_or_above_floor("shadow_probe_rate", 0.10), 0.10)
        self.assertEqual(
            assert_at_or_above_floor("vesting_epochs", 8), 8)

    def test_guard_rejects_below_floor(self):
        with self.assertRaises(ValueError):
            assert_at_or_above_floor("shadow_probe_rate", 0.0)
        with self.assertRaises(ValueError):
            assert_at_or_above_floor("shadow_probe_rate", 0.0499)
        with self.assertRaises(ValueError):
            assert_at_or_above_floor("vesting_epochs", 3)

    def test_guard_refuses_unregistered_parameters(self):
        # an adjustable knob that is not registered must fail loudly,
        # not bypass the floor mechanism.
        with self.assertRaises(KeyError):
            assert_at_or_above_floor("max_batch", 128)


if __name__ == "__main__":
    unittest.main()
