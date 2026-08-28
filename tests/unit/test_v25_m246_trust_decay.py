"""Unit tests for M246: provenance-weighted trust decay.
"""
from __future__ import annotations

import unittest

from geode.attribution.incentives import (
    trust_weight,
    trust_weighted_shares,
)


class TestM246TrustDecay(unittest.TestCase):

    def test_weight_monotonic_decay(self):
        self.assertAlmostEqual(trust_weight(0), 1.0)
        self.assertAlmostEqual(trust_weight(10), 0.5)
        self.assertAlmostEqual(trust_weight(20), 0.25)
        self.assertLess(trust_weight(11), trust_weight(10))

    def test_fresh_dominates_stale_at_equal_v(self):
        shares = trust_weighted_shares(
            {"fresh": (10.0, 0), "stale": (10.0, 10)})
        self.assertGreater(shares["fresh"], shares["stale"])

    def test_shares_renormalise_to_one(self):
        shares = trust_weighted_shares(
            {"a": (1.0, 0), "b": (3.0, 5), "c": (2.0, 30)})
        self.assertAlmostEqual(sum(shares.values()), 1.0)

    def test_zero_total_yields_zeros(self):
        shares = trust_weighted_shares({"a": (0.0, 0), "b": (0.0, 1)})
        self.assertEqual(shares, {"a": 0.0, "b": 0.0})

    def test_negative_age_raises(self):
        with self.assertRaises(ValueError):
            trust_weight(-1)

    def test_nonpositive_half_life_raises(self):
        with self.assertRaises(ValueError):
            trust_weight(0, half_life=0)


if __name__ == "__main__":
    unittest.main()
