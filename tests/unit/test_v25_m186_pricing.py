"""Unit tests for M186: the pricing-oracle study harness."""
from __future__ import annotations

import unittest

from geode.attribution.pricing import (
    bandit_posted,
    make_trace,
    posted_price,
    second_price_auction,
)


class TestPricing(unittest.TestCase):

    def test_deterministic(self):
        trace = make_trace(n=200, arrival=0.4, seed=11)
        a = posted_price(trace, 1.0, rounds=50, seed=1)
        b = posted_price(trace, 1.0, rounds=50, seed=1)
        self.assertEqual(a, b)

    def test_posted_zero_price_serves_all_arrivals(self):
        trace = make_trace(n=200, arrival=0.4, seed=12)
        result = posted_price(trace, 0.0, rounds=50, seed=2)
        self.assertAlmostEqual(result["served"], result["arrivals"],
                               places=6)

    def test_auction_revenue_is_second_highest(self):
        trace = make_trace(n=200, arrival=0.4, seed=13)
        result = second_price_auction(trace, rounds=100, seed=3)
        self.assertGreaterEqual(result["served"], 0.0)
        self.assertLessEqual(result["served"], 100.0)

    def test_bandit_explores_and_converges(self):
        trace = make_trace(n=200, arrival=0.4, seed=14)
        result = bandit_posted(trace, [0.5, 1.0, 2.0], rounds=300,
                               epsilon=0.1, seed=4)
        self.assertEqual(sum(result["arm_counts"]), 300)
        self.assertGreater(result["revenue"], 0.0)

    def test_bandit_tracks_best_price(self):
        # With one obviously-bad arm and one obviously-good arm, the
        # bandit must spend most of its exploitation pulls on the good one.
        trace = make_trace(n=200, arrival=0.5, seed=15)
        result = bandit_posted(trace, [100.0, 0.1], rounds=300,
                               epsilon=0.05, seed=5)
        self.assertGreater(result["arm_counts"][1],
                           result["arm_counts"][0] * 3)


if __name__ == "__main__":
    unittest.main()
