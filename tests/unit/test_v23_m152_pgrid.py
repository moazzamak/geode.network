"""Unit tests for the M152 p-grid cell (promotion-rule helpers only)."""
import unittest

from experiments.tier4.eval_v23_m152_pgrid import (
    FULL_BEST_REFERENCE,
    MARGIN,
)


class TestGateConstants(unittest.TestCase):
    def test_registered_constants(self):
        # the gate arithmetic is the cell's contract; pin it
        self.assertAlmostEqual(FULL_BEST_REFERENCE, 0.27855072463768116,
                               places=15)
        self.assertEqual(MARGIN, 0.005)


class TestPromotionRule(unittest.TestCase):
    def test_only_above_p05_promotes(self):
        p05_best = 0.2273623188405797
        p_screen = {0.25: 0.23, 0.33: 0.20, 0.66: 0.22}
        winners = [p for p, v in p_screen.items() if v > p05_best]
        self.assertEqual(winners, [0.25])

    def test_no_winner_no_promotion(self):
        p05_best = 0.2273623188405797
        p_screen = {0.25: 0.22, 0.33: 0.21}
        winners = [p for p, v in p_screen.items() if v > p05_best]
        self.assertEqual(winners, [])

    def test_gate_arithmetic(self):
        # a full-data cell must clear incumbent + margin
        incumbent = FULL_BEST_REFERENCE
        self.assertFalse(0.2786 >= incumbent + MARGIN)
        self.assertTrue(0.2836 >= incumbent + MARGIN)


if __name__ == "__main__":
    unittest.main()
