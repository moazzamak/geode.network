"""Unit tests for the M155 growth-premise cell (pure helpers only)."""
import unittest

import numpy as np

from experiments.tier4.eval_v23_m155_growth_premise import (
    _block_accuracy,
    _error_rows,
    _floor_ladder,
)


class TestErrorRows(unittest.TestCase):
    def test_positions(self):
        preds = np.array([0, 0, 1, 2, 3])
        labels = np.array([0, 1, 1, 2, 3])
        np.testing.assert_array_equal(_error_rows(preds, labels), [1])


class TestFloorLadder(unittest.TestCase):
    def test_ladder_arithmetic(self):
        # ceil(280000 / 4a) >= 10 -> a <= 7000 -> max power of two 4096
        ladder = _floor_ladder(280_000)
        self.assertEqual(ladder[-1], 4096)
        self.assertEqual(ladder[0], 32)
        self.assertTrue(all(b > a for a, b in zip(ladder, ladder[1:])))

    def test_small_population(self):
        # ceil(2760 / 4a) >= 10 -> a <= 64 (the M145 premise)
        ladder = _floor_ladder(2_760)
        self.assertEqual(ladder[-1], 64)
        self.assertIn(32, ladder)

    def test_tiny_population_empty(self):
        # n_err = 100 -> ceil(100/4a) >= 10 impossible at a >= 32
        self.assertEqual(_floor_ladder(100), [])


class TestBlockAccuracy(unittest.TestCase):
    def test_counts(self):
        rng = np.random.default_rng(0)
        mem = rng.standard_normal((50, 6)).astype(np.float32)
        labels = rng.integers(0, 3, size=50)
        # a constant predictor gets the majority class share
        weights = np.zeros((7, 3))
        weights[-1, 1] = 1.0  # bias -> class 1 always

        class Std:
            def __call__(self, x):
                return x

        acc, n_err = _block_accuracy(weights, Std(), mem, labels, 16, 50)
        expected_acc = float((labels == 1).mean())
        self.assertAlmostEqual(acc, expected_acc, places=6)
        self.assertEqual(n_err, 50 - int((labels == 1).sum()))


if __name__ == "__main__":
    unittest.main()
