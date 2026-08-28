"""Unit tests for the M151 interaction cell (pure helpers only)."""
import unittest

import numpy as np

from experiments.tier4.eval_v23_m151_interaction import (
    _concat_block,
    _score_concat,
)


class TestConcatBlock(unittest.TestCase):
    def test_layout(self):
        spm = np.arange(24, dtype=np.float32).reshape(2, 12)
        ms = np.arange(8, dtype=np.float32).reshape(2, 4)
        out = _concat_block(spm, ms, 0, 2)
        self.assertEqual(out.shape, (2, 16))
        np.testing.assert_array_equal(out[:, :12], spm)
        np.testing.assert_array_equal(out[:, 12:], ms)


class TestScoreConcat(unittest.TestCase):
    def test_accuracy(self):
        rng = np.random.default_rng(0)
        spm = rng.standard_normal((30, 12)).astype(np.float32)
        ms = rng.standard_normal((30, 4)).astype(np.float32)
        labels = rng.integers(0, 3, size=30)
        weights = np.zeros((17, 3))
        weights[-1, 1] = 1.0  # constant -> class 1

        class Std:
            def __call__(self, x):
                return x

        acc = _score_concat(weights, Std(), spm, ms, labels, 16)
        self.assertAlmostEqual(acc, float((labels == 1).mean()), places=6)


if __name__ == "__main__":
    unittest.main()
