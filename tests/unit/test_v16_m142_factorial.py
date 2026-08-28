"""Unit tests for the M142 C1 power-norm transform."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.tier4.eval_v16_m142_factorial import power_norm


class TestPowerNorm(unittest.TestCase):
    def test_p1_identity_up_to_scale(self):
        rng = np.random.default_rng(1)
        xs = rng.standard_normal((20, 24576))
        out = power_norm(xs, 1.0)
        self.assertTrue(np.allclose(out, xs / np.linalg.norm(xs, axis=1,
                                                            keepdims=True),
                                    atol=1e-12))

    def test_rows_unit_norm(self):
        rng = np.random.default_rng(2)
        xs = rng.standard_normal((10, 100))
        out = power_norm(xs, 0.5)
        norms = np.linalg.norm(out, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-12))

    def test_sqrt_sign(self):
        xs = np.array([[4.0, -9.0, 0.0]])
        out = power_norm(xs, 0.5)
        self.assertAlmostEqual(out[0, 0], 2.0 / np.sqrt(4 + 9 + 0), places=10)
        self.assertAlmostEqual(out[0, 1], -3.0 / np.sqrt(4 + 9 + 0), places=10)
        self.assertEqual(out[0, 2], 0.0)

    def test_zero_row_guard(self):
        xs = np.zeros((2, 5))
        out = power_norm(xs, 0.5)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_positive_scaling_invariance_of_argmax(self):
        # the anchor property: p=1.0 row scaling does not change argmax
        rng = np.random.default_rng(3)
        xs = rng.standard_normal((5, 345))
        out = power_norm(xs, 1.0)
        self.assertTrue(np.array_equal(np.argmax(xs, axis=1),
                                       np.argmax(out, axis=1)))


if __name__ == "__main__":
    unittest.main()
