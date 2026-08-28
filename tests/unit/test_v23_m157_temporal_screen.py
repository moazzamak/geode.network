"""Unit tests for the M157 temporal screen (series + feature helpers)."""
import unittest

import numpy as np

from experiments.tier4.eval_v23_m157_temporal_screen import (
    _dyck_series,
    _lorenz,
    _primitive_features,
    _reservoir_states,
)


class TestLorenz(unittest.TestCase):
    def test_shape_and_determinism(self):
        a = _lorenz(200, 10.0, 28.0, 8.0 / 3.0, 0.01, 10, 100, [1, 1, 1], 3)
        b = _lorenz(200, 10.0, 28.0, 8.0 / 3.0, 0.01, 10, 100, [1, 1, 1], 3)
        self.assertEqual(a.shape, (200,))
        np.testing.assert_array_equal(a, b)

    def test_chaotic_variation(self):
        a = _lorenz(300, 10.0, 28.0, 8.0 / 3.0, 0.01, 10, 100, [1, 1, 1], 3)
        self.assertGreater(a.std(), 1.0)


class TestDyck(unittest.TestCase):
    def test_balanced_and_deterministic(self):
        s = _dyck_series(500, 4, 5)
        self.assertEqual(len(s), 500)
        depth = 0
        for t in s:
            depth += 1 if t in (0, 2) else -1
            self.assertGreaterEqual(depth, 0)
        np.testing.assert_array_equal(s, _dyck_series(500, 4, 5))


class TestPrimitiveFeatures(unittest.TestCase):
    def test_values(self):
        s = np.arange(10.0)
        feats, targs = _primitive_features(s, window=4)
        self.assertEqual(feats.shape, (6, 4))
        np.testing.assert_allclose(feats[0, 1], 1.0)  # first difference


class TestReservoirStates(unittest.TestCase):
    def test_echo_property(self):
        rng = np.random.default_rng(0)
        series = rng.standard_normal(500)
        states, targets, rho = _reservoir_states(series, 64, 0.9, 1, 50)
        self.assertLess(rho, 1.0)
        self.assertEqual(states.shape[1], 64)
        self.assertEqual(len(states), len(targets))


if __name__ == "__main__":
    unittest.main()
