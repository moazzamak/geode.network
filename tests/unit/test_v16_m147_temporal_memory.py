"""Unit tests for the M147 temporal-memory screen (unittest, no pytest)."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.tier4.eval_v16_m147_temporal_memory import (
    _delay_matrix,
    _nrmsfe,
    _no_memory_arm,
    _programmatic_arm,
    _reservoir_arm_mg,
    _tap_delay_arm,
    mackey_glass,
)


class TestMackeyGlass(unittest.TestCase):
    def test_deterministic_same_seed(self):
        a = mackey_glass(300, seed=7, discard=50)
        b = mackey_glass(300, seed=7, discard=50)
        self.assertTrue(np.array_equal(a, b))

    def test_different_seed_differs(self):
        a = mackey_glass(300, seed=7, discard=50)
        b = mackey_glass(300, seed=8, discard=50)
        self.assertFalse(np.array_equal(a, b))

    def test_non_constant(self):
        series = mackey_glass(500, seed=7, discard=1000)
        self.assertGreater(float(np.std(series)), 0.05)

    def test_delay_actually_matters(self):
        # a ring-wrap defect makes the DDE collapse to a stable ODE; two
        # different delays must then produce different chaotic trajectories
        a = mackey_glass(500, tau=17.0, seed=7, discard=1000)
        b = mackey_glass(500, tau=23.0, seed=7, discard=1000)
        self.assertFalse(np.array_equal(a, b))


class TestDelayMatrix(unittest.TestCase):
    def test_shapes(self):
        series = np.arange(20, dtype=np.float64)
        for k in (2, 4, 8):
            features, targets = _delay_matrix(series, k)
            self.assertEqual(features.shape, (20 - k, k))
            self.assertEqual(targets.shape, (20 - k,))

    def test_alignment(self):
        series = np.arange(10, dtype=np.float64)
        features, targets = _delay_matrix(series, 3)
        # row 0: [x2, x1, x0] -> x3
        self.assertTrue(np.array_equal(features[0], [2.0, 1.0, 0.0]))
        self.assertEqual(targets[0], 3.0)


class TestExtrapolator(unittest.TestCase):
    def test_exact_on_linear(self):
        t = np.arange(200, dtype=np.float64)
        series = 3.0 * t + 5.0
        result = _programmatic_arm(series[:100], series[100:])
        self.assertEqual(result["winner"], "extrapolator")
        self.assertLess(result["nrmsfe"], 1e-9)


class TestArms(unittest.TestCase):
    def test_no_memory_finite(self):
        series = mackey_glass(400, seed=7, discard=50)
        arm = _no_memory_arm(series[:300], series[300:])
        self.assertTrue(np.isfinite(arm["nrmsfe"]))

    def test_tap_delay_beats_no_memory_on_ar2(self):
        # deterministic AR(2): x_{t+1} = 0.5 x_t + 0.3 x_{t-1}
        rng = np.random.default_rng(5)
        n = 600
        series = np.empty(n)
        series[0], series[1] = 1.0, 0.5
        for t in range(2, n):
            series[t] = 0.5 * series[t - 1] + 0.3 * series[t - 2]
        series = series + 1e-4 * rng.standard_normal(n)
        no_mem = _no_memory_arm(series[:400], series[400:])
        tap = _tap_delay_arm(series[:400], series[400:], k=2)
        self.assertLess(tap["nrmsfe"], no_mem["nrmsfe"])

    def test_reservoir_echo_state_property(self):
        series = mackey_glass(400, seed=7, discard=50)
        arm = _reservoir_arm_mg(series[:300], series[300:], units=64, rho=0.9,
                                seed=21, warmup=20, penalty=1.0)
        self.assertTrue(arm["echo_state_property_ok"])
        self.assertLess(arm["rho_measured"], 1.0)
        self.assertTrue(np.isfinite(arm["nrmsfe"]))


class TestMetric(unittest.TestCase):
    def test_nrmsfe_zero_for_perfect(self):
        rng = np.random.default_rng(1)
        target = rng.standard_normal(100)
        self.assertLess(_nrmsfe(target, target), 1e-12)

    def test_nrmsfe_one_for_constant_predictor(self):
        rng = np.random.default_rng(2)
        target = rng.standard_normal(200)
        target = target - target.mean()  # demeaned: RMSE(zero) == std exactly
        pred = np.zeros_like(target)
        self.assertAlmostEqual(_nrmsfe(pred, target), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
