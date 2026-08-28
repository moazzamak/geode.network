"""Unit tests for the M142 C4 power-norm cell (pure/numpy functions).

Run from the repo root with the CPU environment::

    .\\.venv\\Scripts\\python.exe -m unittest experiments.common.test_v16_m142_c4
"""

import unittest

import numpy as np

from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v16_m142_c4 import _fit_power, _score_power
from experiments.tier4.eval_v16_m142_factorial import power_norm

CLASSES = 345


class PowerNormTests(unittest.TestCase):
    def test_p1_is_positive_row_scale(self):
        rng = np.random.default_rng(0)
        xs = rng.standard_normal((8, 6)).astype(np.float32)
        out = power_norm(xs, 1.0)
        scale = np.linalg.norm(out, axis=1) / (np.linalg.norm(xs, axis=1) + 1e-12)
        np.testing.assert_allclose(out, xs * scale[:, None],
                                   rtol=0, atol=1e-5)
        self.assertTrue((scale > 0).all())

    def test_p05_rows_are_unit_norm(self):
        rng = np.random.default_rng(1)
        xs = rng.standard_normal((8, 6)).astype(np.float32)
        out = power_norm(xs, 0.5)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0,
                                   rtol=0, atol=1e-5)

    def test_raw_fit_matches_accumulator(self):
        # transform=False must reproduce the plain RidgeAccumulator fit
        # (the runner fixes CLASSES=345, so the synthetic uses 345 classes)
        rng = np.random.default_rng(2)
        xs = rng.standard_normal((400, 40)).astype(np.float32)
        labels = rng.integers(0, 345, size=400)
        w_raw, _std = _fit_power(xs, labels, 1.0, [1.0], 400, 64,
                                 transform=False)
        acc = RidgeAccumulator(40, 345)
        acc.add(xs, labels)
        expected = acc.solve(1.0)
        np.testing.assert_allclose(w_raw["1.0"], expected, rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
