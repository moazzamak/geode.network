"""Unit tests for the M145 residual-growth cell (pure helpers only)."""
import unittest

import numpy as np

from experiments.tier4.eval_v16_m143_integration import (
    _split_indices,
    _stacking_fit,
)
from experiments.tier4.eval_v16_m145_growth import (
    _error_rows,
    _floor_ok,
    _growth_dictionaries,
)


class TestSplitHalves(unittest.TestCase):
    def test_m143_split(self):
        fit, ev = _split_indices(34500, 33)
        self.assertEqual(len(fit), 17250)
        self.assertEqual(len(ev), 17250)
        self.assertEqual(len(np.intersect1d(fit, ev)), 0)
        self.assertEqual(sorted(np.concatenate([fit, ev])),
                         list(range(34500)))


class TestErrorRows(unittest.TestCase):
    def test_wrong_rows_only(self):
        preds = np.array([0, 0, 1, 2, 3])
        labels = np.array([0, 1, 1, 2, 3])
        rows = _error_rows(preds, labels)
        np.testing.assert_array_equal(rows, [1])

    def test_none_wrong(self):
        preds = np.array([1, 2, 0])
        labels = np.array([1, 2, 0])
        self.assertEqual(len(_error_rows(preds, labels)), 0)


class TestFloorGate(unittest.TestCase):
    def test_ceil_arithmetic(self):
        # width = 4 * atoms; ratio = ceil(n_err / width)
        self.assertTrue(_floor_ok(13400, 128, 2))   # 27 rows/dim
        self.assertTrue(_floor_ok(13400, 256, 2))   # 14 rows/dim
        self.assertFalse(_floor_ok(13400, 512, 2))  # 7 rows/dim
        self.assertTrue(_floor_ok(20480, 512, 2))   # exactly 10
        self.assertFalse(_floor_ok(18432, 512, 2))  # 9 rows/dim
        self.assertFalse(_floor_ok(512, 128, 2))    # exactly 1 row/dim

    def test_floor_zero_errors(self):
        self.assertFalse(_floor_ok(0, 128, 2))


class TestGrowthDictionaries(unittest.TestCase):
    def test_nested_prefix(self):
        rng = np.random.default_rng(0)
        pool = rng.standard_normal((64, 3)).astype(np.float32)
        dicts = _growth_dictionaries(pool, [16, 32])
        np.testing.assert_array_equal(dicts[32][:16], dicts[16])
        self.assertEqual(dicts[16].shape, (16, 3))
        self.assertEqual(dicts[32].shape, (32, 3))


class TestEightArmStacking(unittest.TestCase):
    def test_shapes_and_determinism(self):
        rng = np.random.default_rng(5)
        train = rng.standard_normal((40, 2760))
        test = rng.standard_normal((10, 2760))
        labels = rng.integers(0, 5, size=40)
        predict = _stacking_fit(train, labels, 1.0)
        preds = predict(test)
        self.assertEqual(preds.shape, (10,))
        np.testing.assert_array_equal(preds, predict(test))


if __name__ == "__main__":
    unittest.main()
