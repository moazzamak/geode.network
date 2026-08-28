"""Unit tests for the M156 growth cell (pure helpers only)."""
import unittest

import numpy as np

from experiments.tier4.eval_v23_m156_growth import (
    _extract_error_rows,
    _floor_ok,
    _nested_prefix,
    _prefix_columns,
    _score_matrix,
)


class TestPrefixColumns(unittest.TestCase):
    def test_interleaved_layout(self):
        # atom a owns {a, A+a, 2A+a, 3A+a} with A = atom_count
        cols = _prefix_columns(6144, 2)
        np.testing.assert_array_equal(
            cols, [0, 1, 6144, 6145, 12288, 12289, 18432, 18433])

    def test_nested_prefixes(self):
        cols256 = _prefix_columns(6144, 256)
        cols2048 = _prefix_columns(6144, 2048)
        self.assertEqual(len(cols256), 1024)
        self.assertEqual(len(cols2048), 8192)
        self.assertTrue(_nested_prefix(cols256, cols2048))
        self.assertFalse(_nested_prefix(cols2048, cols256))

    def test_prefix_touches_all_bins(self):
        # a prefix of g atoms must draw its g columns from EACH of the 4 bins
        cols = _prefix_columns(6144, 256)
        for q in range(4):
            block = np.arange(256) + q * 6144
            self.assertTrue(np.isin(block, cols).all())


class TestExtractErrorRows(unittest.TestCase):
    def test_multi_part_order(self):
        # two parts of 4 rows; error positions 1, 4, 6
        p1 = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
        p2 = np.arange(4 * 3, dtype=np.float32).reshape(4, 3) + 100
        parts = [(p1, None), (p2, None)]
        err_pos = np.array([1, 4, 6])
        out = _extract_error_rows(parts, err_pos, None, block=2)
        np.testing.assert_array_equal(out[0], p1[1])
        np.testing.assert_array_equal(out[1], p2[0])
        np.testing.assert_array_equal(out[2], p2[2])

    def test_column_subset(self):
        p1 = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
        parts = [(p1, None)]
        out = _extract_error_rows(parts, np.array([0, 2]), np.array([0, 2]),
                                  block=4)
        np.testing.assert_array_equal(out[0], [0.0, 2.0])
        np.testing.assert_array_equal(out[1], [8.0, 10.0])


class TestFloor(unittest.TestCase):
    def test_registered_budgets_clear(self):
        # the M155 premise: 76,670 error rows; rungs {256, 2048}
        self.assertTrue(_floor_ok(76670, 256, 10.0))
        self.assertTrue(_floor_ok(76670, 2048, 10.0))
        self.assertFalse(_floor_ok(76670, 4096, 10.0))

    def test_tiny_population(self):
        self.assertFalse(_floor_ok(100, 256, 10.0))


class TestScoreMatrix(unittest.TestCase):
    class _Std:
        def __init__(self):
            self.centre = np.array([0.0, 1.0], dtype=np.float32)
            self.scale = np.array([1.0, 2.0], dtype=np.float32)

        def __call__(self, xs):
            return (xs - self.centre) / self.scale

    def test_score_matrix(self):
        part = np.array([[1.0, 1.0, 1.0, 1.0],
                         [2.0, 2.0, 2.0, 2.0]], dtype=np.float32)
        cols = np.array([0, 2], dtype=np.int64)
        # 2 feature weights + 1 intercept row -> (3, 345); only the first
        # three classes matter, the rest are zero.
        w = np.zeros((3, 345), dtype=np.float64)
        w[0, 0] = 1.0                   # class0 = x0
        w[1, 1] = 1.0                   # class1 = x1
        w[2, 2] = 2.0                   # intercept -> class2 = 2
        out = _score_matrix([(part, None)], cols, w, self._Std(), 1, 2)
        self.assertEqual(out.shape, (2, 345))
        # row 0: class0 = 1, class1 = 0, class2 = 2 -> argmax 2
        # row 1: class0 = 2, class1 = 0.5, class2 = 2 -> tie -> argmax 0
        self.assertEqual(int(np.argmax(out[0])), 2)
        self.assertEqual(int(np.argmax(out[1])), 0)


if __name__ == "__main__":
    unittest.main()
