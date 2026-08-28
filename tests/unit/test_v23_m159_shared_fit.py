"""Unit tests for the M159 shared-fit cell (pure helpers only)."""
import unittest

import numpy as np

from experiments.tier4.eval_v23_m159_shared_fit import (
    _concat_rows,
    _score_matrix,
)


class TestConcatRows(unittest.TestCase):
    def test_layout(self):
        spec = np.arange(6 * 4 * 3, dtype=np.float32).reshape(6, 4, 3)
        glob = np.full((4, 3), 100.0, dtype=np.float32)
        out = _concat_rows(spec, glob, 4, classes=3)
        self.assertEqual(out.shape, (4, 7 * 3))
        # last 3 columns are the global arm
        np.testing.assert_array_equal(out[:, 18:21],
                                      np.full((4, 3), 100.0))
        # row 0, arm 0 = spec[0, 0] = [0, 1, 2]
        np.testing.assert_array_equal(out[0, :3], [0, 1, 2])
        # row 1, arm 1 = spec[1, 1] = [15, 16, 17] at columns 3..6
        np.testing.assert_array_equal(out[1, 3:6], [15, 16, 17])


class TestScoreMatrix(unittest.TestCase):
    class _Std:
        def __init__(self):
            self.centre = np.zeros(2, dtype=np.float32)
            self.scale = np.ones(2, dtype=np.float32)

        def __call__(self, xs):
            return (xs - self.centre) / self.scale

    def test_score_matrix(self):
        codes = np.array([[1.0, 0.0],
                          [0.0, 1.0]], dtype=np.float32)
        w = np.zeros((3, 345), dtype=np.float64)
        w[0, 0] = 2.0    # class0 = 2*x0
        w[1, 1] = 1.0    # class1 = x1
        w[2, 2] = 2.0    # intercept -> class2 = 2
        out = _score_matrix(codes, w, self._Std(), 1)
        self.assertEqual(out.shape, (2, 345))
        # row 0 (x=[1,0]): class0=2, class1=0, class2=2 -> tie -> 0
        self.assertEqual(int(np.argmax(out[0])), 0)
        # row 1 (x=[0,1]): class0=0, class1=1, class2=2 -> 2
        self.assertEqual(int(np.argmax(out[1])), 2)


if __name__ == "__main__":
    unittest.main()
