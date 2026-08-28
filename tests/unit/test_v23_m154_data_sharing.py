"""Unit tests for the M154 data-sharing cell (pure helpers only)."""
import unittest

import numpy as np

from experiments.tier4.eval_v23_m154_data_sharing import (
    _flat_features,
    _gated_features,
)


class TestGatedFeatures(unittest.TestCase):
    def test_gating(self):
        arm_scores = np.zeros((2, 7, 3), dtype=np.float32)  # 2 rows, 7 arms, 3 cls
        arm_scores[:, 0, :] = 1.0
        arm_scores[:, 1, :] = 2.0
        domains = np.array([0, 2])
        out = _gated_features(arm_scores, domains)
        self.assertEqual(out.shape, (2, 7 * 6 * 3))
        # layout: block (a, d) = (a*6 + d)*3
        # row 0 domain 0: arm 0 gate 0 active, arm 1 gate 0 active
        np.testing.assert_array_equal(out[0, :3], [1, 1, 1])
        np.testing.assert_array_equal(out[0, 18:21], [2, 2, 2])
        # arm 0 gate 1 (inactive for row 0): zeros
        np.testing.assert_array_equal(out[0, 3:6], [0, 0, 0])
        # row 1 domain 2: arm 0 gate 2 active, gate 0 inactive
        np.testing.assert_array_equal(out[1, 6:9], [1, 1, 1])
        np.testing.assert_array_equal(out[1, :3], [0, 0, 0])


class TestFlatFeatures(unittest.TestCase):
    def test_flat_width_is_m143b_layout(self):
        # 7 arms x 345 classes = 2,415 columns; the M143b order is
        # [specialists 0..5, global]. A duplicated global arm (the 16 Aug
        # full-run void) would make 2,760 columns here instead.
        arm_scores = np.zeros((7, 5, 3), dtype=np.float32)
        arm_scores[6, :, :] = 7.0
        flat = _flat_features(arm_scores)
        self.assertEqual(flat.shape, (5, 7 * 3))
        # last 3 columns are the global arm's scores
        np.testing.assert_array_equal(flat[:, 18:21],
                                      np.full((5, 3), 7.0, dtype=np.float32))

    def test_full_feature_width(self):
        # flat (2,415) + gated (14,490) = 16,905
        n, arms, classes = 4, 7, 345
        scores = np.arange(n * arms * classes,
                           dtype=np.float32).reshape(7, n, classes)
        flat = _flat_features(scores)
        gated = _gated_features(scores.transpose(1, 0, 2),
                                np.array([0, 1, 2, 3]))
        self.assertEqual(flat.shape[1], 2415)
        self.assertEqual(gated.shape[1], 14490)
        self.assertEqual(flat.shape[1] + gated.shape[1], 16905)


if __name__ == "__main__":
    unittest.main()
