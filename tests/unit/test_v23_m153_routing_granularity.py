"""Unit tests for the M153 routing-granularity cell (pure helpers only)."""
import unittest

import numpy as np

from experiments.tier4.eval_v23_m153_routing_granularity import (
    _child_vectors,
    _class_profiles,
    _group_fusion,
    _kmeans_groups,
)


class TestClassProfiles(unittest.TestCase):
    def test_profiles(self):
        scores = np.zeros((8, 4), dtype=np.float64)
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        scores[labels == 0] = [1, 2, 3, 4]
        scores[labels == 1] = [5, 6, 7, 8]
        scores[labels == 2] = [9, 10, 11, 12]
        prof = _class_profiles(scores, labels)
        np.testing.assert_array_equal(prof[0], [1, 2, 3, 4])
        np.testing.assert_array_equal(prof[1], [5, 6, 7, 8])
        # absent classes get zeros
        np.testing.assert_array_equal(prof[300], np.zeros(4))


class TestKmeansGroups(unittest.TestCase):
    def test_two_blobs(self):
        rng = np.random.default_rng(7)
        profiles = np.concatenate([
            rng.normal(0.0, 0.1, size=(40, 8)),
            rng.normal(5.0, 0.1, size=(30, 8)),
        ])
        groups = _kmeans_groups(profiles, 2, seed=3, runs=2)
        g0 = groups[:40]
        g1 = groups[40:]
        self.assertEqual(len(set(g0)), 1)
        self.assertEqual(len(set(g1)), 1)
        self.assertNotEqual(g0[0], g1[0])


class TestChildVectors(unittest.TestCase):
    def test_masking(self):
        scores = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        groups = np.array([0, 1, 0])  # group 0 = classes {0, 2}
        children = _child_vectors(scores, groups, 2, -1e30)
        # child 0 masks class 1; child 1 masks classes 0 and 2
        np.testing.assert_array_equal(children[0][0], [1.0, -1e30, 3.0])
        np.testing.assert_array_equal(children[1][0], [-1e30, 2.0, -1e30])


class TestGroupFusion(unittest.TestCase):
    def test_train_test_row_split(self):
        # children carry ALL rows; the fusion must slice train rows
        rng = np.random.default_rng(3)
        n_train, n_test = 12, 6
        global_all = rng.standard_normal((n_train + n_test, 4))
        children = _child_vectors(global_all, np.array([0, 1, 0, 0]), 2,
                                  -1e30)
        train_labels = rng.integers(0, 4, size=n_train)
        test_labels = rng.integers(0, 4, size=n_test)
        fused, _pen, _ladder = _group_fusion(
            children, global_all, train_labels, n_train, test_labels)
        self.assertGreaterEqual(fused, 0.0)
        self.assertLessEqual(fused, 1.0)


if __name__ == "__main__":
    unittest.main()
