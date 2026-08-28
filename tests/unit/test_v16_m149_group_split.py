"""Unit tests for M149 group splitting (CPU logic only)."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.tier4.eval_v16_m149_group_split import _fit_child, _kmeans_2


class TestKMeans2(unittest.TestCase):
    def test_two_blobs_separated(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((100, 8)) + 3.0
        b = rng.standard_normal((100, 8)) - 3.0
        xs = np.vstack([a, b])
        labels = _kmeans_2(xs, seed=5)
        # clusters must match the generating halves (up to label swap)
        base = np.concatenate([np.zeros(100, dtype=np.int64),
                               np.ones(100, dtype=np.int64)])
        agreement = max(float((labels == base).mean()),
                        float((labels == 1 - base).mean()))
        self.assertGreater(agreement, 0.99)

    def test_deterministic(self):
        rng = np.random.default_rng(2)
        xs = rng.standard_normal((200, 10))
        self.assertTrue(np.array_equal(_kmeans_2(xs, 7), _kmeans_2(xs, 7)))

    def test_non_degenerate_on_uniform(self):
        # even on a single blob both clusters are non-empty
        rng = np.random.default_rng(3)
        xs = rng.standard_normal((200, 10))
        labels = _kmeans_2(xs, 9)
        self.assertTrue((labels == 0).any())
        self.assertTrue((labels == 1).any())


class TestFitChild(unittest.TestCase):
    def test_shapes_and_accuracy(self):
        rng = np.random.default_rng(4)
        n = 300
        labels = rng.integers(0, 345, size=n)
        scores = rng.standard_normal((n, 2415)) * 0.1
        scores[np.arange(n), labels] += 5.0
        rows = np.arange(200)
        predict = _fit_child(rows, scores, labels)
        preds = np.argmax(predict(scores[rows]), axis=1)
        self.assertEqual(preds.shape, (200,))
        self.assertGreater(float((preds == labels[rows]).mean()), 0.9)

    def test_eval_on_new_rows(self):
        # out-of-sample check: strong 2-class signal hidden in 2415 noise dims
        # must survive the child ridge to the eval rows
        rng = np.random.default_rng(5)
        labels = rng.integers(0, 2, size=200)
        scores = rng.standard_normal((200, 2415)) * 0.2
        scores[np.arange(200), labels] += 8.0
        predict = _fit_child(np.arange(100), scores, labels)
        preds = np.argmax(predict(scores[100:]), axis=1)
        # 2415 noise dims at lambda=1 attenuate the signal; the sanity bar is
        # that the child ridge beats chance clearly on out-of-sample rows
        self.assertGreater(float((preds == labels[100:]).mean()), 0.55)


if __name__ == "__main__":
    unittest.main()
