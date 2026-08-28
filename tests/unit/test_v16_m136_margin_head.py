"""Focused tests for the M136 head-objective machinery (no corpus, no GPU).

Pins the two properties the measurement depends on: the smoothed-target
accumulation equals the manual normal equations, and the smoothing-equivalence
lemma (predictions identical to ridge) holds on synthetic data; plus the batch
hinge fit separating linearly separable data deterministically.
"""

from __future__ import annotations

import unittest

import numpy as np

from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v16_m136_margin_head import (
    _accuracy_per_domain,
    _add_smoothed,
    _fit_batch_hinge,
    _margin_quantiles,
    _standardised_train_block,
)

RNG = np.random.default_rng(11)


class SmoothedTargetTests(unittest.TestCase):
    def test_add_smoothed_matches_manual_normal_equations(self):
        classes = 4
        features = RNG.normal(size=(50, 7))
        labels = RNG.integers(0, classes, size=50)
        epsilon = 0.1

        acc = RidgeAccumulator(7, classes)
        _add_smoothed(acc, features, labels, classes, epsilon)

        targets = np.full((50, classes), epsilon / classes)
        targets[np.arange(50), labels] += 1.0 - epsilon
        self.assertTrue(np.allclose(acc.cross, features.T @ targets))
        self.assertTrue(np.allclose(acc.gram, features.T @ features))
        self.assertTrue(np.allclose(acc.column_sum, features.sum(axis=0)))
        self.assertTrue(np.allclose(acc.class_count, targets.sum(axis=0)))

    def test_smoothing_equivalence_predictions_identical(self):
        """The registered lemma: smoothed targets shift scores by a per-row
        constant, so argmax predictions equal the lambda=1.0 ridge's."""
        classes = 5
        features = RNG.normal(size=(400, 20))
        labels = RNG.integers(0, classes, size=400)
        epsilon = 0.1

        acc_ridge = RidgeAccumulator(20, classes)
        acc_ridge.add(features, labels)
        acc_smooth = RidgeAccumulator(20, classes)
        _add_smoothed(acc_smooth, features, labels, classes, epsilon)

        w_ridge = acc_ridge.solve_many([1.0])[1.0]
        w_smooth = acc_smooth.solve_many([1.0])[1.0]
        test = RNG.normal(size=(80, 20))
        std = acc_ridge.standardiser()

        pred_ridge = np.argmax(std(test) @ w_ridge[:-1] + w_ridge[-1], axis=1)
        pred_smooth = np.argmax(std(test) @ w_smooth[:-1] + w_smooth[-1], axis=1)
        self.assertTrue(np.array_equal(pred_ridge, pred_smooth))


class StandardisationTests(unittest.TestCase):
    def test_standardised_train_block_matches_manual(self):
        rows, width, n = 300, 12, 200
        data = RNG.normal(size=(rows, width))
        xs, centre, scale = _standardised_train_block(data, n, block=64)
        manual = (data[:n] - data[:n].mean(axis=0)) / (data[:n].std(axis=0) + 1e-8)
        self.assertEqual(xs.shape, (n, width))
        self.assertTrue(np.allclose(xs, manual, atol=1e-5))


class BatchHingeTests(unittest.TestCase):
    def test_separates_linearly_separable_data(self):
        rng = np.random.default_rng(7)
        classes = 3
        n_per_class = 40
        width = 8
        xs = []
        labels = []
        for c in range(classes):
            centre = np.zeros(width)
            centre[c] = 3.0
            # Bounded noise keeps a strict margin: the data is linearly
            # separable by construction (Gaussian noise would not be).
            block = rng.uniform(-0.5, 0.5, size=(n_per_class, width)) + centre
            xs.append(block)
            labels.append(np.full(n_per_class, c))
        xs = np.vstack(xs).astype(np.float32)
        labels = np.concatenate(labels)

        weights, objective = _fit_batch_hinge(xs, labels, classes, lam=1e-3,
                                              epochs=8, block=64)
        scores = xs.astype(np.float64) @ weights
        accuracy = float((np.argmax(scores, axis=1) == labels).mean())
        self.assertEqual(accuracy, 1.0)
        self.assertTrue(np.any(np.abs(weights) > 1e-6))

    def test_deterministic_across_runs(self):
        rng = np.random.default_rng(7)
        classes = 3
        xs = rng.normal(size=(90, 6)).astype(np.float32)
        labels = rng.integers(0, classes, size=90)
        first, _ = _fit_batch_hinge(xs, labels, classes, lam=1e-3, epochs=5, block=32)
        second, _ = _fit_batch_hinge(xs, labels, classes, lam=1e-3, epochs=5, block=32)
        self.assertTrue(np.array_equal(first, second))


class MarginAndReadoutTests(unittest.TestCase):
    def test_margin_quantiles_known_values(self):
        true_scores = np.array([2.0, 1.0, -1.0, 0.0])
        max_other = np.array([0.0, 1.5, 0.0, -0.5])
        out = _margin_quantiles(true_scores, max_other)
        margins = true_scores - max_other  # [2, -0.5, -1, 0.5]
        self.assertEqual(out["positive_share"], 0.5)
        self.assertAlmostEqual(out["margin_mean"], float(margins.mean()))
        self.assertAlmostEqual(out["q50"], float(np.median(margins)))

    def test_accuracy_per_domain(self):
        labels = np.array([0, 1, 2, 0, 1])
        domains = np.array([0, 0, 3, 5, 3])
        predictions = np.array([0, 1, 1, 0, 0])
        accuracy, per_domain = _accuracy_per_domain(predictions, labels, domains)
        self.assertAlmostEqual(accuracy, 3 / 5)
        self.assertAlmostEqual(per_domain[0], 1.0)
        self.assertAlmostEqual(per_domain[3], 0.0)
        self.assertAlmostEqual(per_domain[5], 1.0)


if __name__ == "__main__":
    unittest.main()
