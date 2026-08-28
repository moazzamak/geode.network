"""Unit tests for the M298 readouts (LDA/Mahalanobis and the backward
instrument).

Pins, on small synthetic systems: the LDA readout matches a direct
closed-form implementation; the backward instrument flags the full-
system eigen solve as clean; and the class-balanced ridge matches a
direct weighted least-squares solve.
"""
from __future__ import annotations

import unittest

import numpy as np

from experiments.tier4.eval_v26_m298_lda_balanced import (
    CLASSES,
    RESID_TOL,
    _balanced_ridge,
    _lda_direct_agreement,
    lda_readout,
)
from experiments.tier4.eval_v15_m104_experts import Standardiser


def _synthetic(seed: int, n: int = 600, d: int = 30, classes: int = 40):
    rng = np.random.default_rng(seed)
    x_raw = rng.standard_normal((n, d)) * 30 + 50
    labels = rng.integers(0, classes, n)
    labels[:classes] = np.arange(classes)   # every class has rows
    rng.shuffle(labels)
    standardiser = Standardiser(
        x_raw.mean(axis=0).astype(np.float32),
        (x_raw.std(axis=0) + 1e-8).astype(np.float32))
    return x_raw, labels, standardiser, classes


class TestLdaReadout(unittest.TestCase):

    def test_lda_matches_direct_form(self):
        x_raw, labels, standardiser, classes = _synthetic(0)
        n, d = x_raw.shape
        x_std = standardiser(x_raw).astype(np.float64)
        s = x_std.T @ x_std
        vals, vecs = np.linalg.eigh(s)
        t = np.zeros((n, classes))
        t[np.arange(n), labels] = 1.0
        cross = x_std.T @ t
        class_count = t.sum(axis=0)
        lam = 1.0

        scorer, report = lda_readout(vals, vecs, cross, class_count, lam)
        means = cross / class_count[None, :]
        a = s + lam * np.eye(d)
        ainv_m = np.linalg.solve(a, means)
        priors = np.log(class_count / class_count.sum())
        direct_scores = x_std @ ainv_m - 0.5 * np.sum(
            means * ainv_m, axis=0) + priors
        got = scorer(x_std)
        rel = float(np.max(np.abs(got - direct_scores))
                    / max(float(np.max(np.abs(direct_scores))), 1e-12))
        self.assertLessEqual(rel, 1e-8)
        self.assertTrue(report["backward"]["ok"], report["backward"])

    def test_lda_decision_matches_direct(self):
        x_raw, labels, standardiser, classes = _synthetic(1)
        n, d = x_raw.shape
        x_std = standardiser(x_raw).astype(np.float64)
        s = x_std.T @ x_std
        vals, vecs = np.linalg.eigh(s)
        t = np.zeros((n, classes))
        t[np.arange(n), labels] = 1.0
        cross = x_std.T @ t
        class_count = t.sum(axis=0)
        lam = 0.5
        scorer, _ = lda_readout(vals, vecs, cross, class_count, lam)
        means = cross / class_count[None, :]
        ainv_m = np.linalg.solve(s + lam * np.eye(d), means)
        priors = np.log(class_count / class_count.sum())
        direct = np.argmax(x_std @ ainv_m
                           - 0.5 * np.sum(means * ainv_m, axis=0)
                           + priors, axis=1)
        got = np.argmax(scorer(x_std), axis=1)
        self.assertTrue(np.array_equal(direct, got))

    def test_agreement_helper_matches_at_no_drop_lambda(self):
        # M298a: the agreement gate route equals the eigen route where
        # the direct solve is meaningful (no dropped modes)
        x_raw, labels, standardiser, classes = _synthetic(4)
        x_std = standardiser(x_raw).astype(np.float64)
        s = x_std.T @ x_std
        vals, vecs = np.linalg.eigh(s)
        t = np.zeros((len(labels), classes))
        t[np.arange(len(labels)), labels] = 1.0
        cross = x_std.T @ t
        class_count = t.sum(axis=0)
        lam = 5.0        # penalised spectrum all positive
        scorer, report = lda_readout(vals, vecs, cross, class_count, lam)
        self.assertEqual(report["dropped_components"], 0)
        direct, direct_rel = _lda_direct_agreement(
            s, cross, class_count, lam, x_std)
        self.assertIsNotNone(direct)
        self.assertLessEqual(direct_rel, 1e-10)
        rel = float(np.max(np.abs(scorer(x_std) - direct))
                    / max(float(np.max(np.abs(direct))), 1e-12))
        self.assertLessEqual(rel, 1e-8)


class TestBalancedRidge(unittest.TestCase):

    def test_balanced_matches_direct_wls(self):
        x_raw, labels, standardiser, classes = _synthetic(2)
        n, d = x_raw.shape
        x_std = standardiser(x_raw).astype(np.float64)
        counts = np.bincount(labels, minlength=classes).astype(np.float64)
        w = 1.0 / counts[labels]
        t = np.zeros((n, classes))
        t[np.arange(n), labels] = 1.0
        lam = 1.0
        gram = x_std.T @ (w[:, None] * x_std)
        cross_w = x_std.T @ (w[:, None] * t)
        direct = np.linalg.solve(gram + lam * np.eye(d), cross_w)

        weights, report = _balanced_ridge(x_raw, labels, standardiser, lam,
                                          classes=classes)
        rel = float(np.max(np.abs(direct - weights[:-1]))
                    / max(float(np.max(np.abs(direct))), 1e-12))
        self.assertLessEqual(rel, 1e-9)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["dropped_components"], 0)

    def test_balanced_rejects_empty_class(self):
        x_raw, labels, standardiser, classes = _synthetic(3)
        labels_missing = labels.copy()
        labels_missing[labels_missing == 0] = 1   # class 0 vanishes
        result = _balanced_ridge(x_raw, labels_missing, standardiser,
                                 1.0, classes=classes)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
