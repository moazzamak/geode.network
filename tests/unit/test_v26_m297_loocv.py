"""Unit tests for the M297 exact-LOOCV machinery.

The LOOCV formula (hat-matrix form in the eigenbasis of the
standardised symmetric system) is exact; these tests pin it against
brute-force leave-one-out ridge on small synthetic systems, pin the hat
diagonal against the direct form, and pin the registered hat-margin
validity rule on an indefinite system.
"""
from __future__ import annotations

import unittest

import numpy as np

from experiments.tier4.eval_v26_m297_loocv_lambda import (
    CLASSES,
    HAT_MARGIN,
    loocv_ridge,
)
from experiments.tier4.eval_v15_m104_experts import Standardiser


def _brute_force_loocv(x_std: np.ndarray, labels: np.ndarray,
                       lam: float) -> float:
    """Per-fold ridge fits; the same multi-output squared-error
    aggregate the closed form claims."""
    n, d = x_std.shape
    sse = 0.0
    for i in range(n):
        keep = np.arange(n) != i
        g = x_std[keep].T @ x_std[keep] + lam * np.eye(d)
        t = np.zeros((n - 1, CLASSES))
        t[np.arange(n - 1), labels[keep]] = 1.0
        w = np.linalg.solve(g, x_std[keep].T @ t)
        pred = x_std[i] @ w
        tgt = np.zeros(CLASSES)
        tgt[labels[i]] = 1.0
        sse += float(np.sum(np.square(tgt - pred)))
    return sse / (n * CLASSES)


def _synthetic(seed: int, n: int = 300, d: int = 24):
    """Raw features with a strong class signal, the fp32-rounded
    standardiser, and the standardised design."""
    rng = np.random.default_rng(seed)
    x_raw = rng.standard_normal((n, d)) * 40 + 100
    labels = np.clip((x_raw[:, 0] / 18 + x_raw[:, 1] / 9).astype(np.int64),
                     0, CLASSES - 1)
    standardiser = Standardiser(
        x_raw.mean(axis=0).astype(np.float32),
        (x_raw.std(axis=0) + 1e-8).astype(np.float32))
    x_std = standardiser(x_raw).astype(np.float64)
    return x_raw, x_std, labels, standardiser


class TestLoocvRidge(unittest.TestCase):

    def test_loocv_matches_brute_force(self):
        x_raw, x_std, labels, standardiser = _synthetic(0)
        s = x_std.T @ x_std
        vals, vecs = np.linalg.eigh(s)
        t = np.zeros((len(labels), CLASSES))
        t[np.arange(len(labels)), labels] = 1.0
        g = vecs.T @ (x_std.T @ t)
        grid = [0.1, 1.0, 10.0]
        result = loocv_ridge(vals, vecs, g, np.zeros(CLASSES),
                             x_raw, labels, standardiser, grid)
        for lam in grid:
            brute = _brute_force_loocv(x_std, labels, lam)
            rel = abs(result["loocv"][str(lam)] - brute) / brute
            self.assertLessEqual(rel, 1e-8,
                                 f"lambda {lam}: {result['loocv'][str(lam)]} "
                                 f"vs {brute}")
            self.assertTrue(result["valid"][str(lam)])

    def test_hat_diagonal_matches_direct_form(self):
        x_raw, x_std, labels, standardiser = _synthetic(1)
        n, d = x_std.shape
        s = x_std.T @ x_std
        vals, vecs = np.linalg.eigh(s)
        t = np.zeros((n, CLASSES))
        t[np.arange(n), labels] = 1.0
        g = vecs.T @ (x_std.T @ t)
        result = loocv_ridge(vals, vecs, g, np.zeros(CLASSES),
                             x_raw, labels, standardiser, [1.0])
        lam = 1.0
        h_direct = np.diag(x_std @ np.linalg.solve(s + lam * np.eye(d),
                                                   x_std.T))
        h_eigen = np.square(x_std @ vecs) @ (1.0 / (vals + lam))
        self.assertLessEqual(float(np.max(np.abs(h_direct - h_eigen))),
                             1e-10)
        self.assertAlmostEqual(result["max_hat_diagonal"]["1.0"],
                               float(h_eigen.max()), places=10)

    def test_validity_rule_on_indefinite_system(self):
        # an assembled system with a near-penalty negative eigenvalue:
        # a row aligned with that direction pushes h_ii far above 1 at
        # small lambda (INVALID); a large lambda restores positive
        # definiteness (VALID). Deterministic, no flakiness.
        rng = np.random.default_rng(2)
        n, d = 400, 20
        x_std = rng.standard_normal((n, d))
        xtx = x_std.T @ x_std
        big, u = np.linalg.eigh(xtx)
        u = u[:, -1]                    # eigenvector of the LARGEST value
        # the u direction lands at eigenvalue -0.005 exactly
        s = xtx - (float(big[-1]) + 0.005) * np.outer(u, u)
        vals, vecs = np.linalg.eigh(s)
        self.assertAlmostEqual(vals[0], -0.005, places=9)
        # one feature row aligned with that direction: at lambda=0.01
        # its hat term is 0.04 / 0.005 = 8, so the margin is negative;
        # at lambda=30 every penalised eigenvalue is positive.
        aligned = vecs[:, 0] * 0.2
        features = np.vstack([x_std, aligned[None, :]])
        labels = rng.integers(0, CLASSES, n + 1)
        t = np.zeros((n + 1, CLASSES))
        t[np.arange(n + 1), labels] = 1.0
        g = vecs.T @ (features.T @ t)
        identity = Standardiser(np.zeros(d, dtype=np.float32),
                                np.ones(d, dtype=np.float32))
        result = loocv_ridge(vals, vecs, g, np.zeros(CLASSES),
                             features, labels, identity,
                             [0.01, 30.0])
        self.assertFalse(result["valid"]["0.01"])
        self.assertLessEqual(result["min_margin"]["0.01"], HAT_MARGIN)
        self.assertTrue(result["valid"]["30.0"])
        self.assertGreater(result["min_margin"]["30.0"], HAT_MARGIN)

    def test_intercept_carried_in_predictions(self):
        # the closed form must include the intercept in e; brute force
        # with a nonzero intercept must still agree.
        x_raw, x_std, labels, standardiser = _synthetic(3)
        n, d = x_std.shape
        s = x_std.T @ x_std
        vals, vecs = np.linalg.eigh(s)
        t = np.zeros((n, CLASSES))
        t[np.arange(n), labels] = 1.0
        rng = np.random.default_rng(4)
        intercept = rng.standard_normal(CLASSES)

        def brute(lam):
            sse = 0.0
            for i in range(n):
                keep = np.arange(n) != i
                resid = t[keep] - intercept
                w = np.linalg.solve(x_std[keep].T @ x_std[keep]
                                    + lam * np.eye(d),
                                    x_std[keep].T @ resid)
                pred = x_std[i] @ w + intercept
                sse += float(np.sum(np.square(t[i] - pred)))
            return sse / (n * CLASSES)

        g_full = vecs.T @ (x_std.T @ (t - intercept))
        result = loocv_ridge(vals, vecs, g_full, intercept, x_raw,
                             labels, standardiser, [1.0])
        rel = abs(result["loocv"]["1.0"] - brute(1.0)) / brute(1.0)
        self.assertLessEqual(rel, 1e-8)


    def test_noise_direction_does_not_reach_loocv(self):
        # M296d: a planted near-cancellation mode (unpenalised
        # eigenvalue ~ -1, penalised ~ 1e-13 of the scale at lam=1)
        # must be dropped from the hat machinery, so the LOOCV equals
        # the clean system's LOOCV with that direction removed
        # entirely.
        x_raw, x_std, labels, standardiser = _synthetic(5)
        n, d = x_std.shape
        s = x_std.T @ x_std
        vals, vecs = np.linalg.eigh(s)
        planted_vals = vals.copy()
        # plant the near-cancellation on the SMALLEST eigenvalue: the
        # penalty brings the penalised value to 1e-13 of the scale,
        # below the strong-convexity cutoff at lambda = 1.0
        planted_vals[0] = -1.0 + 1e-13 * abs(vals[-1])
        t = np.zeros((n, CLASSES))
        t[np.arange(n), labels] = 1.0
        g = vecs.T @ (x_std.T @ t)
        noisy = loocv_ridge(planted_vals, vecs, g, np.zeros(CLASSES),
                            x_raw, labels, standardiser, [1.0])
        self.assertEqual(noisy["dropped_directions"]["1.0"], 1)
        # the clean system with the same direction removed by hand
        removed = loocv_ridge(vals[1:], vecs[:, 1:], g[1:, :],
                              np.zeros(CLASSES),
                              x_raw, labels, standardiser, [1.0])
        rel = abs(noisy["loocv"]["1.0"] - removed["loocv"]["1.0"]) \
            / removed["loocv"]["1.0"]
        self.assertLessEqual(rel, 1e-10)


if __name__ == "__main__":
    unittest.main()
