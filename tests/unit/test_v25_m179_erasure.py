"""Unit tests for M179: closed-form erasure on frozen components.

Pure numpy; no data, no GPU. Covers LEACE's defining property (all
linear predictability removed), the rank cap, and the certificate's
relative residual.
"""
from __future__ import annotations

import unittest

import numpy as np

from geode.audit.erasure import erasure_certificate, leace_eraser


def _synthetic(n_per_group: int, group_count: int, dim: int,
               seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(group_count), n_per_group)
    directions = rng.standard_normal((group_count, dim))
    features = rng.standard_normal((len(groups), dim)) * 0.1
    features += directions[groups] * 10.0
    return features.astype(np.float32), groups


class TestErasure(unittest.TestCase):

    def test_leace_removes_all_linear_predictability(self):
        features, groups = _synthetic(10, 5, 64, seed=0)
        eraser, removed = leace_eraser(
            features, groups, group_count=5, floor=1e-10,
            singular_tolerance=1e-10)
        self.assertEqual(removed, 4)  # rank cap = group_count - 1
        erased = eraser(features)
        # A ridge probe on the erased features must read chance.
        from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
        acc = RidgeAccumulator(64, 5)
        acc.add(erased, groups)
        weights = acc.solve(1.0)
        scores = acc.standardiser()(erased).astype(np.float64) \
            @ weights[:-1] + weights[-1]
        preds = np.argmax(scores, axis=1)
        acc2 = (preds == groups).mean()
        self.assertLessEqual(acc2, 0.21)  # chance = 0.2

    def test_original_features_readable(self):
        features, groups = _synthetic(10, 5, 64, seed=1)
        from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
        acc = RidgeAccumulator(64, 5)
        acc.add(features, groups)
        weights = acc.solve(1.0)
        scores = acc.standardiser()(features).astype(np.float64) \
            @ weights[:-1] + weights[-1]
        self.assertGreaterEqual((np.argmax(scores, axis=1)
                                 == groups).mean(), 0.99)

    def test_certificate_relative_residual_tiny(self):
        features, groups = _synthetic(10, 5, 64, seed=2)
        eraser, _ = leace_eraser(features, groups, group_count=5,
                                 floor=1e-10, singular_tolerance=1e-10)
        cert = erasure_certificate(features, groups, 5, eraser)
        self.assertLess(cert["relative_mean_gap_residual"], 1e-6)
        self.assertLess(cert["relative_cross_covariance_residual"], 1e-6)

    def test_eraser_returns_caller_dtype(self):
        features, groups = _synthetic(10, 5, 64, seed=3)
        eraser, _ = leace_eraser(features, groups, group_count=5,
                                 floor=1e-10, singular_tolerance=1e-10)
        self.assertEqual(eraser(features).dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
