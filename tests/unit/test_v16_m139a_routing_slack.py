"""Focused tests for the M139a dual accumulator (no corpus, no GPU).

Pins the one property the measurement depends on: the single-pass dual
accumulator produces the SAME normal equations as the sealed RidgeAccumulator
for each target set, and the closed-form solves recover linearly separable
targets.
"""

from __future__ import annotations

import unittest

import numpy as np

from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v16_m139a_routing_slack import DualAccumulator, _score


class DualAccumulatorTests(unittest.TestCase):
    def test_matches_ridge_accumulator_per_target(self):
        rng = np.random.default_rng(11)
        features = rng.normal(size=(120, 9))
        labels = rng.integers(0, 5, size=120)
        domains = rng.integers(0, 6, size=120)

        dual = DualAccumulator(9, 5, 6)
        dual.add(features, labels, domains)
        ref_class = RidgeAccumulator(9, 5)
        ref_class.add(features, labels)
        ref_domain = RidgeAccumulator(9, 6)
        ref_domain.add(features, domains)

        self.assertTrue(np.allclose(dual.gram, ref_class.gram))
        self.assertTrue(np.allclose(dual.gram, ref_domain.gram))
        self.assertTrue(np.allclose(dual.cross_class, ref_class.cross))
        self.assertTrue(np.allclose(dual.cross_domain, ref_domain.cross))
        self.assertTrue(np.allclose(dual.class_count, ref_class.class_count))
        self.assertTrue(np.allclose(dual.domain_count, ref_domain.class_count))

    def test_solutions_match_ridge_accumulator(self):
        rng = np.random.default_rng(23)
        features = rng.normal(size=(200, 10))
        labels = rng.integers(0, 4, size=200)
        domains = rng.integers(0, 3, size=200)

        dual = DualAccumulator(10, 4, 3)
        dual.add(features, labels, domains)
        w_dual = dual.solve(1.0, dual.cross_class, dual.class_count)

        ref = RidgeAccumulator(10, 4)
        ref.add(features, labels)
        w_ref = ref.solve_many([1.0])[1.0]
        self.assertTrue(np.allclose(w_dual, w_ref, atol=1e-10))

    def test_domain_solve_separates_separable_domains(self):
        rng = np.random.default_rng(7)
        width = 6
        blocks = []
        domains = []
        for d in range(3):
            centre = np.zeros(width)
            centre[d] = 5.0
            block = rng.uniform(-0.5, 0.5, size=(60, width)) + centre
            blocks.append(block)
            domains.append(np.full(60, d))
        features = np.vstack(blocks)
        domains = np.concatenate(domains)

        acc = RidgeAccumulator(width, 3)
        acc.add(features, domains)
        weights = acc.solve_many([1.0])[1.0]
        standardise = acc.standardiser()
        predictions = _score(weights, standardise(features))
        self.assertTrue(np.array_equal(predictions, domains))


if __name__ == "__main__":
    unittest.main()
