"""Unit tests for M180: the attribution estimators.

Pure stdlib + math; no data, no GPU. The synthetic 3-arm game has
known hand-computed values, so the estimators are checked against
exact arithmetic, not against each other.
"""
from __future__ import annotations

import unittest

from geode.attribution.attribution import (
    beta_shapley,
    fingerprint_coverage,
    leave_one_out,
    rank_order,
    ranking_stability,
    shapley,
)

PLAYERS = ["a", "b", "c"]
# Hand-computed game:
# V(a)=10, V(b)=20, V(c)=30, V(ab)=40, V(ac)=50, V(bc)=70, V(abc)=100.
V = {
    frozenset(): 0.0,
    frozenset("a"): 10.0,
    frozenset("b"): 20.0,
    frozenset("c"): 30.0,
    frozenset("ab"): 40.0,
    frozenset("ac"): 50.0,
    frozenset("bc"): 70.0,
    frozenset("abc"): 100.0,
}


class TestAttribution(unittest.TestCase):

    def test_shapley_matches_hand_computation(self):
        # classic weights s!(n-s-1)!/n!: sizes 0,1,2 -> 1/3 each size,
        # coalitions of size 1 get 1/6, size 2 get 1/3.
        # phi_a = (V(a)-V())/3 + [(V(ab)-V(b))+(V(ac)-V(c))]/6
        #       + (V(abc)-V(bc))/3
        #       = 10/3 + (20+20)/6 + 30/3 = 3.3333+6.6667+10 = 20
        phi = shapley(V, PLAYERS)
        self.assertAlmostEqual(phi["a"], 20.0, places=10)
        # phi_b = 20/3 + [(40-10)+(70-30)]/6 + (100-50)/3 = 6.6667+11.6667+16.6667 = 35
        self.assertAlmostEqual(phi["b"], 35.0, places=10)
        # phi_c = 30/3 + [(50-10)+(70-20)]/6 + (100-40)/3 = 10+15+20 = 45
        self.assertAlmostEqual(phi["c"], 45.0, places=10)

    def test_shapley_efficiency(self):
        phi = shapley(V, PLAYERS)
        self.assertAlmostEqual(sum(phi.values()), V[frozenset("abc")],
                               places=10)

    def test_beta_1_is_shapley(self):
        self.assertEqual(beta_shapley(V, PLAYERS, beta=1.0),
                         shapley(V, PLAYERS))

    def test_beta_weights_normalized(self):
        from math import comb

        from geode.attribution.attribution import _coalition_weights
        for beta in (1.0, 16.0):
            weights = _coalition_weights(3, beta)
            # per-coalition weights: the probability mass sums with the
            # coalition multiplicities C(n-1, s), not alone.
            mass = sum(comb(2, s) * w for s, w in enumerate(weights))
            self.assertAlmostEqual(mass, 1.0, places=12)
            for w in weights:
                self.assertGreaterEqual(w, 0.0)

    def test_loo_marginals(self):
        loo = leave_one_out(V, PLAYERS)
        self.assertAlmostEqual(loo["a"], 100.0 - 70.0)
        self.assertAlmostEqual(loo["b"], 100.0 - 50.0)
        self.assertAlmostEqual(loo["c"], 100.0 - 40.0)

    def test_coverage_shares_sum_to_one(self):
        cov = fingerprint_coverage([1.0, 0.0],
                                   {"a": [1.0, 0.0], "b": [0.0, 1.0]})
        self.assertAlmostEqual(sum(cov.values()), 1.0)
        self.assertAlmostEqual(cov["a"], 1.0)
        self.assertAlmostEqual(cov["b"], 0.0)

    def test_ranking_stability_identical_rankings(self):
        self.assertEqual(ranking_stability([{"a": 1, "b": 2},
                                            {"a": 1, "b": 2}]), 1.0)

    def test_ranking_stability_reversed(self):
        tau = ranking_stability([{"a": 1, "b": 2}, {"a": 2, "b": 1}])
        self.assertEqual(tau, -1.0)

    def test_rank_order_lexicographic_ties(self):
        self.assertEqual(rank_order({"a": 1.0, "b": 1.0, "c": 2.0}),
                         ["c", "a", "b"])


if __name__ == "__main__":
    unittest.main()
