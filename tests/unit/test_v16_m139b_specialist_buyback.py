"""Focused tests for the M139b arm-assembly logic (no corpus, no GPU)."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.tier4.eval_v16_m139b_specialist_buyback import (
    _assemble_arms,
    _per_domain_accuracy,
)


class PerDomainAccuracyTests(unittest.TestCase):
    def test_per_domain_and_global(self):
        labels = np.array([0, 1, 2, 0, 1, 2, 0])
        domains = np.array([0, 0, 0, 3, 3, 5, 5])
        predictions = np.array([0, 1, 1, 0, 0, 2, 0])
        accuracy, per_domain = _per_domain_accuracy(predictions, labels, domains)
        self.assertAlmostEqual(accuracy, 5 / 7)
        self.assertAlmostEqual(per_domain[0], 2 / 3)
        self.assertAlmostEqual(per_domain[3], 0.5)
        self.assertAlmostEqual(per_domain[5], 1.0)
        self.assertAlmostEqual(per_domain[1], 0.0)  # empty domain -> 0.0


class ArmAssemblyTests(unittest.TestCase):
    def _fixture(self):
        # 6 rows; specialist predictions (6 domains x 6 rows), each specialist
        # predicts a distinctive value so routing choice is observable.
        rows = np.arange(6)
        specialist = np.array([
            [10, 10, 10, 10, 10, 10],
            [11, 11, 11, 11, 11, 11],
            [12, 12, 12, 12, 12, 12],
            [13, 13, 13, 13, 13, 13],
            [14, 14, 14, 14, 14, 14],
            [15, 15, 15, 15, 15, 15],
        ])
        true_domains = np.array([0, 1, 2, 3, 4, 5])
        router = np.array([0, 1, 2, 3, 4, 5])  # perfect routing
        margins = np.array([1.0, 0.5, 0.1, 0.05, 0.2, 0.9])
        global_preds = np.array([100, 100, 100, 100, 100, 100])
        return specialist, true_domains, router, margins, global_preds, rows

    def test_oracle_and_routed_with_perfect_router_agree(self):
        specialist, true_domains, router, margins, global_preds, rows = self._fixture()
        arms = _assemble_arms(specialist, router, margins, global_preds,
                              true_domains, [0.0, 0.2046])
        self.assertTrue(np.array_equal(arms["oracle"], arms["routed"]))
        self.assertTrue(np.array_equal(arms["oracle"],
                                       np.array([10, 11, 12, 13, 14, 15])))

    def test_gated_falls_back_below_threshold(self):
        specialist, true_domains, router, margins, global_preds, rows = self._fixture()
        # Misroute row 2 and give it a tiny margin: below tau=0.2046 it must
        # fall back to the global head.
        router = np.array([0, 1, 0, 3, 4, 5])
        margins = np.array([1.0, 0.5, 0.1, 0.05, 0.2, 0.9])
        arms = _assemble_arms(specialist, router, margins, global_preds,
                              true_domains, [0.2046])
        gated = arms["gated(0.2046)"]
        # Row 2: routed prediction would be specialist 0's value (10); margin
        # 0.1 < 0.2046 -> global (100).
        self.assertEqual(gated[2], 100)
        # Row 4: margin 0.2 < 0.2046? No: 0.2 < 0.2046 -> also global.
        self.assertEqual(gated[4], 100)
        # Rows with margin >= tau stay routed.
        self.assertEqual(gated[0], 10)
        self.assertEqual(gated[1], 11)


if __name__ == "__main__":
    unittest.main()
