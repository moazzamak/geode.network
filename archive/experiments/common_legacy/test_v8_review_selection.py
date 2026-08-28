from __future__ import annotations

import unittest

import numpy as np

from experiments.common.v8_review_selection import (
    boundary_inclusive_indices,
    core_indices,
    kcenter_indices,
    paired_bootstrap_interval,
    random_stratified_indices,
)


class V8ReviewSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = np.random.default_rng(47).normal(size=(120, 8))
        self.margin = np.linspace(-1.0, 1.0, len(self.features))

    def test_all_selectors_respect_equal_budget_and_replay(self):
        selectors = (
            core_indices(self.features, 50),
            kcenter_indices(self.features, 50),
            random_stratified_indices(self.features, 50, 47),
            boundary_inclusive_indices(self.features, self.margin, 50, 47),
        )
        for selected in selectors:
            self.assertEqual(len(selected), 50)
            self.assertEqual(len(set(selected.tolist())), 50)
        self.assertTrue(
            np.array_equal(
                kcenter_indices(self.features, 50),
                kcenter_indices(self.features, 50),
            )
        )

    def test_bootstrap_interval_is_paired_and_deterministic(self):
        first = np.array([0.2, 0.3, 0.4])
        second = np.array([0.1, 0.2, 0.3])
        one = paired_bootstrap_interval(
            first, second, confidence=0.95, n_resamples=1000, seed=47
        )
        two = paired_bootstrap_interval(
            first, second, confidence=0.95, n_resamples=1000, seed=47
        )
        self.assertEqual(one, two)
        self.assertGreater(one["lower"], 0.0)

    def test_invalid_budget_fails_closed(self):
        with self.assertRaises(ValueError):
            core_indices(self.features, len(self.features) + 1)


if __name__ == "__main__":
    unittest.main()
