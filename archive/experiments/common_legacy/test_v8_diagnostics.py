from __future__ import annotations

import unittest

import numpy as np

from experiments.common.v8_diagnostics import (
    boundary_inclusive_indices,
    class_count_threshold,
    representativeness_metrics,
    threshold_for_anchor_lineage,
)


class V8DiagnosticsTests(unittest.TestCase):
    def test_class_count_threshold_is_more_conservative(self):
        transferred = class_count_threshold(10.0, 8, 9)
        self.assertLess(transferred, 10.0)

    def test_class_count_threshold_rejects_non_append_transition(self):
        with self.assertRaises(ValueError):
            class_count_threshold(10.0, 8, 10)

    def test_stale_anchor_lineage_restores_parent_threshold(self):
        self.assertEqual(
            threshold_for_anchor_lineage(
                expected_anchor_hash="a" * 64,
                observed_anchor_hash="b" * 64,
                parent_threshold=10.0,
                recalibrated_threshold=9.0,
            ),
            10.0,
        )

    def test_boundary_selection_has_equal_budget_and_distinct_support(self):
        features = np.arange(120, dtype=np.float64).reshape(60, 2)
        core, inclusive = boundary_inclusive_indices(features, 10)
        self.assertEqual(len(core), 10)
        self.assertEqual(len(inclusive), 10)
        self.assertNotEqual(set(core), set(inclusive))

    def test_representativeness_metrics_are_finite(self):
        rng = np.random.default_rng(46)
        full = rng.normal(size=(80, 6))
        selected = full[:20]
        metrics = representativeness_metrics(full, selected, rank=3)
        self.assertEqual(
            set(metrics),
            {
                "covariance_trace_ratio",
                "low_rank_subspace_variance_ratio",
                "omitted_region_nearest_neighbor_coverage",
                "full_rank_energy_fraction",
            },
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
