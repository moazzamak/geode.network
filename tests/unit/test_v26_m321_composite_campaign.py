"""M321 and coverage-metric unit tests."""
from __future__ import annotations

import unittest

from geode.core.composite_campaign import (
    build_campaign,
    campaign_gate,
)
from geode.core.coverage_adjusted import (
    AxisScore,
    MissingCoverage,
    compare,
    coverage_adjusted_score,
    refuse_missing_coverage,
)


class TestCoverageAdjustedMetric(unittest.TestCase):
    def test_metric_is_accuracy_times_coverage(self):
        self.assertAlmostEqual(coverage_adjusted_score(0.9, 0.5), 0.45)

    def test_monotone_in_coverage_at_fixed_accuracy(self):
        for cov in (0.1, 0.3, 0.6, 1.0):
            self.assertAlmostEqual(
                coverage_adjusted_score(0.8, cov), 0.8 * cov)

    def test_abstention_trades_against_score(self):
        scoped = AxisScore(accuracy=0.901, coverage=129.0 / 601.0)
        full = AxisScore(accuracy=0.5, coverage=1.0)
        # the scoped arm's raw accuracy no longer dominates: the
        # coverage-adjusted comparison ranks the full-coverage arm
        # of lower raw accuracy above it
        self.assertEqual(compare(scoped, full), -1)
        self.assertLess(scoped.coverage_adjusted, full.coverage_adjusted)

    def test_missing_coverage_refused(self):
        with self.assertRaises(MissingCoverage):
            refuse_missing_coverage({"accuracy": 0.9})
        score = refuse_missing_coverage({"accuracy": 0.9,
                                         "coverage": 0.4})
        self.assertEqual(score.coverage, 0.4)


class TestCampaign(unittest.TestCase):
    def test_all_rows_present_with_named_repairs(self):
        report = build_campaign()
        self.assertEqual(len(report.rows), 11)
        for row in report.rows:
            self.assertTrue(row.attack)
            self.assertTrue(row.repair)
            self.assertTrue(row.module)

    def test_h26_8_no_step_remains_profitable(self):
        report = build_campaign()
        gate = campaign_gate(report)
        self.assertTrue(gate["h26_8"], gate)
        self.assertEqual(gate["open"], [])

    def test_every_closure_is_attributed(self):
        report = build_campaign()
        gate = campaign_gate(report)
        self.assertEqual(len(gate["attributions"]),
                         gate["closed"])

    def test_cap_is_the_registered_one(self):
        gate = campaign_gate(build_campaign())
        self.assertEqual(gate["cap_bps"], 2000)


if __name__ == "__main__":
    unittest.main()
