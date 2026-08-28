"""Unit tests for the M319 adjudication rules (A18 selective abort,
A19 quorum-failure resampling)."""
from __future__ import annotations

import unittest

from geode.core.probe_adjudication import (
    LEVEL_DEVIATION,
    LEVEL_DOWNTIME,
    adjudicate_probed_session,
    quorum_failure_plan,
)


class TestAdjudicateProbedSession(unittest.TestCase):

    def test_opened_match_is_clean(self):
        out = adjudicate_probed_session(commit_opened=True, probed=True,
                                        answers_match=True)
        self.assertEqual(out["verdict"], "clean")
        self.assertIsNone(out["ladder_level"])

    def test_opened_mismatch_is_deviation(self):
        out = adjudicate_probed_session(commit_opened=True, probed=True,
                                        answers_match=False)
        self.assertEqual(out["verdict"], "deviation")
        self.assertEqual(out["ladder_level"], LEVEL_DEVIATION)

    def test_unopened_probed_is_deviation(self):
        # A18: the selective abort is a refused inspection, not
        # downtime - it must cost at least as much as a mismatch
        out = adjudicate_probed_session(commit_opened=False, probed=True,
                                        answers_match=False)
        self.assertEqual(out["verdict"], "deviation")
        self.assertEqual(out["ladder_level"], LEVEL_DEVIATION)

    def test_unopened_unprobed_is_downtime(self):
        out = adjudicate_probed_session(commit_opened=False,
                                        probed=False,
                                        answers_match=False)
        self.assertEqual(out["verdict"], "downtime")
        self.assertEqual(out["ladder_level"], LEVEL_DOWNTIME)

    def test_abort_costs_no_less_than_mismatch(self):
        abort = adjudicate_probed_session(False, True, False)
        mismatch = adjudicate_probed_session(True, True, False)
        self.assertEqual(abort["ladder_level"], mismatch["ladder_level"])


class TestQuorumFailurePlan(unittest.TestCase):

    def test_no_op_when_quorum_met(self):
        plan = quorum_failure_plan(responders=6, sampled=9,
                                   quorum_num=2, quorum_den=3,
                                   unspent_budget=100)
        self.assertFalse(plan["quorum_failed"])
        self.assertFalse(plan["resample"])

    def test_resample_and_carry_budget(self):
        plan = quorum_failure_plan(responders=3, sampled=9,
                                   quorum_num=2, quorum_den=3,
                                   unspent_budget=100)
        self.assertTrue(plan["quorum_failed"])
        self.assertTrue(plan["resample"])
        self.assertFalse(plan["new_fee_charged"])
        self.assertEqual(plan["budget_carried_forward"], 100)
        self.assertEqual(plan["needed"], 6)

    def test_demerit_weighted_by_proximity(self):
        far = quorum_failure_plan(responders=0, sampled=9,
                                  quorum_num=2, quorum_den=3,
                                  unspent_budget=10)
        near = quorum_failure_plan(responders=5, sampled=9,
                                   quorum_num=2, quorum_den=3,
                                   unspent_budget=10)
        self.assertEqual(far["demerit_per_non_responder"], 0.0)
        self.assertGreater(near["demerit_per_non_responder"],
                           far["demerit_per_non_responder"])
        self.assertAlmostEqual(near["demerit_per_non_responder"],
                               5 / 9)

    def test_validation(self):
        with self.assertRaises(ValueError):
            quorum_failure_plan(-1, 9, 2, 3, 0)
        with self.assertRaises(ValueError):
            quorum_failure_plan(10, 9, 2, 3, 0)
        with self.assertRaises(ValueError):
            quorum_failure_plan(3, 9, 3, 2, 0)   # quorum not proper
        with self.assertRaises(ValueError):
            quorum_failure_plan(3, 9, 2, 3, -1)


if __name__ == "__main__":
    unittest.main()
