"""Unit tests for v26 M315 takedown containment (A10).

Registered semantics (plan §8.16): the quorum scales with the pool
and never sits below the floor; appeals must cite a registered
evidence class; first ratification suspends, permanence requires
re-ratification; the proposer deposit scales with trailing revenue.
"""
from __future__ import annotations

import unittest

from geode.core.economics import (
    SECURITY_FLOORS,
    assert_at_or_above_floor,
)
from geode.core.takedown_containment import (
    APPEAL_EVIDENCE_CLASSES,
    RESPONDER_SCALE,
    SUSPENSION_EPOCHS,
    appeal_admissible,
    min_responders,
    proposer_deposit,
    takedown_step,
)


class TestMinResponders(unittest.TestCase):

    def test_floor_registered_in_security_floors(self):
        self.assertIn("takedown_min_responders", SECURITY_FLOORS)
        self.assertEqual(
            float(SECURITY_FLOORS["takedown_min_responders"]),
            min_responders(0))

    def test_floor_applies_to_small_pools(self):
        for size in (0, 1, 10, 20):
            self.assertGreaterEqual(min_responders(size), 3)

    def test_scales_with_pool(self):
        import math
        self.assertEqual(min_responders(10), 3)
        self.assertEqual(min_responders(50), 5)
        self.assertEqual(min_responders(100), 10)
        self.assertEqual(min_responders(101), 11)
        self.assertEqual(min_responders(1000),
                         int(math.ceil(RESPONDER_SCALE * 1000)))

    def test_non_decreasing_in_pool_size(self):
        values = [min_responders(s) for s in range(0, 200)]
        self.assertEqual(values, sorted(values))

    def test_rejects_negative_pool(self):
        with self.assertRaises(ValueError):
            min_responders(-1)


class TestAppealPath(unittest.TestCase):

    def test_admissible_with_registered_class(self):
        out = appeal_admissible(["probe_mismatch_record"])
        self.assertTrue(out["admissible"])
        self.assertEqual(out["registered_classes_cited"],
                         ["probe_mismatch_record"])

    def test_inadmissible_without_registered_class(self):
        for cited in ([], ["vibes"], ["anecdote", "opinion"]):
            self.assertFalse(appeal_admissible(cited)["admissible"])

    def test_mixed_citation_lists_both_sides(self):
        out = appeal_admissible(["meter_reading", "opinion"])
        self.assertTrue(out["admissible"])
        self.assertEqual(out["registered_classes_cited"],
                         ["meter_reading"])
        self.assertEqual(out["unregistered_classes_cited"],
                         ["opinion"])

    def test_registered_classes_are_the_v26_vocabulary(self):
        self.assertEqual(APPEAL_EVIDENCE_CLASSES, frozenset((
            "probe_mismatch_record", "session_record",
            "meter_reading", "router_trace",
            "reference_run_record", "admission_draw")))


class TestSuspensionLadder(unittest.TestCase):

    def test_first_ratification_suspends_only(self):
        step = takedown_step(1, False)
        self.assertTrue(step["suspended"])
        self.assertFalse(step["delisted"])
        self.assertEqual(step["suspension_epochs"], SUSPENSION_EPOCHS)

    def test_reratification_after_suspension_delists(self):
        step = takedown_step(2, True)
        self.assertTrue(step["delisted"])
        self.assertFalse(step["suspended"])

    def test_reratification_without_window_suspends_again(self):
        step = takedown_step(2, False)
        self.assertTrue(step["suspended"])
        self.assertFalse(step["delisted"])

    def test_rejects_nonpositive_ratification(self):
        with self.assertRaises(ValueError):
            takedown_step(0, False)


class TestProposerDeposit(unittest.TestCase):

    def test_zero_revenue_zero_deposit(self):
        self.assertEqual(proposer_deposit(0.0), 0.0)

    def test_scales_with_revenue(self):
        self.assertEqual(proposer_deposit(100.0), 50.0)
        self.assertEqual(proposer_deposit(1000.0), 500.0)

    def test_monotone_in_revenue(self):
        revenues = [proposer_deposit(r) for r in (0.0, 1.0, 10.0,
                                                  100.0, 1e6)]
        self.assertEqual(revenues, sorted(revenues))

    def test_rejects_negative_revenue(self):
        with self.assertRaises(ValueError):
            proposer_deposit(-1.0)


if __name__ == "__main__":
    unittest.main()
