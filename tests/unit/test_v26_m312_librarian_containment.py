"""Unit tests for v26 M312 librarian containment (A14).

Registered semantics (plan §8.17): force-inclusion entries past their
window invalidate the chain; replacement needs a recorded reason and
a registered endorsement fraction; liveness statistics flag a stopped
librarian.
"""
from __future__ import annotations

import math
import unittest

from geode.core.librarian_containment import (
    INCLUSION_WINDOW_EPOCHS,
    REPLACEMENT_THRESHOLD,
    chain_valid,
    due_entries,
    incorporate,
    liveness_report,
    post,
    replacement,
)


class TestForceInclusionQueue(unittest.TestCase):

    def test_entry_within_window_keeps_chain_valid(self):
        queue: list = []
        post(queue, "dispute", epoch=5)
        self.assertTrue(chain_valid(queue, epoch=5))
        self.assertTrue(chain_valid(queue, epoch=6))
        self.assertEqual(due_entries(queue, epoch=5), [])
        self.assertEqual(due_entries(queue, epoch=6),
                         [queue[0]])

    def test_entry_past_window_invalidates_chain(self):
        # M312-C1
        queue: list = []
        post(queue, "dispute", epoch=5)
        self.assertTrue(chain_valid(queue, epoch=6))
        self.assertFalse(chain_valid(queue, epoch=7))

    def test_incorporation_within_window_is_clean(self):
        queue: list = []
        post(queue, "e1", epoch=5)
        incorporate(queue, "e1", epoch=6)
        self.assertTrue(chain_valid(queue, epoch=7))
        self.assertFalse(queue[0]["late"])

    def test_late_incorporation_is_recorded(self):
        queue: list = []
        post(queue, "e1", epoch=5)
        incorporate(queue, "e1", epoch=7)
        self.assertTrue(queue[0]["late"])
        self.assertEqual(queue[0]["incorporated_epoch"], 7)

    def test_unknown_entry_raises(self):
        with self.assertRaises(KeyError):
            incorporate([], "nope", epoch=1)


class TestReplacement(unittest.TestCase):

    def test_no_reason_no_replacement(self):
        out = replacement(10, 10, recorded_reason=None)
        self.assertFalse(out["fires"])
        self.assertFalse(out["has_recorded_reason"])

    def test_below_threshold_no_replacement(self):
        # M312-C2
        out = replacement(4, 10, recorded_reason="divergence")
        self.assertFalse(out["fires"])
        self.assertLess(out["endorsement_fraction"],
                        REPLACEMENT_THRESHOLD)

    def test_at_and_above_threshold_fires(self):
        for n, validators in ((5, 10), (6, 10), (10, 10)):
            out = replacement(n, validators,
                              recorded_reason="divergence")
            self.assertTrue(out["fires"])
            self.assertGreaterEqual(out["endorsement_fraction"],
                                    REPLACEMENT_THRESHOLD)

    def test_rejects_bad_counts(self):
        with self.assertRaises(ValueError):
            replacement(11, 10, recorded_reason="x")
        with self.assertRaises(ValueError):
            replacement(0, 0, recorded_reason="x")


class TestLivenessReport(unittest.TestCase):

    def test_healthy_librarian_is_bounded(self):
        report = liveness_report([0, 1, 2, 3, 4], [1, 1, 1, 1])
        self.assertFalse(report["librarian_stopped"])
        self.assertFalse(report["unbounded_latency"])
        self.assertEqual(report["max_anchor_gap"], 1)
        self.assertEqual(report["max_inclusion_latency"], 1)

    def test_stopped_librarian_is_flagged(self):
        # M312-C3
        report = liveness_report([], [])
        self.assertTrue(report["librarian_stopped"])
        self.assertTrue(report["unbounded_latency"])
        self.assertEqual(report["max_anchor_gap"], math.inf)

    def test_no_incorporations_is_unbounded(self):
        report = liveness_report([0, 1, 2], [])
        self.assertFalse(report["librarian_stopped"])
        self.assertTrue(report["unbounded_latency"])


if __name__ == "__main__":
    unittest.main()
