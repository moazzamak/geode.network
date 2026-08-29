"""Unit tests for v26 M312 librarian containment (A14).

Registered semantics (plan §8.17): force-inclusion entries past their
window invalidate the chain; replacement needs a recorded reason and
a registered endorsement level; liveness statistics flag a stopped
librarian.

Amended by review-R2:
- **M365 (G24)** the per-epoch incorporation obligation is capped and
  a backlog rolls forward in posting order; incorporation is FIFO.
- **M366 (G25)** replacement runs on earned weight with the pedigree
  gate, the 20% cap, the diversity floor and the two-thirds bar —
  not on an unweighted headcount of registered validators.
"""
from __future__ import annotations

import math
import unittest

from geode.core.librarian_containment import (
    INCLUSION_WINDOW_EPOCHS,
    MAX_INCORPORATIONS_PER_EPOCH,
    chain_valid,
    deadline_epoch,
    deputy,
    due_entries,
    incorporate,
    liveness_report,
    post,
    replacement,
    successor_order,
)
from geode.privacy.vote_machinery import ratifies


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


class TestCappedObligation(unittest.TestCase):
    """M365 (G24): the obligation is capped, so a spam campaign
    cannot buy chain-invalidity."""

    def test_backlog_within_the_cap_keeps_the_plain_window(self):
        for backlog in range(MAX_INCORPORATIONS_PER_EPOCH):
            self.assertEqual(
                deadline_epoch(5, backlog),
                5 + INCLUSION_WINDOW_EPOCHS)

    def test_backlog_beyond_the_cap_rolls_forward(self):
        cap = MAX_INCORPORATIONS_PER_EPOCH
        self.assertEqual(deadline_epoch(5, cap),
                         5 + INCLUSION_WINDOW_EPOCHS + 1)
        self.assertEqual(deadline_epoch(5, 3 * cap),
                         5 + INCLUSION_WINDOW_EPOCHS + 3)

    def test_spam_campaign_does_not_invalidate_the_chain(self):
        # the pre-repair rule obliged every entry within one window,
        # so this campaign invalidated the chain outright
        queue: list = []
        n = 3 * MAX_INCORPORATIONS_PER_EPOCH
        for k in range(n):
            post(queue, f"spam-{k}", epoch=0)
        # the librarian works at exactly the capped rate
        epoch = 0
        for k in range(n):
            self.assertTrue(chain_valid(queue, epoch=epoch))
            incorporate(queue, f"spam-{k}", epoch=epoch)
            if (k + 1) % MAX_INCORPORATIONS_PER_EPOCH == 0:
                epoch += 1
        self.assertTrue(chain_valid(queue, epoch=epoch))

    def test_a_censored_entry_still_has_a_finite_bound(self):
        queue: list = []
        backlog = 2 * MAX_INCORPORATIONS_PER_EPOCH
        for k in range(backlog):
            post(queue, f"spam-{k}", epoch=0)
        post(queue, "censored", epoch=0)
        entry = queue[-1]
        self.assertEqual(entry["deadline_epoch"],
                         INCLUSION_WINDOW_EPOCHS + 2)
        self.assertLess(entry["deadline_epoch"], math.inf)

    def test_deadlines_are_monotonic_in_posting_order(self):
        queue: list = []
        for k in range(20):
            post(queue, f"e-{k}", epoch=k)
        deadlines = [e["deadline_epoch"] for e in queue]
        self.assertEqual(deadlines, sorted(deadlines))

    def test_incorporation_is_fifo(self):
        queue: list = []
        post(queue, "rival", epoch=0)
        post(queue, "friend", epoch=0)
        with self.assertRaises(ValueError):
            incorporate(queue, "friend", epoch=0)
        incorporate(queue, "rival", epoch=0)
        incorporate(queue, "friend", epoch=0)

    def test_rejects_a_nonpositive_cap(self):
        with self.assertRaises(ValueError):
            deadline_epoch(0, 0, cap=0)


class TestReplacement(unittest.TestCase):
    """M366 (G25): the replacement endorsement runs on earned weight."""

    def setUp(self):
        self.validators = [f"v{i}" for i in range(10)]
        self.weights = {v: 1.0 for v in self.validators}

    def _call(self, reason, endorsers, **over):
        kw = {"verified_weights": self.weights, "responders": 10,
              "pool_size": 10, "ratify": ratifies,
              "pedigreed": self.validators}
        kw.update(over)
        return replacement(reason, endorsers, **kw)

    def test_no_reason_no_replacement(self):
        out = self._call(None, self.validators)
        self.assertFalse(out["fires"])
        self.assertFalse(out["has_recorded_reason"])
        self.assertEqual(out["reason"], "no_recorded_reason")

    def test_below_two_thirds_no_replacement(self):
        out = self._call("divergence", self.validators[:6])
        self.assertFalse(out["fires"])
        self.assertEqual(out["reason"], "below_two_thirds")

    def test_at_and_above_two_thirds_fires(self):
        for k in (7, 8, 10):
            out = self._call("divergence", self.validators[:k])
            self.assertTrue(out["fires"], msg=f"{k} endorsers")
            self.assertGreaterEqual(out["support_fraction"], 2 / 3)

    def test_the_old_headcount_majority_no_longer_suffices(self):
        # the pre-repair rule fired at half of the registered
        # validators; five of ten is now short of two-thirds
        out = self._call("divergence", self.validators[:5])
        self.assertFalse(out["fires"])
        self.assertAlmostEqual(out["support_fraction"], 0.5)

    def test_a_sybil_fleet_with_no_earned_weight_cannot_fire(self):
        # G25's actual attack: identities are cheap, earned weight is
        # not. Forty weightless identities outnumber the validators
        # four to one and still carry nothing.
        sybils = [f"s{i}" for i in range(40)]
        weights = dict(self.weights)
        weights.update({s: 0.0 for s in sybils})
        out = replacement("divergence", sybils,
                          verified_weights=weights, responders=50,
                          pool_size=50, ratify=ratifies,
                          pedigreed=self.validators + sybils)
        self.assertFalse(out["fires"])
        self.assertEqual(out["support_weight"], 0.0)

    def test_unpedigreed_endorsements_are_dropped_and_counted(self):
        out = self._call("divergence",
                         self.validators[:7] + ["stranger"])
        self.assertEqual(out["unpedigreed_dropped"], ["stranger"])
        self.assertNotIn("stranger", out["eligible_endorsers"])
        self.assertTrue(out["fires"])

    def test_the_twenty_percent_cap_binds(self):
        # one whale holding 90% of the weight cannot carry a vote
        weights = {"whale": 90.0}
        weights.update({v: 10.0 / 9 for v in self.validators[:9]})
        out = replacement("divergence", ["whale"],
                          verified_weights=weights, responders=10,
                          pool_size=10, ratify=ratifies,
                          pedigreed=["whale"] + self.validators)
        self.assertFalse(out["fires"])
        self.assertLessEqual(out["support_fraction"], 0.2 + 1e-9)

    def test_the_diversity_floor_binds(self):
        # four identities carry 80% of the weight — clear of
        # two-thirds — but a 50-strong responder pool needs d >= 10
        weights = {f"h{i}": 1.0 for i in range(5)}
        weights.update({f"l{i}": 0.0 for i in range(45)})
        out = replacement("divergence", ["h0", "h1", "h2", "h3"],
                          verified_weights=weights, responders=50,
                          pool_size=50, ratify=ratifies)
        self.assertFalse(out["fires"])
        self.assertGreaterEqual(out["support_fraction"], 2 / 3)
        self.assertEqual(out["reason"], "below_diversity_floor")

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            self._call("x", self.validators, responders=0)
        with self.assertRaises(ValueError):
            self._call("x", self.validators, pool_size=0)
        with self.assertRaises(ValueError):
            self._call("x", ["v0", "v0"])


class TestSuccessorOrderAndDeputy(unittest.TestCase):
    """M388 (M382 remainder): the deterministic successor order and
    the named deputy — the "deterministic successor order" that was
    prose in the launch plan and absent from code."""

    def setUp(self):
        self.roster = [f"v{i}" for i in range(8)]

    def test_order_is_deterministic(self):
        a = successor_order(self.roster, epoch=3,
                            anchor_hash=b"anchor")
        b = successor_order(self.roster, epoch=3,
                            anchor_hash=b"anchor")
        self.assertEqual(a, b)
        self.assertEqual(sorted(a), sorted(self.roster))

    def test_order_depends_on_epoch_and_anchor(self):
        o1 = successor_order(self.roster, epoch=3, anchor_hash=b"a")
        o2 = successor_order(self.roster, epoch=4, anchor_hash=b"a")
        o3 = successor_order(self.roster, epoch=3, anchor_hash=b"b")
        self.assertNotEqual(o1, o2)
        self.assertNotEqual(o1, o3)

    def test_incumbent_is_excluded(self):
        order = successor_order(self.roster, epoch=3, exclude=["v2"])
        self.assertNotIn("v2", order)
        self.assertEqual(len(order), len(self.roster) - 1)

    def test_deputy_is_first_eligible_non_incumbent(self):
        d = deputy(self.roster, epoch=3, anchor_hash=b"anchor",
                   exclude=["v0"], eligible=self.roster)
        order = successor_order(self.roster, epoch=3,
                                anchor_hash=b"anchor", exclude=["v0"])
        self.assertEqual(d, order[0])
        self.assertNotEqual(d, "v0")

    def test_deputy_honours_the_eligibility_gate(self):
        # only identities {v3, v5} are pedigreed: the deputy must be
        # the first of those in the successor order
        eligible = {"v3", "v5"}
        d = deputy(self.roster, epoch=7, anchor_hash=b"x",
                   eligible=eligible)
        order = successor_order(self.roster, epoch=7, anchor_hash=b"x")
        expected = next(i for i in order if i in eligible)
        self.assertEqual(d, expected)

    def test_empty_roster_names_nobody(self):
        self.assertIsNone(deputy([], epoch=1))
        self.assertIsNone(deputy(["only"], epoch=1, exclude=["only"]))

    def test_rejects_a_negative_epoch(self):
        with self.assertRaises(ValueError):
            successor_order(self.roster, epoch=-1)

    def test_replacement_names_the_deputy_when_it_fires(self):
        # M388: replacement() no longer "fires and names nobody" —
        # given the roster and the epoch it names the deputy.
        kw = {"verified_weights": {v: 1.0 for v in self.roster},
              "responders": 8, "pool_size": 8, "ratify": ratifies,
              "pedigreed": self.roster,
              "successor_identities": self.roster, "epoch": 3,
              "anchor_hash": b"anchor", "exclude_librarian": "v0"}
        out = replacement("divergence", self.roster, **kw)
        self.assertTrue(out["fires"])
        self.assertTrue(out["deputy_named"])
        self.assertIsNotNone(out["deputy"])
        self.assertNotEqual(out["deputy"], "v0")
        self.assertEqual(out["deputy"],
                         successor_order(self.roster, epoch=3,
                                         anchor_hash=b"anchor",
                                         exclude=["v0"])[0])

    def test_replacement_without_a_roster_names_nobody(self):
        # backward compatible: existing callers pass no roster, and the
        # result says so rather than pretending
        kw = {"verified_weights": {v: 1.0 for v in self.roster},
              "responders": 8, "pool_size": 8, "ratify": ratifies,
              "pedigreed": self.roster}
        out = replacement("divergence", self.roster, **kw)
        self.assertTrue(out["fires"])
        self.assertFalse(out["deputy_named"])
        self.assertIsNone(out["deputy"])
        self.assertEqual(out["deputy_roster_size"], 0)

    def test_replacement_requires_epoch_with_a_roster(self):
        with self.assertRaises(ValueError):
            replacement("divergence", self.roster,
                        verified_weights={v: 1.0 for v in self.roster},
                        responders=8, pool_size=8, ratify=ratifies,
                        pedigreed=self.roster,
                        successor_identities=self.roster)


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
