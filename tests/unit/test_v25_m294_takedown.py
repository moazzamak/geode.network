"""Unit tests for M294: the quorum takedown mechanism.

Deterministic; covers the registered gates G1 (verdict form),
G2 (no self-dealing), G3 (permanence), G4 (distinctness — no credit
movement), G5 (librarian-only filing).
"""
from __future__ import annotations

import unittest

from geode.core.takedown import (
    OPPOSE,
    SUPPORT,
    QuorumTakedown,
    TakedownError,
)

POOL = [f"v{i}" for i in range(20)]
ARTIFACT = "arm:abc123"


def _fresh() -> QuorumTakedown:
    """A takedown registry whose pool validators registered at
    epoch 0 (activation at 2, full tenure weight at 4) and
    performed a round at epoch 6 (recent at the test epoch 7,
    W=2) — every pool member is eligible and weighs 1.0."""
    q = QuorumTakedown(k=9, librarian="librarian")
    for v in POOL:
        q.note_registration(v, epoch=0)
        q.record_round(v, epoch=6, responded=True)
    return q


def _ratified(q: QuorumTakedown, pid: str) -> dict:
    return q.verdict(pid, epoch=7, artifact_id=ARTIFACT, pool=POOL)


class TestVerdictForm(unittest.TestCase):

    def test_full_support_ratifies(self):
        q = _fresh()
        p = q.propose(ARTIFACT, ["entry:1", "entry:2"], "proposer",
                      10.0)
        sampled = q.sampled_set(7, ARTIFACT, POOL)
        for v in sampled:
            q.vote(p.proposal_id, v, SUPPORT, 7, ARTIFACT, POOL)
        result = _ratified(q, p.proposal_id)
        self.assertTrue(result["ratified"], result)
        self.assertEqual(result["support"], 9)
        self.assertEqual(result["total_weight"], 9.0)
        self.assertEqual(result["need_weight"], 6.0)
        self.assertTrue(result["deposit_returned"])

    def test_below_quorum_fails_closed(self):
        q = _fresh()
        p = q.propose(ARTIFACT, ["entry:1"], "proposer", 10.0)
        sampled = q.sampled_set(7, ARTIFACT, POOL)
        # 5 support, 4 oppose: below the 2/3 bar of 6.
        for i, v in enumerate(sampled):
            q.vote(p.proposal_id, v, SUPPORT if i < 5 else OPPOSE,
                   7, ARTIFACT, POOL)
        result = _ratified(q, p.proposal_id)
        self.assertFalse(result["ratified"], result)
        self.assertFalse(result["deposit_returned"])

    def test_below_min_responders_fails_closed(self):
        q = _fresh()
        p = q.propose(ARTIFACT, ["entry:1"], "proposer", 10.0)
        sampled = q.sampled_set(7, ARTIFACT, POOL)
        # Only 2 responders, both supporting: below the min of 3.
        for v in sampled[:2]:
            q.vote(p.proposal_id, v, SUPPORT, 7, ARTIFACT, POOL)
        result = _ratified(q, p.proposal_id)
        self.assertFalse(result["ratified"], result)

    def test_tiny_pool_fails_closed(self):
        q = _fresh()
        p = q.propose(ARTIFACT, ["entry:1"], "proposer", 10.0)
        for v in ["a", "b"]:
            q.vote(p.proposal_id, v, SUPPORT, 7, ARTIFACT, ["a", "b"])
        result = _ratified(q, p.proposal_id)
        self.assertFalse(result["ratified"])


class TestNoSelfDealing(unittest.TestCase):

    def test_off_sample_votes_ignored(self):
        q = _fresh()
        p = q.propose(ARTIFACT, ["entry:1"], "proposer", 10.0)
        sampled = q.sampled_set(7, ARTIFACT, POOL)
        off_sample = [v for v in POOL if v not in sampled]
        for v in off_sample:
            q.vote(p.proposal_id, v, SUPPORT, 7, ARTIFACT, POOL)
        self.assertEqual(p.votes, {})
        result = _ratified(q, p.proposal_id)
        self.assertFalse(result["ratified"])

    def test_duplicate_vote_counts_once(self):
        q = _fresh()
        p = q.propose(ARTIFACT, ["entry:1"], "proposer", 10.0)
        sampled = q.sampled_set(7, ARTIFACT, POOL)
        for v in sampled:
            q.vote(p.proposal_id, v, SUPPORT, 7, ARTIFACT, POOL)
        # A late oppose from an already-supporting validator is
        # ignored: the first vote stands.
        q.vote(p.proposal_id, sampled[0], OPPOSE, 7, ARTIFACT, POOL)
        self.assertEqual(p.votes[sampled[0]], SUPPORT)
        result = _ratified(q, p.proposal_id)
        self.assertTrue(result["ratified"])

    def test_bad_choice_raises(self):
        q = _fresh()
        p = q.propose(ARTIFACT, ["entry:1"], "proposer", 10.0)
        with self.assertRaises(TakedownError):
            q.vote(p.proposal_id, POOL[0], "abstain", 7, ARTIFACT, POOL)


class TestVoterEligibility(unittest.TestCase):

    def test_fresh_registrations_are_never_sampled(self):
        # Nine validators registered only at epoch 6 (activation 8):
        # at epoch 7 the sampled set is EMPTY and the verdict fails
        # closed regardless of votes.
        q = QuorumTakedown(k=9, librarian="librarian")
        flood = [f"f{i}" for i in range(9)]
        for v in flood:
            q.note_registration(v, epoch=6)
        p = q.propose(ARTIFACT, ["entry:1"], "proposer", 10.0)
        for v in flood:
            q.vote(p.proposal_id, v, SUPPORT, 7, ARTIFACT, flood)
        result = q.verdict(p.proposal_id, 7, ARTIFACT, flood)
        self.assertFalse(result["ratified"], result)
        self.assertEqual(result["sampled"], [])
        self.assertEqual(result["total_weight"], 0.0)

    def test_dormant_validator_fails_the_activity_floor(self):
        # A dedicated registry: v0 registered at epoch 0, sampled
        # four times, never responded once — fails BOTH the activity
        # floor and recency.
        q = QuorumTakedown(librarian="librarian")
        q.note_registration("v0", epoch=0)
        for _ in range(4):
            q.record_round("v0", epoch=6, responded=False)
        self.assertFalse(q.is_active("v0"))
        self.assertFalse(q.is_recent("v0", 7))
        self.assertFalse(q.is_eligible("v0", 7))
        self.assertNotIn("v0", q.sampled_set(7, ARTIFACT, ["v0"]))
        self.assertEqual(q.voter_weight("v0", 7), 0.0)

    def test_silent_veteran_loses_vote_weight(self):
        # Full tenure does not survive silence: v0 last responded at
        # epoch 3; at epoch 7 that is outside the W=2 recency window.
        q = QuorumTakedown(librarian="librarian")
        q.note_registration("v0", epoch=0)
        q.record_round("v0", epoch=2, responded=True)
        q.record_round("v0", epoch=3, responded=True)
        self.assertTrue(q.is_recent("v0", 4))
        self.assertFalse(q.is_recent("v0", 7))
        self.assertFalse(q.is_eligible("v0", 7))
        self.assertEqual(q.voter_weight("v0", 7), 0.0)

    def test_recency_restored_by_work(self):
        q = QuorumTakedown(librarian="librarian")
        q.note_registration("v0", epoch=0)
        q.record_round("v0", epoch=3, responded=True)
        self.assertFalse(q.is_eligible("v0", 7))
        # One performed round inside the window restores eligibility.
        q.record_round("v0", epoch=6, responded=True)
        self.assertTrue(q.is_eligible("v0", 7))
        self.assertEqual(q.voter_weight("v0", 7), 1.0)

    def test_recent_work_never_bypasses_activation(self):
        # A validator that registers AND performs at epoch 6 still
        # cannot be sampled before its activation at epoch 8.
        q = QuorumTakedown(librarian="librarian")
        q.note_registration("f", epoch=6)
        q.record_round("f", epoch=6, responded=True)
        self.assertTrue(q.is_recent("f", 7))
        self.assertFalse(q.is_eligible("f", 7))
        self.assertEqual(q.voter_weight("f", 7), 0.0)

    def test_tenure_weight_ramps_from_zero(self):
        q = _fresh()
        # v0 activated at epoch 2; tenure at epoch 7 is 5 -> full.
        self.assertEqual(q.voter_weight("v0", 7), 1.0)
        # A validator activated at epoch 5 has tenure 2 at epoch 7
        # with T=4 -> weight 0.5 (recent round at epoch 6).
        q2 = QuorumTakedown(librarian="librarian")
        q2.note_registration("new", epoch=3)   # activation 5
        q2.record_round("new", epoch=6, responded=True)
        self.assertEqual(q2.voter_weight("new", 7), 0.5)
        self.assertEqual(q2.voter_weight("new", 5), 0.0)

    def test_flood_cannot_move_the_quorum(self):
        # Established pool: 9 veterans at full weight. The attacker
        # registers 20 fresh validators on the eve of the vote —
        # none are eligible, so the veterans' votes decide alone.
        q = _fresh()
        flood = [f"f{i}" for i in range(20)]
        for v in flood:
            q.note_registration(v, epoch=6)
        pool = POOL + flood
        p = q.propose(ARTIFACT, ["entry:1"], "proposer", 10.0)
        for v in flood:
            q.vote(p.proposal_id, v, OPPOSE, 7, ARTIFACT, pool)
        sampled = q.sampled_set(7, ARTIFACT, pool)
        self.assertEqual(len(sampled), 9)
        self.assertFalse(any(v in flood for v in sampled))
        for v in sampled:
            q.vote(p.proposal_id, v, SUPPORT, 7, ARTIFACT, pool)
        result = q.verdict(p.proposal_id, 7, ARTIFACT, pool)
        self.assertTrue(result["ratified"], result)
        self.assertEqual(result["total_weight"], 9.0)


class TestPermanenceAndDistinctness(unittest.TestCase):

    def test_delist_is_permanent(self):
        q = _fresh()
        q.file_delist(ARTIFACT, "quorum:hash1", by="librarian")
        self.assertTrue(q.is_delisted(ARTIFACT))
        self.assertIn(ARTIFACT, q.delisted())
        # No un-delist API exists; filing again stays delisted.
        q.file_delist(ARTIFACT, "quorum:hash2", by="librarian")
        self.assertTrue(q.is_delisted(ARTIFACT))

    def test_non_librarian_cannot_file(self):
        q = _fresh()
        with self.assertRaises(TakedownError):
            q.file_delist(ARTIFACT, "quorum:hash1", by="attacker")

    def test_takedown_moves_no_credits(self):
        # G4: the module holds no monetary state at all.
        q = _fresh()
        for attr in ("credits_of", "burned_total", "pool", "dev_fund"):
            self.assertFalse(hasattr(q, attr), attr)


class TestProposals(unittest.TestCase):

    def test_proposal_requires_evidence_and_deposit(self):
        q = _fresh()
        with self.assertRaises(TakedownError):
            q.propose(ARTIFACT, [], "proposer", 10.0)
        with self.assertRaises(TakedownError):
            q.propose(ARTIFACT, ["entry:1"], "proposer", 0.0)

    def test_duplicate_proposal_id_raises(self):
        q = _fresh()
        q.propose(ARTIFACT, ["entry:1"], "proposer", 10.0,
                  proposal_id="p1")
        with self.assertRaises(TakedownError):
            q.propose(ARTIFACT, ["entry:2"], "proposer", 10.0,
                      proposal_id="p1")


if __name__ == "__main__":
    unittest.main()
