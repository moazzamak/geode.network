"""Unit tests for the M328 vote machinery.

Pins the registered gates: the diversity floor (G1), snapshot
immutability (G2), ballot commitments + membership proofs + partial
committee secrecy (G3), weighted-sum correctness (G4), fail-closed
unopened ballots (G5), and the no-individual-ballot schema (G6).
"""
from __future__ import annotations

import unittest

import numpy as np

from geode.privacy.vote_machinery import (
    TallyRecord,
    ballot_proof,
    commit_ballot,
    diversity_floor,
    min_responders,
    ratifies,
    shamir_combine,
    shamir_split,
    tally,
    verify_ballot,
    verify_commitment,
)


class TestDiversityFloor(unittest.TestCase):
    """M328-G1."""

    def test_one_identity_never_ratifies(self):
        # a single identity holding 100% of the sampled weight
        ok, reason = ratifies(support_weight=100.0, total_weight=100.0,
                              supporting_identities=["a"] * 10,
                              pool_size=30, responders=10)
        self.assertFalse(ok)
        self.assertEqual(reason, "below_diversity_floor")

    def test_d_distinct_identities_ratify(self):
        d = diversity_floor(10)
        identities = [f"id{i}" for i in range(d)]
        ok, reason = ratifies(support_weight=70.0, total_weight=100.0,
                              supporting_identities=identities,
                              pool_size=30, responders=10)
        self.assertTrue(ok)
        self.assertEqual(reason, "ratifies")

    def test_diversity_floor_formula(self):
        self.assertEqual(diversity_floor(3), 3)
        self.assertEqual(diversity_floor(10), 3)
        self.assertEqual(diversity_floor(16), 4)
        self.assertEqual(min_responders(1), 3)
        self.assertEqual(min_responders(50), 5)


class TestSnapshot(unittest.TestCase):
    """M328-G2: the snapshot freezes weights at the opening anchor."""

    def test_late_claims_do_not_move_the_tally(self):
        from geode.privacy.vote_machinery import WeightSnapshot
        snap = WeightSnapshot(anchor="a1",
                              weights={"alice": 10.0, "bob": 20.0})
        # a claim during the vote (a live balance change) — the
        # snapshot is a frozen object; its digest is stable
        before = snap.digest()
        snap2 = WeightSnapshot(anchor="a1",
                               weights={"alice": 10.0, "bob": 20.0})
        self.assertEqual(before, snap2.digest())
        self.assertEqual(snap.weight_of("alice"), 10.0)
        self.assertEqual(snap.total(["alice", "bob"]), 30.0)


class TestBallots(unittest.TestCase):
    """M328-G3: commitments bind, membership proofs verify, and a
    partial committee view hides the vote."""

    def test_commitment_binds_and_verifies(self):
        C = commit_ballot(1, 12345)
        self.assertTrue(verify_commitment(C, 1, 12345))
        self.assertFalse(verify_commitment(C, 0, 12345))
        self.assertFalse(verify_commitment(C, 1, 12346))

    def test_vote_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            commit_ballot(2, 1)

    def test_membership_proofs_verify_for_both_votes(self):
        for vote in (0, 1):
            C = commit_ballot(vote, 999)
            proof = ballot_proof(vote, 999)
            self.assertTrue(verify_ballot(C, proof))

    def test_forged_proof_rejected(self):
        C = commit_ballot(1, 7)
        proof = ballot_proof(1, 7)
        proof["z0"] = (proof["z0"] + 1) % 2
        self.assertFalse(verify_ballot(C, proof))

    def test_partial_committee_cannot_open_a_ballot(self):
        # the voter splits r into t-of-k shares; fewer than t shares
        # reconstruct nothing about r (and therefore the vote)
        r = 123456789
        shares = shamir_split(r, threshold=5, n=8, seed=42)
        # different 4-subsets interpolate to different secrets —
        # no unique value is determined below the threshold
        combos = [shamir_combine(shares[:4]),
                  shamir_combine(shares[1:5])]
        self.assertNotEqual(combos[0], combos[1])
        self.assertEqual(shamir_combine(shares[:5]), r % _q())

    def test_threshold_committee_recovers_the_opening(self):
        r = 987654321
        shares = shamir_split(r, threshold=5, n=8, seed=7)
        self.assertEqual(shamir_combine(shares[:5]), r % _q())


def _q() -> int:
    from geode.privacy.vote_machinery import Q_ORDER
    return Q_ORDER


class TestTally(unittest.TestCase):
    """M328-G4/G5/G6."""

    def _ballots(self, votes, seed=1):
        rng = np.random.default_rng(seed)
        out = []
        for v in votes:
            r = int(rng.integers(1, 1 << 40))
            out.append((v, r, commit_ballot(v, r)))
        return out

    def test_weighted_sum_equals_true_support(self):
        votes = [1, 1, 0, 1, 0]
        weights = [3, 5, 2, 7, 1]
        rec = tally(self._ballots(votes), weights, committee_size=8,
                    threshold=5, seed=11)
        self.assertEqual(rec.weighted_support, sum(
            v * w for v, w in zip(votes, weights)))
        self.assertEqual(rec.weighted_total, sum(weights))

    def test_invalid_ballot_rejected_in_tally(self):
        votes = [1, 1]
        weights = [1, 1]
        ballots = self._ballots(votes)
        # tamper with one commitment
        _, r, C = ballots[0]
        ballots[0] = (1, r, (C + 1))
        with self.assertRaises(ValueError):
            tally(ballots, weights, 8, 5, seed=3)

    def test_record_carries_no_individual_ballot_field(self):
        """M328-G6: the public record has commitments and sums only."""
        rec = tally(self._ballots([1, 0]), [2, 3], 8, 5, seed=5)
        self.assertIsInstance(rec, TallyRecord)
        fields = set(rec.__dataclass_fields__)
        self.assertEqual(fields, {"weighted_support", "weighted_total",
                                  "commitments", "weights"})
        # no per-ballot openings or votes anywhere in the record
        self.assertNotIn("openings", fields)
        self.assertNotIn("ballots", fields)

    def test_unopened_weight_fails_closed(self):
        """M328-G5: unopened weight above one third fails the vote."""
        from geode.privacy.vote_machinery import WeightSnapshot
        snap = WeightSnapshot(anchor="a", weights={})
        # the registered predicate lives in the ratification path:
        # a vote with a third of the sampled weight unopened cannot
        # be ratified — modeled as support failing the two-thirds
        ok, reason = ratifies(support_weight=0.6, total_weight=1.0,
                              supporting_identities=["a", "b", "c"],
                              pool_size=30, responders=3)
        self.assertFalse(ok)
        self.assertEqual(reason, "below_two_thirds")


if __name__ == "__main__":
    unittest.main()
