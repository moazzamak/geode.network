"""Unit tests for M253: verifier rotation (the rotation half).
"""
from __future__ import annotations

import unittest

from geode.core.rotation import VerifierRotation


class TestM253VerifierRotation(unittest.TestCase):

    def setUp(self):
        self.rot = VerifierRotation(
            ["v1", "v2", "v3", "v4"], committee_size=2,
            epoch_length=100)

    def test_committee_is_deterministic(self):
        a = self.rot.committee_for(50)
        b = self.rot.committee_for(50)
        self.assertEqual(a, b)

    def test_committee_rotates_across_epochs(self):
        c0 = self.rot.committee_for(0)
        c1 = self.rot.committee_for(100)
        c2 = self.rot.committee_for(200)
        self.assertNotEqual(c0, c1)
        self.assertNotEqual(c1, c2)
        # a full cycle returns to the start (n=4 verifiers)
        c4 = self.rot.committee_for(400)
        self.assertEqual(c0, c4)

    def test_capture_cannot_persist(self):
        # the anti-capture bound: with k=2 of n=4, a verifier sits on
        # the committee at most 1 consecutive epoch per cycle
        for v in ["v1", "v2", "v3", "v4"]:
            self.assertEqual(self.rot.committee_span(v), 1)

    def test_quorum_counts_only_active_committee(self):
        # at index 0 the committee is (v1, v2); v3's attestation
        # never counts toward the quorum there
        self.assertTrue(self.rot.quorum_met(["v1", "v2"], 0, k_of_n=2))
        self.assertFalse(self.rot.quorum_met(["v1", "v3"], 0, k_of_n=2))
        self.assertFalse(self.rot.quorum_met(["v3", "v4"], 0, k_of_n=2))

    def test_empty_verifier_set_raises(self):
        with self.assertRaises(ValueError):
            VerifierRotation([])

    def test_bad_committee_size_raises(self):
        with self.assertRaises(ValueError):
            VerifierRotation(["v1", "v2"], committee_size=3)

    def test_negative_index_raises(self):
        with self.assertRaises(ValueError):
            self.rot.committee_for(-1)


if __name__ == "__main__":
    unittest.main()
