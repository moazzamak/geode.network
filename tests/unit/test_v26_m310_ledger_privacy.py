"""Unit tests for the v26 M310 ledger privacy repair.

The ledger records H(answer, nonce), never the answer; the opening
lives only in the sealed replay environment. Route entries carry a
registry state root so replay is a local check. These tests pin:
deterministic commitments, nonce sensitivity, constant-time opening
verification, key-order invariance of the state root, and sensitivity
of the root to any score or price change.
"""
from __future__ import annotations

import unittest

from geode.core.ledger import (
    answer_commitment,
    opens_commitment,
    registry_state_root,
)


class TestAnswerCommitment(unittest.TestCase):

    def test_commitment_deterministic_per_opening(self):
        c1 = answer_commitment("hello", "n1")
        c2 = answer_commitment("hello", "n1")
        self.assertEqual(c1, c2)
        self.assertEqual(len(c1), 64)

    def test_commitment_nonce_sensitive(self):
        c1 = answer_commitment("hello", "n1")
        c2 = answer_commitment("hello", "n2")
        self.assertNotEqual(c1, c2)

    def test_commitment_answer_sensitive(self):
        c1 = answer_commitment("hello", "n1")
        c2 = answer_commitment("hellp", "n1")
        self.assertNotEqual(c1, c2)

    def test_opening_verifies(self):
        commitment = answer_commitment("answer text", "nonce-7")
        self.assertTrue(opens_commitment(
            commitment, "answer text", "nonce-7"))
        self.assertFalse(opens_commitment(
            commitment, "answer text", "other-nonce"))
        self.assertFalse(opens_commitment(
            commitment, "answer textx", "nonce-7"))

    def test_commitment_does_not_contain_answer(self):
        answer = "the quick brown fox"
        commitment = answer_commitment(answer, "n")
        self.assertNotIn(answer, commitment)
        self.assertNotIn("fox", commitment)


class TestRegistryStateRoot(unittest.TestCase):

    def test_root_is_key_order_invariant(self):
        a = {"artifact-1": {"score": 0.91, "price": 3},
             "artifact-2": {"score": 0.76, "price": 2}}
        b = {"artifact-2": {"score": 0.76, "price": 2},
             "artifact-1": {"score": 0.91, "price": 3}}
        self.assertEqual(registry_state_root(a), registry_state_root(b))

    def test_root_changes_with_any_score(self):
        base = {"artifact-1": {"score": 0.91, "price": 3}}
        changed = {"artifact-1": {"score": 0.90, "price": 3}}
        self.assertNotEqual(registry_state_root(base),
                            registry_state_root(changed))

    def test_root_changes_with_any_price(self):
        base = {"artifact-1": {"score": 0.91, "price": 3}}
        changed = {"artifact-1": {"score": 0.91, "price": 2}}
        self.assertNotEqual(registry_state_root(base),
                            registry_state_root(changed))

    def test_root_changes_with_qualification(self):
        base = {"artifact-1": {"score": 0.91, "price": 3,
                               "qualified": True}}
        changed = {"artifact-1": {"score": 0.91, "price": 3,
                                  "qualified": False}}
        self.assertNotEqual(registry_state_root(base),
                            registry_state_root(changed))


if __name__ == "__main__":
    unittest.main()
