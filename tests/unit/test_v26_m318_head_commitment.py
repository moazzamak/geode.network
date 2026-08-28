"""Unit tests for M318 — the Pedersen head commitment as the
registry binding key (R-A15a)."""
from __future__ import annotations

import unittest

import numpy as np

from geode.privacy.head_commitment import (
    bind_artifact,
    commit_head,
    verify_head,
)


class TestHeadCommitment(unittest.TestCase):

    def test_commitment_verifies_for_the_true_head(self):
        rng = np.random.default_rng(0)
        q_W = rng.integers(-(2 ** 16), 2 ** 16, size=(64, 8))
        r = [int(x) for x in rng.integers(1, 1 << 40, size=(8,))]
        C = commit_head(q_W, r)
        self.assertTrue(verify_head(C, q_W, r))

    def test_binding_against_a_different_head(self):
        rng = np.random.default_rng(1)
        q_W = rng.integers(-(2 ** 16), 2 ** 16, size=(64, 8))
        r = [int(x) for x in rng.integers(1, 1 << 40, size=(8,))]
        C = commit_head(q_W, r)
        q_W2 = q_W.copy()
        q_W2[0, 0] += 1
        self.assertFalse(verify_head(C, q_W2, r))

    def test_binding_against_different_openings(self):
        rng = np.random.default_rng(2)
        q_W = rng.integers(-(2 ** 16), 2 ** 16, size=(64, 8))
        r = [int(x) for x in rng.integers(1, 1 << 40, size=(8,))]
        C = commit_head(q_W, r)
        r2 = list(r)
        r2[0] = (r2[0] + 1) % (1 << 40)
        self.assertFalse(verify_head(C, q_W, r2))

    def test_wrong_number_of_openings_rejected(self):
        rng = np.random.default_rng(3)
        q_W = rng.integers(0, 100, size=(8, 4))
        with self.assertRaises(ValueError):
            commit_head(q_W, [1, 2, 3])

    def test_bind_artifact_keeps_the_content_hash(self):
        rng = np.random.default_rng(4)
        q_W = rng.integers(0, 100, size=(16, 4))
        r = [7, 8, 9, 10]
        record = bind_artifact(q_W, r)
        self.assertIn("commitments", record)
        self.assertIn("content_hash", record)
        self.assertEqual(len(record["commitments"]), 4)


if __name__ == "__main__":
    unittest.main()
