"""Unit tests for v26 M307 behavioural artifact identity (A1).

Registered semantics (plan §8.18): a committed Merkle root over the
probe responses; beacon-seeded fresh slices; locality perturbations a
lookup table cannot serve; behavioural dedup above the registered
agreement threshold.
"""
from __future__ import annotations

import unittest

from geode.core.behavioral_identity import (
    DEDUP_AGREEMENT,
    behavioural_dedup_key,
    locality_perturbations,
    merkle_root,
    probe_slice,
    profile_agreement,
    same_artifact,
    verify_slice_answers,
)


class TestMerkleCommitment(unittest.TestCase):

    def test_root_is_deterministic(self):
        values = [b"a", b"b", b"c"]
        self.assertEqual(merkle_root(values), merkle_root(values))

    def test_root_changes_with_content(self):
        values = [b"a", b"b", b"c"]
        self.assertNotEqual(merkle_root(values),
                            merkle_root([b"a", b"b", b"x"]))

    def test_odd_cardinality_pairs_the_last_leaf(self):
        # odd leaf counts are legal: the last leaf pairs with itself
        values = [b"a", b"b", b"c"]
        self.assertEqual(len(merkle_root(values)), 64)

    def test_empty_set_rejected(self):
        with self.assertRaises(ValueError):
            merkle_root([])


class TestProbeSlice(unittest.TestCase):

    def test_fresh_slices_across_epochs(self):
        # M307-C2: consecutive epochs reveal different slices
        s0 = probe_slice("beacon0", 0, 10, 100)
        s1 = probe_slice("beacon0", 1, 10, 100)
        self.assertNotEqual(s0, s1)
        self.assertEqual(len(set(s0)), 10)
        self.assertEqual(len(set(s1)), 10)

    def test_slice_is_seeded_by_beacon(self):
        a = probe_slice("beaconA", 0, 10, 100)
        b = probe_slice("beaconB", 0, 10, 100)
        self.assertNotEqual(a, b)

    def test_rejects_oversize_slice_and_bad_epoch(self):
        with self.assertRaises(ValueError):
            probe_slice("x", 0, 101, 100)
        with self.assertRaises(ValueError):
            probe_slice("x", -1, 5, 100)


class TestSliceVerification(unittest.TestCase):

    def test_correct_answers_open(self):
        # M307-C1
        values = [f"v{i}".encode() for i in range(32)]
        leaves = [_leaf_for(v) for v in values]
        root = merkle_root(leaves)
        slice_indices = probe_slice("beacon", 3, 5, 32)
        answers = [values[i] for i in slice_indices]
        self.assertTrue(verify_slice_answers(leaves, root,
                                             slice_indices, answers))

    def test_tampered_answer_does_not_open(self):
        values = [f"v{i}".encode() for i in range(32)]
        leaves = [_leaf_for(v) for v in values]
        root = merkle_root(leaves)
        slice_indices = probe_slice("beacon", 3, 5, 32)
        answers = [values[i] for i in slice_indices]
        answers[0] = b"tampered"
        self.assertFalse(verify_slice_answers(leaves, root,
                                              slice_indices, answers))

    def test_wrong_root_does_not_open(self):
        values = [f"v{i}".encode() for i in range(32)]
        leaves = [_leaf_for(v) for v in values]
        other = merkle_root([_leaf_for(b"z")] * 32)
        slice_indices = probe_slice("beacon", 3, 5, 32)
        answers = [values[i] for i in slice_indices]
        self.assertFalse(verify_slice_answers(leaves, other,
                                              slice_indices, answers))


class TestLocalityPerturbations(unittest.TestCase):

    def test_neighbours_stay_close(self):
        probe = [1.0, -2.0, 0.5]
        neighbours = locality_perturbations(probe, 5, seed=1)
        self.assertEqual(len(neighbours), 5)
        for nb in neighbours:
            dist = max(abs(a - b) for a, b in zip(probe, nb))
            self.assertLess(dist, 0.05 * max(1.0, max(abs(p)
                                                       for p in probe)))

    def test_neighbours_differ(self):
        probe = [1.0, -2.0, 0.5]
        neighbours = locality_perturbations(probe, 10, seed=2)
        self.assertEqual(len({tuple(round(v, 12) for v in nb)
                              for nb in neighbours}), 10)


class TestBehaviouralDedup(unittest.TestCase):

    def test_identical_profiles_same_artifact(self):
        # M307-C4
        profile = [1, 2, 3, 4, 5] * 20
        out = same_artifact(profile, profile)
        self.assertTrue(out["same_artifact"])
        self.assertEqual(out["agreement"], 1.0)

    def test_bit_flip_copy_is_same_artifact(self):
        # the one-bit-flip copy: behaviourally identical, whatever the
        # weight hashes
        a = [i % 17 for i in range(1000)]
        b = list(a)
        out = same_artifact(a, b)
        self.assertTrue(out["same_artifact"])
        self.assertEqual(out["agreement"], 1.0)

    def test_distinct_profiles_register_separately(self):
        a = [0] * 1000
        b = [1] * 1000
        out = same_artifact(a, b)
        self.assertFalse(out["same_artifact"])
        self.assertEqual(out["agreement"], 0.0)

    def test_threshold_is_the_registered_value(self):
        self.assertEqual(DEDUP_AGREEMENT, 0.95)
        # agreement just below the threshold is NOT the same artifact
        a = [0] * 1000
        b = [0] * 949 + [1] * 51
        self.assertFalse(same_artifact(a, b)["same_artifact"])
        self.assertGreaterEqual(
            profile_agreement(a, [0] * 960 + [1] * 40),
            DEDUP_AGREEMENT)

    def test_dedup_key_differs_across_profiles(self):
        self.assertNotEqual(behavioural_dedup_key([1, 2, 3]),
                            behavioural_dedup_key([1, 2, 4]))

    def test_rejects_mismatched_profiles(self):
        with self.assertRaises(ValueError):
            profile_agreement([1, 2], [1, 2, 3])


def _leaf_for(value: bytes) -> bytes:
    import hashlib
    return hashlib.sha256(b"\x00" + value).digest()


if __name__ == "__main__":
    unittest.main()
