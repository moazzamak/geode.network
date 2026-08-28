"""Unit tests for v26 M308 drawn-challenge admission (A8).

Registered semantics (plan §8.19): the sealed per-axis corpus is
committed; draws are stratified and beacon-seeded; the routable score
is the drawn-challenge fraction; validator-authored challenges are
reported separately and never score.
"""
from __future__ import annotations

import unittest

from geode.core.drawn_challenges import (
    pose_challenge,
    register_corpus,
    routable_score,
    stratified_draw,
    supplementary_stream,
    verify_answer,
)


def _corpus(n: int = 64, classes: int = 4):
    rows = [f"row-{i}".encode() for i in range(n)]
    labels = [i % classes for i in range(n)]
    return register_corpus(rows, labels), rows, labels


class TestCorpusCommitment(unittest.TestCase):

    def test_registration_commits_rows_and_labels(self):
        corpus, _, _ = _corpus()
        self.assertEqual(corpus["row_count"], 64)
        self.assertEqual(corpus["class_count"], 4)
        self.assertEqual(len(corpus["row_root"]), 64)
        self.assertEqual(len(corpus["label_root"]), 64)

    def test_mismatched_registration_rejected(self):
        with self.assertRaises(ValueError):
            register_corpus([b"a"], [1, 2])


class TestStratifiedDraw(unittest.TestCase):

    def test_equal_share_per_class(self):
        # M308-C1: the published rule draws equal per class
        corpus, _, labels = _corpus()
        draw = stratified_draw(corpus, labels, "beacon", 0, 8)
        from collections import Counter
        counts = Counter(labels[i] for i in draw)
        self.assertEqual(counts, {0: 2, 1: 2, 2: 2, 3: 2})

    def test_rotates_across_epochs(self):
        corpus, _, labels = _corpus()
        d0 = stratified_draw(corpus, labels, "beacon", 0, 8)
        d1 = stratified_draw(corpus, labels, "beacon", 1, 8)
        self.assertNotEqual(d0, d1)

    def test_beacon_seed_changes_draw(self):
        corpus, _, labels = _corpus()
        da = stratified_draw(corpus, labels, "beaconA", 0, 8)
        db = stratified_draw(corpus, labels, "beaconB", 0, 8)
        self.assertNotEqual(da, db)

    def test_registered_shares(self):
        corpus, _, labels = _corpus()
        shares = {0: 0.5, 1: 0.25, 2: 0.25, 3: 0.0}
        draw = stratified_draw(corpus, labels, "beacon", 0, 8,
                               class_shares=shares)
        from collections import Counter
        counts = Counter(labels[i] for i in draw)
        self.assertEqual(counts.get(3, 0), 0)
        self.assertEqual(counts.get(0, 0), 4)


class TestVerifyAnswer(unittest.TestCase):

    def test_correct_answer_verifies(self):
        corpus, rows, labels = _corpus()
        self.assertTrue(verify_answer(corpus, rows, labels, 5, 1))
        self.assertFalse(verify_answer(corpus, rows, labels, 5, 3))

    def test_tampered_corpus_rejected(self):
        corpus, rows, labels = _corpus()
        tampered = list(rows)
        tampered[5] = b"row-999"
        self.assertFalse(verify_answer(corpus, tampered, labels, 5, 1))

    def test_pose_reveals_index_in_bounds(self):
        corpus, _, _ = _corpus()
        challenge = pose_challenge(corpus, 3)
        self.assertEqual(challenge["row_index"], 3)
        with self.assertRaises(ValueError):
            pose_challenge(corpus, 99)


class TestRoutableScore(unittest.TestCase):

    def test_score_is_fraction_correct(self):
        out = routable_score([True, True, False, True])
        self.assertEqual(out["score"], 0.75)
        self.assertEqual(out["answered"], 4)
        self.assertEqual(out["correct"], 3)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            routable_score([])


class TestSupplementaryStream(unittest.TestCase):

    def test_authored_never_scores(self):
        out = supplementary_stream([{"challenge": "x"},
                                    {"challenge": "y"}])
        self.assertEqual(out["authored_count"], 2)
        self.assertTrue(out["reported_separately"])
        self.assertFalse(out["enters_routable_score"])


if __name__ == "__main__":
    unittest.main()
