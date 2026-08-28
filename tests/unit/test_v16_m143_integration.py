"""Unit tests for the M143 integration layer (CPU phase-2 functions only)."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.tier4.eval_v16_m143_integration import (
    ARMS,
    CLASSES,
    _arm_accuracy,
    _competence_fit,
    _random_router_accuracy,
    _select_penalty,
    _split_indices,
    _stacking_fit,
)


def _fake_scores(n_rows: int, seed: int, perfect_arm: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, CLASSES, size=n_rows)
    concat = np.empty((n_rows, ARMS * CLASSES), dtype=np.float64)
    for arm in range(ARMS):
        if arm == perfect_arm:
            block = np.zeros((n_rows, CLASSES))
            block[np.arange(n_rows), labels] = 10.0
        else:
            block = rng.standard_normal((n_rows, CLASSES)) * 0.1
        concat[:, arm * CLASSES:(arm + 1) * CLASSES] = block
    return concat, labels


class TestSplit(unittest.TestCase):
    def test_disjoint_halves(self):
        fit, ev = _split_indices(1000, 33)
        self.assertEqual(len(fit), 500)
        self.assertEqual(len(ev), 500)
        self.assertEqual(len(set(fit) | set(ev)), 1000)
        self.assertEqual(len(set(fit) & set(ev)), 0)

    def test_deterministic(self):
        a = _split_indices(1000, 33)
        b = _split_indices(1000, 33)
        self.assertTrue(np.array_equal(a[0], b[0]))
        self.assertTrue(np.array_equal(a[1], b[1]))


class TestStacking(unittest.TestCase):
    def test_shapes(self):
        concat, labels = _fake_scores(300, seed=1)
        predict = _stacking_fit(concat[:200], labels[:200], penalty=1.0)
        preds = predict(concat[200:])
        self.assertEqual(preds.shape, (100,))
        self.assertTrue(set(preds).issubset(range(CLASSES)))

    def test_fusion_recovers_perfect_arm_in_sample(self):
        # with one arm one-hotting the labels and a small penalty, stacking on
        # the FIT half must reach (near-)perfect accuracy: the guarantee that
        # fusion is at least as good as any single arm it contains
        concat, labels = _fake_scores(400, seed=2)
        predict = _stacking_fit(concat[:200], labels[:200], penalty=1e-6)
        preds = predict(concat[:200])
        acc = float((preds == labels[:200]).mean())
        self.assertGreater(acc, 0.95)


class TestCompetence(unittest.TestCase):
    def test_picks_dominant_arm(self):
        # amended target: lowest-index arm whose argmax matches the label;
        # with a perfect arm 2, that is arm 2 for (nearly) every row
        concat, labels = _fake_scores(400, seed=3, perfect_arm=2)
        predict = _competence_fit(concat[:200], labels[:200], penalty=1e-6)
        preds = predict(concat[:200])
        self.assertGreater(float((preds == 2).mean()), 0.9)

    def test_shape_and_bounds(self):
        concat, labels = _fake_scores(300, seed=4)
        predict = _competence_fit(concat[:200], labels[:200], penalty=1.0)
        preds = predict(concat[200:])
        self.assertEqual(preds.shape, (100,))
        self.assertTrue(set(preds).issubset(range(ARMS)))

    def test_target_scale_invariance(self):
        # the amended target is label-derived: scaling one arm's raw scores
        # must not change which arms are correct, hence not the targets
        concat, labels = _fake_scores(300, seed=8, perfect_arm=6)
        scaled = concat.copy()
        scaled[:, 6 * CLASSES:7 * CLASSES] *= 0.001
        a = _competence_fit(concat[:200], labels[:200], penalty=1.0)
        b = _competence_fit(scaled[:200], labels[:200], penalty=1.0)
        self.assertTrue(np.array_equal(a(concat[200:]), b(scaled[200:])))


class TestRandomRouter(unittest.TestCase):
    def test_deterministic_expectation(self):
        concat, labels = _fake_scores(300, seed=5)
        arm_scores = concat.reshape(len(concat), ARMS, CLASSES).transpose(1, 0, 2)
        a = _random_router_accuracy(arm_scores, labels, 44)
        b = _random_router_accuracy(arm_scores, labels, 44)
        self.assertEqual(a, b)
        self.assertTrue(0.0 <= a <= 1.0)

    def test_below_best_arm(self):
        concat, labels = _fake_scores(400, seed=6, perfect_arm=6)
        arm_scores = concat.reshape(len(concat), ARMS, CLASSES).transpose(1, 0, 2)
        best = _arm_accuracy(arm_scores, labels, 6)
        rand = _random_router_accuracy(arm_scores, labels, 44)
        self.assertLess(rand, best)


class TestPenaltySelection(unittest.TestCase):
    def test_selects_argmax(self):
        ladder = [1.0, 10.0, 100.0, 1000.0]
        scores = {1.0: 0.1, 10.0: 0.4, 100.0: 0.9, 1000.0: 0.5}

        def metric(p):
            return scores[p]

        best, recorded = _select_penalty(metric, ladder)
        self.assertEqual(best, 100.0)
        self.assertEqual(recorded["100.0"], 0.9)

    def test_first_max_wins(self):
        ladder = [1.0, 10.0]
        calls = []

        def metric(p):
            calls.append(p)
            return 0.5

        best, _ = _select_penalty(metric, ladder)
        self.assertEqual(best, 1.0)  # ties -> first max wins
        self.assertEqual(calls, ladder)

    def test_fewer_calls_than_full_ladder_not_allowed(self):
        # the ladder must be evaluated fully so the scores dict is complete
        ladder = [1.0, 10.0, 100.0]

        def metric(p):
            return float(p)

        _, recorded = _select_penalty(metric, ladder)
        self.assertEqual(sorted(recorded), ["1.0", "10.0", "100.0"])


if __name__ == "__main__":
    unittest.main()
