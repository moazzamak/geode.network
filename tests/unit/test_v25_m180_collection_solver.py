"""Unit tests for the M180 assembled-LU solver (repair 5).

The block-Schur solver (repair 4) was removed after it voided itself on
real data (gate rel 1.076); these tests pin its replacement: one-pass
block building, Fortran-order assembly, in-place LU (dgetrf+dgetrs),
and the data-streamed residual certificate.
"""
from __future__ import annotations

import unittest

import numpy as np

from experiments.tier4.eval_v25_m180_collection import (
    CLASSES,
    PENALTY,
    _assemble_and_solve,
    _build_blocks,
    _equivalence_gate,
    _fit_coalition,
    _norm_g_inf,
    _residual_certificate,
)


def _random_blocks(k: int, widths: list[int], rng: np.random.Generator):
    """Random standardised-looking blocks: symmetric-PSD diagonals with
    penalty 1.0, arbitrary off-diagonals."""
    blocks = [[None] * k for _ in range(k)]
    for i in range(k):
        for j in range(i, k):
            blocks[i][j] = rng.standard_normal((widths[i], widths[j]))
        blocks[i][i] = blocks[i][i] @ blocks[i][i].T
        blocks[i][i] += np.eye(widths[i]) * PENALTY
    return blocks


class TestAssembleSolve(unittest.TestCase):

    def _assert_matches_direct(self, k, widths, seed):
        rng = np.random.default_rng(seed)
        blocks = _random_blocks(k, widths, rng)
        total = sum(widths)
        full = np.zeros((total, total))
        offsets = np.concatenate([[0], np.cumsum(widths)])
        for i in range(k):
            for j in range(i, k):
                r0, r1 = offsets[i], offsets[i + 1]
                c0, c1 = offsets[j], offsets[j + 1]
                full[r0:r1, c0:c1] = blocks[i][j]
                if i != j:
                    full[c0:c1, r0:r1] = blocks[i][j].T
        cross = [rng.standard_normal((w, CLASSES)) for w in widths]
        full_cross = np.vstack(cross)
        intercept = rng.standard_normal(CLASSES)
        direct = np.linalg.solve(full, full_cross)
        # neutral stats: with zero centres the lower-left convention
        # correction vanishes, so the assembled system equals `full`.
        colsums = [np.zeros(w) for w in widths]
        from experiments.tier4.eval_v15_m104_experts import Standardiser
        stds = [Standardiser(np.zeros(w, dtype=np.float32),
                             np.ones(w, dtype=np.float32))
                for w in widths]
        got = _assemble_and_solve(blocks, list(range(k)), cross,
                                  intercept, stds, colsums)
        rel = float(np.max(np.abs(direct - got[:-1]))
                    / max(float(np.max(np.abs(direct))), 1e-12))
        self.assertLessEqual(rel, 1e-10)
        self.assertLessEqual(
            float(np.max(np.abs(intercept - got[-1]))), 1e-14)

    def test_two_block_matches_direct(self):
        self._assert_matches_direct(2, [5, 3], seed=0)

    def test_three_block_matches_direct(self):
        self._assert_matches_direct(3, [6, 4, 2], seed=1)

    def test_singleton_matches_direct(self):
        self._assert_matches_direct(1, [7], seed=2)


class TestBuildBlocks(unittest.TestCase):

    def test_build_is_deterministic(self):
        rng = np.random.default_rng(3)
        a = (rng.standard_normal((1500, 40)) * 40 + 100).astype(np.float32)
        b = (rng.standard_normal((1500, 20)) * 10 + 30).astype(np.float32)
        labels = rng.integers(0, CLASSES, 1500)
        blocks1, stds1, crosses1, i1, r1, cs1 = _build_blocks(
            [a, b], labels)
        blocks2, stds2, crosses2, i2, r2, cs2 = _build_blocks(
            [a, b], labels)
        for i in range(2):
            for j in range(i, 2):
                self.assertTrue(np.array_equal(blocks1[i][j], blocks2[i][j]))
                self.assertTrue(np.array_equal(crosses1[i], crosses2[i]))
        self.assertTrue(np.array_equal(i1, i2))

    def test_equivalence_gate_self_consistent(self):
        # the registered gate must pass when both paths see the same
        # data. Labels carry a strong linear signal so the decision-
        # level holdout check is not dominated by chance.
        rng = np.random.default_rng(4)
        n_rows, cols = 3000, 256
        a = (rng.standard_normal((n_rows, cols // 2))
             * 50 + 80).astype(np.float32)
        b = (rng.standard_normal((n_rows, cols // 2))
             * 20 + 25).astype(np.float32)
        labels = np.clip((a[:, 0] / 25 + b[:, 0] / 8).astype(np.int64),
                         0, CLASSES - 1)
        gate = _equivalence_gate([a, b], labels, cols=cols,
                                 rows_gate=n_rows)
        self.assertTrue(gate["passed"], gate)
        self.assertLessEqual(gate["weights_rel_delta"], 1e-12)
        self.assertLessEqual(gate["holdout"]["delta"],
                             gate["holdout"]["tolerance"])

    def test_build_agrees_with_ridge_accumulator_at_decision_level(self):
        # last-ulp dgemm-shape differences can flip fp32 centres at
        # ~1e-6 in weights; the meaningful equivalence is at decision
        # level, which is what the sealed anchors pin on the real run.
        from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator

        rng = np.random.default_rng(9)
        n_rows, w0, w1 = 4000, 64, 32
        a = (rng.standard_normal((n_rows, w0)) * 40 + 100).astype(np.float32)
        b = (rng.standard_normal((n_rows, w1)) * 10 + 30).astype(np.float32)
        labels = np.clip((a[:, 0] / 25 + b[:, 0] / 8).astype(np.int64),
                         0, CLASSES - 1)
        blocks, stds, crosses, intercept, _, cs = _build_blocks(
            [a, b], labels)
        weights = _assemble_and_solve(blocks, [0, 1], crosses, intercept,
                                      stds, cs)
        acc = RidgeAccumulator(w0 + w1, CLASSES)
        acc.add(np.concatenate([a.astype(np.float64),
                                b.astype(np.float64)], axis=1), labels)
        ref = acc.solve(PENALTY)
        hold = n_rows // 4
        hold_xs = np.concatenate(
            [a[-hold:].astype(np.float64), b[-hold:].astype(np.float64)],
            axis=1)
        hold_labels = labels[-hold:]
        xs_build = np.concatenate(
            [stds[0](hold_xs[:, :w0]), stds[1](hold_xs[:, w0:])], axis=1)
        scores_build = (xs_build.astype(np.float64) @ weights[:-1]
                        + weights[-1])
        scores_ref = (acc.standardiser()(hold_xs).astype(np.float64)
                      @ ref[:-1] + ref[-1])
        agree = float((np.argmax(scores_build, axis=1)
                       == np.argmax(scores_ref, axis=1)).mean())
        self.assertGreaterEqual(agree, 0.999)


class TestResidualCertificate(unittest.TestCase):

    def test_certificate_accepts_exact_solution_blocks_path(self):
        rng = np.random.default_rng(5)
        parts = [(rng.standard_normal((500, w)) * 30 + 60)
                 .astype(np.float32) for w in (8, 5)]
        labels = rng.integers(0, CLASSES, 500)
        std_blocks, stds, crosses, intercept, _, cs = _build_blocks(
            parts, labels)
        weights = _assemble_and_solve(std_blocks, [0, 1], crosses,
                                      intercept, stds, cs)
        norm_g = _norm_g_inf(std_blocks, [0, 1])
        cert = _residual_certificate(parts, labels, stds, [0, 1], weights,
                                     crosses, norm_g, blocks=std_blocks,
                                     colsums=cs)
        self.assertTrue(cert["passed"], cert)
        self.assertLessEqual(cert["backward_error"], 1e-9)

    def test_certificate_accepts_exact_solution_streamed_path(self):
        # the streamed path carries the registered fp32-centre
        # convention gap; its tolerance is 1e-5.
        rng = np.random.default_rng(8)
        parts = [(rng.standard_normal((500, w)) * 30 + 60)
                 .astype(np.float32) for w in (8, 5)]
        labels = rng.integers(0, CLASSES, 500)
        std_blocks, stds, crosses, intercept, _, cs = _build_blocks(
            parts, labels)
        weights = _assemble_and_solve(std_blocks, [0, 1], crosses,
                                      intercept, stds, cs)
        norm_g = _norm_g_inf(std_blocks, [0, 1])
        cert = _residual_certificate(parts, labels, stds, [0, 1], weights,
                                     crosses, norm_g, blocks=None)
        self.assertTrue(cert["passed"], cert)
        self.assertLessEqual(cert["backward_error"], 1e-5)

    def test_certificate_rejects_garbage(self):
        rng = np.random.default_rng(6)
        parts = [rng.standard_normal((300, 4)).astype(np.float32)
                 for _ in range(2)]
        labels = rng.integers(0, CLASSES, 300)
        blocks, stds, crosses, intercept, _, cs = _build_blocks(
            parts, labels)
        weights = _assemble_and_solve(blocks, [0, 1], crosses, intercept,
                                      stds, cs)
        weights[:-1] = 0.0   # the zero solution is not the solution
        norm_g = _norm_g_inf(blocks, [0, 1])
        cert = _residual_certificate(parts, labels, stds, [0, 1], weights,
                                     crosses, norm_g, blocks=blocks,
                                     colsums=cs)
        self.assertFalse(cert["passed"], cert)


class TestFitCoalition(unittest.TestCase):

    def test_fit_coalition_subset_selection(self):
        # regression (repair 6): a subset selection must index the
        # test parts by part id, not by position in the selection.
        rng = np.random.default_rng(10)
        n_rows = 2000
        widths = (64, 48, 24)
        parts = [(rng.standard_normal((n_rows, w)) * 30 + 50)
                 .astype(np.float32) for w in widths]
        labels = rng.integers(0, CLASSES, n_rows)
        blocks, stds, crosses, intercept, _, cs = _build_blocks(
            parts, labels)
        test_parts = [(rng.standard_normal((100, w)) * 30 + 50)
                      .astype(np.float32) for w in widths]
        test_labels = rng.integers(0, CLASSES, 100)
        fit = _fit_coalition(blocks, stds, crosses, intercept, cs, [1, 2],
                             parts, labels, test_parts, test_labels)
        self.assertTrue(fit["residual"]["passed"], fit["residual"])
        self.assertTrue(0.0 <= fit["accuracy"] <= 1.0)

    def test_fit_coalition_with_free_blocks(self):
        rng = np.random.default_rng(7)
        n_rows = 2000
        parts = [(rng.standard_normal((n_rows, w)) * 30 + 50)
                 .astype(np.float32) for w in (64, 48, 24)]
        labels = rng.integers(0, CLASSES, n_rows)
        blocks, stds, crosses, intercept, _, cs = _build_blocks(
            parts, labels)
        norm_g = _norm_g_inf(blocks, [0, 1, 2])
        test_parts = [(rng.standard_normal((100, w)) * 30 + 50)
                      .astype(np.float32) for w in (64, 48, 24)]
        test_labels = rng.integers(0, CLASSES, 100)
        fit = _fit_coalition(blocks, stds, crosses, intercept, cs,
                             [0, 1, 2], parts, labels, test_parts,
                             test_labels, norm_g=norm_g, free_blocks=True)
        self.assertTrue(fit["residual"]["passed"], fit["residual"])
        self.assertTrue(0.0 <= fit["accuracy"] <= 1.0)
        # blocks were released as copied
        self.assertIsNone(blocks[0][0])
        self.assertIsNone(blocks[1][2])


if __name__ == "__main__":
    unittest.main()
