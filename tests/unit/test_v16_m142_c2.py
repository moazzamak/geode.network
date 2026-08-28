"""Unit tests for the M142 C2 spatial-pyramid cell (pure/numpy functions).

Run from the repo root with the CPU environment::

    .\\.venv\\Scripts\\python.exe -m unittest experiments.common.test_v16_m142_c2
"""

import unittest

import numpy as np
import torch

from experiments.tier4.eval_v15_m103_atoms import _pool
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v16_m142_c2 import (
    _pair_spm_atoms,
    _pool_inference_macs,
    _score_weights,
    _spm_inference_macs,
    _spm_pool,
)

GRID = 27
DIMENSION = 108
CLASSES = 345


class MatchedPairArithmeticTests(unittest.TestCase):
    def test_pair_solver_registered_output(self):
        a = _pair_spm_atoms(2062)
        self.assertEqual(a, 1923,
                         "the registered pair solver must give a=1923 for "
                         "b=2062; if this changes, the config's spm_atoms "
                         "is wrong")

    def test_pair_is_cost_matched(self):
        a, b = 1923, 2062
        pool = _pool_inference_macs(b, GRID, DIMENSION, CLASSES)["total"]
        spm = _spm_inference_macs(a, GRID, DIMENSION, CLASSES)["total"]
        self.assertLessEqual(abs(spm - pool) / pool, 0.005)

    def test_pair_equation_within_one_atom(self):
        # b*POOL_PER_ATOM == a*SPM_PER_ATOM up to the registered rounding
        patches = GRID * GRID
        a, b = 1923, 2062
        lhs = b * (patches * DIMENSION + patches + 4 * CLASSES)
        rhs = a * (patches * DIMENSION + patches + 21 * CLASSES)
        self.assertLessEqual(abs(lhs - rhs),
                             max(lhs, rhs) // a,
                             "pair equation off by more than one atom")

    def test_both_arms_count_pool_adds(self):
        # pool_adds must appear in BOTH per-atom constants (both arms sum
        # the same 729-activation map)
        patches = GRID * GRID
        self.assertEqual(_pool_inference_macs(1, GRID, DIMENSION, CLASSES)
                         ["pool_adds"],
                         _spm_inference_macs(1, GRID, DIMENSION, CLASSES)
                         ["pool_adds"])
        self.assertEqual(patches, 729)

    def test_widths_feasible_for_direct_fit(self):
        # peak of the sealed fit path ~ 3*width^2*8 bytes must stay well
        # under the installed 63 GB
        spm_width = 21 * 1923
        pool_width = 4 * 2062
        for w in (spm_width, pool_width):
            self.assertLess(3 * w * w * 8, 55e9,
                            f"width {w} would not fit the sealed fit path")


class SpmPoolTests(unittest.TestCase):
    def test_layout_and_level2_is_sealed_pool(self):
        rng = np.random.default_rng(0)
        count, atoms, grid = 5, 3, 9
        act = torch.from_numpy(
            rng.standard_normal((count * grid * grid, atoms)).astype(np.float32)
        )
        spm = _spm_pool(act, count, grid)
        self.assertEqual(spm.shape, (count, 21 * atoms))
        level2 = spm[:, atoms:5 * atoms]
        sealed = _pool(act, count, grid, 2)
        self.assertTrue(torch.equal(level2, sealed),
                        "the 2x2 SPM level must be bit-identical to the "
                        "sealed _pool(pool_grid=2)")

    def test_bins_match_manual_edges(self):
        # For each level, each bin must sum exactly the region the
        # round(grid*i/level) rule delimits.
        rng = np.random.default_rng(1)
        count, atoms, grid = 2, 2, 27
        raw = rng.standard_normal((count, grid, grid, atoms)).astype(np.float32)
        # the encoder path hands _spm_pool a 2D (count*patches, atoms) tensor
        act = torch.from_numpy(raw.reshape(count * grid * grid, atoms))
        spm = _spm_pool(act, count, grid).numpy().reshape(count, 21, atoms)
        offset = 0
        for level in (1, 2, 4):
            edges = [round(grid * i / level) for i in range(level + 1)]
            for iy in range(level):
                for ix in range(level):
                    expected = raw[:, edges[iy]:edges[iy + 1],
                                   edges[ix]:edges[ix + 1]].sum(axis=(1, 2))
                    np.testing.assert_allclose(spm[:, offset], expected,
                                               rtol=1e-6, atol=1e-4)
                    offset += 1
        self.assertEqual(offset, 21)

    def test_grid27_level4_edges(self):
        # round() is banker's rounding: edges must be exactly [0,7,14,20,27].
        level = 4
        edges = [round(GRID * i / level) for i in range(level + 1)]
        self.assertEqual(edges, [0, 7, 14, 20, 27])


class ScoreWeightsTests(unittest.TestCase):
    def test_accuracy_and_per_domain_with_standardiser(self):
        rng = np.random.default_rng(6)
        n, width, classes = 500, 20, 4
        xs = rng.standard_normal((n, width)).astype(np.float32)
        labels = rng.integers(0, classes, size=n)
        domains = rng.integers(0, 6, size=n)
        acc = RidgeAccumulator(width, classes)
        acc.add(xs, labels)
        weights = acc.solve(1.0)
        std = acc.standardiser()
        scored = _score_weights(xs, labels, domains, weights, std)
        self.assertAlmostEqual(scored["accuracy"],
                               (np.argmax(std(xs).astype(np.float64)
                                          @ weights[:-1] + weights[-1],
                                          axis=1) == labels).mean(),
                               places=12)
        self.assertEqual(scored["test_rows"], n)
        for d in range(6):
            sel = domains == d
            if sel.sum():
                self.assertGreaterEqual(scored["per_domain"][d], 0.0)
                self.assertLessEqual(scored["per_domain"][d], 1.0)


if __name__ == "__main__":
    unittest.main()
