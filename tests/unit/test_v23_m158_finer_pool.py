"""Unit tests for the M158 finer-pooling cell (pure helpers only)."""
import unittest

import numpy as np
import torch

from experiments.tier4.eval_v16_m142_c2 import _spm_pool
from experiments.tier4.eval_v23_m158_finer_pool import (
    _head_macs,
    _pool_levels,
)


class TestPoolLevels(unittest.TestCase):
    def test_levels_124_matches_c2(self):
        # _pool_levels(levels=(1,2,4)) must equal the sealed C2 _spm_pool;
        # the input is 2-D (count*patches, atoms), the C2 convention
        count, grid, atoms = 3, 9, 5
        act = torch.arange(count * grid * grid * atoms,
                           dtype=torch.float32).reshape(
                               count * grid * grid, atoms)
        a = _pool_levels(act, count, grid, (1, 2, 4))
        b = _spm_pool(act, count, grid)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(torch.equal(a, b))

    def test_level8_bins(self):
        # on a 16x16 grid the level-8 edges rule gives 2x2-patch bins:
        # each of the 64 bins sums 4 patches of 1.0 -> 4.0
        count, grid, atoms = 2, 16, 3
        act = torch.ones(count * grid * grid, atoms, dtype=torch.float32)
        out = _pool_levels(act, count, grid, (8,))
        self.assertEqual(out.shape, (count, 64 * atoms))
        self.assertTrue(torch.equal(out, torch.full((count, 64 * atoms),
                                                    4.0)))

    def test_edge_rule_identity(self):
        # level 1 pooling sums the whole map (the edges rule at level 1)
        count, grid, atoms = 2, 27, 2
        act = torch.rand(count * grid * grid, atoms)
        out = _pool_levels(act, count, grid, (1,))
        np.testing.assert_allclose(out.numpy(),
                                   act.reshape(count, grid, grid, atoms)
                                   .sum(dim=(1, 2)).numpy(),
                                   rtol=1e-6)


class TestHeadMacs(unittest.TestCase):
    def test_head_cost_delta_within_tolerance(self):
        # 64 x 481 x 345 = 10,620,480 vs 16 x 1,923 x 345 = 10,614,960:
        # the 8x8 arm pays +5,520 head MACs (+0.052%), within the
        # family's 0.5% cost-tolerance rule (registered in the re-scope)
        m8 = _head_macs(481, 27, 108, (8,))
        m4 = _head_macs(1923, 27, 108, (4,))
        self.assertEqual(m8["head"], 10620480)
        self.assertEqual(m4["head"], 10614960)
        delta = m8["head"] - m4["head"]
        self.assertLessEqual(delta / m4["head"], 0.005)


if __name__ == "__main__":
    unittest.main()
