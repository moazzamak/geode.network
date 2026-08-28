"""Unit tests for the M142 C3 multi-scale cell (pure/numpy functions).

Run from the repo root with the CPU environment::

    .\\.venv\\Scripts\\python.exe -m unittest experiments.common.test_v16_m142_c3
"""

import unittest

from experiments.tier4.eval_v16_m142_c3 import (
    _scale_atom_split,
    _scale_grid,
    _scale_macs,
)

SIZE = 32
CLASSES = 345
POOL_TOTAL = 175_197_198


class ScaleSplitTests(unittest.TestCase):
    def test_grids(self):
        self.assertEqual(_scale_grid(SIZE, 3), 30)
        self.assertEqual(_scale_grid(SIZE, 5), 28)
        self.assertEqual(_scale_grid(SIZE, 7), 26)

    def test_registered_atom_split(self):
        got = _scale_atom_split(POOL_TOTAL, SIZE, CLASSES)
        self.assertEqual(got, {3: 1950, 5: 850, 7: 511},
                         "the registered equal-MAC-share split must give "
                         "{3:1950, 5:850, 7:511}; if this changes, the "
                         "config's atoms_by_scale is wrong")

    def test_split_is_cost_matched(self):
        macs = _scale_macs(_scale_atom_split(POOL_TOTAL, SIZE, CLASSES),
                           SIZE, CLASSES)
        self.assertLessEqual(abs(macs["total"] - POOL_TOTAL) / POOL_TOTAL,
                             0.005)

    def test_pool_adds_counted_per_scale(self):
        atoms = {3: 1950, 5: 850, 7: 511}
        macs = _scale_macs(atoms, SIZE, CLASSES)
        for patch in (3, 5, 7):
            grid = _scale_grid(SIZE, patch)
            self.assertEqual(macs[str(patch)]["pool_adds"],
                             grid * grid * atoms[patch])

    def test_whitening_sum_is_exact(self):
        expected = (30 * 30 * 27 * 27
                    + 28 * 28 * 75 * 75
                    + 26 * 26 * 147 * 147)
        atoms = {3: 1950, 5: 850, 7: 511}
        macs = _scale_macs(atoms, SIZE, CLASSES)
        got = sum(macs[str(p)]["whitening"] for p in (3, 5, 7))
        self.assertEqual(got, expected)


class EdgesRuleTests(unittest.TestCase):
    def test_pool2_edges_per_grid(self):
        # the m107 rule round(grid*i/pool_grid); Python round is banker's
        for grid, expected in ((30, [0, 15, 30]), (28, [0, 14, 28]),
                               (26, [0, 13, 26])):
            edges = [round(grid * i / 2) for i in range(3)]
            self.assertEqual(edges, expected)


if __name__ == "__main__":
    unittest.main()
