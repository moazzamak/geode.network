"""Unit tests for the M150 rank-sweep cell (pure helpers only)."""
import unittest

import numpy as np

from experiments.tier4.eval_v23_m150_rank_sweep import (
    _participation_ratio,
    _spectrum_shares,
)


class TestParticipationRatio(unittest.TestCase):
    def test_rank_k_identity(self):
        vals = np.concatenate([np.ones(8), np.zeros(92)]) + 1e-9
        self.assertAlmostEqual(_participation_ratio(vals), 8.0, places=4)

    def test_flat_spectrum(self):
        vals = np.ones(10) + 1e-9
        self.assertAlmostEqual(_participation_ratio(vals), 10.0, places=4)

    def test_decaying_spectrum_smaller_than_count(self):
        vals = np.array([4.0, 2.0, 1.0, 0.5, 0.25] + [0.0] * 95)
        r = _participation_ratio(vals)
        self.assertLess(r, 5.0)
        self.assertGreater(r, 1.0)


class TestSpectrumShares(unittest.TestCase):
    def test_shares(self):
        vals = np.array([4.0, 2.0, 1.0, 0.5] + [0.0] * 6)
        keep = vals > 1e-12
        out = _spectrum_shares(vals, keep, top=2)
        self.assertAlmostEqual(out["top1_share"], 4.0 / 7.5, places=6)
        self.assertAlmostEqual(out["top2_share"], 6.0 / 7.5, places=6)
        self.assertAlmostEqual(out["condition"], 4.0 / 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
