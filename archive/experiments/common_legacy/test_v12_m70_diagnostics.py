from __future__ import annotations

import unittest

import numpy as np

from experiments.tier4.eval_v12_m70_d1_sample_sensitivity import (
    _bootstrap_thresholds,
    _expanded_tangent_probes,
)
from experiments.common.v11_directional_envelope import DirectionalTube


class V12M70DiagnosticTests(unittest.TestCase):
    def test_sample_sensitivity_marks_only_extrapolated_sizes(self) -> None:
        result = _bootstrap_thresholds(
            np.linspace(0.5, 1.5, 600),
            sample_sizes=[100, 200, 400, 800],
            miscoverage=0.08,
            resamples=20,
            seed=70,
        )
        self.assertFalse(result["400"]["extrapolated_with_replacement"])
        self.assertTrue(result["800"]["extrapolated_with_replacement"])
        self.assertGreater(result["800"]["threshold_ratio"], 0.0)

    def test_expanded_probes_raise_per_patch_count_eightfold(self) -> None:
        tube = DirectionalTube(
            center=np.asarray([1.0, 0.0, 0.0, 0.0]),
            basis=np.asarray(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
            ),
            residual_scale=0.01,
            tangent_extents=np.asarray([0.1, 0.2]),
            outer_scales=np.asarray([0.1, 0.1]),
            penalty_weight=16.0,
            class_label=0,
        )
        points, owners = _expanded_tangent_probes(
            [tube],
            multiplier=4.0,
            replicates_per_axis_sign=8,
            seed=70,
        )
        self.assertEqual(len(points), 2 * tube.rank * 8)
        np.testing.assert_array_equal(owners, np.zeros(len(points), dtype=np.int64))
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
