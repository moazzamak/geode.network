from __future__ import annotations

import unittest

import numpy as np

from experiments.common.v8_local_residual import residual_predictions
from experiments.common.v7_adaptation import fit_gaussian_bundle


class V8LocalResidualTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(49)
        x = np.concatenate((rng.normal(-2, 0.2, (30, 4)), rng.normal(2, 0.2, (30, 4))))
        y = np.repeat((0, 1), 30)
        self.bundle = fit_gaussian_bundle(x, y, rank=2, threshold=100.0)
        self.features = x[:10]

    def test_temperature_residual_preserves_unaffected_predictions(self):
        baseline, _ = self.bundle.predict(self.features)
        affected = np.zeros(len(self.features), dtype=bool)
        affected[:3] = True
        predictions, _ = residual_predictions(
            self.bundle,
            self.features,
            affected,
            target_label=0,
            target_temperature=2.0,
        )
        np.testing.assert_array_equal(predictions[~affected], baseline[~affected])

    def test_residual_requires_exactly_one_family(self):
        with self.assertRaises(ValueError):
            residual_predictions(
                self.bundle,
                self.features,
                np.ones(len(self.features), dtype=bool),
                target_label=0,
            )


if __name__ == "__main__":
    unittest.main()
