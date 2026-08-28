from __future__ import annotations

import unittest

import numpy as np

from experiments.tier4.eval_v12_m71_gaussian_classifier import (
    _decomposition_audit,
    _fit_gaussian,
    _gaussian_outputs,
    _head_metrics,
)


class V12GaussianClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(71)
        self.features = np.vstack(
            [
                rng.normal(loc=-1.0, scale=0.1, size=(40, 6)),
                rng.normal(loc=1.0, scale=0.1, size=(40, 6)),
            ]
        )
        self.labels = np.repeat([0, 1], 40)

    def test_gaussian_classifier_and_decomposition_are_exact(self) -> None:
        classes, primitives = _fit_gaussian(self.features, self.labels, rank=2)
        outputs = _gaussian_outputs(classes, primitives, self.features)
        self.assertGreater(np.mean(outputs["predictions"] == self.labels), 0.99)
        audit = _decomposition_audit(primitives, self.features)
        self.assertTrue(audit["exact_score_decomposition"])
        self.assertLessEqual(audit["maximum_absolute_residual"], 1e-9)

    def test_open_set_metrics_use_calibration_only_threshold(self) -> None:
        predictions = np.asarray([0, 1, 0, 1])
        novelty = np.asarray([0.1, 0.2, 2.0, 3.0])
        labels = np.asarray([0, 1, 2, 2])
        metrics = _head_metrics(
            np.asarray([0.1, 0.2, 0.3, 0.4]),
            predictions,
            novelty,
            labels,
            known_classes=np.asarray([0, 1]),
            coverage=0.75,
        )
        self.assertEqual(metrics["novelty_threshold"], 0.4)
        self.assertEqual(metrics["known_coverage"], 1.0)
        self.assertEqual(metrics["unknown_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
