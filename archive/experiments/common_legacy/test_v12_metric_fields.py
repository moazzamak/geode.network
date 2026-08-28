from __future__ import annotations

import unittest

import numpy as np

from experiments.common.v12_metric_fields import (
    initialize_metric_fields,
    initialize_projected_metric_fields,
    projection_diagnostics,
    train_metric_fields,
    train_projected_metric_fields,
)
from experiments.tier4.eval_v12_m74_confirmation_transfer import (
    _domainnet_partitions,
)
from experiments.tier4.eval_v12_m75_inspectability import (
    _exact_decomposition,
    _state_from_dict,
)


class V12MetricFieldTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(72)
        self.features = np.vstack(
            [
                rng.normal(loc=-1.0, scale=0.2, size=(40, 6)),
                rng.normal(loc=1.0, scale=0.2, size=(40, 6)),
            ]
        )
        self.labels = np.repeat([0, 1], 40)

    def test_score_decomposition_and_gradient_are_exact(self) -> None:
        state = initialize_metric_fields(self.features, self.labels, rank=2)
        points = self.features[:5]
        tangent, residual = state.score_terms(points)
        np.testing.assert_allclose(
            state.scores(points) ** 2,
            np.sum(tangent, axis=2) + residual,
            atol=1e-10,
        )
        epsilon = 1e-6
        perturbed = points.copy()
        perturbed[:, 0] += epsilon
        finite = (state.scores(perturbed) - state.scores(points)) / epsilon
        np.testing.assert_allclose(
            finite, state.score_gradients(points)[:, :, 0], atol=2e-4
        )

    def test_training_is_deterministic_and_updates_state(self) -> None:
        initial = initialize_metric_fields(self.features, self.labels, rank=2)
        arguments = {
            "epochs": 2,
            "batch_size": 20,
            "learning_rate": 1e-3,
            "classification_temperature": 2.0,
            "target_score": 1.0,
            "separation_margin": 1.0,
            "probe_margin": 2.0,
            "loss_weights": {
                "classification": 1.0,
                "eikonal": 0.1,
                "probe": 0.1,
                "distribution": 0.1,
                "separation": 0.1,
            },
            "probe_families": ("axis_tangent", "normal"),
            "seed": 72,
        }
        first, first_history = train_metric_fields(
            initial, self.features, self.labels, **arguments
        )
        second, second_history = train_metric_fields(
            initial, self.features, self.labels, **arguments
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first_history, second_history)
        self.assertFalse(np.array_equal(initial.centers, first.centers))

    def test_projected_training_is_deterministic_and_constrained(self) -> None:
        initial = initialize_projected_metric_fields(
            self.features, self.labels, output_dimension=4, rank=2
        )
        arguments = {
            "epochs": 2,
            "batch_size": 20,
            "learning_rate": 1e-3,
            "classification_temperature": 2.0,
            "target_score": 1.0,
            "separation_margin": 1.0,
            "probe_margin_multiplier": 2.0,
            "loss_weights": {
                "classification": 1.0,
                "eikonal": 0.1,
                "probe": 0.1,
                "distribution": 0.1,
                "separation": 0.1,
            },
            "collapse_weight": 1.0,
            "probe_families": ("axis_tangent", "normal"),
            "seed": 73,
        }
        first, first_history = train_projected_metric_fields(
            initial, self.features, self.labels, **arguments
        )
        second, second_history = train_projected_metric_fields(
            initial, self.features, self.labels, **arguments
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first_history, second_history)
        self.assertFalse(np.array_equal(initial.projection, first.projection))
        self.assertEqual(first.transform(self.features).shape, (80, 4))
        diagnostics = projection_diagnostics(first, initial, self.features)
        self.assertEqual(diagnostics["effective_rank"], 4)
        self.assertGreater(diagnostics["minimum_singular_value"], 0.0)

    def test_m74_domainnet_partitions_are_disjoint(self) -> None:
        features = np.arange(40 * 3, dtype=np.float64).reshape(40, 3)
        labels = np.repeat(np.arange(4), 10)
        config = {
            "domainnet_transfer": {
                "known_class_count": 2,
                "geometry_per_class": 4,
                "calibration_per_class": 3,
                "evaluation_per_class": 3,
                "unknown_class_start": 2,
                "unknown_class_stop": 4,
                "unknown_evaluation_per_class": 3,
            }
        }
        partitions = _domainnet_partitions(features, labels, config)
        self.assertEqual(partitions["geometry_fit"][0].shape, (8, 3))
        self.assertEqual(partitions["score_calibration"][0].shape, (6, 3))
        self.assertEqual(partitions["known_evaluation"][0].shape, (6, 3))
        self.assertEqual(partitions["unknown_evaluation"][0].shape, (6, 3))
        row_sets = [
            {tuple(row) for row in values[0]} for values in partitions.values()
        ]
        for first in range(len(row_sets)):
            for second in range(first + 1, len(row_sets)):
                self.assertFalse(row_sets[first] & row_sets[second])

    def test_m75_state_replay_and_decomposition_are_exact(self) -> None:
        initial = initialize_projected_metric_fields(
            self.features, self.labels, output_dimension=4, rank=2
        )
        replayed = _state_from_dict(initial.to_dict())
        self.assertEqual(initial.to_dict(), replayed.to_dict())
        audit = _exact_decomposition(
            replayed,
            replayed.transform(self.features[:8]),
            tolerance=1e-9,
        )
        self.assertTrue(audit["passed"])
        self.assertLessEqual(audit["maximum_absolute_residual"], 1e-9)


if __name__ == "__main__":
    unittest.main()
