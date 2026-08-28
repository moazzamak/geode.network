from __future__ import annotations

import unittest

import numpy as np

from experiments.common.v6_boundary_distillation import (
    fit_boundary_distilled_student,
    generate_margin_sphere_candidates,
    predict_boundary_student,
    student_to_geode_models,
)
from experiments.tier4.eval_v5_frozen_space_heads import predict_geode


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    first = rng.normal(loc=(-1.0, -1.0), scale=0.35, size=(30, 2))
    second = rng.normal(loc=(1.0, 1.0), scale=0.35, size=(30, 2))
    features = np.vstack([first, second])
    labels = np.repeat(np.array([0, 1], dtype=np.int64), 30)
    logits = np.column_stack(
        [
            -np.sum((features - np.array([-1.0, -1.0])) ** 2, axis=1),
            -np.sum((features - np.array([1.0, 1.0])) ** 2, axis=1),
        ]
    )
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return features, labels, probabilities, np.array([0, 1], dtype=np.int64)


class BoundaryDistillationTests(unittest.TestCase):
    def _candidates(self):
        features, labels, probabilities, classes = _fixture()
        candidates = generate_margin_sphere_candidates(
            features,
            labels,
            probabilities,
            classes,
            candidates_per_class=5,
            seed_size=4,
            anchor_fraction=0.4,
        )
        return features, labels, probabilities, classes, candidates

    def test_candidate_generation_is_deterministic(self):
        first = self._candidates()[-1]
        second = self._candidates()[-1]
        self.assertEqual(
            [candidate.to_dict() for candidate in first],
            [candidate.to_dict() for candidate in second],
        )
        self.assertEqual({candidate.support_size for candidate in first}, {4})

    def test_fit_is_monotone_and_budget_bounded(self):
        features, labels, probabilities, classes, candidates = self._candidates()
        student = fit_boundary_distilled_student(
            features,
            labels,
            probabilities,
            classes,
            candidates,
            np.arange(len(features), dtype=np.int64),
            component_limit=8,
        )
        trajectory = student["objective_trajectory"]
        self.assertTrue(
            all(next_value < value for value, next_value in zip(trajectory, trajectory[1:]))
        )
        self.assertLessEqual(len(student["selected_candidates"]), 8)
        self.assertTrue(all(count >= 1 for count in student["component_counts"]))

    def test_serialized_student_matches_geode_readout(self):
        features, labels, probabilities, classes, candidates = self._candidates()
        student = fit_boundary_distilled_student(
            features,
            labels,
            probabilities,
            classes,
            candidates,
            np.arange(len(features), dtype=np.int64),
            component_limit=8,
        )
        predictions, student_probabilities = predict_boundary_student(student, features)
        models = student_to_geode_models(student)
        geode_predictions, geode_probabilities = predict_geode(
            models, features, classes
        )
        np.testing.assert_array_equal(predictions, geode_predictions)
        np.testing.assert_allclose(
            student_probabilities, geode_probabilities, rtol=0.0, atol=1e-12
        )

    def test_invalid_teacher_probabilities_fail_closed(self):
        features, labels, probabilities, classes = _fixture()
        probabilities[0, 0] = -1.0
        with self.assertRaises(ValueError):
            generate_margin_sphere_candidates(
                features,
                labels,
                probabilities,
                classes,
                candidates_per_class=5,
                seed_size=4,
                anchor_fraction=0.4,
            )


if __name__ == "__main__":
    unittest.main()
