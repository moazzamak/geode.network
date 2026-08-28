from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from experiments.common.v6_boundary_distillation import SphereCandidate
from experiments.common.v6_factorial import (
    fit_global_temperature,
    local_edit_rollback_evidence,
    predict_factorial_student,
    select_coverage_candidates,
    select_predictive_candidates,
    serialize_factorial_student,
    validate_fractional_factorial,
)


ROOT = Path(__file__).resolve().parents[2]


class V6FactorialTests(unittest.TestCase):
    def test_registered_fractional_factorial_has_full_main_effect_rank(self):
        config = json.loads(
            (
                ROOT / "experiments" / "configs" / "v6" / "m31_factorial_s2.json"
            ).read_text(encoding="utf-8")
        )
        result = validate_fractional_factorial(
            config["cells"], config["factorial_baselines"]
        )
        self.assertEqual(result["design_rank"], 8)
        self.assertTrue(result["main_effects_identifiable"])

    def test_predictive_and_coverage_selection_fill_exact_budget(self):
        fields = np.array(
            [
                [-1.0, 0.2, 2.0, 1.5],
                [0.1, -1.0, 1.4, 2.0],
                [2.0, 1.5, -1.0, 0.2],
                [1.5, 2.0, 0.2, -1.0],
            ]
        )
        labels = np.array([0, 0, 1, 1])
        classes = np.array([0, 1])
        probabilities = np.eye(2)[labels] * 0.9 + 0.05
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        candidate_labels = [0, 0, 1, 1]
        predictive = select_predictive_candidates(
            fields,
            candidate_labels,
            probabilities,
            labels,
            classes,
            objective="direct",
            score="normalized_radial",
            component_limit=4,
            initial_components_per_class=1,
            minimum_improvement=0.0,
        )
        coverage = select_coverage_candidates(
            fields,
            candidate_labels,
            labels,
            classes,
            component_limit=4,
            initial_components_per_class=1,
        )
        self.assertEqual(len(predictive["selected_candidate_indices"]), 4)
        self.assertEqual(len(coverage["selected_candidate_indices"]), 4)

    def test_temperature_prediction_edit_and_rollback(self):
        candidates = [
            SphereCandidate(0, np.array([-1.0, 0.0]), 0.8, 0, 3),
            SphereCandidate(1, np.array([1.0, 0.0]), 0.8, 1, 3),
        ]
        selection = {
            "selected_candidate_indices": [0, 1],
            "component_counts": [1, 1],
            "objective_trajectory": [1.0],
        }
        student = serialize_factorial_student(
            cell={
                "id": "fixture",
                "objective": "direct",
                "primitive": "sphere",
                "score": "teacher_softmin",
                "budget": "component",
            },
            classes=np.array([0, 1]),
            candidates=candidates,
            selection=selection,
            parent_representation_hash="a" * 64,
            directional_representation_hash=None,
            class_priors=np.array([0.5, 0.5]),
        )
        features = np.array([[-1.1, 0.0], [-0.8, 0.1], [0.9, 0.0], [1.2, -0.1]])
        labels = np.array([0, 0, 1, 1])
        temperature = fit_global_temperature(
            student, features, labels, minimum=0.05, maximum=20.0
        )
        self.assertGreaterEqual(temperature, 0.05)
        predictions, probabilities = predict_factorial_student(student, features)
        np.testing.assert_array_equal(predictions, labels)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        lifecycle = local_edit_rollback_evidence(student, features)
        self.assertTrue(lifecycle["exact_json_rollback"])
        self.assertTrue(lifecycle["rollback_restored_predictions"])


if __name__ == "__main__":
    unittest.main()
