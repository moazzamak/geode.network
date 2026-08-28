from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from experiments.common.v61_weighted_readout import (
    _objective_and_gradient,
    fit_weighted_readout,
    normalized_class_weights,
    predict_weighted_student,
    readout_collapse_summary,
    serialize_weighted_student,
    validate_weighted_student,
    weighted_class_logits,
    weighted_local_edit_rollback_evidence,
    weighted_student_parameter_count,
)
from experiments.tier4.eval_v61_weighted_s1 import (
    DEFAULT_CONFIG,
    _validate_config,
)
from src.subspace_primitive import SubspacePrimitive, fit_subspace_primitive


class WeightedReadoutTests(unittest.TestCase):
    def _fields(self) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray]:
        fields = np.array(
            [
                [0.1, 0.9, 2.0, 2.5],
                [0.2, 1.0, 2.2, 2.1],
                [2.2, 2.0, 0.1, 0.8],
                [2.4, 2.1, 0.2, 0.9],
            ],
            dtype=np.float64,
        )
        return fields, [0, 0, 1, 1], np.array([0, 0, 1, 1]), np.array([0, 1])

    def _parent_student(self) -> dict:
        rng = np.random.default_rng(31)
        candidates = []
        for class_label, offset in ((0, -1.0), (0, -0.5), (1, 0.5), (1, 1.0)):
            points = rng.normal(size=(34, 34)) * 0.1 + offset
            primitive = fit_subspace_primitive(
                points,
                32,
                class_label=class_label,
                anchor_index=len(candidates),
            )
            candidates.append(
                {"family": "subspace_r32", "payload": primitive.to_dict()}
            )
        return {
            "schema_version": 1,
            "cell": {
                "budget": "component",
                "id": "direct_subspace_radial_component",
                "objective": "direct",
                "primitive": "subspace_r32",
                "score": "normalized_radial",
            },
            "classes": [0, 1],
            "selected_candidates": candidates,
            "selected_candidate_indices": [0, 1, 2, 3],
            "component_counts": [2, 2],
            "objective_trajectory": [1.0, 0.5],
            "parent_representation_hash": "a" * 64,
            "directional_representation_hash": None,
            "class_priors": [0.5, 0.5],
            "global_temperature": 1.0,
        }

    def _fitted(self, *, weighted: bool = True) -> dict:
        fields, candidate_labels, labels, classes = self._fields()
        return fit_weighted_readout(
            fields,
            candidate_labels,
            labels,
            classes,
            regularization=1e-4,
            maximum_iterations=500,
            gradient_tolerance=1e-8,
            minimum_temperature=0.05,
            maximum_temperature=20.0,
            fit_component_weights=weighted,
        )

    def test_weights_are_nonnegative_and_normalized_per_class(self):
        fitted = self._fitted()
        weights = np.asarray(fitted["component_weights"])
        self.assertTrue(np.all(weights >= 0.0))
        np.testing.assert_allclose(weights[:2].sum(), 1.0)
        np.testing.assert_allclose(weights[2:].sum(), 1.0)

    def test_equal_weights_reproduce_log_mean_exp_and_zero_weights_are_ignored(self):
        fields, candidate_labels, _, classes = self._fields()
        equal = normalized_class_weights(
            np.zeros(4), candidate_labels, classes
        )
        logits = weighted_class_logits(
            fields,
            candidate_labels,
            classes,
            equal,
            global_temperature=1.0,
        )
        expected = np.column_stack(
            [
                log_mean_exp(-fields[:, :2]),
                log_mean_exp(-fields[:, 2:]),
            ]
        )
        np.testing.assert_allclose(logits, expected)
        zero_weight = np.array([1.0, 0.0, 0.0, 1.0])
        zero_logits = weighted_class_logits(
            fields,
            candidate_labels,
            classes,
            zero_weight,
            global_temperature=1.0,
        )
        np.testing.assert_allclose(zero_logits[:, 0], -fields[:, 0])
        np.testing.assert_allclose(zero_logits[:, 1], -fields[:, 3])

    def test_single_component_classes_are_supported(self):
        fields = np.array([[0.2, 1.0], [1.0, 0.2]])
        logits = weighted_class_logits(
            fields,
            [0, 1],
            np.array([0, 1]),
            np.ones(2),
            global_temperature=2.0,
        )
        np.testing.assert_allclose(logits, -fields / 2.0)

    def test_component_permutation_is_equivariant(self):
        fields, candidate_labels, _, classes = self._fields()
        log_weights = np.array([0.3, -0.2, 0.7, -0.4])
        weights = normalized_class_weights(log_weights, candidate_labels, classes)
        expected = weighted_class_logits(
            fields,
            candidate_labels,
            classes,
            weights,
            global_temperature=1.3,
        )
        order = np.array([2, 0, 3, 1])
        actual = weighted_class_logits(
            fields[:, order],
            np.asarray(candidate_labels)[order],
            classes,
            weights[order],
            global_temperature=1.3,
        )
        np.testing.assert_allclose(actual, expected)

    def test_optimizer_is_deterministic_and_equal_parent_is_nested(self):
        first = self._fitted()
        second = self._fitted()
        self.assertEqual(first, second)
        equal = self._fitted(weighted=False)
        np.testing.assert_allclose(equal["log_weights"], 0.0)
        np.testing.assert_allclose(equal["component_weights"], 0.5)
        self.assertLessEqual(
            first["optimizer"]["final_objective"],
            equal["optimizer"]["final_objective"] + 1e-8,
        )

    def test_optimizer_gradient_matches_finite_difference(self):
        fields, candidate_labels, labels, classes = self._fields()
        parameters = np.array([0.2, -0.1, 0.3, -0.4, np.log(1.2)])
        target_columns = labels.copy()
        _, analytic = _objective_and_gradient(
            parameters,
            fields=fields,
            candidate_labels=np.asarray(candidate_labels),
            classes=classes,
            target_columns=target_columns,
            regularization=1e-4,
        )
        step = 1e-6
        numeric = np.empty_like(parameters)
        for index in range(len(parameters)):
            delta = np.zeros_like(parameters)
            delta[index] = step
            upper, _ = _objective_and_gradient(
                parameters + delta,
                fields=fields,
                candidate_labels=np.asarray(candidate_labels),
                classes=classes,
                target_columns=target_columns,
                regularization=1e-4,
            )
            lower, _ = _objective_and_gradient(
                parameters - delta,
                fields=fields,
                candidate_labels=np.asarray(candidate_labels),
                classes=classes,
                target_columns=target_columns,
                regularization=1e-4,
            )
            numeric[index] = (upper - lower) / (2.0 * step)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-7)

    def test_serialization_accounting_collapse_and_schema_guards(self):
        parent = self._parent_student()
        student = serialize_weighted_student(
            parent,
            self._fitted(),
            parent_student_sha256="b" * 64,
        )
        validate_weighted_student(student)
        expected = sum(
            SubspacePrimitive.from_dict(item["payload"]).parameter_count
            for item in student["selected_candidates"]
        ) + 4
        self.assertEqual(weighted_student_parameter_count(student), expected)
        collapse = readout_collapse_summary(student)
        self.assertEqual(collapse["class_count"], 2)
        broken = copy.deepcopy(student)
        broken["readout_contract"]["temperature_policy"] = "per_class"
        with self.assertRaises(ValueError):
            validate_weighted_student(broken)
        broken = copy.deepcopy(student)
        broken["selected_candidates"].reverse()
        with self.assertRaises(ValueError):
            validate_weighted_student(broken)

    def test_prediction_lineage_local_edit_and_exact_rollback(self):
        parent = self._parent_student()
        student = serialize_weighted_student(
            parent,
            self._fitted(),
            parent_student_sha256="b" * 64,
        )
        features = np.vstack(
            [
                np.full((20, 34), -0.8),
                np.full((20, 34), 0.8),
            ]
        )
        predictions, probabilities = predict_weighted_student(
            student, features, parent_representation_hash="a" * 64
        )
        self.assertEqual(predictions.shape, (40,))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        evidence = weighted_local_edit_rollback_evidence(
            student, features, parent_representation_hash="a" * 64
        )
        self.assertTrue(evidence["exact_json_rollback"])
        self.assertTrue(evidence["rollback_restored_predictions"])
        self.assertGreaterEqual(evidence["unaffected_prediction_preservation"], 0.999)
        with self.assertRaises(ValueError):
            predict_weighted_student(
                student, features, parent_representation_hash="c" * 64
            )


class WeightedReadoutConfigTests(unittest.TestCase):
    def test_registered_config_is_strict_and_test_sealed(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        _validate_config(config)
        config["test_labels_opened"] = True
        with self.assertRaises(ValueError):
            _validate_config(config)

    def test_optimizer_budget_and_tangent_scope_drift_fail_closed(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["readout"]["regularization"] = 0.001
        with self.assertRaises(ValueError):
            _validate_config(config)
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["budget"]["parameter_limit"] += 1
        with self.assertRaises(ValueError):
            _validate_config(config)
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["tangent_result"]["required_advancement_status"] = True
        with self.assertRaises(ValueError):
            _validate_config(config)


def log_mean_exp(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=1)
    return maximum + np.log(
        np.mean(np.exp(values - maximum[:, None]), axis=1)
    )


if __name__ == "__main__":
    unittest.main()
