from __future__ import annotations

import copy
import unittest

import numpy as np

from experiments.common.v6_factorial import select_predictive_candidates
from experiments.common.v6_directional_distillation import (
    normalized_representation_hash,
)
from experiments.common.v61_tangent import (
    generate_tangent_cap_candidates,
    predict_tangent_student,
    serialize_tangent_student,
    tangent_field_matrix,
    tangent_local_edit_rollback_evidence,
)
from experiments.tier4.eval_v61_tangent_s1 import (
    DEFAULT_CONFIG,
    _validate_config,
)
from src.directional_primitive import l2_normalize
from src.tangent_cap_primitive import (
    TangentCapPrimitive,
    fit_tangent_cap,
    sphere_log_map,
)


class TangentCapPrimitiveTests(unittest.TestCase):
    def _support(self, count: int = 10) -> np.ndarray:
        rng = np.random.default_rng(17)
        points = np.column_stack(
            [
                np.ones(count),
                0.12 * rng.normal(size=(count, 4)),
            ]
        )
        return l2_normalize(points)

    def test_log_map_zero_small_angle_and_near_antipode(self):
        mean = np.array([1.0, 0.0, 0.0])
        angle = 1e-6
        points = np.array(
            [
                [1.0, 0.0, 0.0],
                [np.cos(angle), np.sin(angle), 0.0],
                [-1.0, 1e-10, 0.0],
            ]
        )
        mapped = sphere_log_map(mean, points)
        np.testing.assert_allclose(mapped[0], 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(mapped[1])), angle, places=10)
        self.assertAlmostEqual(float(np.linalg.norm(mapped[2])), np.pi, places=8)
        self.assertTrue(np.all(np.isfinite(mapped)))
        np.testing.assert_allclose(mapped @ mean, 0.0, atol=1e-12)

    def test_fit_has_unit_mean_and_tangent_orthonormal_basis(self):
        primitive = fit_tangent_cap(self._support(), 2)
        self.assertAlmostEqual(float(np.linalg.norm(primitive.mean_direction)), 1.0)
        np.testing.assert_allclose(
            primitive.basis.T @ primitive.basis, np.eye(2), atol=1e-10
        )
        np.testing.assert_allclose(
            primitive.mean_direction @ primitive.basis, 0.0, atol=1e-10
        )

    def test_fit_is_scale_invariant_deterministic_and_serializable(self):
        support = self._support()
        first = fit_tangent_cap(
            support,
            2,
            class_label=3,
            anchor_index=7,
            support_indices=tuple(range(len(support))),
        )
        second = fit_tangent_cap(
            support * np.arange(1, len(support) + 1)[:, None],
            2,
            class_label=3,
            anchor_index=7,
            support_indices=tuple(range(len(support))),
        )
        np.testing.assert_allclose(first.mean_direction, second.mean_direction)
        np.testing.assert_allclose(first.basis, second.basis)
        np.testing.assert_allclose(
            first.radial_field(support), second.radial_field(support)
        )
        replay = TangentCapPrimitive.from_dict(first.to_dict())
        self.assertEqual(first.to_dict(), replay.to_dict())

    def test_radial_gradient_matches_finite_difference(self):
        primitive = fit_tangent_cap(self._support(), 2)
        point = l2_normalize(np.array([[1.0, 0.07, -0.04, 0.02, 0.03]]))
        analytic = primitive.radial_gradient(point)[0]
        step = 1e-7
        numeric = np.empty(primitive.dimension)
        for axis in range(primitive.dimension):
            delta = np.zeros_like(point)
            delta[0, axis] = step
            numeric[axis] = (
                primitive.radial_field(point + delta)[0]
                - primitive.radial_field(point - delta)[0]
            ) / (2.0 * step)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-4, atol=2e-4)

    def test_rank_support_and_parameter_budget_fail_closed(self):
        with self.assertRaises(ValueError):
            fit_tangent_cap(self._support(3), 2)
        rng = np.random.default_rng(29)
        primitive = fit_tangent_cap(
            rng.normal(size=(34, 384)),
            32,
            support_indices=tuple(range(34)),
        )
        self.assertEqual(primitive.parameter_count, 12706)
        self.assertEqual(46 * primitive.parameter_count, 584476)
        broken = primitive.to_dict()
        broken["support_indices"] = broken["support_indices"][:-1]
        with self.assertRaises(ValueError):
            TangentCapPrimitive.from_dict(broken)

    def test_deterministic_basis_signs_are_positive_at_pivots(self):
        primitive = fit_tangent_cap(self._support(), 2)
        for column in range(primitive.rank):
            pivot = int(np.argmax(np.abs(primitive.basis[:, column])))
            self.assertGreaterEqual(primitive.basis[pivot, column], 0.0)


class TangentCapStudentTests(unittest.TestCase):
    def _fixture(self):
        rng = np.random.default_rng(41)
        class_zero = np.column_stack(
            [np.ones(40), 0.08 * rng.normal(size=(40, 4))]
        )
        class_one = np.column_stack(
            [-np.ones(40), 0.08 * rng.normal(size=(40, 4))]
        )
        features = np.vstack([class_zero, class_one])
        labels = np.repeat([0, 1], 40)
        probabilities = np.full((80, 2), 0.1)
        probabilities[np.arange(80), labels] = 0.9
        return features, labels, probabilities, np.array([0, 1])

    def test_candidates_use_r_plus_2_support_and_fixed_anchor_policy(self):
        features, labels, probabilities, classes = self._fixture()
        normalized = l2_normalize(features)
        first = generate_tangent_cap_candidates(
            normalized,
            labels,
            probabilities,
            classes,
            rank=2,
            candidates_per_class=3,
            anchor_fraction=0.5,
        )
        second = generate_tangent_cap_candidates(
            normalized,
            labels,
            probabilities,
            classes,
            rank=2,
            candidates_per_class=3,
            anchor_fraction=0.5,
        )
        self.assertEqual([item.to_dict() for item in first], [
            item.to_dict() for item in second
        ])
        self.assertTrue(all(len(item.support_indices) == 4 for item in first))
        self.assertEqual(
            tangent_field_matrix(
                first, normalized, "normalized_tangent_radial"
            ).shape,
            (80, 6),
        )
        likelihood = tangent_field_matrix(
            first, normalized, "tangent_gaussian_log_likelihood"
        )
        self.assertTrue(np.all(np.isfinite(likelihood)))

    def test_student_lineage_replay_local_edit_and_rollback(self):
        features, labels, probabilities, classes = self._fixture()
        normalized = l2_normalize(features)
        candidates = generate_tangent_cap_candidates(
            normalized,
            labels,
            probabilities,
            classes,
            rank=2,
            candidates_per_class=3,
            anchor_fraction=0.5,
        )
        fields = tangent_field_matrix(
            candidates, normalized, "normalized_tangent_radial"
        )
        selection = select_predictive_candidates(
            fields,
            [int(item.class_label) for item in candidates],
            probabilities,
            labels,
            classes,
            objective="direct",
            score="normalized_radial",
            component_limit=4,
            initial_components_per_class=1,
            minimum_improvement=1e-8,
        )
        parent_hash = "a" * 64
        student = serialize_tangent_student(
            classes=classes,
            candidates=candidates,
            selection=selection,
            parent_representation_hash=parent_hash,
            directional_representation_hash=normalized_representation_hash(
                parent_hash
            ),
            cohort_indices=np.arange(len(features)),
            configuration={"rank": 2},
        )
        first = predict_tangent_student(
            student, features, parent_representation_hash=parent_hash
        )
        second = predict_tangent_student(
            student, features * 7.0, parent_representation_hash=parent_hash
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        evidence = tangent_local_edit_rollback_evidence(
            student, features, parent_representation_hash=parent_hash
        )
        self.assertTrue(evidence["exact_json_rollback"])
        self.assertTrue(evidence["rollback_restored_predictions"])
        self.assertGreaterEqual(evidence["unaffected_prediction_preservation"], 0.999)
        with self.assertRaises(ValueError):
            predict_tangent_student(
                student, features, parent_representation_hash="b" * 64
            )
        broken = copy.deepcopy(student)
        broken["normalization_policy"]["dtype"] = "float32"
        with self.assertRaises(ValueError):
            predict_tangent_student(
                broken, features, parent_representation_hash=parent_hash
            )


class TangentCapA1ConfigTests(unittest.TestCase):
    def test_registered_config_is_strict_and_test_sealed(self):
        import json

        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        _validate_config(config)
        config["test_labels_opened"] = True
        with self.assertRaises(ValueError):
            _validate_config(config)

    def test_rank_or_budget_drift_fails_closed(self):
        import json

        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["primitive"]["rank"] = 16
        with self.assertRaises(ValueError):
            _validate_config(config)
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["budget"]["parameter_limit"] += 1
        with self.assertRaises(ValueError):
            _validate_config(config)


if __name__ == "__main__":
    unittest.main()
