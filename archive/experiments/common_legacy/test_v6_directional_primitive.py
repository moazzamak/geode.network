from __future__ import annotations

import unittest

import numpy as np

from experiments.common.v6_boundary_distillation import (
    fit_distilled_candidate_fields,
)
from experiments.common.v5_statistics import paired_seed_t_interval
from experiments.common.v6_directional_distillation import (
    directional_field_matrix,
    generate_paired_directional_candidates,
    normalized_representation_hash,
    predict_directional_student,
    serialize_directional_student,
)
from src.directional_primitive import (
    SphericalCapPrimitive,
    fit_spherical_cap,
    l2_normalize,
)


class DirectionalPrimitiveTests(unittest.TestCase):
    def test_paired_seed_interval_is_directional_and_validated(self):
        interval = paired_seed_t_interval(
            np.array([0.84, 0.85, 0.86]),
            np.array([0.80, 0.81, 0.82]),
        )
        self.assertAlmostEqual(interval["difference"], 0.04)
        self.assertGreater(interval["lower"], 0.0)
        with self.assertRaises(ValueError):
            paired_seed_t_interval(np.array([0.8]), np.array([0.7]))

    def test_angular_distance_fixtures_and_monotonicity(self):
        cap = SphericalCapPrimitive(np.array([1.0, 0.0]), np.pi / 4.0)
        points = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 0.0]])
        np.testing.assert_allclose(
            cap.angles(points), [0.0, np.pi / 4.0, np.pi / 2.0, np.pi]
        )
        self.assertTrue(np.all(np.diff(cap.angular_field(points)) > 0.0))

    def test_score_is_scale_invariant_and_rejects_zero_vectors(self):
        cap = SphericalCapPrimitive(np.array([1.0, 0.0]), 0.5)
        points = np.array([[1.0, 1.0], [2.0, -1.0]])
        np.testing.assert_allclose(
            cap.angular_field(points), cap.angular_field(points * [[3.0], [7.0]])
        )
        with self.assertRaises(ValueError):
            l2_normalize(np.array([[0.0, 0.0]]))

    def test_fit_preserves_unit_direction_and_serialization(self):
        support = np.array([[1.0, 0.1], [1.0, -0.1], [2.0, 0.0]])
        cap = fit_spherical_cap(
            support,
            class_label=3,
            anchor_index=4,
            support_indices=(1, 4, 7),
        )
        self.assertAlmostEqual(float(np.linalg.norm(cap.mean_direction)), 1.0)
        replay = SphericalCapPrimitive.from_dict(cap.to_dict())
        self.assertEqual(cap.to_dict(), replay.to_dict())
        self.assertEqual(cap.parameter_count, 3)

    def test_gradient_is_finite_near_zero_and_matches_finite_difference(self):
        cap = SphericalCapPrimitive(np.array([1.0, 0.0]), 0.4)
        point = np.array([[1.0, 1e-5]])
        gradient = cap.angular_gradient(point)[0]
        self.assertTrue(np.all(np.isfinite(gradient)))
        step = 1e-7
        numerical = np.empty(2)
        for axis in range(2):
            offset = np.zeros((1, 2))
            offset[0, axis] = step
            numerical[axis] = (
                cap.angular_field(point + offset)[0]
                - cap.angular_field(point - offset)[0]
            ) / (2.0 * step)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-3, atol=2e-3)
        self.assertTrue(
            np.all(np.isfinite(cap.angular_gradient(np.array([[-1.0, 0.0]]))))
        )

    def test_paired_candidates_use_identical_anchors_and_supports(self):
        rng = np.random.default_rng(3)
        features = rng.normal(size=(24, 4))
        labels = np.repeat([0, 1], 12)
        probabilities = np.full((24, 2), 0.2)
        probabilities[np.arange(24), labels] = 0.8
        normalized = l2_normalize(features)
        spheres, caps = generate_paired_directional_candidates(
            normalized,
            labels,
            probabilities,
            np.array([0, 1]),
            candidates_per_class=3,
            seed_size=5,
            anchor_fraction=0.5,
        )
        self.assertEqual(
            [candidate.anchor_index for candidate in spheres],
            [candidate.anchor_index for candidate in caps],
        )
        self.assertTrue(
            all(sphere.support_size == len(cap.support_indices) == 5
                for sphere, cap in zip(spheres, caps))
        )
        for sphere, cap in zip(spheres, caps):
            expected_center = normalized[list(cap.support_indices)].mean(axis=0)
            np.testing.assert_allclose(sphere.center, expected_center)
        self.assertEqual(
            directional_field_matrix(spheres, normalized, "euclidean_sphere").shape,
            directional_field_matrix(caps, normalized, "cosine_cap").shape,
        )

    def test_student_replay_and_hash_mismatch_rejection(self):
        rng = np.random.default_rng(5)
        features = rng.normal(size=(20, 3))
        labels = np.repeat([0, 1], 10)
        probabilities = np.full((20, 2), 0.1)
        probabilities[np.arange(20), labels] = 0.9
        classes = np.array([0, 1])
        normalized = l2_normalize(features)
        _, caps = generate_paired_directional_candidates(
            normalized,
            labels,
            probabilities,
            classes,
            candidates_per_class=3,
            seed_size=4,
            anchor_fraction=0.5,
        )
        selection = fit_distilled_candidate_fields(
            directional_field_matrix(caps, normalized, "cosine_cap"),
            [int(candidate.class_label) for candidate in caps],
            probabilities,
            labels,
            classes,
            component_limit=4,
            exact_component_count=True,
        )
        self.assertEqual(len(selection["selected_candidate_indices"]), 4)
        parent_hash = "a" * 64
        student = serialize_directional_student(
            geometry="cosine_cap",
            classes=classes,
            candidates=caps,
            selection=selection,
            parent_representation_hash=parent_hash,
            directional_representation_hash=normalized_representation_hash(parent_hash),
            cohort_indices=np.arange(20),
            configuration={},
        )
        first = predict_directional_student(
            student, features, parent_representation_hash=parent_hash
        )
        second = predict_directional_student(
            student, features * 3.0, parent_representation_hash=parent_hash
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        with self.assertRaises(ValueError):
            predict_directional_student(
                student, features, parent_representation_hash="b" * 64
            )


if __name__ == "__main__":
    unittest.main()
