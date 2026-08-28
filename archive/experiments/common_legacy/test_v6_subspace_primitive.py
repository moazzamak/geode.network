from __future__ import annotations

import unittest

import numpy as np

from experiments.common.v6_boundary_distillation import (
    fit_distilled_candidate_fields,
)
from src.subspace_primitive import SubspacePrimitive, fit_subspace_primitive


class SubspacePrimitiveTests(unittest.TestCase):
    def _points(self) -> np.ndarray:
        rng = np.random.default_rng(13)
        latent = rng.normal(size=(12, 2))
        basis = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, -0.5]]
        )
        return latent @ basis.T + np.array([2.0, -1.0, 0.5, 3.0])

    def test_fit_is_deterministic_and_serializable(self):
        points = self._points()
        first = fit_subspace_primitive(points, 2)
        second = fit_subspace_primitive(points, 2)
        np.testing.assert_array_equal(first.center, second.center)
        np.testing.assert_array_equal(first.basis, second.basis)
        restored = SubspacePrimitive.from_dict(first.to_dict())
        np.testing.assert_array_equal(
            first.radial_field(points), restored.radial_field(points)
        )

    def test_translation_and_rotation_invariance(self):
        points = self._points()
        primitive = fit_subspace_primitive(points, 2)
        query = points[:4] + np.array([0.1, -0.2, 0.05, 0.3])
        rng = np.random.default_rng(5)
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        translation = np.array([3.0, -2.0, 1.0, 0.5])
        transformed_points = points @ rotation + translation
        transformed_query = query @ rotation + translation
        transformed = fit_subspace_primitive(transformed_points, 2)
        np.testing.assert_allclose(
            primitive.quadratic_form(query),
            transformed.quadratic_form(transformed_query),
            rtol=1e-9,
            atol=1e-9,
        )

    def test_orthogonal_distance_increases_field(self):
        primitive = fit_subspace_primitive(self._points(), 2)
        on_subspace = primitive.center[None, :]
        projector = primitive.basis @ primitive.basis.T
        normal = np.eye(primitive.dimension) - projector
        direction = normal[np.argmax(np.linalg.norm(normal, axis=1))]
        direction /= np.linalg.norm(direction)
        off_subspace = on_subspace + direction[None, :]
        self.assertLess(
            float(primitive.radial_field(on_subspace)[0]),
            float(primitive.radial_field(off_subspace)[0]),
        )

    def test_quadratic_gradient_matches_finite_difference(self):
        primitive = fit_subspace_primitive(self._points(), 2)
        point = np.array([[1.8, -0.4, 0.7, 2.5]], dtype=np.float64)
        analytic = primitive.quadratic_gradient(point)[0]
        epsilon = 1e-6
        numeric = np.empty(primitive.dimension)
        for dimension in range(primitive.dimension):
            delta = np.zeros_like(point)
            delta[0, dimension] = epsilon
            numeric[dimension] = (
                primitive.quadratic_form(point + delta)[0]
                - primitive.quadratic_form(point - delta)[0]
            ) / (2.0 * epsilon)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-5)

    def test_support_and_rank_boundaries_fail_closed(self):
        points = self._points()
        with self.assertRaises(ValueError):
            fit_subspace_primitive(points[:3], 2)
        with self.assertRaises(ValueError):
            fit_subspace_primitive(points, 4)

    def test_generic_distillation_initializes_three_per_class(self):
        fields = np.array(
            [
                [0.0, 0.2, 0.4, 2.0, 2.2, 2.4],
                [2.0, 2.2, 2.4, 0.0, 0.2, 0.4],
            ]
        )
        teacher = np.array([[0.9, 0.1], [0.1, 0.9]])
        selection = fit_distilled_candidate_fields(
            fields,
            [0, 0, 0, 1, 1, 1],
            teacher,
            np.array([0, 1]),
            np.array([0, 1]),
            component_limit=6,
            initial_components_per_class=3,
        )
        self.assertEqual(selection["component_counts"], [3, 3])
        self.assertEqual(len(selection["selected_candidate_indices"]), 6)


if __name__ == "__main__":
    unittest.main()
