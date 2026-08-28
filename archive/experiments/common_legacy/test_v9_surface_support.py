from __future__ import annotations

import unittest

import numpy as np

from experiments.common.v9_surface_support import (
    BoundedTubePrimitive,
    class_minimum_fields,
    class_signed_depths,
    deterministic_equal_mass_bands,
    metric_corrected_field,
    normalized_shell,
    permuted_labels,
    random_orientation,
    validate_disjoint_partitions,
    fit_bounded_tube,
)
from src.runtime.schemas import SurfaceSupportDiagnostic
from src.subspace_primitive import SubspacePrimitive


def _primitive(variances: tuple[float, ...] = (4.0, 4.0)) -> SubspacePrimitive:
    return SubspacePrimitive(
        center=np.zeros(3),
        basis=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        tangent_variances=np.asarray(variances),
        residual_variance=4.0,
        class_label=0,
        support_size=10,
    )


class V9SurfaceSupportTests(unittest.TestCase):
    def test_sphere_metric_shell_equals_radial_gap(self):
        primitive = _primitive()
        points = np.asarray([[2.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
        corrected = metric_corrected_field(primitive, points, eta=1e-12)
        expected = np.asarray([0.0, 2.0])
        np.testing.assert_allclose(np.abs(corrected), expected, atol=1e-10)

    def test_anisotropic_metric_correction_matches_gradient_definition(self):
        primitive = _primitive((1.0, 4.0))
        point = np.asarray([[1.0, 2.0, 3.0]])
        field = primitive.radial_field(point)[0]
        radius = np.sqrt(primitive.quadratic_form(point))[0]
        gradient = primitive.quadratic_gradient(point)[0] / (2.0 * radius)
        expected = field / (np.linalg.norm(gradient) + 1e-9)
        self.assertAlmostEqual(
            metric_corrected_field(primitive, point, eta=1e-9)[0], expected
        )

    def test_absolute_shell_is_sign_invariant_but_volume_is_not(self):
        fields = np.asarray([-2.0, -0.25, 1.5])
        np.testing.assert_array_equal(normalized_shell(fields), normalized_shell(-fields))
        self.assertFalse(np.array_equal(fields, -fields))

    def test_equal_mass_bands_are_deterministic(self):
        values = np.asarray([-0.1, -0.8, -0.3, -0.5, -0.7, 0.2])
        self.assertEqual(
            deterministic_equal_mass_bands(values, fraction=0.4),
            deterministic_equal_mass_bands(values[::-1], fraction=0.4),
        )

    def test_partition_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_disjoint_partitions({"geometry": ["a", "b"], "eval": ["b"]})

    def test_negative_controls_replay(self):
        first = random_orientation(_primitive(), seed=51)
        second = random_orientation(_primitive(), seed=51)
        np.testing.assert_array_equal(first.basis, second.basis)
        labels = [0, 0, 1, 1, 2]
        np.testing.assert_array_equal(
            permuted_labels(labels, seed=51), permuted_labels(labels, seed=51)
        )

    def test_center_and_zero_gradient_are_finite(self):
        corrected = metric_corrected_field(
            _primitive(), np.zeros((1, 3)), eta=1e-6
        )
        self.assertTrue(np.all(np.isfinite(corrected)))

    def test_softmin_normalization_is_component_count_invariant(self):
        fields_one = np.asarray([[0.4]])
        fields_two = np.asarray([[0.4, 0.4]])
        one = class_signed_depths(fields_one, [0], np.asarray([0]), np.asarray([1.0]))
        two = class_signed_depths(
            fields_two, [0, 0], np.asarray([0]), np.asarray([0.5, 0.5])
        )
        np.testing.assert_allclose(one, two, atol=1e-12)

    def test_component_occupancy_uses_nearest_same_class_field(self):
        fields = np.asarray([[0.4, -0.2, -1.0]])
        scores = class_minimum_fields(fields, [0, 0, 1], np.asarray([0, 1]))
        np.testing.assert_array_equal(scores, np.asarray([[-0.2, -1.0]]))

    def test_surface_diagnostic_round_trips(self):
        pairs = (
            ("near_surface", 0.8),
            ("deep_interior", 0.6),
            ("exterior", 0.1),
        )
        record = SurfaceSupportDiagnostic(
            component_hash="a" * 64,
            representation_hash="b" * 64,
            score_variant="normalized",
            score_direction="lower_is_stronger_support",
            class_label=1,
            seed=11,
            partition_id="development_eval",
            normalized_signed_depth_quantiles=(-2.0, -1.0, 0.0, 1.0, 2.0),
            metric_signed_depth_quantiles=(-3.0, -1.0, 0.0, 1.0, 3.0),
            stratum_counts=(
                ("near_surface", 8),
                ("deep_interior", 6),
                ("exterior", 1),
            ),
            own_class_precision=pairs,
            competing_class_occupancy=pairs,
            unknown_occupancy=pairs,
            width_selection_provenance="geometry_fit",
            selected_ids=("dev:1",),
            replay_hash="c" * 64,
        )
        self.assertEqual(
            SurfaceSupportDiagnostic.from_dict(record.to_dict()), record
        )

    def test_unbounded_residual_ignores_tangent_distance(self):
        tube = BoundedTubePrimitive(
            center=np.zeros(3),
            basis=np.asarray([[1.0], [0.0], [0.0]]),
            residual_variance=1.0,
            tangent_extents=np.asarray([1.0]),
            tangent_scales=np.asarray([0.5]),
            penalty_weight=1.0,
            class_label=0,
        )
        points = np.asarray([[1.0, 0.2, 0.0], [8.0, 0.2, 0.0]])
        np.testing.assert_allclose(tube.unbounded_score(points), [0.04, 0.04])
        self.assertGreater(tube.score(points)[1], tube.score(points)[0])

    def test_tube_penalty_is_zero_inside_and_monotonic_outside(self):
        tube = BoundedTubePrimitive(
            center=np.zeros(2),
            basis=np.asarray([[1.0], [0.0]]),
            residual_variance=1.0,
            tangent_extents=np.asarray([2.0]),
            tangent_scales=np.asarray([1.0]),
            penalty_weight=2.0,
            class_label=0,
        )
        scores = tube.score(np.asarray([[1.0, 0.0], [3.0, 0.0], [4.0, 0.0]]))
        np.testing.assert_allclose(scores, [0.0, 2.0, 8.0])

    def test_bounded_tube_fit_replays(self):
        rng = np.random.default_rng(53)
        geometry = rng.normal(size=(80, 6))
        calibration = rng.normal(size=(20, 6))
        first = fit_bounded_tube(
            geometry,
            calibration,
            rank=2,
            extent_quantile=0.95,
            scale_quantile=0.75,
            penalty_weight=1.0,
            class_label=0,
        )
        second = fit_bounded_tube(
            geometry,
            calibration,
            rank=2,
            extent_quantile=0.95,
            scale_quantile=0.75,
            penalty_weight=1.0,
            class_label=0,
        )
        self.assertEqual(first.to_dict(), second.to_dict())


if __name__ == "__main__":
    unittest.main()
