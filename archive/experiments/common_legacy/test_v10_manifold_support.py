from __future__ import annotations

import unittest

import numpy as np

from experiments.common.v10_manifold_support import (
    DimensionlessTube,
    SafetyPenaltySelectionError,
    deterministic_patch_assignments,
    estimate_registered_rank,
    fit_dimensionless_tube,
    generate_safety_probes,
    probe_acceptance,
    select_smallest_safety_penalty,
)
from experiments.common.v9_surface_support import validate_disjoint_partitions
from src.runtime.schemas import TubeCalibrationRecord, TubeSafetyEvidence


def _tube(*, rank: int = 2, penalty: float = 16.0) -> DimensionlessTube:
    basis = np.eye(4, rank)
    return DimensionlessTube(
        center=np.zeros(4),
        basis=basis,
        residual_scale=4.0,
        tangent_extents=np.full(rank, 2.0),
        outer_scales=np.full(rank, 1.0),
        penalty_weight=penalty,
        class_label=0,
    )


class V10ManifoldSupportTests(unittest.TestCase):
    def test_registered_rank_recovery_and_volume_control(self):
        rng = np.random.default_rng(57)
        basis, _ = np.linalg.qr(rng.normal(size=(64, 16)))
        manifold = rng.normal(size=(320, 16)) @ basis.T
        manifold += rng.normal(scale=0.01, size=manifold.shape)
        recovered = estimate_registered_rank(
            manifold,
            rank_grid=(8, 16, 32),
            explained_variance_target=0.95,
        )
        self.assertEqual(recovered["selected_rank"], 16)
        volume = estimate_registered_rank(
            rng.normal(size=(320, 64)),
            rank_grid=(8, 16, 32),
            explained_variance_target=0.95,
        )
        self.assertGreater(volume["residual_fraction_at_max_rank"], 0.25)

    def test_common_feature_rescaling_preserves_scores(self):
        rng = np.random.default_rng(56)
        geometry = rng.normal(size=(80, 6))
        calibration = rng.normal(size=(30, 6))
        query = rng.normal(size=(20, 6))
        first = fit_dimensionless_tube(
            geometry,
            calibration,
            rank=2,
            extent_quantile=0.95,
            outer_scale_policy="interquantile_range",
            penalty_weight=16.0,
            class_label=0,
        )
        second = fit_dimensionless_tube(
            7.0 * geometry,
            7.0 * calibration,
            rank=2,
            extent_quantile=0.95,
            outer_scale_policy="interquantile_range",
            penalty_weight=16.0,
            class_label=0,
        )
        np.testing.assert_allclose(first.score(query), second.score(7.0 * query))

    def test_rank_normalization_preserves_equal_per_axis_overshoot(self):
        rank_one = _tube(rank=1)
        rank_two = _tube(rank=2)
        one = np.asarray([[3.0, 0.0, 0.0, 0.0]])
        two = np.asarray([[3.0, 3.0, 0.0, 0.0]])
        self.assertAlmostEqual(rank_one.score_terms(one)[1][0], 1.0)
        self.assertAlmostEqual(rank_two.score_terms(two)[1][0], 1.0)

    def test_penalty_is_zero_inside_and_monotonic_outside(self):
        tube = _tube(rank=1)
        points = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]]
        )
        tangent = tube.score_terms(points)[1]
        np.testing.assert_allclose(tangent, [0.0, 1.0, 4.0])

    def test_system_acceptance_is_reported_separately_from_source_patch(self):
        source = _tube(rank=1, penalty=64.0)
        masking = DimensionlessTube(
            center=np.asarray([8.0, 0.0, 0.0, 0.0]),
            basis=np.eye(4, 1),
            residual_scale=4.0,
            tangent_extents=np.asarray([20.0]),
            outer_scales=np.asarray([1.0]),
            penalty_weight=64.0,
            class_label=1,
        )
        probes = generate_safety_probes([source, masking], seed=56)
        evidence = probe_acceptance([source, masking], probes, threshold=10.0)
        self.assertIn("source_patch", evidence)
        self.assertIn("system", evidence)
        self.assertGreaterEqual(
            evidence["system"]["axis_tangent"],
            evidence["source_patch"]["axis_tangent"],
        )

    def test_pca_clustering_extents_and_probes_replay(self):
        rng = np.random.default_rng(57)
        geometry = rng.normal(size=(80, 6))
        calibration = rng.normal(size=(30, 6))
        first = fit_dimensionless_tube(
            geometry,
            calibration,
            rank=2,
            extent_quantile=0.95,
            outer_scale_policy="interquantile_range",
            penalty_weight=16.0,
            class_label=0,
        )
        second = fit_dimensionless_tube(
            geometry,
            calibration,
            rank=2,
            extent_quantile=0.95,
            outer_scale_policy="interquantile_range",
            penalty_weight=16.0,
            class_label=0,
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        np.testing.assert_array_equal(
            deterministic_patch_assignments(geometry, patch_count=2, seed=57),
            deterministic_patch_assignments(geometry, patch_count=2, seed=57),
        )
        first_probes = generate_safety_probes([first], seed=57)
        second_probes = generate_safety_probes([second], seed=57)
        for name in first_probes:
            np.testing.assert_array_equal(first_probes[name][0], second_probes[name][0])
            np.testing.assert_array_equal(first_probes[name][1], second_probes[name][1])

    def test_partition_overlap_and_invalid_patches_fail_closed(self):
        with self.assertRaises(ValueError):
            validate_disjoint_partitions({"geometry": ["x"], "calibration": ["x"]})
        with self.assertRaises(ValueError):
            fit_dimensionless_tube(
                np.zeros((3, 4)),
                np.zeros((2, 4)),
                rank=2,
                extent_quantile=0.95,
                outer_scale_policy="interquantile_range",
                penalty_weight=1.0,
                class_label=0,
            )
        with self.assertRaises(ValueError):
            deterministic_patch_assignments(
                np.zeros((2, 4)), patch_count=4, seed=1
            )

    def test_penalty_selection_uses_smallest_feasible_grid_value(self):
        tube = _tube(rank=1, penalty=1.0)
        calibration = np.zeros((20, 4))
        selected = select_smallest_safety_penalty(
            [tube], calibration, penalty_grid=(1.0, 4.0, 16.0, 64.0)
        )
        self.assertIn(selected["selected_penalty"], (1.0, 4.0, 16.0, 64.0))
        self.assertTrue(selected["attempts"][-1]["feasible"])
        self.assertEqual(
            selected["selected_penalty"],
            min(
                attempt["penalty"]
                for attempt in selected["attempts"]
                if attempt["feasible"]
            ),
        )

    def test_infeasible_penalty_selection_preserves_attempt_evidence(self):
        masking = DimensionlessTube(
            center=np.asarray([8.0, 0.0, 0.0, 0.0]),
            basis=np.eye(4, 1),
            residual_scale=4.0,
            tangent_extents=np.asarray([20.0]),
            outer_scales=np.asarray([1.0]),
            penalty_weight=1.0,
            class_label=1,
        )
        with self.assertRaises(SafetyPenaltySelectionError) as context:
            select_smallest_safety_penalty(
                [_tube(rank=1, penalty=1.0), masking],
                np.zeros((20, 4)),
                penalty_grid=(1.0, 4.0),
            )
        self.assertEqual(len(context.exception.attempts), 2)
        self.assertTrue(
            all(not attempt["feasible"] for attempt in context.exception.attempts)
        )

    def test_versioned_records_round_trip_and_keep_labels_sealed(self):
        calibration = TubeCalibrationRecord(
            geometry_hash="a" * 64,
            representation_hash="b" * 64,
            partition_hash="c" * 64,
            rank=16,
            patch_count=1,
            extent_quantile=0.95,
            outer_scale_policy="interquantile_range",
            penalty_grid=(1.0, 4.0, 16.0),
            selected_penalty=4.0,
            known_coverage_target=0.92,
            calibrated_threshold=2.0,
            calibration_known_coverage=0.92,
            selected_before_development=True,
            final_labels_opened=False,
            replay_hash="d" * 64,
        )
        self.assertEqual(
            TubeCalibrationRecord.from_dict(calibration.to_dict()), calibration
        )
        pairs_i = tuple((name, 2) for name in (
            "axis_tangent", "corner_tangent", "normal", "mixed", "bridge",
            "cross_class_bridge", "random_direction"
        ))
        pairs_f = tuple((name, 0.0) for name, _ in pairs_i)
        safety = TubeSafetyEvidence(
            calibration_replay_hash="e" * 64,
            probe_generator_hash="f" * 64,
            probe_counts=pairs_i,
            source_patch_acceptance=pairs_f,
            system_acceptance=pairs_f,
            tangent_acceptance_by_multiplier=(
                ("0.5", 1.0), ("1", 1.0), ("2", 0.0), ("4", 0.0), ("8", 0.0)
            ),
            parameter_count=100,
            fit_work_units=200,
            latency_seconds=0.01,
            peak_temporary_bytes=4096,
            exact_replay=True,
        )
        self.assertEqual(TubeSafetyEvidence.from_dict(safety.to_dict()), safety)


if __name__ == "__main__":
    unittest.main()
