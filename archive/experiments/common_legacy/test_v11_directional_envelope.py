from __future__ import annotations

import unittest

import numpy as np

from experiments.common.v11_directional_envelope import (
    CONTRAST_MARGINS,
    PROBE_FAMILIES,
    DirectionalTube,
    calibrate_class_thresholds,
    class_score_matrix,
    composite_endpoint_records,
    contrast_acceptance,
    deterministic_directional_patch_assignments,
    directional_replay_hash,
    estimate_directional_rank,
    fit_delegated_rbf_head,
    fit_directional_tube,
    generate_geodesic_probes,
    negative_guided_extents,
    spherical_exp_map,
    spherical_log_map,
    split_conformal_quantile,
    verify_delegated_head_lineage,
)
from experiments.common.v9_surface_support import validate_disjoint_partitions
from src.runtime.schemas import (
    ConformalCalibrationRecord,
    ContrastAcceptanceRecord,
    DirectionalGeometryRecord,
)


def _directions(
    rng: np.random.Generator,
    *,
    count: int,
    dimension: int = 8,
    rank: int = 2,
) -> np.ndarray:
    center = np.eye(dimension)[0]
    vectors = np.zeros((count, dimension), dtype=np.float64)
    vectors[:, 1 : rank + 1] = rng.normal(scale=0.08, size=(count, rank))
    vectors[:, rank + 1 :] = rng.normal(
        scale=0.002, size=(count, dimension - rank - 1)
    )
    return spherical_exp_map(center, vectors)


def _tube(
    *, class_label: int = 0, center_axis: int = 0, dimension: int = 6
) -> DirectionalTube:
    center = np.eye(dimension)[center_axis]
    basis_axis = 1 if center_axis != 1 else 0
    return DirectionalTube(
        center=center,
        basis=np.eye(dimension)[:, [basis_axis]],
        residual_scale=0.01,
        tangent_extents=np.asarray([0.2]),
        outer_scales=np.asarray([0.1]),
        penalty_weight=16.0,
        class_label=class_label,
    )


class V11DirectionalEnvelopeTests(unittest.TestCase):
    def test_common_rescaling_preserves_directional_scores(self):
        rng = np.random.default_rng(63)
        geometry = _directions(rng, count=80)
        calibration = _directions(rng, count=40)
        query = _directions(rng, count=20)
        first = fit_directional_tube(
            geometry,
            calibration,
            rank=2,
            extent_policy="quantile",
            extent_quantile=0.95,
            penalty_weight=16.0,
            class_label=0,
        )
        second = fit_directional_tube(
            11.0 * geometry,
            11.0 * calibration,
            rank=2,
            extent_policy="quantile",
            extent_quantile=0.95,
            penalty_weight=16.0,
            class_label=0,
        )
        np.testing.assert_allclose(first.score(query), second.score(11.0 * query))

    def test_split_conformal_uses_registered_finite_sample_rank_and_stable_ties(self):
        rng = np.random.default_rng(64)
        scores = rng.uniform(size=49)
        threshold = split_conformal_quantile(scores, miscoverage=0.08)
        expected_count = int(np.ceil((len(scores) + 1) * 0.92))
        self.assertEqual(int(np.sum(scores <= threshold)), expected_count)
        tied = np.asarray([0.0, 1.0, 1.0, 1.0])
        self.assertEqual(split_conformal_quantile(tied, miscoverage=0.4), 1.0)

    def test_contrast_rejects_near_tie_accepted_by_minimum_score_rule(self):
        scores = np.asarray([[0.92, 0.97]])
        decision = contrast_acceptance(
            scores,
            np.ones(2),
            np.asarray([0, 1]),
            margin=0.1,
        )
        self.assertTrue(scores.min() <= 1.0)
        self.assertFalse(bool(decision["accepted"][0]))

    def test_negative_guided_extents_stay_between_floor_and_policy_one(self):
        values = np.linspace(0.1, 0.6, 18)
        own = np.column_stack(
            [
                np.concatenate([values, [0.8, 1.0]]),
                np.concatenate([values, [0.8, 1.0]]),
            ]
        )
        negatives = np.asarray([[0.85, 0.85], [2.0, 2.0]])
        extents = negative_guided_extents(
            own, negatives, upper_quantile=0.95
        )
        floor = np.quantile(np.abs(own), 0.90, axis=0, method="higher")
        upper = np.quantile(np.abs(own), 0.95, axis=0, method="higher")
        self.assertTrue(np.all(extents >= floor))
        self.assertTrue(np.all(extents <= upper))
        self.assertFalse(np.any(np.all(np.abs(negatives) <= extents, axis=1)))

    def test_tangent_penalty_is_zero_inside_and_monotonic_outside(self):
        tube = _tube()
        vectors = np.zeros((3, 6))
        vectors[:, 1] = [0.1, 0.3, 0.4]
        points = spherical_exp_map(tube.center, vectors)
        tangent = tube.score_terms(points)[1]
        np.testing.assert_allclose(tangent, [0.0, 1.0, 4.0], atol=1e-12)

    def test_fit_probes_geodesics_and_replay_are_deterministic(self):
        rng = np.random.default_rng(65)
        geometry = _directions(rng, count=80)
        calibration = _directions(rng, count=40)
        first = fit_directional_tube(
            geometry,
            calibration,
            rank=2,
            extent_policy="quantile",
            extent_quantile=0.95,
            penalty_weight=16.0,
            class_label=0,
        )
        second = fit_directional_tube(
            geometry,
            calibration,
            rank=2,
            extent_policy="quantile",
            extent_quantile=0.95,
            penalty_weight=16.0,
            class_label=0,
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        competing = _tube(class_label=1, center_axis=2, dimension=8)
        first_probes = generate_geodesic_probes([first, competing], seed=65)
        second_probes = generate_geodesic_probes([second, competing], seed=65)
        for name in PROBE_FAMILIES:
            for left, right in zip(
                first_probes[name], second_probes[name], strict=True
            ):
                np.testing.assert_array_equal(left, right)
            points = first_probes[name][0]
            if len(points):
                np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0)
        log = spherical_log_map(first.center, geometry[:5])
        np.testing.assert_allclose(
            spherical_exp_map(first.center, log), geometry[:5], atol=1e-12
        )

    def test_directional_rank_and_patch_assignments_replay(self):
        rng = np.random.default_rng(64)
        directions = _directions(
            rng, count=160, dimension=16, rank=8
        )
        recovered = estimate_directional_rank(
            directions,
            rank_grid=(4, 8, 12),
            explained_variance_target=0.995,
        )
        self.assertEqual(recovered["selected_rank"], 8)
        first = deterministic_directional_patch_assignments(
            directions, patch_count=2, seed=64
        )
        second = deterministic_directional_patch_assignments(
            directions, patch_count=2, seed=64
        )
        np.testing.assert_array_equal(first, second)

    def test_conformal_class_scores_and_composite_endpoints_are_separate(self):
        tubes = [_tube(class_label=0, center_axis=0), _tube(class_label=1, center_axis=2)]
        points = np.vstack([tube.center for tube in tubes])
        scores, classes = class_score_matrix(tubes, points)
        thresholds = calibrate_class_thresholds(
            scores + 0.01,
            np.asarray([0, 1]),
            classes,
        )
        decision = contrast_acceptance(
            scores + 0.01, thresholds, classes, margin=0.0
        )
        records = composite_endpoint_records(
            decision,
            np.asarray([0, 0]),
            np.asarray([0, 1]),
        )
        self.assertEqual(records[0]["envelope_class"], 0)
        self.assertEqual(records[1]["envelope_class"], 1)
        self.assertTrue(records[0]["head_correct"])
        self.assertFalse(records[1]["head_correct"])
        self.assertEqual(len(records), 2)

    def test_lineage_partition_and_invalid_patches_fail_closed(self):
        metadata = {
            "family": "rbf_svm",
            "representation_hash": "a" * 64,
            "training_split_hash": "b" * 64,
            "predictions_sha256": "c" * 64,
        }
        verify_delegated_head_lineage(
            metadata,
            representation_hash="a" * 64,
            training_split_hash="b" * 64,
            predictions_hash="c" * 64,
        )
        with self.assertRaises(ValueError):
            verify_delegated_head_lineage(
                metadata,
                representation_hash="d" * 64,
                training_split_hash="b" * 64,
                predictions_hash="c" * 64,
            )
        with self.assertRaises(ValueError):
            validate_disjoint_partitions({"geometry_fit": ["x"], "unknown_eval": ["x"]})
        with self.assertRaises(ValueError):
            fit_directional_tube(
                np.ones((2, 8)),
                np.ones((2, 8)),
                rank=2,
                extent_policy="quantile",
                extent_quantile=0.95,
                penalty_weight=1.0,
                class_label=0,
            )

    def test_delegated_head_uses_only_known_fit_and_calibration_classes(self):
        rng = np.random.default_rng(63)
        geometry = np.vstack(
            [
                rng.normal(loc=-1.0, scale=0.1, size=(30, 4)),
                rng.normal(loc=1.0, scale=0.1, size=(30, 4)),
            ]
        )
        labels = np.repeat([0, 1], 30)
        calibration = np.vstack(
            [
                rng.normal(loc=-1.0, scale=0.1, size=(12, 4)),
                rng.normal(loc=1.0, scale=0.1, size=(12, 4)),
            ]
        )
        calibration_labels = np.repeat([0, 1], 12)
        result = fit_delegated_rbf_head(
            geometry,
            labels,
            calibration,
            calibration_labels,
            calibration,
            known_classes=(0, 1),
            seed=63,
        )
        self.assertEqual(result["fit_class_count"], 2)
        self.assertEqual(result["calibration_class_count"], 2)
        self.assertTrue(result["support_vectors_unchanged_by_calibration"])
        self.assertEqual(set(result["predictions"]), {0, 1})

    def test_directional_objects_and_runtime_schemas_round_trip_exactly(self):
        tube = _tube()
        self.assertEqual(DirectionalTube.from_dict(tube.to_dict()).to_dict(), tube.to_dict())
        replay_hash = directional_replay_hash(
            [tube], np.asarray([1.0]), miscoverage=0.08, contrast_margin=0.0
        )
        geometry = DirectionalGeometryRecord(
            geometry_hash="a" * 64,
            representation_hash="b" * 64,
            partition_hash="c" * 64,
            rank=8,
            patch_count=1,
            extent_policy="quantile",
            extent_quantile=0.95,
            parameter_count=100,
            fit_work_units=200,
            replay_hash=replay_hash,
        )
        conformal = ConformalCalibrationRecord(
            geometry_replay_hash=replay_hash,
            delegated_head_hash="d" * 64,
            miscoverage=0.08,
            class_counts=(("0", 49), ("1", 49)),
            class_thresholds=(("0", 1.0), ("1", 1.1)),
            selected_before_development=True,
            final_labels_opened=False,
            replay_hash="e" * 64,
        )
        zero_probabilities = tuple((name, 0.0) for name in PROBE_FAMILIES)
        contrast = ContrastAcceptanceRecord(
            calibration_replay_hash="e" * 64,
            margin_grid=CONTRAST_MARGINS,
            selected_margin=0.0,
            probe_counts=tuple((name, 1) for name in PROBE_FAMILIES),
            source_patch_acceptance=zero_probabilities,
            system_acceptance=zero_probabilities,
            endpoint_count=2,
            latency_seconds=0.0,
            peak_temporary_bytes=1024,
            exact_replay=True,
        )
        self.assertEqual(
            DirectionalGeometryRecord.from_dict(geometry.to_dict()), geometry
        )
        self.assertEqual(
            ConformalCalibrationRecord.from_dict(conformal.to_dict()), conformal
        )
        self.assertEqual(
            ContrastAcceptanceRecord.from_dict(contrast.to_dict()), contrast
        )


if __name__ == "__main__":
    unittest.main()
