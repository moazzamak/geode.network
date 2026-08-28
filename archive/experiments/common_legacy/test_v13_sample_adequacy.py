"""Focused tests for the v13 M78 sample-adequacy forensics."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from experiments.common.v13_sample_adequacy import (
    basis_stability,
    fit_shared_projection,
    mean_principal_angle_degrees,
    random_subspace_angle_degrees,
)
from experiments.tier4.eval_v13_m78_sample_adequacy import (
    _aggregate,
    _cell_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "logs" / "results" / "v13"
EVIDENCE = RESULTS / "m78_sample_adequacy" / "evidence.json"
VOID_EVIDENCE = RESULTS / "m78_sample_adequacy_void_r1" / "evidence.json"
CONFIG = REPO_ROOT / "experiments" / "configs" / "v13" / "m78_sample_adequacy.json"


def _synthetic_corpus(
    *, rank: int, per_class: int, dimension: int, noise: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    features = []
    labels = []
    for class_label in range(4):
        basis = np.linalg.qr(generator.normal(size=(dimension, rank)))[0]
        coefficients = generator.normal(scale=10.0, size=(per_class, rank))
        block = coefficients @ basis.T
        block = block + generator.normal(scale=noise, size=(per_class, dimension))
        features.append(block + generator.normal(scale=6.0, size=dimension))
        labels.append(np.full(per_class, class_label))
    return np.vstack(features), np.concatenate(labels)


class CellConfigTests(unittest.TestCase):
    def test_cell_config_does_not_mutate_the_base(self) -> None:
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        base["domainnet_transfer"]["geometry_per_class"] = 60
        cell = _cell_config(base, geometry_per_class=20, rank=4)
        self.assertEqual(cell["rank"], 4)
        self.assertEqual(cell["domainnet_transfer"]["geometry_per_class"], 20)
        self.assertEqual(base["domainnet_transfer"]["geometry_per_class"], 60)

    def test_registered_grid_respects_the_available_sample_ceiling(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        transfer = config["domainnet_transfer"]
        budget = (
            100
            - transfer["calibration_per_class"]
            - transfer["evaluation_per_class"]
        )
        self.assertEqual(budget, 60)
        self.assertEqual(config["axis_a_rank_sensitivity"]["geometry_per_class"], 60)
        self.assertTrue(
            all(
                count <= budget
                for count in config["axis_b_sample_sensitivity"]["geometry_per_class"]
            )
        )


class PrincipalAngleTests(unittest.TestCase):
    def test_identical_bases_have_zero_angle(self) -> None:
        generator = np.random.default_rng(11)
        basis = np.linalg.qr(generator.normal(size=(20, 5)))[0]
        # arccos loses half of the available precision near unit cosine, so a
        # numerically exact match still lands around 1e-7 degrees.
        self.assertLess(mean_principal_angle_degrees(basis, basis), 1e-5)

    def test_rotation_within_the_span_has_zero_angle(self) -> None:
        generator = np.random.default_rng(12)
        basis = np.linalg.qr(generator.normal(size=(20, 5)))[0]
        rotation = np.linalg.qr(generator.normal(size=(5, 5)))[0]
        self.assertLess(
            mean_principal_angle_degrees(basis, basis @ rotation), 1e-5
        )

    def test_random_reference_shrinks_as_rank_fills_the_space(self) -> None:
        narrow = random_subspace_angle_degrees(
            dimension=64, rank=2, trials=32, seed=3
        )
        wide = random_subspace_angle_degrees(
            dimension=64, rank=32, trials=32, seed=3
        )
        self.assertGreater(narrow, 45.0)
        self.assertLess(wide, narrow)
        self.assertLess(wide, 90.0)


class SharedProjectionTests(unittest.TestCase):
    def test_projection_rows_are_orthonormal_and_sign_pinned(self) -> None:
        features, _ = _synthetic_corpus(
            rank=3, per_class=200, dimension=24, noise=0.1, seed=99
        )
        _, projection = fit_shared_projection(features, output_dimension=8)
        np.testing.assert_allclose(projection @ projection.T, np.eye(8), atol=1e-9)
        for row in projection:
            self.assertGreater(row[int(np.argmax(np.abs(row)))], 0.0)


class BasisStabilityTests(unittest.TestCase):
    def test_identified_subspace_is_recovered_from_both_halves(self) -> None:
        features, labels = _synthetic_corpus(
            rank=2, per_class=400, dimension=16, noise=0.05, seed=7801
        )
        report = basis_stability(
            features, labels, output_dimension=8, rank=2, random_trials=32
        )
        self.assertTrue(report["shared_projection"])
        self.assertLess(report["mean_principal_angle_degrees"], 10.0)
        self.assertGreater(report["identifiability"], 0.8)

    def test_unidentified_subspace_approaches_the_random_reference(self) -> None:
        generator = np.random.default_rng(7802)
        features = generator.normal(size=(80, 64))
        labels = np.repeat(np.arange(4), 20)
        report = basis_stability(
            features, labels, output_dimension=32, rank=8, random_trials=32
        )
        self.assertLess(report["identifiability"], 0.25)
        self.assertGreater(
            report["mean_principal_angle_degrees"],
            0.75 * report["random_subspace_angle_degrees"],
        )

    def test_effective_rank_is_reported_when_the_guard_clips(self) -> None:
        features, labels = _synthetic_corpus(
            rank=2, per_class=12, dimension=32, noise=0.1, seed=44
        )
        report = basis_stability(
            features, labels, output_dimension=16, rank=32, random_trials=8
        )
        self.assertEqual(report["requested_rank"], 32)
        self.assertLess(report["effective_rank"], 32)
        self.assertEqual(report["half_sample_count_per_class"], 6)


class AggregateTests(unittest.TestCase):
    def test_aggregate_groups_by_cell_and_averages_seeds(self) -> None:
        def cell(accuracy: float, angle: float) -> dict:
            return {
                "geometry_per_class": 60,
                "requested_rank": 8,
                "fitted_rank": 8,
                "samples_per_fitted_dimension": 7.5,
                "known_balanced_accuracy": accuracy,
                "unknown_recall": 0.01,
                "logistic_known_balanced_accuracy": 0.74,
                "logistic_unknown_recall": 0.20,
                "subspace_stability": {
                    "mean_principal_angle_degrees": angle,
                    "random_subspace_angle_degrees": 70.0,
                    "identifiability": 1.0 - angle / 70.0,
                    "effective_rank": 8,
                },
            }

        summary = _aggregate([cell(0.70, 60.0), cell(0.72, 64.0)])
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["seed_count"], 2)
        self.assertAlmostEqual(summary[0]["mean_known_balanced_accuracy"], 0.71)
        self.assertAlmostEqual(
            summary[0]["mean_subspace_principal_angle_degrees"], 62.0
        )
        self.assertAlmostEqual(summary[0]["mean_identifiability"], 1.0 - 62.0 / 70.0)


class VoidRunTests(unittest.TestCase):
    @unittest.skipUnless(VOID_EVIDENCE.exists(), "void R1 evidence is absent")
    def test_void_run_is_retained_for_the_record(self) -> None:
        void = json.loads(VOID_EVIDENCE.read_text(encoding="utf-8"))
        self.assertEqual(void["milestone"], "M78")
        self.assertEqual(void["schema_version"], 1)


@unittest.skipUnless(EVIDENCE.exists(), "M78 evidence has not been generated")
class M78EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_corpus_ceiling_is_recorded(self) -> None:
        corpus = self.evidence["corpus"]
        self.assertEqual(corpus["observations_per_class"], 100)
        self.assertEqual(
            self.evidence["gate"]["maximum_available_geometry_per_class"], 60
        )
        self.assertEqual(self.evidence["registration_amendments"], ["R1", "R2"])

    def test_stability_uses_the_shared_projection(self) -> None:
        for cell in self.evidence["cells"]:
            stability = cell["subspace_stability"]
            self.assertTrue(stability["shared_projection"])
            self.assertIn("random_subspace_angle_degrees", stability)

    def test_every_cell_reports_its_sample_adequacy_ratio(self) -> None:
        for cell in self.evidence["cells"]:
            self.assertGreater(cell["samples_per_fitted_dimension"], 0.0)

    def test_the_void_run_is_explicitly_superseded(self) -> None:
        self.assertEqual(
            self.evidence["supersedes"],
            "logs/results/v13/m78_sample_adequacy_void_r1",
        )
        self.assertFalse(self.evidence["gate"]["final_labels_opened"])


if __name__ == "__main__":
    unittest.main()
