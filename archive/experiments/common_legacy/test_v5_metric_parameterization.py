import copy
import json
import unittest

import numpy as np

from experiments.tier1.eval_v5_metric_support_sweep import (
    DEFAULT_CONFIG,
    _evaluate_policy,
    _select_policy,
    run_sweep,
)
from experiments.tier4.eval_v5_metric_policy import evaluate_frozen_metric_policy
from src.greedy_constructor import GreedyConstructor
from src.metric_parameterization import (
    METRIC_FAMILIES,
    PrecisionMetric,
    fit_class_precision_metrics,
    fit_precision_metric,
)
from src.sdf_engine import EllipsoidExpert


def _points(seed: int = 11, count: int = 80, dimension: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    covariance = (basis * np.linspace(2.0, 0.2, dimension)) @ basis.T
    return rng.multivariate_normal(np.zeros(dimension), covariance, size=count)


class PrecisionMetricTests(unittest.TestCase):
    def test_all_families_are_positive_definite_and_replay_exactly(self):
        class_points = {0: _points(), 1: _points(seed=23) + 1.0}
        for family in METRIC_FAMILIES:
            with self.subTest(family=family):
                fits = fit_class_precision_metrics(class_points, family, rank=2)
                for fit in fits.values():
                    eigenvalues = np.linalg.eigvalsh(fit.metric.dense_precision())
                    self.assertGreater(eigenvalues[0], 0.0)
                    restored = PrecisionMetric.from_dict(fit.metric.to_dict())
                    np.testing.assert_array_equal(
                        restored.dense_precision(),
                        fit.metric.dense_precision(),
                    )

    def test_factorized_and_dense_scores_match(self):
        rng = np.random.default_rng(37)
        deltas = rng.normal(size=(20, 6))
        for rank in (0, 2, 6):
            with self.subTest(rank=rank):
                metric = fit_precision_metric(
                    _points(), "diagonal_low_rank", rank=rank
                ).metric
                dense = np.einsum(
                    "ni,ij,nj->n", deltas, metric.dense_precision(), deltas
                )
                np.testing.assert_allclose(
                    metric.quadratic_form(deltas), dense, rtol=1e-12, atol=1e-12
                )

    def test_gradient_matches_finite_difference(self):
        metric = fit_precision_metric(
            _points(), "diagonal_low_rank", rank=3
        ).metric
        point = np.linspace(-0.5, 0.5, metric.dimension)
        analytic = metric.gradient(point[None, :])[0]
        epsilon = 1e-6
        numeric = np.empty(metric.dimension)
        for index in range(metric.dimension):
            offset = np.zeros(metric.dimension)
            offset[index] = epsilon
            numeric[index] = (
                metric.quadratic_form((point + offset)[None, :])[0]
                - metric.quadratic_form((point - offset)[None, :])[0]
            ) / (2.0 * epsilon)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-7)

    def test_rank_zero_and_full_rank_boundaries_are_stable(self):
        zero = fit_precision_metric(
            _points(), "diagonal_low_rank", rank=0
        ).metric
        full = fit_precision_metric(
            _points(), "diagonal_low_rank", rank=6
        ).metric
        oversized = fit_precision_metric(
            _points(), "diagonal_low_rank", rank=20
        )
        self.assertEqual(zero.rank, 0)
        self.assertGreater(np.ptp(zero.diagonal), 0.0)
        self.assertLessEqual(full.rank, 6)
        self.assertIn("rank_clipped_to_dimension", oversized.warnings)

    def test_covariance_floor_is_explicit(self):
        points = np.ones((8, 3))
        fit = fit_precision_metric(points, "full")
        self.assertIn("covariance_eigenvalues_floored", fit.warnings)
        self.assertTrue(np.all(np.isfinite(fit.metric.dense_precision())))

    def test_shared_fit_propagates_pooled_warnings(self):
        classes = {0: np.ones((8, 3)), 1: np.ones((8, 3)) * 2.0}
        fits = fit_class_precision_metrics(
            classes, "shared_low_rank_diagonal", rank=8
        )
        for fit in fits.values():
            self.assertIn("rank_clipped_to_dimension", fit.warnings)
            self.assertIn("covariance_eigenvalues_floored", fit.warnings)

    def test_ellipsoid_conversion_preserves_quadratic_score(self):
        fit = fit_precision_metric(_points(), "diagonal_low_rank", rank=2)
        ellipsoid = EllipsoidExpert.from_precision_metric(fit.center, fit.metric)
        query = _points(seed=53, count=10)
        expected = np.sqrt(fit.metric.quadratic_form(query - fit.center)) - 1.0
        np.testing.assert_allclose(ellipsoid.compute_sdf(query), expected)


class MetricConstructorTests(unittest.TestCase):
    def test_metric_constructor_is_opt_in_and_sphere_is_default(self):
        default = GreedyConstructor()
        metric = GreedyConstructor(metric_family="diagonal_low_rank", metric_rank=2)
        self.assertIsNone(default.metric_family)
        self.assertEqual(default.primitive_family, "sphere")
        self.assertEqual(default._minimal_seed_size(4), 6)
        self.assertEqual(metric._minimal_seed_size(4), 9)
        candidate = metric._generate_candidate(_points(dimension=4, count=9))
        self.assertTrue(np.all(candidate.radii > 0.0))
        self.assertTrue(np.all(np.isfinite(candidate.orientation)))

    def test_invalid_metric_configuration_fails_closed(self):
        with self.assertRaises(ValueError):
            GreedyConstructor(metric_family="unknown")
        with self.assertRaises(ValueError):
            GreedyConstructor(metric_family="full", metric_rank=-1)


class MetricPolicyTests(unittest.TestCase):
    def _records(self):
        records = []
        for cell, samples in (("low", 8), ("high", 80)):
            for candidate, nll, parameters in (
                ("spherical", 0.8 if cell == "low" else 0.9, 2),
                ("full", 1.0 if cell == "low" else 0.5, 20),
            ):
                records.append(
                    {
                        "cell_id": cell,
                        "candidate": candidate,
                        "samples_per_class": samples,
                        "dimension": 4,
                        "intrinsic_rank": 1,
                        "negative_log_likelihood": nll,
                        "balanced_accuracy": 0.8,
                        "parameter_count": parameters,
                        "parameter_bytes": parameters * 8,
                        "serialized_bytes": parameters * 8,
                        "fit_work_units": parameters * 100,
                    }
                )
        return records

    def test_policy_selection_is_deterministic(self):
        records = self._records()
        bins = {
            "sample_dimension_edges": [3.0, 8.0],
            "rank_fraction_edge": 0.25,
        }
        first = _select_policy(records, ["spherical", "full"], bins, 0.001)
        second = _select_policy(records, ["spherical", "full"], bins, 0.001)
        self.assertEqual(first, second)
        self.assertEqual(first["support_bin_selections"]["low:low_rank"], "spherical")
        self.assertEqual(first["support_bin_selections"]["high:low_rank"], "full")

    def test_unseen_bin_fallback_emits_warning(self):
        records = self._records()[:2]
        policy = {
            "global_candidate": "spherical",
            "support_bin_selections": {},
            "support_bins": {
                "sample_dimension_edges": [3.0, 8.0],
                "rank_fraction_edge": 0.25,
            },
        }
        evaluation = _evaluate_policy(records, policy)
        self.assertEqual(
            evaluation["fallback_warnings"][0]["warning"],
            "unseen_support_bin_used_global_candidate",
        )

    def test_bounded_sweep_freezes_policy_from_development_only(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        for split in ("development", "test"):
            for key in config[split]:
                config[split][key] = [config[split][key][0]]
        config["candidate_ranks"] = [0]
        first = run_sweep(copy.deepcopy(config))
        second = run_sweep(copy.deepcopy(config))
        self.assertEqual(first, second)
        self.assertEqual(first["policy"]["selected_from"], "development_only")
        self.assertEqual(len(first["development_records"]), 18)
        self.assertEqual(len(first["test_records"]), 18)

    def test_tier4_policy_requires_disjoint_splits(self):
        features = np.vstack([_points(count=10), _points(seed=23, count=10) + 1.0])
        labels = np.repeat([0, 1], 10)
        policy = {
            "global_candidate": "diagonal",
            "support_bin_selections": {},
            "support_bins": {
                "sample_dimension_edges": [3.0, 8.0],
                "rank_fraction_edge": 0.25,
            },
        }
        with self.assertRaises(ValueError):
            evaluate_frozen_metric_policy(
                features,
                labels,
                np.arange(12),
                np.arange(10, 20),
                policy,
                intrinsic_rank=2,
            )
        with self.assertRaises(ValueError):
            evaluate_frozen_metric_policy(
                features,
                labels,
                np.array([-1, -2, -3, -4]),
                np.arange(16, 20),
                policy,
                intrinsic_rank=2,
            )


if __name__ == "__main__":
    unittest.main()
