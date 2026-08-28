from __future__ import annotations

import unittest

import numpy as np

from experiments.common.v7_acceptance import evaluate_acceptance_heads


class AcceptanceBakeoffTests(unittest.TestCase):
    def test_bakeoff_is_deterministic_and_rejects_shifted_unknowns(self) -> None:
        rng = np.random.default_rng(7)
        train_features = np.concatenate(
            [rng.normal(label * 4.0, 0.3, size=(80, 12)) for label in range(4)]
        )
        train_labels = np.repeat(np.arange(4), 80)
        evaluation_features = np.concatenate(
            [rng.normal(label * 4.0, 0.3, size=(20, 12)) for label in range(4)]
        )
        evaluation_labels = np.repeat(np.arange(4), 20)
        config = {
            "proxy_unknown_classes": [3],
            "calibration_fraction": 0.2,
            "calibration_known_coverage_target": 0.92,
            "corruption_noise_scale_fraction": 0.01,
            "review_budget_per_1000": 50,
            "knn_support_per_class": 32,
            "evm_support_per_class": 32,
            "gaussian_rank": 4,
            "sdf_rank": 3,
            "sdf_components_per_class": 2,
            "rbf_svm": {"C": 1.0, "gamma": "scale"},
        }
        result = evaluate_acceptance_heads(
            train_features,
            train_labels,
            evaluation_features,
            evaluation_labels,
            config,
            seed=11,
        )
        self.assertEqual(
            set(result["heads"]),
            {
                "maximum_posterior",
                "knn_support",
                "low_rank_gaussian",
                "evm_style_weibull_margin",
                "weighted_affine_sdf",
                "rbf_svm_evidence",
            },
        )
        self.assertTrue(
            all(value["exact_replay"] for value in result["heads"].values())
        )
        self.assertEqual(result["heads"]["knn_support"]["unknown_recall"], 1.0)
