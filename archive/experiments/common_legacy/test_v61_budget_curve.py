from __future__ import annotations

import unittest

import numpy as np

from experiments.common.v61_budget_curve import (
    REGISTERED_COMPONENT_COUNTS,
    classify_capacity_curve,
    marginal_accuracy_per_ten,
    probability_margin_error,
)
from experiments.tier4.eval_v61_budget_curve_s1 import (
    DEFAULT_CONFIG,
    _validate_config,
)


def _rows(accuracies: list[float], nlls: list[float] | None = None):
    if nlls is None:
        nlls = [0.4] * len(accuracies)
    return [
        {
            "component_count": count,
            "development_balanced_accuracy": accuracy,
            "development_nll": nll,
        }
        for count, accuracy, nll in zip(
            REGISTERED_COMPONENT_COUNTS, accuracies, nlls
        )
    ]


class BudgetCurveTests(unittest.TestCase):
    def test_registered_config_is_diagnostic_and_budget_immutable(self):
        import json

        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        _validate_config(config)
        config["a2_component_count_immutable"] = 120
        with self.assertRaises(ValueError):
            _validate_config(config)

    def test_count_or_test_policy_drift_fails_closed(self):
        import json

        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["selection"]["exact_component_counts"][-1] = 121
        with self.assertRaises(ValueError):
            _validate_config(config)
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["test_labels_opened"] = True
        with self.assertRaises(ValueError):
            _validate_config(config)

    def test_marginal_accuracy_normalizes_unequal_count_intervals(self):
        rows = _rows([0.80, 0.81, 0.82, 0.836, 0.85, 0.87, 0.89, 0.90])
        marginal = marginal_accuracy_per_ten(rows)
        self.assertIsNone(marginal[0]["marginal_accuracy_per_ten"])
        self.assertAlmostEqual(
            float(marginal[3]["marginal_accuracy_per_ten"]), 0.01
        )
        self.assertAlmostEqual(
            float(marginal[-1]["marginal_accuracy_per_ten"]), 0.005
        )

    def test_budget_limited_classification_uses_80_to_120_slope(self):
        result = classify_capacity_curve(
            _rows([0.80, 0.82, 0.84, 0.86, 0.87, 0.88, 0.884, 0.888]),
            minimum_high_budget_slope_per_ten=0.001,
            material_accuracy_reversal=0.005,
            material_nll_reversal_fraction=0.05,
        )
        self.assertEqual(result["classification"], "budget-limited")
        self.assertAlmostEqual(result["endpoint_slope_per_ten"], 0.002)

    def test_saturated_classification_requires_no_material_reversal(self):
        result = classify_capacity_curve(
            _rows([0.80, 0.82, 0.84, 0.86, 0.87, 0.88, 0.881, 0.882]),
            minimum_high_budget_slope_per_ten=0.001,
            material_accuracy_reversal=0.005,
            material_nll_reversal_fraction=0.05,
        )
        self.assertEqual(result["classification"], "saturated")

    def test_accuracy_or_nll_reversal_is_unstable(self):
        accuracy_result = classify_capacity_curve(
            _rows([0.80, 0.82, 0.84, 0.86, 0.87, 0.90, 0.89, 0.91]),
            minimum_high_budget_slope_per_ten=0.001,
            material_accuracy_reversal=0.005,
            material_nll_reversal_fraction=0.05,
        )
        self.assertEqual(accuracy_result["classification"], "unstable")
        nll_result = classify_capacity_curve(
            _rows(
                [0.80, 0.82, 0.84, 0.86, 0.87, 0.90, 0.901, 0.902],
                [0.4, 0.38, 0.36, 0.34, 0.33, 0.30, 0.32, 0.29],
            ),
            minimum_high_budget_slope_per_ten=0.001,
            material_accuracy_reversal=0.005,
            material_nll_reversal_fraction=0.05,
        )
        self.assertEqual(nll_result["classification"], "unstable")

    def test_unregistered_counts_fail_closed(self):
        rows = _rows([0.8] * 8)
        rows[-1]["component_count"] = 121
        with self.assertRaises(ValueError):
            marginal_accuracy_per_ten(rows)
        with self.assertRaises(ValueError):
            classify_capacity_curve(
                rows,
                minimum_high_budget_slope_per_ten=0.001,
                material_accuracy_reversal=0.005,
                material_nll_reversal_fraction=0.05,
            )

    def test_probability_margin_error_matches_definition(self):
        student = np.array([[0.7, 0.3], [0.4, 0.6]])
        teacher = np.array([[0.8, 0.2], [0.45, 0.55]])
        self.assertAlmostEqual(
            probability_margin_error(student, teacher), 0.15
        )


if __name__ == "__main__":
    unittest.main()
