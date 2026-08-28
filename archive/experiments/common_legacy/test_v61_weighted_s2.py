from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from experiments.tier4.eval_v61_weighted_s2 import (
    DEFAULT_CONFIG,
    _evaluate_gate,
    _paired_intervals,
    _validate_config,
)


def _seed_result(
    *,
    replay: bool = True,
    rollback: bool = True,
    preservation: float = 1.0,
    resources: bool = True,
) -> dict:
    return {
        "exact_replay": replay,
        "resource_passed": resources,
        "lifecycle": {
            "exact_json_rollback": rollback,
            "rollback_restored_predictions": rollback,
            "unaffected_prediction_preservation": preservation,
        },
    }


class WeightedS2ConfigTests(unittest.TestCase):
    def test_registered_config_is_strict_and_test_sealed(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        _validate_config(config)
        config["test_labels_opened"] = True
        with self.assertRaises(ValueError):
            _validate_config(config)

    def test_seed_mechanism_and_budget_drift_fail_closed(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["seeds"] = [11, 23]
        with self.assertRaises(ValueError):
            _validate_config(config)
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["retained_mechanisms"].append("tangent_cap_rank32")
        with self.assertRaises(ValueError):
            _validate_config(config)
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["budget"]["component_count"] = 100
        with self.assertRaises(ValueError):
            _validate_config(config)

    def test_readout_and_gate_drift_fail_closed(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["readout"]["regularization"] = 0.001
        with self.assertRaises(ValueError):
            _validate_config(config)
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["parity_gate"]["maximum_same_space_gap"] = 0.03
        with self.assertRaises(ValueError):
            _validate_config(config)


class WeightedS2GateTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.results = [_seed_result() for _ in range(3)]

    def _evaluate(self, **overrides):
        arguments = {
            "weighted_mean": 0.95,
            "strongest_same_space_mean": 0.97,
            "m31_mean": 0.91,
            "seed_results": self.results,
            "gate": self.config["parity_gate"],
        }
        arguments.update(overrides)
        return _evaluate_gate(**arguments)

    def test_exact_parity_boundary_passes(self):
        result = self._evaluate()
        self.assertTrue(result["passed"])
        self.assertTrue(all(value for key, value in result.items() if key != "passed"))

    def test_same_space_gap_and_m31_regression_fail(self):
        self.assertFalse(self._evaluate(weighted_mean=0.949)["passed"])
        self.assertFalse(
            self._evaluate(
                weighted_mean=0.90,
                strongest_same_space_mean=0.91,
                m31_mean=0.903,
            )["m31_non_regression"]
        )

    def test_every_seed_replay_rollback_locality_and_resources_are_required(self):
        mutations = (
            {"replay": False},
            {"rollback": False},
            {"preservation": 0.998},
            {"resources": False},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                results = copy.deepcopy(self.results)
                results[1] = _seed_result(**mutation)
                self.assertFalse(self._evaluate(seed_results=results)["passed"])

    def test_intervals_resolve_nested_same_space_controls(self):
        results = []
        for seed in range(3):
            labels = np.array([0, 1, 0, 1])
            weighted = np.array([0, 1, 0, seed % 2])
            rbf = np.array([0, 1, 0, 1])
            results.append(
                {
                    "weighted": {
                        "development": {"balanced_accuracy": 0.75 + seed * 0.01}
                    },
                    "controls": {"rbf": {"balanced_accuracy": 1.0}},
                    "development_labels": labels,
                    "predictions": {"weighted": weighted, "rbf": rbf},
                }
            )
        intervals = _paired_intervals(
            results,
            "weighted",
            "rbf",
            {
                "confidence": 0.95,
                "bootstrap_resamples": 10,
                "bootstrap_seed": 61,
            },
        )
        self.assertEqual(intervals["seed_paired_t"]["seed_count"], 3)
        self.assertEqual(
            intervals["pooled_per_example_bootstrap"]["n_resamples"], 10
        )


if __name__ == "__main__":
    unittest.main()
