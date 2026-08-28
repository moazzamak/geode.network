from __future__ import annotations

import unittest

from experiments.common.v61_lifecycle_frontier import (
    classify_outcome_c,
    complete_frontier_point,
    dominates,
    non_dominated_models,
)
from experiments.tier4.eval_v61_lifecycle_frontier import (
    DEFAULT_CONFIG,
    _validate_config,
)


def _point(accuracy: float, latency: float = 1.0) -> dict:
    return {
        "balanced_accuracy": accuracy,
        "unaffected_prediction_preservation": 1.0,
        "rollback_reliability": 1.0,
        "accepted_edit_evidence_count": 10,
        "edit_latency_seconds": latency,
        "inference_latency_seconds": latency,
    }


class LifecycleFrontierTests(unittest.TestCase):
    def test_registered_config_is_strict_and_test_sealed(self):
        import json

        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        _validate_config(config)
        config["test_labels_opened"] = True
        with self.assertRaises(ValueError):
            _validate_config(config)

    def test_task_or_edit_contract_drift_fails_closed(self):
        import json

        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["tasks"].pop()
        with self.assertRaises(ValueError):
            _validate_config(config)
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["explicit_edit_contract"][
            "bounded_shift_temperature_scale"
        ] = 1.02
        with self.assertRaises(ValueError):
            _validate_config(config)

    def test_dominance_requires_complete_points(self):
        complete = _point(0.9)
        incomplete = {**complete, "edit_latency_seconds": None}
        self.assertTrue(complete_frontier_point(complete))
        self.assertFalse(complete_frontier_point(incomplete))
        with self.assertRaises(ValueError):
            dominates(complete, incomplete)

    def test_non_dominated_frontier_ignores_unsupported_costs(self):
        records = {
            "explicit": _point(0.9, 0.5),
            "slow": _point(0.9, 1.0),
            "unsupported": {**_point(0.99), "edit_latency_seconds": None},
        }
        self.assertEqual(non_dominated_models(records), ["explicit"])

    def test_unsupported_accuracy_superior_control_blocks_tradeoff_claim(self):
        records = {
            "explicit": _point(0.92),
            "rbf": {**_point(0.97), "edit_latency_seconds": None},
        }
        result = classify_outcome_c(
            records,
            retained_model="explicit",
            exact_rollback_every_seed_and_task=True,
            locality_contract_passed=True,
            predictive_deficit_reported=True,
            paired_advantage_controls=[],
        )
        self.assertFalse(result["specialized_tradeoff_claim_passed"])
        self.assertEqual(result["status"], "lifecycle_safety_qualification_only")
        self.assertEqual(result["unsupported_accuracy_superior_controls"], ["rbf"])

    def test_complete_advantage_can_pass(self):
        records = {
            "explicit": _point(0.92, 0.5),
            "rbf": {
                **_point(0.97, 1.0),
                "unaffected_prediction_preservation": 0.9,
            },
        }
        result = classify_outcome_c(
            records,
            retained_model="explicit",
            exact_rollback_every_seed_and_task=True,
            locality_contract_passed=True,
            predictive_deficit_reported=True,
            paired_advantage_controls=["rbf"],
        )
        self.assertTrue(result["specialized_tradeoff_claim_passed"])

    def test_failed_locality_narrows_status_to_rollback_only(self):
        records = {"explicit": _point(0.92)}
        result = classify_outcome_c(
            records,
            retained_model="explicit",
            exact_rollback_every_seed_and_task=True,
            locality_contract_passed=False,
            predictive_deficit_reported=True,
            paired_advantage_controls=[],
        )
        self.assertFalse(result["specialized_tradeoff_claim_passed"])
        self.assertEqual(result["status"], "rollback_qualification_only")


if __name__ == "__main__":
    unittest.main()
