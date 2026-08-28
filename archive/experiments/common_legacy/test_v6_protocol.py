from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.common.v5_artifacts import payload_hash
from experiments.common.v6_protocol import (
    BudgetSpec,
    PrimitiveMetadata,
    TeacherLineage,
    enumerate_budget_table,
    require_teacher_compatibility,
    select_boundary_cohort,
    validate_baseline_locks,
    validate_prediction_baseline,
    validate_v6_protocol_config,
)
from experiments.tier1.eval_v6_protocol_s0 import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    run_s0,
    verify_s0,
)


class V6ProtocolTests(unittest.TestCase):
    def _config(self) -> dict:
        return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def _teacher(self, **updates: object) -> TeacherLineage:
        values = {
            "family": "rbf_svm",
            "representation_hash": payload_hash({"representation": 1}),
            "training_split_hash": payload_hash({"split": "train"}),
            "development_split_hash": payload_hash({"split": "development"}),
            "checkpoint_hash": payload_hash({"teacher": 1}),
            "prediction_hash": payload_hash({"predictions": 1}),
            "selection_metric": "development_balanced_accuracy",
            "test_labels_used_for_selection": False,
        }
        values.update(updates)
        return TeacherLineage(**values)

    def test_config_and_frozen_baselines_validate(self):
        config = self._config()
        validate_v6_protocol_config(config)
        locks = validate_baseline_locks(config["baseline_locks"], REPO_ROOT)
        self.assertEqual(len(locks), 4)
        baseline = validate_prediction_baseline(
            config["prediction_baseline"], REPO_ROOT
        )
        self.assertEqual(
            set(baseline["observed_metrics"]),
            {"current_geode", "rbf_svm", "weighted_knn"},
        )

    def test_config_rejects_test_label_selection(self):
        config = self._config()
        config["teacher"]["test_labels_used_for_selection"] = True
        with self.assertRaises(PermissionError):
            validate_v6_protocol_config(config)
        with self.assertRaises(PermissionError):
            self._teacher(test_labels_used_for_selection=True).validate()

    def test_frozen_baseline_hash_mismatch_fails_closed(self):
        config = self._config()
        config["baseline_locks"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_baseline_locks(config["baseline_locks"], REPO_ROOT)

    def test_teacher_lineage_mismatch_fails_closed(self):
        teacher = self._teacher()
        with self.assertRaises(ValueError):
            require_teacher_compatibility(
                teacher,
                representation_hash=payload_hash({"representation": 2}),
                training_split_hash=teacher.training_split_hash,
                development_split_hash=teacher.development_split_hash,
            )

    def test_boundary_cohort_is_deterministic_and_index_tied(self):
        probabilities = np.array(
            [[0.51, 0.49], [0.49, 0.51], [0.8, 0.2], [0.6, 0.4]],
            dtype=np.float64,
        )
        first = select_boundary_cohort(
            probabilities, fraction=0.5, minimum_count=1
        )
        second = select_boundary_cohort(
            probabilities, fraction=0.5, minimum_count=1
        )
        self.assertEqual(first, second)
        self.assertEqual(first["selected_indices"], [0, 1])

    def test_primitive_metadata_is_family_specific(self):
        PrimitiveMetadata(
            family="subspace",
            minimum_seed_rule="r_plus_2",
            score_semantics="gaussian_log_likelihood",
            local_rank=16,
            residual_scale="isotropic",
        ).validate()
        with self.assertRaises(ValueError):
            PrimitiveMetadata(
                family="sphere",
                minimum_seed_rule="d_plus_2",
                score_semantics="normalized_radial",
                local_rank=16,
            ).validate()

    def test_budget_table_is_deterministic(self):
        budgets = [
            BudgetSpec("component_matched", 70, None),
            BudgetSpec("parameter_matched", None, 27020),
        ]
        first = enumerate_budget_table(budgets, ["sphere", "subspace"])
        second = enumerate_budget_table(budgets, ["sphere", "subspace"])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)


class V6S0GateTests(unittest.TestCase):
    def test_s0_matrix_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            self.assertEqual(run_s0(DEFAULT_CONFIG, first), run_s0(DEFAULT_CONFIG, second))
            first_files = {
                path.relative_to(first).as_posix(): path.read_bytes()
                for path in first.rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(second).as_posix(): path.read_bytes()
                for path in second.rglob("*")
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)

    def test_s0_verifier_writes_complete_artifact_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "verified"
            summary = verify_s0(DEFAULT_CONFIG, output)
            self.assertTrue(summary["byte_identical_replay"])
            self.assertEqual(summary["baseline_lock_count"], 4)
            self.assertEqual(summary["prediction_head_count"], 3)
            self.assertEqual(summary["primitive_count"], 5)
            self.assertEqual(summary["budget_cell_count"], 10)
            index = json.loads(
                (output / "artifact_index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["artifact_count"], len(index["artifacts"]))
            self.assertIn(
                "verification.json", {item["path"] for item in index["artifacts"]}
            )


if __name__ == "__main__":
    unittest.main()
