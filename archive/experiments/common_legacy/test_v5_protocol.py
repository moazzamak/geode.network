import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    parameter_count,
    payload_hash,
    require_representation_match,
    serialized_size,
    validate_migration_report,
)
from experiments.common.v5_protocol import (
    DataStage,
    GateOperand,
    LabelUse,
    RepresentationLineage,
    SplitRole,
    require_label_use,
    seeds_for_stage,
    validate_protocol_config,
)
from experiments.common.v5_registry import (
    ExperimentCell,
    expand_matrix,
    validate_matched_comparison,
    validate_required_controls,
)
from experiments.common.v5_statistics import (
    paired_prediction_interval,
    pareto_dominates,
)
from experiments.tier1.eval_v5_protocol_s0 import DEFAULT_CONFIG, run_s0, verify_s0


class V5ProtocolTests(unittest.TestCase):
    def test_seed_binding_and_final_label_guard(self):
        self.assertEqual(seeds_for_stage(DataStage.S3), (11, 23, 37, 53, 71))
        with self.assertRaises(ValueError):
            seeds_for_stage(DataStage.S2, (11, 23))
        with self.assertRaises(ValueError):
            seeds_for_stage(DataStage.S4, (999,))
        require_label_use(SplitRole.TEST, LabelUse.OBSERVE)
        with self.assertRaises(PermissionError):
            require_label_use(SplitRole.TEST, LabelUse.SELECT)

    def test_protocol_schema_rejects_missing_provenance(self):
        payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        validate_protocol_config(payload)
        del payload["datasets"]
        with self.assertRaises(ValueError):
            validate_protocol_config(payload)

    def test_protocol_schema_requires_every_data_stage(self):
        payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        del payload["stages"]["S4"]
        with self.assertRaises(ValueError):
            validate_protocol_config(payload)
        payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        payload["datasets"] = [
            item for item in payload["datasets"] if item["stage"] != "S4"
        ]
        with self.assertRaises(ValueError):
            validate_protocol_config(payload)

    def test_lineage_hash_is_stable_and_interface_is_paired(self):
        digest = payload_hash({"value": 1})
        lineage = RepresentationLineage(
            backbone_id="test",
            weights_hash=digest,
            preprocessing_hash=digest,
            output_dimension=4,
        )
        self.assertEqual(lineage.digest, lineage.digest)
        with self.assertRaises(ValueError):
            RepresentationLineage(
                backbone_id="test",
                weights_hash=digest,
                preprocessing_hash=digest,
                output_dimension=4,
                interface_id="missing-hash",
            ).validate()

    def test_gate_operand_records_explicit_result(self):
        self.assertEqual(
            GateOperand("accuracy", 0.8, "ge", 0.75).to_dict(),
            {
                "name": "accuracy",
                "value": 0.8,
                "operator": "ge",
                "threshold": 0.75,
                "passed": True,
            },
        )


class V5RegistryTests(unittest.TestCase):
    def _cells(self):
        digest = payload_hash({"split": 1})
        return expand_matrix(
            milestone="M16",
            stage=DataStage.S0,
            dataset="toy",
            representations=["identity"],
            heads=["linear", "geode"],
            readouts=["raw"],
            split_hashes={"identity": digest},
            feature_hashes={"identity": payload_hash({"feature": 1})},
        )

    def test_matrix_expansion_and_naming_are_deterministic(self):
        first = self._cells()
        second = self._cells()
        self.assertEqual(first, second)
        self.assertEqual([cell.cell_id for cell in first], [cell.cell_id for cell in second])
        validate_required_controls(first, {"linear", "geode"})

    def test_missing_control_and_split_mismatch_are_rejected(self):
        cells = self._cells()
        with self.assertRaises(ValueError):
            validate_required_controls(cells, {"linear", "rbf", "geode"})
        mismatched = [
            cells[0],
            ExperimentCell(
                **{
                    **cells[1].__dict__,
                    "split_hash": payload_hash({"split": 2}),
                }
            ),
        ]
        with self.assertRaises(ValueError):
            validate_matched_comparison(mismatched)
        cross_dataset = [
            cells[0],
            ExperimentCell(
                **{
                    **cells[1].__dict__,
                    "dataset": "different",
                }
            ),
        ]
        with self.assertRaises(ValueError):
            validate_matched_comparison(cross_dataset)

    def test_cross_representation_split_mismatch_is_rejected(self):
        first_hash = payload_hash({"split": 1})
        second_hash = payload_hash({"split": 2})
        cells = expand_matrix(
            milestone="M16",
            stage=DataStage.S0,
            dataset="toy",
            representations=["first", "second"],
            heads=["linear", "geode"],
            readouts=["raw"],
            split_hashes={"first": first_hash, "second": second_hash},
            feature_hashes={
                "first": payload_hash({"feature": 1}),
                "second": payload_hash({"feature": 2}),
            },
        )
        with self.assertRaises(ValueError):
            validate_required_controls(cells, {"linear", "geode"})


class V5ArtifactTests(unittest.TestCase):
    def test_parameter_and_byte_counts(self):
        payload = {
            "first": np.zeros((2, 3), dtype=np.float64),
            "second": [np.ones(4, dtype=np.float64)],
        }
        self.assertEqual(parameter_count(payload), 10)
        self.assertEqual(serialized_size({"a": [1, 2]}), len('{"a":[1,2]}\n'))

    def test_artifact_index_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "b.txt").write_text("b\n", encoding="utf-8")
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            first = build_artifact_index(root)
            first_bytes = (root / "artifact_index.json").read_bytes()
            second = build_artifact_index(root)
            self.assertEqual(first, second)
            self.assertEqual(first_bytes, (root / "artifact_index.json").read_bytes())
            self.assertEqual(
                [item["path"] for item in first["artifacts"]],
                ["a.txt", "b.txt"],
            )

    def test_lineage_mismatch_and_invalid_migration_fail_closed(self):
        first = payload_hash({"representation": 1})
        second = payload_hash({"representation": 2})
        with self.assertRaises(ValueError):
            require_representation_match(
                active_hash=first,
                artifact_hash=second,
                artifact_kind="calibration",
            )
        with self.assertRaises(ValueError):
            require_representation_match(
                active_hash=first.upper(),
                artifact_hash=first,
                artifact_kind="calibration",
            )
        report = {
            "schema_version": 1,
            "source_representation_hash": first,
            "target_representation_hash": second,
            "component_correspondence": [],
            "edit_survival": [],
            "invalidated_artifacts": ["calibration", "support_profile"],
            "rollback_bundle_hash": payload_hash({"bundle": 1}),
        }
        validate_migration_report(report)
        report["target_representation_hash"] = first
        with self.assertRaises(ValueError):
            validate_migration_report(report)


class V5StatisticsTests(unittest.TestCase):
    def test_paired_interval_matches_hand_computed_difference(self):
        truth = np.array([0, 0, 1, 1])
        first = np.array([0, 0, 1, 1])
        second = np.array([0, 1, 1, 0])
        result = paired_prediction_interval(
            truth,
            first,
            second,
            n_resamples=100,
            seed=7,
        )
        self.assertEqual(result["difference"], 0.5)
        self.assertEqual(result["n_resamples"], 100)

    def test_pareto_dominance_uses_declared_directions(self):
        directions = {"accuracy": "higher", "latency": "lower"}
        self.assertTrue(
            pareto_dominates(
                {"accuracy": 0.9, "latency": 2.0},
                {"accuracy": 0.8, "latency": 2.0},
                directions,
            )
        )
        self.assertFalse(
            pareto_dominates(
                {"accuracy": 0.9, "latency": 3.0},
                {"accuracy": 0.8, "latency": 2.0},
                directions,
            )
        )


class V5S0GateTests(unittest.TestCase):
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

    def test_s0_verification_rejects_mismatch_and_regenerates_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "verified"
            summary = verify_s0(DEFAULT_CONFIG, output)
            self.assertTrue(summary["byte_identical_replay"])
            self.assertTrue(summary["split_mismatch_rejected"])
            self.assertEqual(summary["cell_count"], 10)
            index = json.loads((output / "artifact_index.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["artifact_count"], len(index["artifacts"]))
            self.assertIn("verification.json", {item["path"] for item in index["artifacts"]})


if __name__ == "__main__":
    unittest.main()
