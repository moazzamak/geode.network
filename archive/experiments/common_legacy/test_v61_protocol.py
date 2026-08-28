from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.common.v61_protocol import (
    validate_indexed_parent_locks,
    validate_parent_file_locks,
    validate_representation_lineage,
    validate_v61_config,
)
from experiments.tier1.eval_v61_parent_lock_a0 import (
    DEFAULT_CONFIG,
    M30_EVIDENCE,
    REPO_ROOT,
    run_a0,
    verify_a0,
)


class V61ProtocolTests(unittest.TestCase):
    def _config(self) -> dict:
        return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def _m30_evidence(self) -> dict:
        return json.loads(M30_EVIDENCE.read_text(encoding="utf-8"))

    def test_registered_config_and_parent_locks_validate(self):
        config = self._config()
        validate_v61_config(config)
        self.assertEqual(
            len(validate_parent_file_locks(config["parent_file_locks"], REPO_ROOT)),
            6,
        )
        indexed = validate_indexed_parent_locks(
            config["indexed_parent_locks"], REPO_ROOT
        )
        self.assertEqual([item["id"] for item in indexed], [
            "m29_subspace_s1",
            "m30_directional_s2",
            "m31_factorial_s2",
        ])
        self.assertGreater(sum(item["artifact_count"] for item in indexed), 40)

    def test_parent_file_hash_mismatch_fails_closed(self):
        config = self._config()
        config["parent_file_locks"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_parent_file_locks(config["parent_file_locks"], REPO_ROOT)

    def test_parent_representation_mismatch_fails_closed(self):
        config = self._config()
        config["representation_lineage"][0]["parent_representation_hash"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_representation_lineage(
                config["representation_lineage"],
                m30_evidence=self._m30_evidence(),
            )

    def test_split_mismatch_fails_closed(self):
        config = self._config()
        config["representation_lineage"][1]["development_split_hash"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_representation_lineage(
                config["representation_lineage"],
                m30_evidence=self._m30_evidence(),
            )

    def test_normalization_mismatch_fails_closed(self):
        config = self._config()
        config["amended_primitive"]["normalization"] = "implicit"
        with self.assertRaises(ValueError):
            validate_v61_config(config)

    def test_rank_mismatch_fails_closed(self):
        config = self._config()
        config["amended_primitive"]["rank"] = 31
        with self.assertRaises(ValueError):
            validate_v61_config(config)

    def test_readout_schema_mismatch_fails_closed(self):
        config = self._config()
        config["weighted_readout"]["constraint"] = "unconstrained"
        with self.assertRaises(ValueError):
            validate_v61_config(config)

    def test_final_test_access_fails_closed(self):
        config = self._config()
        config["test_labels_opened"] = True
        with self.assertRaises(PermissionError):
            validate_v61_config(config)

    def test_missing_selected_student_fails_closed(self):
        config = self._config()
        broken = copy.deepcopy(config["indexed_parent_locks"])
        broken[2]["required_paths"].append("seed_11/missing_student.json")
        with self.assertRaises(ValueError):
            validate_indexed_parent_locks(broken, REPO_ROOT)


class V61A0GateTests(unittest.TestCase):
    def test_parent_lock_replays_byte_identically_without_training_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first_summary = run_a0(DEFAULT_CONFIG, first)
            second_summary = run_a0(DEFAULT_CONFIG, second)
            self.assertEqual(first_summary, second_summary)
            self.assertFalse(first_summary["training_data_loaded"])
            self.assertFalse(first_summary["test_labels_opened"])
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

    def test_verifier_writes_complete_artifact_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "verified"
            summary = verify_a0(DEFAULT_CONFIG, output)
            self.assertTrue(summary["byte_identical_replay"])
            self.assertEqual(summary["representation_lineage_count"], 3)
            self.assertEqual(summary["claim_count"], 6)
            index = json.loads(
                (output / "artifact_index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["artifact_count"], len(index["artifacts"]))
            self.assertIn(
                "verification.json", {item["path"] for item in index["artifacts"]}
            )


if __name__ == "__main__":
    unittest.main()
