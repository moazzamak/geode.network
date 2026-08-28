from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.common.v5_artifacts import sha256_file
from experiments.tier1.verify_v61_artifacts import (
    DEFAULT_CONFIG,
    _validate_artifact_index,
    _validate_config,
    _value_at,
    verify_replay,
)


class V61ArtifactReplayTests(unittest.TestCase):
    def test_registered_config_is_closed_and_data_sealed(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        _validate_config(config)
        config["training_data_allowed"] = True
        with self.assertRaises(ValueError):
            _validate_config(config)

    def test_operand_lookup_fails_closed(self):
        self.assertEqual(_value_at({"a": {"b": 3}}, "a.b"), 3)
        with self.assertRaises(ValueError):
            _value_at({"a": {}}, "a.b")

    def test_index_validation_detects_artifact_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "value.json"
            artifact.write_text("{}\n", encoding="utf-8")
            index = {
                "schema_version": 1,
                "artifacts": [
                    {
                        "path": artifact.name,
                        "sha256": sha256_file(artifact),
                        "bytes": artifact.stat().st_size,
                    }
                ],
            }
            self.assertEqual(_validate_artifact_index(root / "index.json", index), 1)
            artifact.write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                _validate_artifact_index(root / "index.json", index)

    def test_full_replay_is_byte_identical_and_loads_no_training_data(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = verify_replay(DEFAULT_CONFIG, Path(directory) / "output")
            self.assertTrue(summary["byte_identical_replay"])
            self.assertFalse(summary["training_data_loaded"])
            self.assertFalse(summary["test_labels_opened"])
            self.assertEqual(summary["predictive_outcome"], "Outcome D")

    def test_operand_drift_in_fixture_fails_comparison(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        operand = copy.deepcopy(config["operands"][0])
        operand["expected"] = not operand["expected"]
        payload = {"advancement_passed": False}
        self.assertNotEqual(
            _value_at(payload, operand["path"]), operand["expected"]
        )


if __name__ == "__main__":
    unittest.main()
