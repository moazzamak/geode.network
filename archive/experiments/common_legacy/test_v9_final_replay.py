from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.common.v5_artifacts import sha256_file, write_canonical_json
from experiments.tier4.verify_v9_final import _load_locked, run_verification


class V9FinalReplayTests(unittest.TestCase):
    def test_locked_artifact_rejects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.json"
            write_canonical_json(artifact, {"value": 1})
            with patch(
                "experiments.tier4.verify_v9_final.REPO_ROOT", root
            ):
                with self.assertRaises(ValueError):
                    _load_locked(
                        {"path": "artifact.json", "sha256": "0" * 64}
                    )

    def test_final_replay_is_deterministic_and_data_free(self):
        with tempfile.TemporaryDirectory() as first_directory:
            with tempfile.TemporaryDirectory() as second_directory:
                first = run_verification(output_dir=first_directory)
                second = run_verification(output_dir=second_directory)
                self.assertEqual(first, second)
                self.assertFalse(first["training_data_loaded"])
                self.assertFalse(first["final_labels_opened"])
                self.assertEqual(first["outcome"], "D")
                self.assertEqual(
                    sha256_file(Path(first_directory) / "evidence.json"),
                    sha256_file(Path(second_directory) / "evidence.json"),
                )

    def test_final_configuration_keeps_labels_sealed(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "v9"
            / "m55_final_replay.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(payload["final_labels_opened"])


if __name__ == "__main__":
    unittest.main()
