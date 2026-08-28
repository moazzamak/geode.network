from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.tier1.eval_v8_final_replay import (
    DEFAULT_CONFIG,
    verify_final_replay,
)


class V8FinalReplayTests(unittest.TestCase):
    def test_final_outcome_replays_without_data_access(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = verify_final_replay(
                DEFAULT_CONFIG, Path(directory) / "v8-final"
            )
        self.assertEqual(summary["outcome"], "D")
        self.assertEqual(summary["verified_index_count"], 4)
        self.assertEqual(summary["verified_conclusion_count"], 6)
        self.assertTrue(summary["byte_identical_replay"])
        self.assertFalse(summary["training_data_loaded"])
        self.assertFalse(summary["final_labels_opened"])


if __name__ == "__main__":
    unittest.main()
