from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.tier1.eval_v7_final_replay import (
    DEFAULT_CONFIG,
    build_replay,
)


class V7FinalReplayTests(unittest.TestCase):
    def test_outcome_c_replays_without_data_or_final_labels(self) -> None:
        config = json.loads(Path(DEFAULT_CONFIG).read_text(encoding="utf-8"))
        replay = build_replay(config)
        self.assertEqual(replay["outcome"], "C")
        self.assertEqual(replay["verified_index_count"], 6)
        self.assertFalse(replay["training_data_loaded"])
        self.assertFalse(replay["final_labels_opened"])
        self.assertTrue(all(replay["conclusions"].values()))
