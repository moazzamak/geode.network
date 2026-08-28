"""Unit tests for the M182 registered repair overlay + digest gates."""
import json
import unittest
from pathlib import Path

import numpy as np

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.tier4.eval_v25_m182_contributions import (
    F6144_WIDTH,
    _load_repair_overlay,
    _part_block,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPAIR_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                 / "m182_contributions_repaired.json")

configure_external_cache_environment()
CACHE_PRESENT = data_cache_root().exists()


@unittest.skipUnless(CACHE_PRESENT,
                     "the corpus cache (GEODE_CACHE_DIR) is not "
                     "present — CI runs without the research data")
class TestM182Repair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(REPAIR_CONFIG.read_text(encoding="utf-8"))
        cls.root = data_cache_root()

    def test_patch_shape_and_start(self):
        patch, start = _load_repair_overlay(self.config)
        self.assertIsNotNone(patch)
        self.assertEqual(patch.shape, (251, F6144_WIDTH))
        self.assertEqual(start, 137749)

    def test_overlay_at_block_boundaries(self):
        patch, start = _load_repair_overlay(self.config)
        # synthetic mem with distinctive row values
        mem = np.zeros((140000, F6144_WIDTH), dtype=np.float32)
        mem[:] = 7.0
        # block fully inside the healthy region: untouched
        b = _part_block(mem, 0, 0, 4096, patch, start)
        self.assertTrue((b == 7.0).all())
        # block overlapping the patch start (137744:138000)
        b = _part_block(mem, 0, 135168, 138000, patch, start)
        lo = max(135168, start) - 135168
        self.assertTrue((b[:lo] == 7.0).all())
        self.assertTrue((b[lo:] == patch).all())
        # patch not applied to part 2
        b = _part_block(mem, 1, 0, 4096, patch, start)
        self.assertTrue((b == 7.0).all())

    def test_registered_digests_present(self):
        repair = self.config["repair"]
        for key in ("original_sha256", "patch_sha256", "test_sha256"):
            self.assertIn(key, repair)
            self.assertEqual(len(repair[key]), 64)


if __name__ == "__main__":
    unittest.main()
