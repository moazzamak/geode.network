from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from experiments.e2e.generate_e11_public_study import reproduce_public_study


class E11PublicStudyTests(unittest.TestCase):
    def test_artifact_only_reproduction_is_byte_identical(self):
        config = Path("experiments/configs/e11_public_study.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "artifact_index.json"
            first = reproduce_public_study(
                config, root / "first", lock_path=lock, refresh_lock=True,
            )
            second = reproduce_public_study(
                config, root / "second", lock_path=lock,
            )
            self.assertEqual(first, second)
            self.assertEqual(len(first), 5)

            locked = lock.read_text(encoding="utf-8")
            self.assertIn(
                '"E7": "local_small_complete_multihost_blocked"', locked,
            )
            self.assertIn('"sha256"', locked)


if __name__ == "__main__":
    unittest.main()