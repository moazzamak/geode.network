from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from experiments.e2e.run_e9_transactional_adaptation import run_qualification


class E9TransactionalAdaptationTests(unittest.TestCase):
    def test_qualification_is_deterministic_and_rolls_back(self):
        config = Path("experiments/configs/e9_transactional_adaptation.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = run_qualification(config, root / "registry-1")
            second = run_qualification(config, root / "registry-2")
            self.assertTrue(first["gate_passed"])
            self.assertEqual(first["review"], second["review"])
            self.assertEqual(first["registry"], second["registry"])
            self.assertEqual(
                first["registry"]["current_bundle_id"],
                first["registry"]["parent_bundle_id"],
            )


if __name__ == "__main__":
    unittest.main()