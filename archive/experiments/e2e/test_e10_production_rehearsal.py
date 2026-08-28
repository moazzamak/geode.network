from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from experiments.e2e.run_e10_production_rehearsal import run_qualification


class E10ProductionRehearsalTests(unittest.TestCase):
    def test_bad_canary_and_coordinator_loss_restore_parent(self):
        config = Path("experiments/configs/e10_production_rehearsal.json")
        with tempfile.TemporaryDirectory() as directory:
            result = run_qualification(config, Path(directory) / "registry")
            self.assertTrue(result["gate_passed"])
            self.assertEqual(
                result["bundles"]["current"], result["bundles"]["production"],
            )
            self.assertEqual(result["shadow"]["agreement"], 0.0)
            self.assertEqual(result["telemetry"]["replicas"], 2)


if __name__ == "__main__":
    unittest.main()