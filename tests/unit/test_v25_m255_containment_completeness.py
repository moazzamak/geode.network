"""Unit tests for M255: the containment-completeness fix
(cold_start and serve obey the freeze + OOD guards; the safety
surface is exported).
"""
from __future__ import annotations

import unittest

from geode.core.freeze import FreezeRegistry
from geode.core.ood import OodGate
from geode.core.orchestrator import Orchestrator
from geode.core.router import Router

TASK_FP = [0.9, 0.3, 0.2, 0.1]


def _arm(arm_id: str, fp, general=True) -> dict:
    return {
        "arm_id": arm_id, "fingerprint": fp,
        "output_contract": {"kind": "class"},
        "held_out_accuracy": 0.5, "availability": {"healthy": True},
        "price": 1.0, "general": general, "primitive": False,
    }


class TestM255ContainmentCompleteness(unittest.TestCase):

    def setUp(self):
        self.router = Router()
        self.router.add_arm(_arm("g1", TASK_FP))
        self.freeze = FreezeRegistry(k_of_n=2, default_ttl=100)
        self.freeze.freeze("e1", frozenset({"v1", "v2"}), 0,
                           reason="incident")
        self.guard = OodGate(threshold=3.0)
        self.guard.fit_profile([[0.0, 0.0], [1.0, 1.0]])

    def test_cold_start_empty_while_frozen(self):
        out = self.router.cold_start(freeze=self.freeze, as_of_index=5)
        self.assertEqual(out, {})
        out = self.router.cold_start(freeze=self.freeze, as_of_index=200)
        self.assertIn("arm_id", out)

    def test_cold_start_empty_on_ood(self):
        out = self.router.cold_start(ood_guard=self.guard,
                                     input_vec=[100.0, 100.0])
        self.assertEqual(out, {})
        out = self.router.cold_start(ood_guard=self.guard,
                                     input_vec=[0.5, 0.5])
        self.assertIn("arm_id", out)
        # missing input with a guard present: fail closed
        self.assertEqual(self.router.cold_start(ood_guard=self.guard),
                         {})

    def test_serve_empty_while_frozen_and_ledger_recorded(self):
        orch = Orchestrator()
        orch.router.add_arm(_arm("g1", TASK_FP))
        out = orch.serve("q1", TASK_FP, freeze=self.freeze,
                         as_of_index=5)
        self.assertEqual(out, [])
        rec = orch.ledger.to_dict()["records"][0]["content"]
        self.assertEqual(rec["chosen"], [])
        self.assertTrue(rec["contained"])

    def test_safety_surface_exported(self):
        import geode
        for name in ["FreezeRegistry", "OodGate", "ConstraintRegistry",
                     "Prohibition", "BehaviorDiffGate",
                     "VerifierRotation", "OverrideLedger", "DriftGate",
                     "RefusalCapability", "median_vector", "quorum",
                     "Demerit", "safety_adjusted_value", "trust_weight"]:
            self.assertTrue(hasattr(geode, name), f"{name} missing")


if __name__ == "__main__":
    unittest.main()
