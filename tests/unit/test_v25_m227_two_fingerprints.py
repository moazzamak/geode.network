"""Unit tests for M227: the two-fingerprint architecture (task
fingerprint gates admission, empirical fingerprint ranks selection,
absent until trained; provisional contributor arms).
"""
from __future__ import annotations

import unittest

from geode.core.fingerprint import EmpiricalFingerprintEncoder
from geode.core.router import Router

TASK_FP = [0.9, 0.3, 0.2, 0.1]
EMP_FP = [0.1, 0.2, 0.3, 0.9]


def _arm(arm_id: str, fp, emp=None, provisional=False) -> dict:
    spec = {
        "arm_id": arm_id, "fingerprint": fp,
        "output_contract": {"kind": "class"},
        "held_out_accuracy": 0.5, "availability": {"healthy": True},
        "price": 1.0, "general": False, "primitive": False,
        "provisional": provisional,
    }
    if emp is not None:
        spec["empirical_profile"] = emp
    return spec


class TestTwoFingerprintArchitecture(unittest.TestCase):

    def setUp(self):
        self.router = Router()
        self.router.add_arm(_arm("a_task_close", TASK_FP,
                                 emp=[0.9, 0.2, 0.1, 0.1]))
        self.router.add_arm(_arm("b_task_close_emp_far", TASK_FP,
                                 emp=[0.1, 0.2, 0.3, 0.9]))
        self.router.add_arm(_arm("c_contributor", TASK_FP,
                                 provisional=True))

    def test_empirical_ranks_when_present(self):
        # empirical cosine: EMP_FP ~ b's emp (0.1*0.9*3+0.9*0.9)
        recs = self.router.route(TASK_FP, k=2, emp_fp=EMP_FP)
        self.assertEqual(recs[0]["arm_id"], "b_task_close_emp_far")
        self.assertEqual(recs[0]["ranked_by"], "empirical")

    def test_task_ranks_when_empirical_absent(self):
        recs = self.router.route(TASK_FP, k=1)
        self.assertEqual(recs[0]["ranked_by"], "task")
        self.assertFalse(recs[0]["provisional"])

    def test_provisional_marked(self):
        recs = self.router.route(TASK_FP, k=len(self.router.list_arms()))
        prov = [r for r in recs if r["arm_id"] == "c_contributor"]
        self.assertEqual(len(prov), 1)
        self.assertTrue(prov[0]["provisional"])

    def test_empirical_encoder_absent_until_trained(self):
        enc = EmpiricalFingerprintEncoder()
        self.assertFalse(enc.trained)
        from geode.core.descriptor import normalise
        desc = normalise({"input.modality": "image",
                          "output.kind": "class"})
        self.assertIsNone(enc.encode(desc))

    def test_empirical_training_gated(self):
        enc = EmpiricalFingerprintEncoder()
        with self.assertRaises(RuntimeError):
            enc.train_on_measured([{"task": "x", "arm": "y",
                                    "outcome": 0.5}])


if __name__ == "__main__":
    unittest.main()
