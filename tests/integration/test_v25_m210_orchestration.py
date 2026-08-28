"""Unit tests for the M210 model-agnostic orchestration closure."""
import unittest

from geode.core.arm import (
    arm_from_sealed_head,
    validate_arm_spec,
)
from geode.core.dnn_admission import AdmissionRegistry, DNNSubmission
from geode.core.orchestrator import Orchestrator


def _submission() -> DNNSubmission:
    return DNNSubmission(
        architecture_hash="a" * 64, seed_hash="b" * 64,
        data_digest="c" * 64, software_hash="d" * 64,
        weights_hash="e" * 64, training_log_digest="f" * 64,
        eval_report={"split": "test", "n_test": 34500, "accuracy": 0.5})


class TestArmAdapters(unittest.TestCase):
    def test_sealed_head_valid(self):
        spec = arm_from_sealed_head("h", "spm", 40383, 0.26, "ev.json",
                                    per_task={"d3": 0.36})
        self.assertEqual(validate_arm_spec(spec), [])
        self.assertEqual(spec["held_out_accuracy"]["d3"], 0.36)
        self.assertEqual(spec["param_count"], 40383 * 345 + 345)

    def test_dnn_arm_requires_admission(self):
        sub = _submission()
        registry = AdmissionRegistry()
        result = registry.admit(sub)
        from geode.core.arm import arm_from_admission
        spec = arm_from_admission(sub, result, "m", 0.5, param_count=12)
        self.assertEqual(validate_arm_spec(spec), [])
        # a duplicate submission on the SAME registry is rejected and
        # can never route
        dup = registry.admit(sub)
        self.assertFalse(dup.admitted)
        with self.assertRaises(ValueError):
            arm_from_admission(sub, dup, "m2", 0.5)

    def test_size_agnostic(self):
        huge = arm_from_sealed_head("big", "synthetic", 100_000, 0.5,
                                    "synthetic")
        huge["kind"] = "dnn"
        huge["replay_hash"] = "0" * 64
        huge["param_count"] = 1_000_000_000
        huge["size_bytes"] = 4_000_000_000
        self.assertEqual(validate_arm_spec(huge), [])
        self.assertTrue(validate_arm_spec({"arm_id": "bad"}))


class TestOrchestrator(unittest.TestCase):
    def test_register_route_attribute_record(self):
        orch = Orchestrator()
        for name, acc in [("a", 0.20), ("b", 0.30), ("c", 0.25)]:
            orch.register(arm_from_sealed_head(
                name, "fam", 1000, acc, f"ev_{name}.json",
                per_task={"d0": acc, "d1": acc + 0.01}))
        routed = orch.serve("q1", [], task_id="d1")
        self.assertEqual(routed[0]["arm_id"], "b")  # 0.31 wins
        self.assertEqual(orch.chain_verify()["ok"], True)
        self.assertEqual(orch.chain_verify()["record_count"], 4)
        att = orch.attribute()
        self.assertAlmostEqual(att["v_all"], 0.30)
        self.assertAlmostEqual(att["loo_marginals"]["b"], 0.05)
        self.assertAlmostEqual(att["loo_marginals"]["c"], 0.0)

    def test_determinism(self):
        def build():
            orch = Orchestrator()
            orch.register(arm_from_sealed_head("x", "f", 100, 0.5,
                                               "ev.json"))
            orch.serve("q", [], task_id=None)
            return orch
        o1, o2 = build(), build()
        self.assertEqual(o1.content_hash(), o2.content_hash())
        self.assertEqual(o1.chain_verify()["tip"],
                         o2.chain_verify()["tip"])


if __name__ == "__main__":
    unittest.main()
