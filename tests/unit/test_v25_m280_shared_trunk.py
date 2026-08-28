"""M280 — shared-trunk program tests: trunks are appended
once, arms must reference registered trunks, a new trunk needs
measured gap evidence, and primitives are exempt."""
import unittest

from geode.core.shared_trunk import TrunkRegistry, validate_arm_trunk


class TestSharedTrunk(unittest.TestCase):
    def setUp(self):
        self.registry = TrunkRegistry()
        self.registry.register_trunk("bert", "Apache-2.0", 110.0)
        self.registry.register_trunk("dinov2-small", "Apache-2.0",
                                     22.0)

    def test_duplicate_trunk_rejected(self):
        with self.assertRaises(ValueError):
            self.registry.register_trunk("bert", "Apache-2.0", 110.0)

    def test_trunk_reuse_needs_no_evidence(self):
        decision = self.registry.admit_trunk("bert", "Apache-2.0",
                                             110.0, None)
        self.assertTrue(decision["admitted"])
        self.assertEqual(decision["reason"], "trunk_reuse")

    def test_new_trunk_without_gap_rejected(self):
        decision = self.registry.admit_trunk("whisper-large",
                                             "Apache-2.0", 1550.0,
                                             None)
        self.assertFalse(decision["admitted"])
        self.assertIn("measured_gap", decision["reason"])

    def test_new_trunk_with_gap_admitted(self):
        gap = {"task": "asr-long-audio", "measured_gap": 0.31,
               "evidence_path": "logs/results/v25/mxxx/evidence.json"}
        decision = self.registry.admit_trunk("whisper-large",
                                             "Apache-2.0", 1550.0,
                                             gap)
        self.assertTrue(decision["admitted"])
        self.assertEqual(decision["reason"], "gap_measured")

    def test_arm_needs_registered_trunk(self):
        self.assertEqual(validate_arm_trunk(
            {"trunk_id": "bert"}, self.registry), [])
        self.assertEqual(validate_arm_trunk(
            {"trunk_id": "gpt99"}, self.registry),
            ["trunk 'gpt99' not registered (M280)"])

    def test_primitive_exempt(self):
        self.assertEqual(validate_arm_trunk(
            {"primitive": True}, self.registry), [])


if __name__ == "__main__":
    unittest.main()
