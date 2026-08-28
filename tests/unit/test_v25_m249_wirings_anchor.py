"""Unit tests for M249 (probe suite), the M252/M247/M250 integration
wirings, M253-staking structure, and M254 anchoring structure
(incl. the zk-adjudicated dispute, the M256 cell 2 shape).
"""
from __future__ import annotations

import unittest

from geode.core.anchor import (
    AnchorSpec,
    anchor_from_ledger,
    verify_anchor_entry,
)
from geode.core.constraints import ConstraintRegistry, Prohibition
from geode.core.ledger import AppendOnlyLedger
from geode.core.orchestrator import Orchestrator
from geode.core.probes import ProbeSuite
from geode.core.refusal import (
    RefusalRecord,
    augment_measured_tags,
)
from geode.core.router import Router
from geode.settlement.slashing import SlashLedger

TASK_FP = [0.9, 0.3, 0.2, 0.1]


def _arm(arm_id: str, fp, violations=None) -> dict:
    spec = {
        "arm_id": arm_id, "fingerprint": fp,
        "output_contract": {"kind": "class"},
        "held_out_accuracy": 0.5, "availability": {"healthy": True},
        "price": 1.0, "general": False, "primitive": False,
    }
    if violations is not None:
        spec["known_violations"] = violations
    return spec


class TestM249ProbeSuite(unittest.TestCase):

    def setUp(self):
        self.suite = ProbeSuite()

    def test_artifact_integrity(self):
        report = self.suite.verify_integrity()
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["probe_count"], 15)

    def test_tampered_probe_detected(self):
        suite = ProbeSuite()
        suite.probes["honesty_audit_1"]["text"] = "tampered"
        report = suite.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertIn("honesty_audit_1", report["broken_probes"])

    def test_full_refusal_admits(self):
        responses = {pid: (True, frozenset({"v1", "v2"}))
                     for pid in self.suite.probes}
        out = self.suite.evaluate(responses)
        self.assertTrue(out["admitted"])

    def test_below_quorum_never_counts(self):
        responses = {pid: (True, frozenset({"v1"}))
                     for pid in self.suite.probes}
        out = self.suite.evaluate(responses)
        self.assertFalse(out["admitted"])
        for cat in out["categories"].values():
            self.assertEqual(cat["refusal_rate"], 0.0)

    def test_mixed_arm_rejected_per_category(self):
        responses = {}
        for pid in self.suite.probes:
            refused = not pid.startswith("harm_")
            responses[pid] = (refused, frozenset({"v1", "v2"}))
        out = self.suite.evaluate(responses)
        self.assertFalse(out["admitted"])
        self.assertFalse(out["categories"]["harm_refusal"]["admitted"])


class TestIntegrationWirings(unittest.TestCase):

    def test_constraints_tier_excludes_violators(self):
        reg = ConstraintRegistry(min_authors=1)
        proh = Prohibition(action="emit", subject="pii")
        cid = reg.commit("a1", "s", proh)
        reg.reveal(cid, "a1", "s", proh)
        router = Router()
        router.add_arm(_arm("clean", TASK_FP))
        router.add_arm(_arm("violator", TASK_FP,
                            violations=[{"action": "emit",
                                          "subject": "pii",
                                          "condition": ""}]))
        recs = router.route(TASK_FP, k=10, constraints=reg)
        self.assertEqual({r["arm_id"] for r in recs}, {"clean"})
        cold = router.cold_start(constraints=reg)
        self.assertNotEqual(cold.get("arm_id"), "violator")

    def test_refusal_tag_augmentation(self):
        arm = {"arm_id": "a"}
        good = [RefusalRecord(probe_id="p1", refusal_rate=0.95,
                              attestations=frozenset({"v1", "v2"}))]
        out = augment_measured_tags(arm, good)
        self.assertIn("refusal", out["measured_tags"])
        bad = [RefusalRecord(probe_id="p1", refusal_rate=0.4,
                             attestations=frozenset({"v1", "v2"}))]
        out2 = augment_measured_tags(arm, bad)
        self.assertNotIn("refusal", out2.get("measured_tags", []))
        # a tag once measured is never removed by augmentation
        out3 = augment_measured_tags(out, bad)
        self.assertIn("refusal", out3["measured_tags"])

    def test_behavior_diff_admission_wiring(self):
        from geode.core.behavior_diff import BehaviorDiffGate
        gate = BehaviorDiffGate(drift_bound=0.2, k_of_n=2)
        orch = Orchestrator()
        decision = orch.admit_behavior_update(
            "arm", [1.0, 0.0], frozenset({"v1", "v2"}), 5, gate)
        self.assertEqual(decision["reason"], "baseline_established")
        with self.assertRaises(ValueError):
            orch.admit_behavior_update(
                "arm", [0.0, 1.0], frozenset({"v1", "v2"}), 6, gate)


class TestSlashingAndAnchor(unittest.TestCase):

    def test_slash_accused_when_challenger_verifies(self):
        ledger = SlashLedger()
        ledger.deposit("accused", 100.0)
        ledger.deposit("challenger", 50.0)
        out = ledger.dispute(
            "d1", "accused", "challenger", "ref-1",
            accused_proof=None, challenger_proof="ok",
            verify_fn=lambda proof, ref: proof == "ok")
        self.assertEqual(out["verdict"], "slash_accused")
        self.assertEqual(out["slashed"], 100.0)
        self.assertEqual(ledger.stake_of("accused"), 0.0)

    def test_false_accusation_slashes_challenger(self):
        ledger = SlashLedger()
        ledger.deposit("accused", 100.0)
        ledger.deposit("challenger", 50.0)
        out = ledger.dispute(
            "d2", "accused", "challenger", "ref-1",
            accused_proof="ok", challenger_proof="ok",
            verify_fn=lambda proof, ref: proof == "ok")
        self.assertEqual(out["verdict"], "slash_challenger")
        self.assertEqual(ledger.stake_of("challenger"), 0.0)

    def test_unresolved_quarantined(self):
        ledger = SlashLedger()
        out = ledger.dispute(
            "d3", "a", "c", "ref-1", accused_proof=None,
            challenger_proof=None,
            verify_fn=lambda proof, ref: False)
        self.assertEqual(out["verdict"], "unresolved")
        self.assertTrue(ledger.verify()["ok"])

    def test_anchor_verify(self):
        ledger = AppendOnlyLedger()
        ledger.append({"kind": "x", "key": "k"})
        spec = anchor_from_ledger(ledger)
        self.assertTrue(verify_anchor_entry(spec.to_dict(), spec)["ok"])
        tampered = spec.to_dict()
        tampered["values"]["tip"] = "deadbeef"
        self.assertFalse(verify_anchor_entry(tampered, spec)["ok"])
        self.assertEqual(len(spec.digest()), 64)

    def test_anchor_submit_gated(self):
        from geode.core.anchor import AnchorClient
        client = AnchorClient()
        spec = AnchorSpec(tip="t", record_count=1, last_record_hash="h")
        with self.assertRaises(NotImplementedError):
            client.submit(spec)


if __name__ == "__main__":
    unittest.main()
