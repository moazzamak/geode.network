"""M273 — the pending wirings (M247/M250/M252) tests.

The ninth-wave gap audit found the M247 measured-tag assembly and
the M250 behaviour-diff admission structure-shipped but not wired
into admission; M252 constraint-tier consumption was already live in
the router. These tests pin the wirings:
- register(refusal_records=...) augments measured_tags BEFORE
  admission (adds-only);
- admit_behavior_update receipts every decision on the ledger and
  rejects drifted updates with no state change;
- the M252 active-prohibition exclusion stays live in routing.
"""
import unittest

from geode.core.behavior_diff import BehaviorDiffGate
from geode.core.arm import arm_from_sealed_head
from geode.core.constraints import ConstraintRegistry, Prohibition
from geode.core.orchestrator import Orchestrator
from geode.core.refusal import RefusalRecord


def _arm(name, acc=0.5, fp=None, **kw):
    return arm_from_sealed_head(name, "fam", 100, acc,
                                f"ev_{name}.json", fingerprint=fp, **kw)


def _refusal_records(admitted: bool):
    # admitted: quorum (2 attestations) at rate 1.0 over 2 probes
    if admitted:
        return [RefusalRecord(probe_id=f"p{i}", refusal_rate=1.0,
                              attestations=frozenset({"a1", "a2"}))
                for i in range(2)]
    return [RefusalRecord(probe_id="p0", refusal_rate=0.3,
                          attestations=frozenset({"a1"}))]


class TestM247Wiring(unittest.TestCase):
    def test_admitted_refusal_records_add_tag(self):
        orch = Orchestrator()
        orch.register(_arm("a"), refusal_records=_refusal_records(True))
        arm = orch.router._arms["a"]
        self.assertIn("refusal", arm.get("measured_tags") or [])
        rec = orch.ledger.to_dict()["records"][0]["content"]
        self.assertIn("refusal", rec["measured_tags"])

    def test_failing_refusal_records_add_nothing(self):
        orch = Orchestrator()
        orch.register(_arm("b"), refusal_records=_refusal_records(False))
        arm = orch.router._arms["b"]
        self.assertNotIn("refusal", arm.get("measured_tags") or [])

    def test_no_records_leaves_tags_untouched(self):
        orch = Orchestrator()
        spec = _arm("c")
        spec["measured_tags"] = ["existing"]
        orch.register(spec)
        arm = orch.router._arms["c"]
        self.assertEqual(arm.get("measured_tags"), ["existing"])


class TestM250Wiring(unittest.TestCase):
    def test_first_update_establishes_baseline_with_receipt(self):
        orch = Orchestrator()
        gate = BehaviorDiffGate()
        decision = orch.admit_behavior_update(
            "a", [1.0, 2.0], frozenset({"x"}), 1, gate)
        self.assertTrue(decision["admitted"])
        receipts = [r["content"] for r in
                    orch.ledger.to_dict()["records"]
                    if r["content"]["kind"] == "behavior_update"]
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]["reason"], "baseline_established")

    def test_drifted_update_rejected_with_receipt(self):
        orch = Orchestrator()
        gate = BehaviorDiffGate()
        orch.admit_behavior_update("a", [1.0, 0.0], frozenset({"x"}),
                                   1, gate)
        # drift is 1 - cosine (M242): orthogonal vectors drift 1.0
        with self.assertRaises(ValueError):
            orch.admit_behavior_update(
                "a", [0.0, 1.0], frozenset({"x"}), 2, gate, bound=0.5)
        receipts = [r["content"] for r in
                    orch.ledger.to_dict()["records"]
                    if r["content"]["kind"] == "behavior_update"]
        self.assertEqual(len(receipts), 2)
        self.assertFalse(receipts[-1]["admitted"])
        # the baseline is untouched by the rejected update
        self.assertEqual(gate.latest("a")[0], [1.0, 0.0])


class TestM252StillLive(unittest.TestCase):
    def test_active_prohibition_excludes_violator(self):
        constraints = ConstraintRegistry(min_authors=1)
        proh = Prohibition(action="serve",
                           subject="measured_tags:refusal",
                           condition="")
        cid = constraints.commit("author", "salt", proh)
        constraints.reveal(cid, "author", "salt", proh)
        orch = Orchestrator()
        spec = _arm("v")
        spec["known_violations"] = [{
            "action": "serve", "subject": "measured_tags:refusal",
            "condition": ""}]
        orch.register(spec)
        # M252: an arm whose quorum-measured violations match an
        # ACTIVE prohibition is excluded from the route — hard,
        # never down-ranked — so cold_start serves nothing.
        out = orch.router.cold_start(constraints=constraints)
        self.assertEqual(out, {})


if __name__ == "__main__":
    unittest.main()
