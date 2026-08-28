"""Unit tests for M248-M252 (the trustless containment tranche):
emergency freeze, OOD input guard, typed constraints with
commitment-based authorship, behavioural diffing, and their router
integration.
"""
from __future__ import annotations

import unittest

from geode.core.behavior_diff import BehaviorDiffGate
from geode.core.constraints import (
    ConstraintRegistry,
    Prohibition,
    commit_hash,
)
from geode.core.freeze import FreezeError, FreezeRegistry
from geode.core.ood import OodGate
from geode.core.router import Router

TASK_FP = [0.9, 0.3, 0.2, 0.1]


def _arm(arm_id: str, fp) -> dict:
    return {
        "arm_id": arm_id, "fingerprint": fp,
        "output_contract": {"kind": "class"},
        "held_out_accuracy": 0.5, "availability": {"healthy": True},
        "price": 1.0, "general": False, "primitive": False,
    }


class TestM248Freeze(unittest.TestCase):

    def setUp(self):
        self.freeze = FreezeRegistry(k_of_n=2, default_ttl=100)
        self.freeze.freeze("e1", frozenset({"v1", "v2"}), 10,
                           reason="incident")

    def test_quorum_freeze_covers_window(self):
        self.assertTrue(self.freeze.is_frozen(10))
        self.assertTrue(self.freeze.is_frozen(109))
        self.assertFalse(self.freeze.is_frozen(110))  # auto-expiry

    def test_below_quorum_inert(self):
        self.freeze.freeze("e2", frozenset({"v1"}), 200)
        self.assertFalse(self.freeze.is_frozen(200))

    def test_unfreeze_requires_quorum_and_specific_event(self):
        self.freeze.unfreeze("e1", frozenset({"v3", "v4"}), 50)
        self.assertFalse(self.freeze.is_frozen(50))
        with self.assertRaises(FreezeError):
            self.freeze.unfreeze("e1", frozenset({"v3"}), 20)

    def test_freeze_is_time_bounded(self):
        # a permanent freeze is refused by construction
        with self.assertRaises(ValueError):
            self.freeze.freeze("e3", frozenset({"v1", "v2"}), 0, ttl=0)

    def test_router_returns_empty_while_frozen(self):
        router = Router()
        router.add_arm(_arm("a", TASK_FP))
        out = router.route(TASK_FP, freeze=self.freeze, as_of_index=10)
        self.assertEqual(out, [])
        out = router.route(TASK_FP, freeze=self.freeze, as_of_index=110)
        self.assertEqual(len(out), 1)

    def test_admission_rejects_while_frozen(self):
        router = Router()
        with self.assertRaises(FreezeError):
            router.add_arm(_arm("a", TASK_FP), freeze=self.freeze,
                           as_of_index=10)
        router.add_arm(_arm("a", TASK_FP))  # no freeze -> fine


class TestM251OodGuard(unittest.TestCase):

    def setUp(self):
        self.guard = OodGate(threshold=3.0)
        self.guard.fit_profile([[0.0, 0.0], [1.0, 0.0],
                                [0.0, 1.0], [1.0, 1.0]])

    def test_in_distribution_admitted(self):
        decision = self.guard.admits([0.5, 0.5])
        self.assertTrue(decision["admitted"])
        self.assertEqual(decision["reason"], "in_distribution")

    def test_out_of_distribution_rejected(self):
        decision = self.guard.admits([100.0, 100.0])
        self.assertFalse(decision["admitted"])
        self.assertEqual(decision["reason"], "out_of_distribution")

    def test_unfitted_guard_fails_closed(self):
        guard = OodGate()
        decision = guard.admits([1.0, 1.0])
        self.assertFalse(decision["admitted"])
        self.assertEqual(decision["reason"], "guard_unfitted")

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.guard.score([1.0, 2.0, 3.0])

    def test_router_ood_escalation(self):
        router = Router()
        router.add_arm(_arm("a", TASK_FP))
        out = router.route(TASK_FP, ood_guard=self.guard,
                           input_vec=[100.0, 100.0])
        self.assertEqual(out, [])
        out = router.route(TASK_FP, ood_guard=self.guard,
                           input_vec=[0.5, 0.5])
        self.assertEqual(len(out), 1)
        # missing input_vec with a guard present: fail closed
        self.assertEqual(router.route(TASK_FP, ood_guard=self.guard), [])


class TestM252Constraints(unittest.TestCase):

    def setUp(self):
        self.reg = ConstraintRegistry(min_authors=2)
        self.proh = Prohibition(action="emit", subject="pii",
                                condition="input_has_pii")

    def test_commit_then_reveal(self):
        cid = self.reg.commit("a1", "salt1", self.proh)
        self.assertFalse(self.reg.reveal(cid, "a1", "salt1", self.proh))
        cid2 = self.reg.commit("a2", "salt2", self.proh)
        self.assertTrue(self.reg.reveal(cid2, "a2", "salt2", self.proh))
        self.assertIn(self.proh, self.reg.active())

    def test_reveal_without_commit_raises(self):
        cid = commit_hash("a1", "salt1", self.proh)
        with self.assertRaises(ValueError):
            self.reg.reveal(cid, "a1", "salt1", self.proh)

    def test_tampered_salt_raises(self):
        cid = self.reg.commit("a1", "salt1", self.proh)
        with self.assertRaises(ValueError):
            self.reg.reveal(cid, "a1", "saltX", self.proh)

    def test_below_min_authors_inactive(self):
        cid = self.reg.commit("a1", "salt1", self.proh)
        self.reg.reveal(cid, "a1", "salt1", self.proh)
        self.assertEqual(self.reg.active(), [])

    def test_violations_match(self):
        self.reg.min_authors = 1
        cid = self.reg.commit("a1", "salt1", self.proh)
        self.reg.reveal(cid, "a1", "salt1", self.proh)
        arm = {"known_violations": [{"action": "emit", "subject": "pii",
                                     "condition": "input_has_pii"}]}
        self.assertEqual(self.reg.violations(arm), [self.proh])
        arm_clean = {"known_violations": []}
        self.assertEqual(self.reg.violations(arm_clean), [])

    def test_unconditional_matches_any_condition(self):
        broad = Prohibition(action="emit", subject="pii")
        self.assertTrue(broad.matches("emit", "pii", "anything"))


class TestM250BehaviorDiff(unittest.TestCase):

    def setUp(self):
        self.gate = BehaviorDiffGate(drift_bound=0.2, k_of_n=2)

    def test_first_update_establishes_baseline(self):
        decision = self.gate.admits_update("arm", [1.0, 0.0], 5)
        self.assertTrue(decision["admitted"])
        self.assertEqual(decision["reason"], "baseline_established")

    def test_drifted_update_gated(self):
        self.gate.admits_update("arm", [1.0, 0.0], 5)
        decision = self.gate.admits_update("arm", [0.0, 1.0], 6)
        self.assertFalse(decision["admitted"])
        self.assertTrue(decision["reason"].startswith("behavior_drift"))

    def test_close_update_admitted_and_rebaselines(self):
        self.gate.admits_update("arm", [1.0, 0.0], 5)
        decision = self.gate.admits_update("arm", [0.99, 0.01], 6)
        self.assertTrue(decision["admitted"])
        self.assertEqual(self.gate.latest("arm")[1], 6)

    def test_below_quorum_snapshot_never_baselines(self):
        ok = self.gate.record_snapshot("arm", [1.0, 0.0],
                                       frozenset({"v1"}), 5)
        self.assertFalse(ok)
        self.assertIsNone(self.gate.latest("arm"))
        # so the next update is the baseline, not a gated drift
        decision = self.gate.admits_update("arm", [0.0, 1.0], 6)
        self.assertEqual(decision["reason"], "baseline_established")


if __name__ == "__main__":
    unittest.main()
