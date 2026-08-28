"""Unit tests for M241-M245: the alignment + Byzantine-tolerance
tranche (routing constraint tier + abstention, empirical drift gate,
override ledger, demerits + safety-adjusted credit, Byzantine
measurement aggregation).
"""
from __future__ import annotations

import unittest

from geode.attribution.incentives import (
    Demerit,
    safety_adjusted_value,
)
from geode.core.byzantine import (
    admitted_facts,
    median_vector,
    quorum,
)
from geode.core.fingerprint import DriftGate
from geode.core.override import OverrideLedger
from geode.core.router import Router

TASK_FP = [0.9, 0.3, 0.2, 0.1]


def _arm(arm_id: str, fp, vetted=False, measured_tags=None,
         provisional=False) -> dict:
    spec = {
        "arm_id": arm_id, "fingerprint": fp,
        "output_contract": {"kind": "class"},
        "held_out_accuracy": 0.5, "availability": {"healthy": True},
        "price": 1.0, "general": False, "primitive": False,
        "vetted": vetted,
        "provisional": provisional,
    }
    if measured_tags is not None:
        spec["measured_tags"] = measured_tags
    return spec


class TestM241RoutingConstraints(unittest.TestCase):
    """M241: hard constraints + abstention; declared fields are never
    trusted for safety admission."""

    def setUp(self):
        self.router = Router()
        self.router.add_arm(_arm("vetted_ok", TASK_FP, vetted=True,
                                 measured_tags=["refusal", "pii"]))
        self.router.add_arm(_arm("vetted_partial", TASK_FP, vetted=True,
                                 measured_tags=["refusal"]))
        self.router.add_arm(_arm("unvetted", TASK_FP, vetted=False))
        self.router.add_arm(_arm("provisional_liar", TASK_FP,
                                 vetted=False,
                                 provisional=True,
                                 measured_tags=["refusal", "pii"]))

    def test_flagged_task_excludes_unmeasured(self):
        recs = self.router.route(TASK_FP, k=10,
                                 required_tags=["refusal"])
        ids = {r["arm_id"] for r in recs}
        self.assertIn("vetted_ok", ids)
        self.assertIn("vetted_partial", ids)
        self.assertNotIn("unvetted", ids)
        self.assertNotIn("provisional_liar", ids)

    def test_missing_tag_excludes_even_vetted(self):
        recs = self.router.route(TASK_FP, k=10,
                                 required_tags=["refusal", "pii"])
        ids = {r["arm_id"] for r in recs}
        self.assertEqual(ids, {"vetted_ok"})

    def test_declared_safety_cannot_help(self):
        # provisional_liar carries measured_tags but is not vetted:
        # declared coverage never counts toward safety admission.
        recs = self.router.route(TASK_FP, k=10,
                                 required_tags=["refusal"])
        self.assertNotIn("provisional_liar",
                         {r["arm_id"] for r in recs})

    def test_abstain_floor_returns_empty(self):
        recs = self.router.route([0.99, 0.01, 0.01, 0.01], k=1,
                                 required_tags=["refusal"],
                                 abstain_below=0.95)
        self.assertEqual(recs, [])

    def test_unflagged_defaults_unchanged(self):
        # no safety kwargs -> the pre-M241 behaviour (everything
        # eligible that matches a fingerprint).
        recs = self.router.route(TASK_FP, k=10)
        self.assertEqual(len(recs), 4)

    def test_flagged_chain_skips_fallback_tiers(self):
        self.router.add_arm(_arm("general_arm", TASK_FP))
        # force 'general' on one arm for the tier test
        gen = self.router._arms["general_arm"]
        gen["general"] = True
        gen["fingerprint"] = []
        chain = self.router.chain(TASK_FP, required_tags=["refusal"])
        self.assertNotIn("general_arm", [r["arm_id"] for r in chain])


class TestM242DriftGate(unittest.TestCase):
    """M242: drift bound + ledger-index staleness, deterministic."""

    def setUp(self):
        self.gate = DriftGate(drift_bound=0.2, staleness_window=100)

    def test_within_bound_admitted(self):
        ok, reason = self.gate.admits([1.0, 0.0], [0.99, 0.01],
                                      measured_index=10, as_of_index=20)
        self.assertTrue(ok)
        self.assertEqual(reason, "admitted")

    def test_beyond_bound_rejected(self):
        ok, reason = self.gate.admits([1.0, 0.0], [0.0, 1.0],
                                      measured_index=10, as_of_index=20)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("drift_exceeded"))

    def test_stale_rejected(self):
        ok, reason = self.gate.admits([1.0, 0.0], [1.0, 0.0],
                                      measured_index=10, as_of_index=200)
        self.assertFalse(ok)
        self.assertEqual(reason, "stale_measurement")

    def test_zero_vector_is_maximum_drift(self):
        self.assertEqual(self.gate.drift([1.0, 0.0], [0.0, 0.0]), 1.0)

    def test_bound_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            DriftGate(drift_bound=1.5)


class TestM243OverrideLedger(unittest.TestCase):
    """M243: overrides require justification + counterfactual and are
    hash-chained."""

    def test_blank_justification_rejected(self):
        ledger = OverrideLedger()
        with self.assertRaises(ValueError):
            ledger.record("operator", "kill_switch", "   ",
                          {"would_have": "routed to arm_a"})

    def test_missing_counterfactual_rejected(self):
        ledger = OverrideLedger()
        with self.assertRaises(ValueError):
            ledger.record("operator", "admission_exception",
                          "incident review approved", {})

    def test_unknown_action_rejected(self):
        ledger = OverrideLedger()
        with self.assertRaises(ValueError):
            ledger.record("operator", "whatever", "reason",
                          {"would_have": "x"})

    def test_record_and_verify(self):
        ledger = OverrideLedger()
        idx = ledger.record("operator", "manual_rerank",
                            "flagged by the safety review",
                            {"would_have": "routed to arm_a"})
        self.assertEqual(idx, 0)
        self.assertTrue(ledger.verify()["ok"])
        rec = ledger.to_dict()["records"][0]["content"]
        self.assertEqual(rec["action"], "manual_rerank")
        self.assertEqual(rec["counterfactual"],
                         {"would_have": "routed to arm_a"})


class TestM244Demerits(unittest.TestCase):
    """M244: only quorum-attested harm discounts credit."""

    def test_quorum_demerit_discounts(self):
        demerits = [Demerit(arm="a", harm=3.0,
                            attestations=frozenset({"v1", "v2"}))]
        self.assertEqual(safety_adjusted_value(10.0, demerits,
                                               k_of_n=2), 7.0)

    def test_single_source_quarantined(self):
        demerits = [Demerit(arm="a", harm=3.0,
                            attestations=frozenset({"v1"}))]
        self.assertEqual(safety_adjusted_value(10.0, demerits,
                                               k_of_n=2), 10.0)

    def test_floor_respected(self):
        demerits = [Demerit(arm="a", harm=20.0,
                            attestations=frozenset({"v1", "v2"}))]
        self.assertEqual(safety_adjusted_value(10.0, demerits,
                                               k_of_n=2, floor=2.0), 2.0)


class TestM245Byzantine(unittest.TestCase):
    """M245: median aggregation resists a minority of liars."""

    def test_median_resists_one_byzantine_of_three(self):
        honest = [[1.0, 2.0], [1.2, 1.9]]
        byzantine = [[9.0, 9.0]]
        med = median_vector(honest + byzantine)
        self.assertEqual(med, [1.2, 2.0])

    def test_even_n_takes_lower_middle(self):
        med = median_vector([[1.0], [9.0], [1.0], [9.0]])
        self.assertEqual(med, [1.0])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            median_vector([])

    def test_quorum_counting(self):
        att = {"fact_a": frozenset({"v1", "v2", "v3"}),
               "fact_b": frozenset({"v1"})}
        report = quorum(att, k_of_n=2)
        self.assertTrue(report["fact_a"]["quorum"])
        self.assertFalse(report["fact_b"]["quorum"])
        self.assertEqual(admitted_facts(att, 2), ["fact_a"])


if __name__ == "__main__":
    unittest.main()
