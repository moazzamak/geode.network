"""Unit tests for the M212 settlement wire (orchestrator ->
CreditLedger attribution batches)."""
import unittest

from geode.core.arm import arm_from_sealed_head
from geode.core.orchestrator import Orchestrator
from geode.settlement import (
    MAX_BATCH,
    address_of,
    build_credit_batches,
    deposit_split,
    mask_for,
    recompute_batch_hash,
    verify_batch_rules,
)


def _orchestrator() -> Orchestrator:
    orch = Orchestrator()
    for name, acc in [("arm_a", 0.20), ("arm_b", 0.30), ("arm_c", 0.25)]:
        orch.register(arm_from_sealed_head(
            name, "fam", 1000, acc, f"ev_{name}.json",
            per_task={"d0": acc, "d1": acc + 0.01}))
    return orch


class TestM372NoPerPayerBudgetField(unittest.TestCase):
    """G8 (M372): the per-payer budget is a gateway-local rule; no
    per-payer budget field appears in any ledger entry type, and a
    session's route + payment replays without one."""

    def _report(self):
        orch = _orchestrator()
        for i in range(4):
            orch.serve(f"q{i}", [], task_id="d1")
        return build_credit_batches(
            orch, price_per_query=100,
            payer_of=lambda qid: address_of(f"payer:{qid}"))

    def test_entries_carry_no_budget_field(self):
        report = self._report()
        budget_names = {"cap", "used", "rate", "budget", "quota"}
        for batch in report["batches"]:
            for entry in batch["entries"]:
                keys = set(entry)
                self.assertTrue(
                    budget_names.isdisjoint(keys),
                    f"per-payer budget field leaked into an entry: "
                    f"{keys & budget_names}")
        # the batch object itself is a budget-free schema too
        for batch in report["batches"]:
            self.assertTrue(budget_names.isdisjoint(set(batch)))

    def test_replay_succeeds_without_budget(self):
        report = self._report()
        # the route+payment replay validates from the report alone;
        # the gateway-local budget ledger is nowhere in it
        self.assertEqual(verify_batch_rules(
            report, pool=report["pool_expected"]), [])
        self.assertEqual(recompute_batch_hash(report),
                         report["batch_hash"])


class TestDepositSplit(unittest.TestCase):
    def test_contract_arithmetic(self):
        # 2.5% dev cut first, floor division, as in the Solidity
        self.assertEqual(deposit_split(1000), (975, 25))
        self.assertEqual(deposit_split(1), (1, 0))  # floor: 1*25//1000=0
        self.assertEqual(deposit_split(40), (39, 1))

    def test_pool_never_exceeds_975_per_1000(self):
        for amount in range(0, 2001):
            pool_part, dev = deposit_split(amount)
            self.assertEqual(pool_part + dev, amount)
            self.assertEqual(dev, amount * 25 // 1000)


class TestAddressAndMask(unittest.TestCase):
    def test_address_deterministic(self):
        self.assertEqual(address_of("arm_a"), address_of("arm_a"))
        self.assertNotEqual(address_of("arm_a"), address_of("arm_b"))
        self.assertEqual(len(address_of("arm_a")), 42)  # 0x + 40 hex

    def test_masks_cover_both_kinds(self):
        from geode.settlement import (BIT_DNN, BIT_ENCODER, BIT_HEAD,
                                      BIT_ORCH)
        dnn = mask_for("dnn")
        self.assertTrue(dnn & BIT_DNN)
        self.assertTrue(dnn & BIT_ENCODER)
        head = mask_for("sealed_head")
        self.assertTrue(head & BIT_HEAD)
        self.assertFalse(head & BIT_DNN)
        self.assertTrue(head & BIT_ORCH)


class TestBuildAndRules(unittest.TestCase):
    def test_batch_conforms_to_contract_rules(self):
        orch = _orchestrator()
        for i in range(5):
            orch.serve(f"q{i}", [], task_id="d1")
        report = build_credit_batches(
            orch, price_per_query=100,
            payer_of=lambda qid: address_of(f"payer:{qid}"))
        self.assertEqual(verify_batch_rules(report,
                                            pool=report["pool_expected"]),
                         [])
        self.assertEqual(len(report["batches"]), 1)
        self.assertEqual(len(report["batches"][0]["entries"]), 5)
        # each query priced 100 -> pool 97 per query (floor of 100*25/1000
        # is 2, so 98 to the pool actually)
        self.assertEqual(report["batches"][0]["entries"][0]["amount"],
                         deposit_split(100)[0])

    def test_self_payment_excluded(self):
        # C1: the self-payment exclusion keys on the PAYOUT address —
        # a payer that IS the credited payout address is skipped by
        # the contract (skip-and-emit), and the builder mirrors that.
        orch = _orchestrator()
        for i in range(3):
            orch.serve(f"q{i}", [], task_id="d1")
        top_payout = address_of("arm_b")  # arm_b tops task d1
        report = build_credit_batches(
            orch, price_per_query=100,
            payer_of=lambda qid: top_payout if qid == "q1"
            else address_of(f"payer:{qid}"))
        self.assertEqual(verify_batch_rules(report,
                                            pool=report["pool_expected"]),
                         [])
        entries = [e for b in report["batches"] for e in b["entries"]]
        self.assertEqual(len(entries), 2)  # q1's entry excluded
        self.assertEqual(len(report["expected"]["skipped"]), 1)
        self.assertEqual(report["expected"]["skipped"][0]["reason"],
                         "self-payment")
        self.assertIn("artifactId",
                      report["expected"]["skipped"][0])

    def test_report_carries_registrations(self):
        orch = _orchestrator()
        for i in range(2):
            orch.serve(f"q{i}", [], task_id="d1")
        report = build_credit_batches(
            orch, price_per_query=100,
            payer_of=lambda qid: address_of(f"payer:{qid}"),
            registration_fee=7)
        self.assertEqual(report["registration_fee"], 7)
        self.assertEqual(len(report["registrations"]), 3)
        first = report["registrations"][0]
        self.assertIn("artifactId", first)
        self.assertIn("operator", first)
        self.assertIn("payoutAddress", first)
        self.assertIn("sealedClaim", first)
        entries = report["batches"][0]["entries"]
        self.assertTrue(all("artifactId" in e for e in entries))
        self.assertTrue(all("proofHash" in e for e in entries))
        self.assertIn("proof_hash", report["batches"][0])

    def test_deterministic(self):
        def build():
            orch = _orchestrator()
            for i in range(4):
                orch.serve(f"q{i}", [], task_id="d1")
            return build_credit_batches(
                orch, price_per_query=40,
                payer_of=lambda qid: address_of(f"payer:{qid}"))
        r1, r2 = build(), build()
        self.assertEqual(r1["batch_hash"], r2["batch_hash"])
        self.assertEqual(r1["batches"], r2["batches"])

    def test_tamper_changes_hash(self):
        orch = _orchestrator()
        orch.serve("q0", [], task_id="d1")
        report = build_credit_batches(
            orch, price_per_query=100,
            payer_of=lambda qid: address_of(f"payer:{qid}"))
        report["batches"][0]["entries"][0]["amount"] += 1
        self.assertNotEqual(report["batch_hash"],
                            recompute_batch_hash(report))
        self.assertIn("batch_hash does not recompute from content",
                      verify_batch_rules(report))

    def test_pool_violation_detected(self):
        orch = _orchestrator()
        orch.serve("q0", [], task_id="d1")
        report = build_credit_batches(
            orch, price_per_query=100,
            payer_of=lambda qid: address_of(f"payer:{qid}"))
        small_pool = report["batches"][0]["entries"][0]["amount"] - 1
        self.assertTrue(verify_batch_rules(report, pool=small_pool))

    def test_anchor_fields_match_ledger(self):
        orch = _orchestrator()
        orch.serve("q0", [], task_id="d1")
        report = build_credit_batches(
            orch, price_per_query=100,
            payer_of=lambda qid: address_of(f"payer:{qid}"))
        self.assertEqual(report["anchor"]["ledger_tip"],
                         orch.ledger.tip())
        self.assertEqual(report["anchor"]["record_count"],
                         orch.ledger.to_dict()["record_count"])

    def test_batches_capped_at_max_batch(self):
        orch = Orchestrator()
        orch.register(arm_from_sealed_head("only", "fam", 10, 0.5,
                                           "ev.json"))
        for i in range(MAX_BATCH + 3):
            orch.serve(f"q{i}", [], task_id=None)
        report = build_credit_batches(
            orch, price_per_query=100,
            payer_of=lambda qid: address_of(f"payer:{qid}"))
        self.assertEqual(len(report["batches"]), 2)
        self.assertLessEqual(len(report["batches"][0]["entries"]),
                             MAX_BATCH)


if __name__ == "__main__":
    unittest.main()
