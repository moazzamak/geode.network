"""M309 unit tests - eval custody: rows never leave the sealed
scoring environment, and no shard is purchasable."""
from __future__ import annotations

import unittest

from geode.core.eval_custody import (
    CustodyLedger,
    PurchasableShardError,
    QueryRecord,
    RowEgressError,
    SealedScoringEnvironment,
    Shard,
    assert_not_purchasable,
    canary_detection,
    identity_economics,
    shard_purchasability,
)


def _env() -> SealedScoringEnvironment:
    return SealedScoringEnvironment([
        Shard(shard_id="s1", axis="image",
              rows=tuple(range(1000)), canary_rows=(990, 991)),
        Shard(shard_id="s2", axis="image",
              rows=tuple(range(1000, 2000)), canary_rows=(990, 991)),
    ])


class TestSealedEnvironment(unittest.TestCase):
    def test_query_returns_aggregate_only(self):
        env = _env()
        verdict = env.score_query("val-1", "image", "q-1")
        self.assertIsInstance(verdict, float)
        # four significant digits: the verdict is bounded, not raw
        self.assertLessEqual(abs(verdict), 1.0)

    def test_row_egress_raises(self):
        env = _env()
        with self.assertRaises(RowEgressError):
            env.shard_rows("s1")
        with self.assertRaises(RowEgressError):
            env.assert_no_row_egress((1, 2, 3))
        env.assert_no_row_egress("just a string")  # not row material

    def test_ledger_records_queries_never_rows(self):
        env = _env()
        env.score_query("val-1", "image", "q-1")
        env.score_query("val-2", "image", "q-2")
        self.assertEqual(len(env.ledger.entries), 2)
        with self.assertRaises(RowEgressError):
            env.ledger.record(QueryRecord(
                validator="val-3", axis="image",
                query_hash="q-3", verdict=0.5, digits=4,
                note="row: 7"))

    def test_verdict_precision_bounded_by_environment(self):
        env = _env()
        for i in range(5):
            verdict = env.score_query("val-1", "image", f"q-{i}")
            text = repr(verdict)
            self.assertLessEqual(
                len(text.replace("0.", "").rstrip("0") or "0"), 5)


class TestPurchasability(unittest.TestCase):
    def test_identity_is_a_cost_not_a_yield(self):
        eco = identity_economics(
            registration_fee=10.0, earnings_per_challenge=0.01,
            challenges_per_epoch=50.0, epochs=8)
        self.assertTrue(eco["service_not_yield"])
        self.assertLess(eco["net_cashflow"], 0.0)

    def test_purchasable_shard_raises(self):
        # shard worth one fee: the A9 defect
        self.assertEqual(shard_purchasability(10.0, 10.0), 1.0)
        with self.assertRaises(PurchasableShardError):
            assert_not_purchasable(10.0, 10.0)
        with self.assertRaises(PurchasableShardError):
            assert_not_purchasable(5.0, 10.0)

    def test_unpurchasable_shard_passes(self):
        assert_not_purchasable(shard_value=100.0, identity_cost=10.0)


class TestCanaryFallback(unittest.TestCase):
    def test_canary_detects_private_use_signature(self):
        # quiet on overlap (other holders agree), loud on private
        # rows (overfit) - the registered signature
        self.assertTrue(canary_detection(
            divergence_on_overlap=0.001, divergence_on_private=0.12))
        # no signature: both quiet
        self.assertFalse(canary_detection(
            divergence_on_overlap=0.001, divergence_on_private=0.002))


if __name__ == "__main__":
    unittest.main()
