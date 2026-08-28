"""Unit tests for v26 M313 economic repairs (A11, A20).

Registered semantics (plan §8.15): the per-axis bond is sized to the
compute saving over the exposure window; claim delay tracks open
exposure; L3 burns vested-but-unclaimed only and is reachable because
claims were frozen; tenure credit accrues only from sampled, verified
work.
"""
from __future__ import annotations

import unittest

from geode.core.economic_repairs import (
    SLASH_LADDER,
    VERIFIED_SOURCES,
    claim_delay_epochs,
    conviction_burn,
    per_axis_bond,
    slash_decision,
    tenure_weight,
    verified_activity,
)


class TestPerAxisBond(unittest.TestCase):

    def test_bond_is_saving_times_exposure(self):
        self.assertEqual(per_axis_bond(0.02, 400.0), 0.02 * 400.0)
        self.assertEqual(per_axis_bond(0.0, 10.0), 0.0)

    def test_bond_rejects_negative_saving_and_bad_exposure(self):
        with self.assertRaises(ValueError):
            per_axis_bond(-0.01, 10.0)
        with self.assertRaises(ValueError):
            per_axis_bond(0.01, 0.0)


class TestClaimDelay(unittest.TestCase):

    def test_delay_ceil_in_epochs(self):
        self.assertEqual(claim_delay_epochs(0.0, 100.0), 0)
        self.assertEqual(claim_delay_epochs(1.0, 100.0), 1)
        self.assertEqual(claim_delay_epochs(100.0, 100.0), 1)
        self.assertEqual(claim_delay_epochs(101.0, 100.0), 2)

    def test_delay_monotone_in_exposure(self):
        # M313-C3: non-decreasing in open exposure units
        delays = [claim_delay_epochs(u, 100.0)
                  for u in (0.0, 50.0, 100.0, 250.0, 1000.0)]
        self.assertEqual(delays, sorted(delays))

    def test_delay_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            claim_delay_epochs(-1.0, 100.0)
        with self.assertRaises(ValueError):
            claim_delay_epochs(10.0, 0.0)


class TestConvictionBurn(unittest.TestCase):

    def test_burn_is_vested_minus_claimed(self):
        self.assertEqual(conviction_burn(10.0, 4.0), 6.0)

    def test_burn_zero_when_fully_claimed(self):
        # M313-C4: zero for fully-claimed accounts
        self.assertEqual(conviction_burn(10.0, 10.0), 0.0)

    def test_burn_never_negative_nor_above_vested(self):
        # M313-C4
        self.assertGreaterEqual(conviction_burn(0.0, 0.0), 0.0)
        self.assertLessEqual(conviction_burn(7.0, 2.0), 7.0)

    def test_burn_rejects_claimed_above_vested(self):
        with self.assertRaises(ValueError):
            conviction_burn(3.0, 4.0)


class TestSlashLadder(unittest.TestCase):

    def test_ladder_has_four_honest_rungs(self):
        self.assertEqual(len(SLASH_LADDER), 4)

    def test_l3_burns_and_delists(self):
        decision = slash_decision(3, 10.0, 6.0)
        self.assertTrue(decision["delist"])
        self.assertTrue(decision["freeze_claims"])
        self.assertEqual(decision["burn"], 4.0)

    def test_l0_does_nothing(self):
        decision = slash_decision(0, 10.0, 9.0)
        self.assertFalse(decision["delist"])
        self.assertFalse(decision["freeze_claims"])
        self.assertEqual(decision["burn"], 0.0)

    def test_l1_freezes_without_burn(self):
        decision = slash_decision(1, 10.0, 9.0)
        self.assertTrue(decision["freeze_claims"])
        self.assertFalse(decision["delist"])
        self.assertEqual(decision["burn"], 0.0)

    def test_rejects_unknown_level(self):
        with self.assertRaises(ValueError):
            slash_decision(4, 1.0, 0.0)


class TestVerifiedActivity(unittest.TestCase):

    def test_only_sampled_verified_sources_accrue(self):
        records = [
            {"source": "sampled_challenge", "weight": 1.0},
            {"source": "probe_reference", "weight": 1.0},
            {"source": "self_served", "weight": 1.0},
            {"source": "self_purchased", "weight": 1.0},
            {"source": "unknown", "weight": 1.0},
        ]
        kept = verified_activity(records)
        self.assertEqual([r["source"] for r in kept],
                         ["sampled_challenge", "probe_reference"])

    def test_registered_source_names(self):
        # the source vocabulary is registered; new kinds are opt-in
        self.assertEqual(VERIFIED_SOURCES,
                         frozenset(("sampled_challenge",
                                    "probe_reference")))


class TestTenureWeight(unittest.TestCase):

    def test_wash_ring_accrues_zero(self):
        # M313-C2: wash-ring records (self-generated volume under any
        # address) yield zero tenure weight
        wash_ring = [
            {"source": "self_served", "weight": 1e6,
             "address": "0xaa"},
            {"source": "self_served", "weight": 1e6,
             "address": "0xbb"},
        ]
        self.assertEqual(tenure_weight(wash_ring), 0.0)

    def test_verified_work_accrues_positive(self):
        records = [
            {"source": "sampled_challenge", "weight": 3.0},
            {"source": "probe_reference", "weight": 2.0},
            {"source": "self_served", "weight": 100.0},
        ]
        self.assertEqual(tenure_weight(records), 5.0)

    def test_default_weight_is_one(self):
        self.assertEqual(
            tenure_weight([{"source": "sampled_challenge"}]), 1.0)


if __name__ == "__main__":
    unittest.main()
