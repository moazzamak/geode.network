"""M327 unit tests - bootstrap governance."""
from __future__ import annotations

import unittest

from geode.core.bootstrap_council import (
    BootstrapCouncil,
    CharterCapAudit,
    CouncilViolation,
    EarnedWeightLedger,
    register_with_zero_stake,
)


class TestCouncil(unittest.TestCase):
    def test_council_is_multi_party(self):
        with self.assertRaises(CouncilViolation):
            BootstrapCouncil(members={"dev"}, sunset_epoch=4,
                             developer="dev")

    def test_vote_only_while_active(self):
        council = BootstrapCouncil(
            members={"dev", "op-1", "val-1"}, sunset_epoch=4,
            developer="dev")
        self.assertTrue(council.active_at(3))
        self.assertTrue(council.vote(3, "op-1", "p-1"))
        self.assertFalse(council.active_at(4))
        with self.assertRaises(CouncilViolation):
            council.vote(4, "op-1", "p-1")

    def test_sunset_is_timelocked(self):
        council = BootstrapCouncil(
            members={"dev", "op-1", "val-1"}, sunset_epoch=4,
            developer="dev")
        with self.assertRaises(CouncilViolation):
            council.sunset(3)
        self.assertTrue(council.sunset(4))
        self.assertTrue(council.sunset(5))

    def test_no_fund_routing_to_members(self):
        council = BootstrapCouncil(
            members={"dev", "op-1", "val-1"}, sunset_epoch=4,
            developer="dev")
        with self.assertRaises(CouncilViolation):
            council.assert_no_fund_routing({"zakat", "op-1"})
        council.assert_no_fund_routing({"zakat", "education"})


class TestEarnedWeight(unittest.TestCase):
    def test_genesis_is_zero(self):
        ledger = EarnedWeightLedger()
        ledger.assert_genesis_zero()

    def test_weight_accrues_only_by_verified_work(self):
        ledger = EarnedWeightLedger()
        ledger.earn("a", 5.0)
        ledger.earn("b", 3.0)
        self.assertEqual(ledger.total(), 8.0)
        with self.assertRaises(CouncilViolation):
            ledger.earn("a", -1.0)

    def test_cap_clips_excess_to_zero(self):
        ledger = EarnedWeightLedger()
        ledger.earn("hoarder", 100.0)
        ledger.earn("b", 2.0)
        ledger.earn("c", 2.0)
        capped = ledger.capped()
        ceiling = 104.0 * 0.2
        self.assertAlmostEqual(capped["hoarder"], ceiling)
        self.assertEqual(capped["b"], 2.0)

    def test_cap_under_extreme_concentration(self):
        # one identity holds everything: it is clipped to 20% of the
        # RAW total and the excess counts at zero (the registered
        # rule - three capped identities reach only 60%)
        ledger = EarnedWeightLedger()
        ledger.earn("hoarder", 1000.0)
        ledger.earn("small-1", 1.0)
        ledger.earn("small-2", 1.0)
        raw_total = ledger.total()
        capped = ledger.capped()
        self.assertAlmostEqual(capped["hoarder"], raw_total * 0.2)
        self.assertEqual(capped["small-1"], 1.0)
        self.assertEqual(capped["small-2"], 1.0)
        # the excess (800.0) contributes zero to any tally
        self.assertAlmostEqual(sum(capped.values()),
                               raw_total * 0.2 + 2.0)


class TestCharterCapAudit(unittest.TestCase):
    def test_cap_has_no_mutator(self):
        audit = CharterCapAudit()
        audit.register_mutator("floors_raise")
        audit.register_mutator("zakat_rule")
        audit.assert_cap_unmutable()

    def test_cap_mutator_is_a_violation(self):
        audit = CharterCapAudit()
        audit.register_mutator("voting_cap_raise")
        with self.assertRaises(CouncilViolation):
            audit.assert_cap_unmutable()


class TestZeroStakeAdmission(unittest.TestCase):
    def test_admission_paths_are_measured_only(self):
        council = BootstrapCouncil(
            members={"dev", "op-1", "val-1"}, sunset_epoch=4,
            developer="dev")
        self.assertTrue(register_with_zero_stake(
            {"challenge_session", "registration_fee", "per_axis_bond"},
            council, epoch=1))

    def test_stake_referencing_path_is_rejected(self):
        council = BootstrapCouncil(
            members={"dev", "op-1", "val-1"}, sunset_epoch=4,
            developer="dev")
        with self.assertRaises(CouncilViolation):
            register_with_zero_stake(
                {"challenge_session", "council_vote"}, council, epoch=1)


if __name__ == "__main__":
    unittest.main()
