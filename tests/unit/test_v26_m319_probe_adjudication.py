"""Unit tests for the M319 adjudication rules (A18 selective abort,
A19 quorum-failure resampling)."""
from __future__ import annotations

import unittest

from geode.core.probe_adjudication import (
    ABORT_ESCALATION_MIN_ABORTS,
    ABORT_SELECTIVITY_ALPHA,
    LEVEL_DEVIATION,
    LEVEL_DOWNTIME,
    abort_allowance,
    abort_selectivity,
    adjudicate_epoch_aborts,
    adjudicate_probed_session,
    expected_host_profit,
    quorum_failure_plan,
)

# the registered economic cell for the M364 profit sweep
SWEEP = dict(sessions=10_000, unit_price=1.0, honest_cost=0.6,
             cheat_cost_ratio=0.1, probe_rate=0.05, burn=16_000.0)


class TestAdjudicateProbedSession(unittest.TestCase):

    def test_opened_match_is_clean(self):
        out = adjudicate_probed_session(commit_opened=True, probed=True,
                                        answers_match=True)
        self.assertEqual(out["verdict"], "clean")
        self.assertIsNone(out["ladder_level"])

    def test_opened_mismatch_is_deviation(self):
        out = adjudicate_probed_session(commit_opened=True, probed=True,
                                        answers_match=False)
        self.assertEqual(out["verdict"], "deviation")
        self.assertEqual(out["ladder_level"], LEVEL_DEVIATION)

    def test_unopened_is_an_abort_at_no_level_yet(self):
        # M364: one session cannot say whether an unopened commit was
        # a refused inspection or a denied service. It is charged the
        # unit price and its level waits for the epoch.
        for probed in (True, False):
            out = adjudicate_probed_session(commit_opened=False,
                                            probed=probed,
                                            answers_match=False)
            self.assertEqual(out["verdict"], "abort")
            self.assertIsNone(out["ladder_level"])
            self.assertTrue(out["charge_unit_price"])

    def test_only_an_opened_mismatch_is_slashed_per_session(self):
        # the one verdict a single session can still reach on its own
        mismatch = adjudicate_probed_session(True, True, False)
        self.assertEqual(mismatch["ladder_level"], LEVEL_DEVIATION)
        self.assertFalse(mismatch["charge_unit_price"])


class TestAbortSelectivity(unittest.TestCase):
    """M364 (G23): the statistic that separates a denied service from
    a refused inspection."""

    def test_all_aborts_probed_is_selective(self):
        out = abort_selectivity(aborts_probed=5, aborts_total=5,
                                probe_rate=0.05)
        self.assertTrue(out["selective"])
        self.assertAlmostEqual(out["p_value"], 0.05 ** 5)

    def test_aborts_at_the_probe_rate_are_not_selective(self):
        out = abort_selectivity(aborts_probed=5, aborts_total=100,
                                probe_rate=0.05)
        self.assertFalse(out["selective"])
        self.assertGreater(out["p_value"], ABORT_SELECTIVITY_ALPHA)

    def test_no_aborts_is_never_selective(self):
        out = abort_selectivity(0, 0, 0.05)
        self.assertFalse(out["selective"])
        self.assertEqual(out["p_value"], 1.0)

    def test_the_test_is_one_sided(self):
        # aborting only UNprobed sessions is not evidence of dodging
        out = abort_selectivity(aborts_probed=0, aborts_total=20,
                                probe_rate=0.05)
        self.assertFalse(out["selective"])
        self.assertEqual(out["p_value"], 1.0)

    def test_rejects_impossible_counts(self):
        with self.assertRaises(ValueError):
            abort_selectivity(6, 5, 0.05)
        with self.assertRaises(ValueError):
            abort_selectivity(1, 5, 1.5)


class TestEpochAborts(unittest.TestCase):
    """M364 (G23): a third party must not be able to fire a burn."""

    def test_the_defect_reproduces_under_the_old_rule(self):
        # A18 as first written: `if probed: return L1` for every
        # unopened commit. Applied to one DoS campaign, side by side
        # with M364 on the identical campaign.
        def old_rule(probed: bool) -> int | None:
            # verbatim shape of the branch M364 removed
            return LEVEL_DEVIATION if probed else LEVEL_DOWNTIME

        committed, probe_rate = 10_000, 0.05
        aborts_probed, aborts_unprobed = 50, 950
        campaign = ([True] * aborts_probed
                    + [False] * aborts_unprobed)

        old_burns = sum(1 for p in campaign
                        if old_rule(p) == LEVEL_DEVIATION)
        self.assertEqual(old_burns, aborts_probed)
        self.assertGreater(old_burns, 0)  # the third-party trigger

        new = adjudicate_epoch_aborts(
            committed_sessions=committed,
            probed_sessions=int(committed * probe_rate),
            aborts_probed=aborts_probed,
            aborts_unprobed=aborts_unprobed, unit_price=1.0)
        self.assertEqual(new["ladder_level"], LEVEL_DOWNTIME)
        self.assertFalse(new["escalates"])

    def test_a_dos_campaign_produces_zero_burns_at_any_size(self):
        # the attacker drops a fixed fraction of sessions, blind to
        # the probe flag, so the probed share of aborts is the probe
        # rate. Campaign size does not change the verdict.
        for aborts_total in (10, 100, 1_000, 5_000, 9_000):
            probed = round(aborts_total * 0.05)
            out = adjudicate_epoch_aborts(
                committed_sessions=10_000, probed_sessions=500,
                aborts_probed=probed,
                aborts_unprobed=aborts_total - probed,
                unit_price=1.0)
            self.assertFalse(
                out["escalates"],
                f"campaign of {aborts_total} escalated")
            self.assertEqual(out["ladder_level"], LEVEL_DOWNTIME)

    def test_the_victim_pays_exactly_the_refunded_sessions(self):
        out = adjudicate_epoch_aborts(
            committed_sessions=10_000, probed_sessions=500,
            aborts_probed=15, aborts_unprobed=285, unit_price=7.0)
        self.assertEqual(out["refunded_sessions"], 300)
        self.assertAlmostEqual(out["unit_price_charge"], 300 * 7.0)
        self.assertFalse(out["escalates"])

    def test_a_dodging_host_is_escalated(self):
        # aborts only on probed sessions
        out = adjudicate_epoch_aborts(
            committed_sessions=1_000, probed_sessions=50,
            aborts_probed=30, aborts_unprobed=0, unit_price=1.0)
        self.assertTrue(out["selective"])
        self.assertTrue(out["escalates"])
        self.assertEqual(out["ladder_level"], LEVEL_DEVIATION)

    def test_the_dodges_allowed_before_escalation_are_computed(self):
        # published, not assumed: how many probed-only aborts a host
        # gets before the test fires, at the registered alpha
        committed, probed = 1_000, 50
        allowed = 0
        for k in range(1, 40):
            out = adjudicate_epoch_aborts(
                committed_sessions=committed, probed_sessions=probed,
                aborts_probed=k, aborts_unprobed=0, unit_price=1.0)
            if out["escalates"]:
                break
            allowed = k
        self.assertEqual(allowed, ABORT_ESCALATION_MIN_ABORTS)
        # and every one of those dodges was charged full price
        charged = adjudicate_epoch_aborts(
            committed_sessions=committed, probed_sessions=probed,
            aborts_probed=allowed, aborts_unprobed=0, unit_price=3.0)
        self.assertAlmostEqual(charged["unit_price_charge"],
                               allowed * 3.0)

    def test_the_floor_does_not_scale_with_traffic(self):
        # a proportional floor would let a large host hide a large
        # number of dodges; the sweep below measures why that matters
        self.assertEqual(abort_allowance(100),
                         abort_allowance(10_000_000))

    def test_the_small_sample_floor_never_licenses_a_free_abort(self):
        out = adjudicate_epoch_aborts(
            committed_sessions=100, probed_sessions=5,
            aborts_probed=1, aborts_unprobed=0, unit_price=2.0)
        self.assertFalse(out["escalates"])
        self.assertAlmostEqual(out["unit_price_charge"], 2.0)

    def test_both_conditions_are_needed_to_escalate(self):
        # significant share but at or below the floor: no escalation
        below = adjudicate_epoch_aborts(
            committed_sessions=100, probed_sessions=5,
            aborts_probed=1, aborts_unprobed=0, unit_price=1.0)
        self.assertFalse(below["above_floor"])
        self.assertFalse(below["escalates"])
        # above the floor but unaimed: no escalation
        unaimed = adjudicate_epoch_aborts(
            committed_sessions=100, probed_sessions=5,
            aborts_probed=1, aborts_unprobed=40, unit_price=1.0)
        self.assertTrue(unaimed["above_floor"])
        self.assertFalse(unaimed["selective"])
        self.assertFalse(unaimed["escalates"])

    def test_false_escalation_rate_respects_alpha(self):
        # an honest host under attack, over many epochs, drawn from
        # the null the test assumes
        import random
        rng = random.Random(364)
        committed, probe_rate, aborts = 10_000, 0.05, 200
        epochs, escalated = 2_000, 0
        for _ in range(epochs):
            probed_aborts = sum(1 for _ in range(aborts)
                                if rng.random() < probe_rate)
            out = adjudicate_epoch_aborts(
                committed_sessions=committed,
                probed_sessions=int(committed * probe_rate),
                aborts_probed=probed_aborts,
                aborts_unprobed=aborts - probed_aborts,
                unit_price=1.0)
            escalated += out["escalates"]
        rate = escalated / epochs
        self.assertLessEqual(rate, 10 * ABORT_SELECTIVITY_ALPHA,
                             f"false escalation rate {rate}")

    def test_rejects_incoherent_epochs(self):
        with self.assertRaises(ValueError):
            adjudicate_epoch_aborts(10, 20, 0, 0, 1.0)
        with self.assertRaises(ValueError):
            adjudicate_epoch_aborts(10, 5, 6, 6, 1.0)
        with self.assertRaises(ValueError):
            adjudicate_epoch_aborts(10, 5, 1, 1, -1.0)


class TestProfitSweep(unittest.TestCase):
    """M364: the thesis "the only profitable behaviour is serving the
    artifact every time" is swept over cheat rates, not asserted."""

    def _best(self, strategy):
        return max((expected_host_profit(i / 2000.0, strategy, **SWEEP)
                    for i in range(401)),
                   key=lambda r: r["profit"])

    def test_camouflage_and_open_mismatch_are_never_profitable(self):
        base = expected_host_profit(0.0, "honest", **SWEEP)["profit"]
        for strategy in ("camouflage", "open_mismatch"):
            best = self._best(strategy)
            self.assertLessEqual(best["profit"], base,
                                 f"{strategy} beat honest serving")

    def test_the_dodge_residual_is_bounded_and_disclosed(self):
        # NOT zero. A statistical test needs a minimum sample, so a
        # host that keeps its aborts under the floor hides a few
        # cheats per epoch. The number is measured here rather than
        # waved at, and it is what the paper must state.
        base = expected_host_profit(0.0, "honest", **SWEEP)["profit"]
        best = self._best("dodge")
        gain = best["profit"] - base
        self.assertGreater(gain, 0.0)          # the residual is real
        self.assertLessEqual(gain / base, 0.01)
        self.assertLessEqual(best["aborted"], ABORT_ESCALATION_MIN_ABORTS)
        self.assertLessEqual(best["cheated"] / SWEEP["sessions"], 0.01)

    def test_a_proportional_floor_would_be_far_worse(self):
        # why the floor is a constant: with a 1%-of-traffic floor the
        # same host hides two orders of magnitude more
        base = expected_host_profit(0.0, "honest", **SWEEP)["profit"]
        proportional = int(SWEEP["sessions"] * 0.01)
        best = max((expected_host_profit(
            i / 2000.0, "dodge", escalation_min_aborts=proportional,
            **SWEEP) for i in range(401)),
            key=lambda r: r["profit"])
        constant = self._best("dodge")
        self.assertGreater(best["profit"] - base,
                           10 * (constant["profit"] - base))

    def test_scaling_the_burn_does_not_remove_the_residual(self):
        # the residual is a small-sample property, not a pricing one:
        # the dodger never gets burned, so a larger burn cannot reach it
        harsh = dict(SWEEP, burn=SWEEP["burn"] * 1000)
        base = expected_host_profit(0.0, "honest", **harsh)["profit"]
        best = max((expected_host_profit(i / 2000.0, "dodge", **harsh)
                    for i in range(401)), key=lambda r: r["profit"])
        self.assertFalse(best["burned"])
        self.assertGreater(best["profit"] - base, 0.0)

    def test_rejects_an_unknown_strategy(self):
        with self.assertRaises(ValueError):
            expected_host_profit(0.1, "bribe_the_probe", **SWEEP)


class TestQuorumFailurePlan(unittest.TestCase):

    def test_no_op_when_quorum_met(self):
        plan = quorum_failure_plan(responders=6, sampled=9,
                                   quorum_num=2, quorum_den=3,
                                   unspent_budget=100)
        self.assertFalse(plan["quorum_failed"])
        self.assertFalse(plan["resample"])

    def test_resample_and_carry_budget(self):
        plan = quorum_failure_plan(responders=3, sampled=9,
                                   quorum_num=2, quorum_den=3,
                                   unspent_budget=100)
        self.assertTrue(plan["quorum_failed"])
        self.assertTrue(plan["resample"])
        self.assertFalse(plan["new_fee_charged"])
        self.assertEqual(plan["budget_carried_forward"], 100)
        self.assertEqual(plan["needed"], 6)

    def test_demerit_weighted_by_proximity(self):
        far = quorum_failure_plan(responders=0, sampled=9,
                                  quorum_num=2, quorum_den=3,
                                  unspent_budget=10)
        near = quorum_failure_plan(responders=5, sampled=9,
                                   quorum_num=2, quorum_den=3,
                                   unspent_budget=10)
        self.assertEqual(far["demerit_per_non_responder"], 0.0)
        self.assertGreater(near["demerit_per_non_responder"],
                           far["demerit_per_non_responder"])
        self.assertAlmostEqual(near["demerit_per_non_responder"],
                               5 / 9)

    def test_validation(self):
        with self.assertRaises(ValueError):
            quorum_failure_plan(-1, 9, 2, 3, 0)
        with self.assertRaises(ValueError):
            quorum_failure_plan(10, 9, 2, 3, 0)
        with self.assertRaises(ValueError):
            quorum_failure_plan(3, 9, 3, 2, 0)   # quorum not proper
        with self.assertRaises(ValueError):
            quorum_failure_plan(3, 9, 2, 3, -1)


if __name__ == "__main__":
    unittest.main()
