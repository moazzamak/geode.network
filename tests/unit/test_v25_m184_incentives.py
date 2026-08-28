"""Unit tests for M184 + M199: the incentive simulation harness.

Pure stdlib + numpy; deterministic seeds. Covers the registered
synthetic gates: H1 (shared beats solo), H3 (wash loses), H8
(availability honesty by construction), and the M199 corner-case arms
(collusion rings, inference farms, Sybil duplicates).
"""
from __future__ import annotations

import unittest

from geode.attribution.incentives import (
    Agent,
    collusion_ring_gate,
    dust_storm_gate,
    farm_gate,
    h1_gate,
    h3_gate,
    h8_gate,
    run_round,
    self_payment_wash_gate,
    structural_form_checks,
    sybil_duplicate_gate,
)


def _h1_agents() -> list[Agent]:
    return [
        Agent("coop_a", "cooperative", cost=1.0, contribution=3.0,
              solo_progress=1.2),
        Agent("coop_b", "cooperative", cost=1.0, contribution=2.5,
              solo_progress=1.0),
        Agent("defector_1", "defector", cost=0.5, solo_progress=1.5),
        Agent("defector_2", "defector", cost=0.5, solo_progress=1.4),
    ]


class TestH1(unittest.TestCase):

    def test_shared_beats_solo(self):
        result = h1_gate(_h1_agents(), rounds=10, demand=100.0,
                         lag_sweep=[0, 2, 5], seed=1)
        self.assertTrue(result["passes"])

    def test_shared_loses_when_solo_dominates(self):
        agents = _h1_agents()
        for a in agents:
            if a.kind == "defector":
                a.solo_progress = 100.0
        result = h1_gate(agents, rounds=10, demand=100.0,
                         lag_sweep=[0], seed=1)
        self.assertFalse(result["passes"])


class TestH3(unittest.TestCase):

    def test_wash_loses_honest_gains(self):
        result = h3_gate(Agent("wash", "wash"), Agent("honest", "cooperative"),
                         rounds=20, demand=100.0, lag=5, seed=2)
        self.assertTrue(result["passes"], result)


class TestH8(unittest.TestCase):

    def test_gamer_self_report_ignored(self):
        agents = [
            Agent("down_gamer", "gamer", actually_healthy=False,
                  self_reported_healthy=True),
            Agent("well", "cooperative", actually_healthy=True),
        ]
        result = h8_gate(agents, seed=3)
        self.assertTrue(result["passes"])
        self.assertEqual(result["selected"], "well")

    def test_all_down_selects_none(self):
        agents = [Agent("down", "gamer", actually_healthy=False)]
        self.assertEqual(h8_gate(agents, seed=4)["selected"], None)


class TestRounds(unittest.TestCase):

    def test_run_round_vests_only_cooperative(self):
        agents = [Agent("coop", "cooperative", cost=0.5, contribution=2.0),
                  Agent("free", "free_ride")]
        run_round(agents, demand=100.0, lag=0,
                  rng=__import__("numpy").random.default_rng(5))
        self.assertGreater(agents[0].vested, 0.0)


class TestM199CollusionRing(unittest.TestCase):

    def test_ring_loses_money_in_aggregate(self):
        ring = [Agent("ring_a", "gamer", cash=0.0),
                Agent("ring_b", "gamer", cash=0.0),
                Agent("ring_c", "gamer", cash=0.0)]
        result = collusion_ring_gate(ring, rounds=30, demand=100.0,
                                     lag=5, seed=11)
        self.assertTrue(result["passes"], result)
        self.assertLess(result["ring_net_change"], 0.0)
        for member, cash in result["final_cash"].items():
            self.assertLessEqual(cash, 0.0, (member, cash))

    def test_ring_members_fall_but_receiver_side_gets_some_back(self):
        ring = [Agent("ring_a", "gamer", cash=100.0),
                Agent("ring_b", "gamer", cash=100.0)]
        result = collusion_ring_gate(ring, rounds=10, demand=100.0,
                                     lag=0, seed=12)
        # aggregate loss still strictly negative (taxes per hop)
        self.assertLess(result["ring_net_change"], 0.0)


class TestM199InferenceFarm(unittest.TestCase):

    def test_low_quality_farm_loses_high_quality_thaws(self):
        farm = Agent("farm", "gamer", hosting_cost=2.0)
        result = farm_gate(farm, rounds=25, demand=100.0, lag=5,
                           quality_floor=0.5, seed=13)
        self.assertTrue(result["passes"], result)
        self.assertLess(result["low_quality_final_cash"], 0.0)
        self.assertGreater(result["high_quality_thawed"], 0.0)
        self.assertEqual(result["low_quality_thawed"], 0.0)


class TestM199SybilDuplicate(unittest.TestCase):

    def test_duplicate_digest_earns_zero(self):
        original = Agent("orig", "cooperative", contribution=2.0,
                         content_digest="sha:abc")
        sybil = Agent("copy", "cooperative", contribution=2.0,
                      content_digest="sha:abc")
        result = sybil_duplicate_gate(original, sybil, seed=14)
        self.assertTrue(result["passes"], result)
        self.assertEqual(result["sybil_share"], 0.0)
        self.assertGreater(result["original_share"], 0.0)

    def test_distinct_digests_both_count(self):
        original = Agent("orig", "cooperative", contribution=2.0,
                         content_digest="sha:abc")
        other = Agent("other", "cooperative", contribution=1.0,
                      content_digest="sha:def")
        result = sybil_duplicate_gate(original, other, seed=15)
        self.assertFalse(result["passes"])
        self.assertEqual(result["sybil_share"], 1.0)


class TestM199Closure(unittest.TestCase):

    def test_self_payment_loses_entire_spend(self):
        result = self_payment_wash_gate(sessions=10, demand=100.0,
                                        seed=21)
        self.assertTrue(result["passes"], result)
        self.assertEqual(result["stack_net"], -1000.0)
        self.assertEqual(result["own_arm_credit"], 0.0)
        self.assertEqual(result["baseline_net"], 0.0)

    def test_self_payment_bad_input_raises(self):
        with self.assertRaises(ValueError):
            self_payment_wash_gate(sessions=0, demand=100.0, seed=21)

    def test_dust_storm_loses_and_earns_no_liveness(self):
        for size in (1, 5, 50):
            result = dust_storm_gate(storm_size=size,
                                     min_session_fee=5.0, seed=22)
            self.assertTrue(result["passes"], (size, result))
            self.assertLess(result["net_with_fee"], 0.0)
            self.assertEqual(result["liveness_credit"], 0.0)
            self.assertEqual(result["storm_cost_with_fee"],
                             5.0 * size)

    def test_structural_cases_closed_by_construction(self):
        result = structural_form_checks(seed=23)
        self.assertTrue(result["passes"], result)
        self.assertEqual(result["front_run_edge"], 0.0)
        self.assertEqual(result["washer_control_of_dev_fund"], 0.0)


if __name__ == "__main__":
    unittest.main()
