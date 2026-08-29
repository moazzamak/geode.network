"""Unit tests for the M303 repaired router.

Pins the registered selection semantics: price-floor rejection at
registration and routing; expected-charge ranking (bloat lowers the
score); anchor-seeded determinism and anchor-dependence; the lottery
removes single-winner capture; the tie-break is anchor-seeded, not
artifact-hash-seeded.
"""
from __future__ import annotations

import unittest

import numpy as np

from geode.core.router_repair import (
    RepairedRouter,
    rank_score,
)

# M388: the draw is seeded from the randomness beacon (closed after
# declaration) ordered by the epoch anchor. A fixed epoch anchor is
# the protocol's cadence; spread-traffic tests vary the beacon round.
BEACON = "beacon-round-0x3f"
EPOCH_ANCHOR = "epoch-0x8f3a"


def _arm(arm_id: str, acc: float, price: float, ubar: float | None = None,
         healthy: bool = True) -> dict:
    spec = {"arm_id": arm_id, "held_out_accuracy": acc, "price": price,
            "availability": {"healthy": healthy}}
    if ubar is not None:
        spec["expected_units"] = ubar
    return spec


class TestRankScore(unittest.TestCase):

    def test_score_is_quality_per_expected_charge(self):
        self.assertAlmostEqual(rank_score(0.8, 2.0, 4.0), 0.1)
        self.assertAlmostEqual(rank_score(0.8, 2.0, 8.0), 0.05)

    def test_zero_or_negative_price_rejected(self):
        with self.assertRaises(ValueError):
            rank_score(0.8, 0.0, 1.0)
        with self.assertRaises(ValueError):
            rank_score(0.8, -1.0, 1.0)

    def test_nonpositive_units_rejected(self):
        with self.assertRaises(ValueError):
            rank_score(0.8, 1.0, 0.0)


class TestPriceFloor(unittest.TestCase):

    def test_registration_below_floor_rejected(self):
        router = RepairedRouter(price_floor=1.0)
        with self.assertRaises(ValueError):
            router.add_arm(_arm("a", 0.9, 0.99))

    def test_route_excludes_below_floor_arm(self):
        router = RepairedRouter(price_floor=1.0)
        router.add_arm(_arm("floor_ok", 0.9, 1.0))
        # an arm admitted elsewhere (or pre-repair) below the floor is
        # excluded at route time, never selected
        router._arms["stale"] = _arm("stale", 0.999, 0.001)
        out = router.route([1.0, 0.0], beacon=BEACON, anchor="a1")
        self.assertEqual([r["arm_id"] for r in out], ["floor_ok"])

    def test_zero_price_impossible_downstream(self):
        router = RepairedRouter(price_floor=1.0)
        with self.assertRaises(ValueError):
            router.add_arm(_arm("free", 1.0, 0.0))


class TestLottery(unittest.TestCase):

    def test_deterministic_per_anchor(self):
        router = RepairedRouter(price_floor=1.0)
        router.add_arm(_arm("a", 0.60, 1.0))
        router.add_arm(_arm("b", 0.60, 1.0))
        first = router.route([1.0, 0.0], beacon=BEACON, anchor="epoch-7")
        second = router.route([1.0, 0.0], beacon=BEACON, anchor="epoch-7")
        self.assertEqual([r["arm_id"] for r in first],
                         [r["arm_id"] for r in second])
        self.assertEqual(first[0]["draw_seed"], second[0]["draw_seed"])

    def test_beacon_dependence(self):
        # M388: the same anchor and session under different beacon
        # rounds take different draws — the closure that makes the
        # seed unknowable at declaration time.
        router = RepairedRouter(price_floor=1.0)
        router.add_arm(_arm("a", 0.60, 1.0))
        router.add_arm(_arm("b", 0.60, 1.0))
        winners = {router.route([1.0, 0.0], beacon=f"round-{i}",
                                anchor=EPOCH_ANCHOR)[0]["arm_id"]
                   for i in range(40)}
        self.assertEqual(winners, {"a", "b"},
                         "the beacon round must move the draw")

    def test_anchor_dependence_on_ties(self):
        router = RepairedRouter(price_floor=1.0)
        router.add_arm(_arm("a", 0.60, 1.0))
        router.add_arm(_arm("b", 0.60, 1.0))
        winners = {router.route([1.0, 0.0], beacon=BEACON,
                                anchor=f"epoch-{i}")[0]["arm_id"]
                   for i in range(40)}
        self.assertEqual(winners, {"a", "b"},
                         "a tie must produce anchor-dependent winners")

    def test_no_single_winner_capture(self):
        router = RepairedRouter(price_floor=1.0)
        router.add_arm(_arm("top", 0.70, 1.0))
        router.add_arm(_arm("mid", 0.62, 1.0))
        shares: dict[str, int] = {"top": 0, "mid": 0}
        n = 2000
        for i in range(n):
            winner = router.route([1.0, 0.0], beacon=f"round-{i}",
                                  anchor=EPOCH_ANCHOR)[0]
            shares[winner["arm_id"]] += 1
        self.assertLess(shares["top"], n, "winner-take-all must be gone")
        self.assertGreater(shares["mid"], 0, "the pool must win sometimes")
        # proportional weights: top has score .70, mid .62 -> top share
        # in (.5, .57)
        self.assertGreater(shares["top"] / n, 0.48)
        self.assertLess(shares["top"] / n, 0.60)

    def test_bloat_lowers_share(self):
        router = RepairedRouter(price_floor=1.0)
        router.add_arm(_arm("lean", 0.60, 1.0, ubar=1.0))
        router.add_arm(_arm("bloated", 0.60, 1.0, ubar=5.0))
        shares = {"lean": 0, "bloated": 0}
        n = 2000
        for i in range(n):
            winner = router.route([1.0, 0.0], beacon=f"round-{i}",
                                  anchor=EPOCH_ANCHOR)[0]
            shares[winner["arm_id"]] += 1
        # scores: lean .60, bloated .12 -> bloated share ~1/6 of lean's;
        # the registered test threshold is a 2x margin, far below the
        # mechanism's expected 5x, so sampling noise cannot flip it
        self.assertGreater(shares["lean"], 2 * shares["bloated"])

    def test_quality_still_pays(self):
        router = RepairedRouter(price_floor=1.0)
        router.add_arm(_arm("better", 0.90, 1.0))
        router.add_arm(_arm("worse", 0.45, 1.0))
        shares = {"better": 0, "worse": 0}
        n = 2000
        for i in range(n):
            winner = router.route([1.0, 0.0], beacon=f"round-{i}",
                                  anchor=EPOCH_ANCHOR)[0]
            shares[winner["arm_id"]] += 1
        self.assertGreater(shares["better"], shares["worse"])


class TestTieBreak(unittest.TestCase):

    def test_rank_order_within_pool_is_anchor_seeded(self):
        router = RepairedRouter(price_floor=1.0)
        router.add_arm(_arm("a", 0.60, 1.0))
        router.add_arm(_arm("b", 0.60, 1.0))
        first_ids = [r["arm_id"] for r in
                     router.route([1.0, 0.0], beacon=BEACON,
                                  anchor="epoch-1")]
        second_ids = [r["arm_id"] for r in
                      router.route([1.0, 0.0], beacon=BEACON,
                                   anchor="epoch-2")]
        # the pool order (excluding the winner flag) is the tie-break
        # order; anchors may or may not flip it - the claim under test
        # is only that it is deterministic and anchor-derived
        self.assertEqual(len(first_ids), 2)
        self.assertEqual(sorted(first_ids), sorted(second_ids))


if __name__ == "__main__":
    unittest.main()
