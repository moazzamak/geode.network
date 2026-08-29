"""M354 - per-session entropy in the route seed.

The regression this file exists for: without a session identifier
every seed field is constant within an epoch for a given task, so
the weighted lottery degenerates to a per-epoch winner-take-all.
The degenerate case is asserted first, against the same router,
so the test fails if the session field is quietly removed.
"""
from __future__ import annotations

import unittest

from geode.core.router_repair import RepairedRouter, draw_seed

FP = [1.0, 0.0]
EPOCH_ANCHOR = "epoch-0x8f3a"


def _router() -> RepairedRouter:
    router = RepairedRouter(price_floor=1.0)
    for arm_id, acc in (("leader", 0.70), ("follower1", 0.69),
                        ("follower2", 0.60)):
        router.add_arm({"arm_id": arm_id, "held_out_accuracy": acc,
                        "price": 1.0,
                        "availability": {"healthy": True}})
    return router


class TestWithoutSessionEntropy(unittest.TestCase):

    def test_fixed_epoch_anchor_is_winner_take_all(self) -> None:
        router = _router()
        winners = {router.route(FP, anchor=EPOCH_ANCHOR)[0]["arm_id"]
                   for _ in range(200)}
        self.assertEqual(len(winners), 1)


class TestSessionEntropy(unittest.TestCase):

    def test_distinct_sessions_spread_traffic(self) -> None:
        router = _router()
        counts: dict[str, int] = {}
        n = 3000
        for i in range(n):
            winner = router.route(FP, anchor=EPOCH_ANCHOR,
                                  session_id=f"session-{i}")[0]["arm_id"]
            counts[winner] = counts.get(winner, 0) + 1
        self.assertEqual(len(counts), 3)
        share = counts["leader"] / n
        self.assertGreater(share, 0.28)
        self.assertLess(share, 0.40)

    def test_the_same_session_replays_exactly(self) -> None:
        router = _router()
        first = router.route(FP, anchor=EPOCH_ANCHOR,
                             session_id="session-7")
        again = router.route(FP, anchor=EPOCH_ANCHOR,
                             session_id="session-7")
        self.assertEqual(first[0]["arm_id"], again[0]["arm_id"])
        self.assertEqual(first[0]["draw_seed"], again[0]["draw_seed"])

    def test_distinct_sessions_take_distinct_seeds(self) -> None:
        state = _router().state_root()
        self.assertNotEqual(
            draw_seed(EPOCH_ANCHOR, "t", state, FP, "session-1"),
            draw_seed(EPOCH_ANCHOR, "t", state, FP, "session-2"))

    def test_seed_carries_no_arm_identifier(self) -> None:
        """A host must not be able to bias its own draw."""
        state = _router().state_root()
        seed = draw_seed(EPOCH_ANCHOR, "t", state, FP, "session-1")
        self.assertNotIn("leader", seed)
        self.assertEqual(
            seed, draw_seed(EPOCH_ANCHOR, "t", state, FP, "session-1"))


if __name__ == "__main__":
    unittest.main()
