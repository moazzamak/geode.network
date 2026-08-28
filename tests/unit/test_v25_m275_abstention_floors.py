"""M275 — per-arm abstention floors tests.

Unknown-INPUT queries abstain instead of guessing: an arm whose own
guard score exceeds its registered floor is excluded (hard, never
down-ranked); an arm WITH a floor but WITHOUT a score is excluded
(fail-closed); no arm_scores -> floors inactive (backwards
compatible). The floor itself comes from the registered
split-conformal order statistic of the arm's in-distribution scores.
"""
import unittest

from geode.core.arm import arm_from_sealed_head
from geode.core.guard_composition import abstention_floor_from_scores
from geode.core.router import Router


def _arm(name, acc=0.5, fp=None, **kw):
    return arm_from_sealed_head(name, "fam", 100, acc,
                                f"ev_{name}.json", fingerprint=fp, **kw)


class TestFloorRule(unittest.TestCase):
    def test_conformal_rank(self):
        scores = [1.0, 2.0, 3.0, 4.0]
        # rank = ceil(5 * 0.95) = 5 -> capped to 4 -> floor 4.0
        self.assertEqual(abstention_floor_from_scores(scores, 0.95),
                         4.0)
        # rank = ceil(5 * 0.75) = 4 -> floor 4.0
        self.assertEqual(abstention_floor_from_scores(scores, 0.75),
                         4.0)

    def test_empirical_quantile_undershoots_but_conformal_does_not(self):
        # 19 scores, coverage 0.95: empirical rank 18 (0.9474);
        # conformal rank ceil(20*0.95)=19 (0.95) — the registered
        # statistics lesson.
        scores = list(range(1, 20))
        floor = abstention_floor_from_scores(scores, 0.95)
        self.assertEqual(floor, 19.0)


class TestPerArmFloors(unittest.TestCase):
    def _router(self):
        r = Router()
        spec_a = _arm("a", fp=[1.0, 0.0])
        spec_a["abstention_floor"] = 2.0
        r.add_arm(spec_a)
        spec_b = _arm("b", fp=[0.0, 1.0])
        r.add_arm(spec_b)
        return r

    def test_above_floor_input_abstains_for_that_arm(self):
        r = self._router()
        out = r.route([1.0, 0.0], k=2, arm_scores={"a": 5.0,
                                                    "b": 1.0})
        self.assertEqual([rec["arm_id"] for rec in out], ["b"])

    def test_fail_closed_when_floor_arm_has_no_score(self):
        r = self._router()
        out = r.route([1.0, 0.0], k=2, arm_scores={"b": 1.0})
        # arm a has a floor but no score -> excluded; only b routes
        self.assertEqual([rec["arm_id"] for rec in out], ["b"])

    def test_floors_inactive_without_scores(self):
        r = self._router()
        out = r.route([1.0, 0.0], k=2)
        self.assertEqual({rec["arm_id"] for rec in out}, {"a", "b"})

    def test_chain_and_cold_start_honor_floors(self):
        r = self._router()
        chain = r.chain([1.0, 0.0], arm_scores={"a": 5.0, "b": 1.0})
        self.assertNotIn("a", [rec["arm_id"] for rec in chain])
        cold = r.cold_start(arm_scores={"a": 5.0, "b": 1.0})
        self.assertEqual(cold["arm_id"], "b")


if __name__ == "__main__":
    unittest.main()
