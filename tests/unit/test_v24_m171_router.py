"""Unit tests for M171: the GEODE router (v24 Phase C).

Pure stdlib + the sealed artifact helpers; no data, no GPU. Covers the
registered MVP contract: deterministic nearest-arm routing, the
redundant-capability selection ordering, the failover chain, and the
cold-start fallback (I4).
"""
from __future__ import annotations

import unittest

from geode.core.router import Router


def _arm(arm_id: str, fp=None, kind="regression", dim=1, acc=0.0,
         healthy=True, price=0.0, general=False, primitive=False):
    return {
        "arm_id": arm_id,
        "fingerprint": fp or [],
        "output_contract": {"kind": kind, "dim": dim},
        "held_out_accuracy": {arm_id: acc},
        "selection_accuracy": acc,
        "availability": {"contract_hash": "c1", "payload_hash": "p1",
                         "healthy": healthy},
        "price": price,
        "general": general,
        "primitive": primitive,
    }


class TestRouterRoute(unittest.TestCase):

    def test_nearest_arm_by_cosine(self):
        r = Router()
        r.add_arm(_arm("a", fp=[1.0, 0.0], acc=0.5))
        r.add_arm(_arm("b", fp=[0.0, 1.0], acc=0.9))
        r.add_arm(_arm("c", fp=[-1.0, 0.0], acc=0.7))
        top = r.route([1.0, 0.1], k=3)
        self.assertEqual([t["arm_id"] for t in top], ["a", "b", "c"])

    def test_tie_break_selection_score_accuracy_first(self):
        r = Router()
        r.add_arm(_arm("lo", fp=[1.0, 0.0], acc=0.3))
        r.add_arm(_arm("hi", fp=[1.0, 0.0], acc=0.8))
        top = r.route([1.0, 0.0], k=2)
        self.assertEqual([t["arm_id"] for t in top], ["hi", "lo"])

    def test_tie_break_availability_then_price(self):
        r = Router()
        r.add_arm(_arm("unhealthy", fp=[1.0], acc=0.8, healthy=False))
        r.add_arm(_arm("expensive", fp=[1.0], acc=0.8, price=9.0))
        r.add_arm(_arm("cheap", fp=[1.0], acc=0.8, price=1.0))
        top = r.route([1.0], k=3)
        self.assertEqual([t["arm_id"] for t in top],
                         ["cheap", "expensive", "unhealthy"])

    def test_determinism_identical_repeat(self):
        r = Router()
        for i in range(5):
            r.add_arm(_arm(f"a{i}", fp=[float(i), 1.0], acc=0.1 * i))
        self.assertEqual(r.route([0.5, 0.5], k=5), r.route([0.5, 0.5], k=5))
        self.assertEqual(r.chain([0.5, 0.5]), r.chain([0.5, 0.5]))

    def test_add_arm_validation(self):
        r = Router()
        with self.assertRaises(ValueError):
            r.add_arm({"arm_id": "x"})
        r.add_arm(_arm("ok"))
        r.add_arm(_arm("ok"))  # idempotent
        self.assertEqual(len(r.list_arms()), 1)


class TestRouterChain(unittest.TestCase):

    def test_chain_order_and_primitives_bottom(self):
        r = Router()
        r.add_arm(_arm("specialist", fp=[1.0, 0.0], acc=0.9))
        r.add_arm(_arm("general", acc=0.4, general=True))
        r.add_arm(_arm("primitive", acc=0.0, primitive=True))
        chain = r.chain([1.0, 0.0])
        self.assertEqual([c["arm_id"] for c in chain],
                         ["specialist", "general", "primitive"])

    def test_unhealthy_skipped_in_chain(self):
        r = Router()
        r.add_arm(_arm("sick", fp=[1.0, 0.0], acc=0.9, healthy=False))
        r.add_arm(_arm("well", fp=[1.0, 0.0], acc=0.8))
        r.add_arm(_arm("primitive", acc=0.0, primitive=True))
        chain = r.chain([1.0, 0.0])
        self.assertEqual([c["arm_id"] for c in chain], ["well", "primitive"])

    def test_general_tier_ordered_by_selection_score(self):
        r = Router()
        r.add_arm(_arm("g_lo", acc=0.2, general=True))
        r.add_arm(_arm("g_hi", acc=0.6, general=True))
        chain = r.chain([1.0, 0.0])
        self.assertEqual([c["arm_id"] for c in chain], ["g_hi", "g_lo"])


class TestRouterColdStart(unittest.TestCase):
    def test_strongest_general_by_kind_then_overall(self):
        r = Router()
        r.add_arm(_arm("g_reg", acc=0.5, general=True, kind="regression"))
        r.add_arm(_arm("g_cls", acc=0.7, general=True, kind="classification"))
        self.assertEqual(r.cold_start("classification")["arm_id"], "g_cls")
        self.assertEqual(r.cold_start("regression")["arm_id"], "g_reg")
        self.assertEqual(r.cold_start()["arm_id"], "g_cls")

    def test_no_general_falls_back_to_specialist_then_primitive(self):
        r = Router()
        r.add_arm(_arm("primitive", acc=0.0, primitive=True))
        r.add_arm(_arm("specialist", fp=[1.0], acc=0.4))
        self.assertEqual(r.cold_start()["arm_id"], "specialist")
        r2 = Router()
        r2.add_arm(_arm("primitive", acc=0.0, primitive=True))
        self.assertEqual(r2.cold_start()["arm_id"], "primitive")

    def test_empty_registry_returns_none(self):
        self.assertEqual(Router().cold_start(), {})


class TestRouterContractGuard(unittest.TestCase):
    """M175 cell C: a task's contract excludes wrong-modality arms."""

    def _two_kind_registry(self) -> Router:
        r = Router()
        r.add_arm(_arm("vis_a", fp=[1.0, 0.0], acc=0.5,
                       kind="classification-vision"))
        r.add_arm(_arm("vis_b", fp=[0.5, 0.5], acc=0.7,
                       kind="classification-vision"))
        r.add_arm(_arm("txt_a", fp=[1.0, 0.0], acc=0.6,
                       kind="next-token-text"))
        r.add_arm(_arm("g_vis", acc=0.3, general=True,
                       kind="classification-vision"))
        r.add_arm(_arm("g_txt", acc=0.4, general=True,
                       kind="next-token-text"))
        r.add_arm(_arm("prim", acc=0.0, primitive=True))
        return r

    def test_route_filters_by_contract(self):
        r = self._two_kind_registry()
        top = r.route([1.0, 0.0], k=10, contract_kind="next-token-text")
        kinds = {t["output_contract"]["kind"] for t in top}
        self.assertEqual(kinds, {"next-token-text"})

    def test_chain_all_tiers_filtered_by_contract(self):
        r = self._two_kind_registry()
        chain = r.chain([1.0, 0.0], contract_kind="next-token-text")
        for arm in chain:
            self.assertEqual(arm["output_contract"]["kind"],
                             "next-token-text")

    def test_cross_contract_query_never_yields_wrong_kind(self):
        # vision fingerprint queried with a text contract: only text arms
        # may appear, never a vision arm (the guard, not the cosine, rules).
        r = self._two_kind_registry()
        chain = r.chain([1.0, 0.0], contract_kind="next-token-text")
        ids = [a["arm_id"] for a in chain]
        self.assertNotIn("vis_a", ids)
        self.assertNotIn("vis_b", ids)
        self.assertTrue(all(
            a["output_contract"]["kind"] == "next-token-text"
            for a in chain))

    def test_no_contract_is_backward_compatible(self):
        r = self._two_kind_registry()
        top = r.route([1.0, 0.0], k=10)
        self.assertEqual(len(top), 3)  # the three fp-carrying arms


class TestRouterHash(unittest.TestCase):

    def test_content_hash_deterministic_and_sensitive(self):
        r1, r2 = Router(), Router()
        r1.add_arm(_arm("a", fp=[1.0], acc=0.3))
        r2.add_arm(_arm("a", fp=[1.0], acc=0.3))
        self.assertEqual(r1.content_hash(), r2.content_hash())
        r2.add_arm(_arm("b", fp=[2.0], acc=0.4))
        self.assertNotEqual(r1.content_hash(), r2.content_hash())


if __name__ == "__main__":
    unittest.main()
