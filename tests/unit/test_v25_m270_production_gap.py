"""M270 — production-gap implementation (M264 spec) tests.

C1 cache (containment-first), C2 canary determinism, C3 streaming
records, C4 signed requests, C6 license field. The mechanisms are
prior art; what is tested here is their deterministic, ledger-audited
composition (see the spec's section 10).
"""
import unittest

from geode.core.arm import arm_from_sealed_head, validate_arm_spec
from geode.core.auth import (
    generate_keypair,
    sign_request,
    verify_request,
)
from geode.core.orchestrator import Orchestrator


def _arm(name, acc=0.5, fp=None, **kw):
    return arm_from_sealed_head(name, "fam", 100, acc,
                                f"ev_{name}.json", fingerprint=fp, **kw)


class _Frozen:
    """A frozen registry stub (M248 containment)."""

    def is_frozen(self, as_of_index):
        return True


class TestC6LicenseField(unittest.TestCase):
    def test_builders_default_to_empty_license(self):
        spec = _arm("a")
        self.assertEqual(spec["license"],
                         {"code": "", "weights": "", "data": ""})
        self.assertEqual(validate_arm_spec(spec), [])

    def test_missing_license_rejected(self):
        spec = _arm("a")
        del spec["license"]
        reasons = validate_arm_spec(spec)
        self.assertTrue(any("license" in r for r in reasons))

    def test_custom_license_recorded(self):
        lic = {"code": "Apache-2.0", "weights": "MIT", "data": "CC-BY-4.0"}
        spec = _arm("a", license=lic)
        self.assertEqual(spec["license"], lic)
        self.assertEqual(validate_arm_spec(spec), [])

    def test_malformed_license_rejected(self):
        spec = _arm("a")
        spec["license"] = {"code": "MIT"}
        self.assertTrue(validate_arm_spec(spec))


class TestC1Cache(unittest.TestCase):
    def test_hit_returns_stored_decision_and_records(self):
        orch = Orchestrator()
        orch.register(_arm("a", acc=0.9))
        cache = {}
        first = orch.serve("q1", [], cache=cache)
        self.assertEqual(first[0]["arm_id"], "a")
        records = orch.ledger.to_dict()["records"]
        kinds = [r["content"]["kind"] for r in records]
        self.assertIn("cache_store", kinds)
        again = orch.serve("q2", [], cache=cache)
        self.assertEqual(again[0]["arm_id"], "a")
        kinds = [r["content"]["kind"]
                 for r in orch.ledger.to_dict()["records"]]
        self.assertIn("cache_hit", kinds)
        self.assertEqual(orch.chain_verify()["ok"], True)

    def test_containment_never_caches(self):
        orch = Orchestrator()
        orch.register(_arm("a", acc=0.9))
        cache = {}
        routed = orch.serve("q1", [], cache=cache, freeze=_Frozen(),
                            as_of_index=0)
        self.assertEqual(routed, [])
        kinds = [r["content"]["kind"]
                 for r in orch.ledger.to_dict()["records"]]
        self.assertNotIn("cache_store", kinds)
        self.assertNotIn("cache_hit", kinds)

    def test_registry_change_invalidates(self):
        orch = Orchestrator()
        orch.register(_arm("a", acc=0.9))
        cache = {}
        orch.serve("q1", [], cache=cache)
        orch.register(_arm("b", acc=0.95))
        third = orch.serve("q2", [], cache=cache)
        kinds = [r["content"]["kind"]
                 for r in orch.ledger.to_dict()["records"]]
        self.assertEqual(kinds.count("cache_store"), 2)
        self.assertNotIn("cache_hit", kinds)
        self.assertEqual(third[0]["arm_id"], "b")


class TestC2Canary(unittest.TestCase):
    def test_bucket_deterministic(self):
        fp = [0.1, 0.2, 0.3]
        b1 = Orchestrator.rollout_bucket(fp, "g", "v1")
        b2 = Orchestrator.rollout_bucket(list(fp), "g", "v1")
        self.assertEqual(b1, b2)
        self.assertEqual(b1, Orchestrator.rollout_bucket(fp, "g", "v1"))
        self.assertTrue(0 <= b1 < 1_000_000)

    def test_policy_remaps_within_permille(self):
        orch = Orchestrator()
        orch.register(_arm("stable", acc=0.8, fp=[1.0]))
        orch.register(_arm("canary", acc=0.1, fp=[1.0]))
        orch.register_rollout_policy("v1", "g1", "stable", "canary",
                                     1_000_000)
        routed = orch.serve("q1", [1.0])
        self.assertEqual(routed[0]["arm_id"], "canary")
        kinds = [r["content"]["kind"]
                 for r in orch.ledger.to_dict()["records"]]
        self.assertIn("rollout", kinds)
        self.assertIn("rollout_policy", kinds)
        self.assertEqual(orch.chain_verify()["ok"], True)

    def test_zero_permille_keeps_stable_and_records(self):
        orch = Orchestrator()
        orch.register(_arm("stable", acc=0.8, fp=[1.0]))
        orch.register(_arm("canary", acc=0.1, fp=[1.0]))
        orch.register_rollout_policy("v1", "g1", "stable", "canary", 0)
        routed = orch.serve("q1", [1.0])
        self.assertEqual(routed[0]["arm_id"], "stable")
        rollouts = [r["content"] for r in orch.ledger.to_dict()["records"]
                    if r["content"]["kind"] == "rollout"]
        self.assertEqual(len(rollouts), 1)
        self.assertEqual(rollouts[0]["effective_arm_id"], "stable")

    def test_unknown_arms_rejected(self):
        orch = Orchestrator()
        orch.register(_arm("stable", acc=0.8, fp=[1.0]))
        with self.assertRaises(ValueError):
            orch.register_rollout_policy("v1", "g1", "stable", "ghost",
                                         500)


class TestC3Streaming(unittest.TestCase):
    def test_stream_sequence_replays(self):
        orch = Orchestrator()
        orch.register(_arm("a"))
        orch.serve("q1", [])
        orch.stream_begin("q1", route_record_index=1, seed="s42")
        orch.stream_chunk("q1", 0, "h0")
        orch.stream_chunk("q1", 1, "h1")
        orch.stream_end("q1", total_chunks=2, final_payload_hash="hf",
                        status="complete")
        kinds = [r["content"]["kind"]
                 for r in orch.ledger.to_dict()["records"]]
        self.assertEqual(kinds, ["arm_register", "route", "stream_begin",
                                 "stream_chunk", "stream_chunk",
                                 "stream_end"])
        self.assertEqual(orch.chain_verify()["ok"], True)

    def test_aborted_stream_is_terminal(self):
        orch = Orchestrator()
        orch.stream_begin("q2", 0, "s")
        orch.stream_end("q2", 0, "", "aborted")
        records = orch.ledger.to_dict()["records"]
        self.assertEqual(records[-1]["content"]["status"], "aborted")

    def test_bad_status_rejected(self):
        orch = Orchestrator()
        with self.assertRaises(ValueError):
            orch.stream_end("q3", 0, "", "meh")


class TestC4Auth(unittest.TestCase):
    def test_roundtrip(self):
        kp = generate_keypair()
        sig = sign_request(kp["private_key"], "POST", "/serve", "ph", "n1",
                           0, 9_999_999_999)
        verdict = verify_request(kp["public_key"], "POST", "/serve", "ph",
                                 "n1", 0, 9_999_999_999, sig)
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["outcome"], "ok")

    def test_tampered_path_fails(self):
        kp = generate_keypair()
        sig = sign_request(kp["private_key"], "POST", "/serve", "ph", "n1",
                           0, 9_999_999_999)
        verdict = verify_request(kp["public_key"], "POST", "/other", "ph",
                                 "n1", 0, 9_999_999_999, sig)
        self.assertEqual(verdict["outcome"], "bad_signature")

    def test_replayed_nonce(self):
        kp = generate_keypair()
        sig = sign_request(kp["private_key"], "POST", "/serve", "ph", "n1",
                           0, 9_999_999_999)
        store = {}
        first = verify_request(kp["public_key"], "POST", "/serve", "ph",
                               "n1", 0, 9_999_999_999, sig, store)
        self.assertTrue(first["ok"])
        replay = verify_request(kp["public_key"], "POST", "/serve", "ph",
                                "n1", 0, 9_999_999_999, sig, store)
        self.assertEqual(replay["outcome"], "replayed_nonce")

    def test_expired_window(self):
        kp = generate_keypair()
        sig = sign_request(kp["private_key"], "POST", "/serve", "ph", "n1",
                           100, 200)
        verdict = verify_request(kp["public_key"], "POST", "/serve", "ph",
                                 "n1", 100, 200, sig, now=500)
        self.assertEqual(verdict["outcome"], "expired")

    def test_requester_recorded_but_never_routes(self):
        orch = Orchestrator()
        orch.register(_arm("a", acc=0.5))
        orch.serve("q1", [], requester="tenant-7")
        route = orch.ledger.get("route:q1")["content"]
        self.assertEqual(route["requester"], "tenant-7")
        orch2 = Orchestrator()
        orch2.register(_arm("a", acc=0.5))
        orch2.serve("q1", [])
        self.assertEqual(route["chosen"],
                         orch2.ledger.get("route:q1")["content"]["chosen"])


if __name__ == "__main__":
    unittest.main()
