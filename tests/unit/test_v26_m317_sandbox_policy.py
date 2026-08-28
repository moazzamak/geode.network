"""Unit tests for v26 M317 standard-library sandboxing (A24).

Registered semantics (plan §8.20): the pre-repair model exposes a key
path even under hash pinning; the post-repair model has none; uniform
terms are enforced.
"""
from __future__ import annotations

import copy
import unittest

from geode.core.sandbox_policy import (
    SANDBOX_TERMS,
    assert_uniform_terms,
    post_repair_reachable,
    pre_repair_reachable,
)


class TestReachabilityModel(unittest.TestCase):

    def test_pre_repair_path_exists(self):
        # M317-C1: the defect is demonstrated in the model
        self.assertTrue(pre_repair_reachable("cas_engine", "key"))

    def test_pre_repair_path_survives_hash_pinning(self):
        # trusted-by-hash runs the intended code, including its
        # intended bugs, on attacker-chosen input
        self.assertTrue(pre_repair_reachable("pinned_cas", "key"))

    def test_post_repair_no_path(self):
        # M317-C2: sandboxing on identical terms removes every path
        primitives = ["cas_engine", "pure_fn_engine", "torch_like"]
        self.assertFalse(post_repair_reachable(primitives, "key"))

    def test_post_repair_no_path_even_for_hosted_cas(self):
        self.assertFalse(post_repair_reachable(["cas_engine"], "key"))


class TestUniformTerms(unittest.TestCase):

    def test_registered_terms_match(self):
        uniform = dict(SANDBOX_TERMS)
        for primitive in ("cas_engine", "pure_fn_engine"):
            assert_uniform_terms([primitive], {primitive: uniform})

    def test_elevated_terms_rejected(self):
        # M317-C3: any primitive with elevated terms must be refused
        elevated = copy.deepcopy(SANDBOX_TERMS)
        elevated["settlement_key_access"] = True
        with self.assertRaises(ValueError):
            assert_uniform_terms(["cas_engine"],
                                 {"cas_engine": elevated})

    def test_missing_terms_rejected(self):
        with self.assertRaises(ValueError):
            assert_uniform_terms(["cas_engine"], {})

    def test_terms_deny_key_access(self):
        self.assertFalse(SANDBOX_TERMS["settlement_key_access"])
        self.assertTrue(SANDBOX_TERMS["memory_isolation"])


if __name__ == "__main__":
    unittest.main()
