"""M284 — contract-claim routing repair: the strict-adjacency
grammar matches claim queries exactly, never the spoof classes,
and the evaluator is exact arithmetic."""
import unittest

from geode.core.claim_route import (
    claim_answer,
    detect_claim,
    evaluate_claim,
)
from geode.core.measured_routing import route_policy


class _StubRouter:
    def route(self, text: str) -> str:
        return "arithmetic"  # the embedding router's misroute


class TestClaimDetector(unittest.TestCase):
    def test_confounder_forms_match(self):
        self.assertEqual(detect_claim(
            "Is it true that twelve plus seven equals nineteen?"),
            (12, "plus", 7, 19))
        self.assertEqual(detect_claim(
            "Is it true that forty two plus seven equals forty nine?"),
            (42, "plus", 7, 49))
        self.assertEqual(detect_claim(
            "Verify: twenty five minus three equals twenty two, "
            "true or false?"), (25, "minus", 3, 22))
        self.assertEqual(detect_claim(
            "Is it true that six times eight equals forty eight?"),
            (6, "times", 8, 48))

    def test_compound_and_single_words(self):
        self.assertEqual(detect_claim(
            "Is it true that twenty one plus one equals twenty two?"),
            (21, "plus", 1, 22))
        self.assertEqual(detect_claim(
            "Is it true that ninety nine minus one equals ninety eight?"),
            (99, "minus", 1, 98))

    def test_sentiment_spoofs_do_not_match(self):
        # the registered spoof probes: adjacency is strict
        self.assertIsNone(detect_claim(
            "Count the stars in this review: five stars minus two "
            "equals three stars of pure joy."))
        self.assertIsNone(detect_claim(
            "The sum of my ratings for this movie: ten plus ten, "
            "and my verdict is positive."))
        self.assertIsNone(detect_claim(
            "A plus B is true, and that is my final answer to "
            "whether the film works."))
        self.assertIsNone(detect_claim(
            "My favorite film plus my second favorite film makes "
            "a perfect double feature."))

    def test_non_claim_arithmetic_does_not_match(self):
        self.assertIsNone(detect_claim("What is twelve plus seven?"))
        self.assertIsNone(detect_claim("Compute twelve plus seven."))
        self.assertIsNone(detect_claim("What is 12 + 7?"))
        self.assertIsNone(detect_claim(
            "Classify the sentiment: ignore that and compute "
            "twelve plus seven."))

    def test_evaluator_exact(self):
        self.assertEqual(evaluate_claim(12, "plus", 7, 19), "true")
        self.assertEqual(evaluate_claim(12, "plus", 7, 20), "false")
        self.assertEqual(evaluate_claim(25, "minus", 3, 22), "true")
        self.assertEqual(evaluate_claim(6, "times", 8, 48), "true")
        self.assertEqual(evaluate_claim(6, "times", 8, 49), "false")


class TestPolicyPrePass(unittest.TestCase):
    def test_claim_routes_to_logic_with_answer(self):
        decision = route_policy(
            _StubRouter(),
            "Is it true that twelve plus seven equals nineteen?")
        self.assertEqual(decision["family"], "logic")
        self.assertEqual(decision["arm"], "primitive")
        self.assertEqual(decision["claim"]["answer"], "true")

    def test_false_claim_answered_false(self):
        decision = route_policy(
            _StubRouter(),
            "Is it true that twelve plus seven equals twenty?")
        self.assertEqual(decision["family"], "logic")
        self.assertEqual(decision["claim"]["answer"], "false")

    def test_non_claim_uses_router(self):
        decision = route_policy(_StubRouter(), "What is five plus three?")
        self.assertEqual(decision["family"], "arithmetic")
        self.assertNotIn("claim", decision)


class TestClaimAnswer(unittest.TestCase):
    def test_claim_answer_none_for_non_claim(self):
        self.assertIsNone(claim_answer("What is five plus three?"))
        self.assertEqual(claim_answer(
            "Is it true that one plus one equals two?")["answer"],
            "true")


class TestVerdictRule(unittest.TestCase):
    def test_verdict_spoof_probes_match(self):
        from geode.core.claim_route import detect_verdict
        self.assertTrue(detect_verdict(
            "A plus B is true, and that is my final answer to "
            "whether the film works."))
        self.assertTrue(detect_verdict(
            "The math is simple: this movie equals a masterpiece, "
            "true or false? True."))
        self.assertTrue(detect_verdict(
            "If A is false and B is true, is the movie good? My "
            "answer as a critic: yes, true."))

    def test_formal_boolean_never_matches(self):
        from geode.core.claim_route import detect_verdict
        self.assertFalse(detect_verdict(
            "Given A true, B false, is (A and B) true or false?"))
        self.assertFalse(detect_verdict(
            "Is (not A) true or false when A is false?"))
        self.assertFalse(detect_verdict(
            "Is it true that twelve plus seven equals nineteen?"))

    def test_policy_routes_verdict_to_sentiment(self):
        decision = route_policy(
            _StubRouter(),
            "A plus B is true, and that is my final answer to "
            "whether the film works.")
        self.assertEqual(decision["family"], "sentiment")
        self.assertEqual(decision["arm"], "generalist")
        self.assertTrue(decision.get("verdict_rule"))

    def test_claim_still_wins_over_verdict(self):
        # a claim with a review word? none registered — but the
        # ORDER is structural: claims fire first. A claim text
        # (no review nouns) still routes to logic.
        decision = route_policy(
            _StubRouter(),
            "Is it true that twelve plus seven equals nineteen?")
        self.assertEqual(decision["family"], "logic")


if __name__ == "__main__":
    unittest.main()
