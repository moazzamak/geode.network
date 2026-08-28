"""M272 — measured routing module tests.

The policy under test: the embedding nearest-centroid router
(cell-4 rule, 0 misroutes) + the measured per-type arm rules.
Determinism and the tie rule are structural; the full cell-4
reproduction is the evidence runner, not this file.
"""
import unittest

import numpy as np

from geode.core.measured_routing import (
    EmbeddingRouter,
    FAMILY_ORDER,
    FAMILY_SAMPLES,
    MEASURED_ARM_RULES,
    route_policy,
)


def _fake_embedder(texts):
    """Deterministic keyword embedding: sentiment texts mention
    'movie/film/review', arithmetic texts mention digits words,
    logic texts mention 'true/false'."""
    if isinstance(texts, str):
        texts = [texts]
    out = []
    for text in texts:
        t = text.lower()
        vec = np.zeros(3)
        if any(w in t for w in ("movie", "film", "review", "cinema")):
            vec[0] = 1.0
        if any(w in t for w in ("what is", "compute", "plus",
                                "multiply", "minus")):
            vec[1] = 1.0
        if "true" in t or "false" in t:
            vec[2] = 1.0
        out.append(vec)
    return np.array(out, dtype=np.float64)


class TestMeasuredRules(unittest.TestCase):
    def test_rules_carry_measured_numbers(self):
        self.assertEqual(MEASURED_ARM_RULES["sentiment"]["arm"],
                         "generalist")
        self.assertEqual(MEASURED_ARM_RULES["arithmetic"]["arm"],
                         "primitive")
        self.assertEqual(MEASURED_ARM_RULES["logic"]["arm"],
                         "primitive")
        self.assertEqual(MEASURED_ARM_RULES["code"]["arm"], "coder")
        for family, rule in MEASURED_ARM_RULES.items():
            self.assertIn("source", rule)

    def test_centroid_fit_deterministic(self):
        r1 = EmbeddingRouter(_fake_embedder)
        r2 = EmbeddingRouter(_fake_embedder)
        for family in FAMILY_ORDER:
            self.assertTrue(np.allclose(r1.centroids[family],
                                        r2.centroids[family]))


class TestEmbeddingRouter(unittest.TestCase):
    def test_routes_by_family(self):
        router = EmbeddingRouter(_fake_embedder)
        self.assertEqual(
            router.route("The film was wonderful and the review is good"),
            "sentiment")
        self.assertEqual(router.route("What is five plus three?"),
                         "arithmetic")
        self.assertEqual(
            router.route("Suppose A is true. Is A or B true?"), "logic")

    def test_policy_assigns_measured_arm(self):
        router = EmbeddingRouter(_fake_embedder)
        decision = route_policy(router, "What is five plus three?")
        self.assertEqual(decision["family"], "arithmetic")
        self.assertEqual(decision["arm"], "primitive")
        decision = route_policy(router, "The movie was great")
        self.assertEqual(decision["arm"], "generalist")

    def test_unmeasured_family_defaults_to_generalist(self):
        router = EmbeddingRouter(_fake_embedder)
        decision = route_policy(router, "some entirely unknown text")
        self.assertEqual(decision["arm"], "generalist")

    def test_family_samples_frozen(self):
        self.assertEqual(list(FAMILY_SAMPLES),
                         FAMILY_ORDER)
        for family, samples in FAMILY_SAMPLES.items():
            self.assertEqual(len(samples), 5)


if __name__ == "__main__":
    unittest.main()
