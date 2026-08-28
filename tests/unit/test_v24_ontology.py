"""Tests for the frozen ontology v0 and its consistency with the code."""
from __future__ import annotations

import unittest

from geode.core.ontology import check_consistency, load_ontology


class TestOntologyV0(unittest.TestCase):

    def test_frozen_artifact_loads(self):
        onto = load_ontology()
        self.assertEqual(onto["version"], "v0")
        self.assertIn("axes", onto)
        self.assertIn("similarity_positive_controls", onto)

    def test_consistent_with_code_schema(self):
        self.assertEqual(check_consistency(), [])

    def test_similar_and_dissimilar_disjoint(self):
        onto = load_ontology()
        similar = {tuple(p) for p in
                   onto["similarity_positive_controls"]["known_similar"]}
        dissimilar = {tuple(p) for p in
                      onto["similarity_positive_controls"]["known_dissimilar"]}
        self.assertFalse(similar & dissimilar)

    def test_llm_error_corrected(self):
        onto = load_ontology()
        similar = onto["similarity_positive_controls"]["known_similar"]
        pair = ["CIFAR-10 image classification", "Mackey-Glass forecasting"]
        self.assertNotIn(pair, similar)


if __name__ == "__main__":
    unittest.main()
