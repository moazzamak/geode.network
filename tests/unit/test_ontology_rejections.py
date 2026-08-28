"""Unit tests for the ontology consistency checker's rejection
branches (the frozen artifact is exercised separately)."""
import unittest
from unittest import mock

from geode.core import ontology
from geode.core.ontology import check_consistency


class TestConsistencyRejections(unittest.TestCase):
    def _onto_with(self, **changes):
        onto = ontology.load_ontology()
        onto.update(changes)
        return onto

    def test_axes_mismatch_detected(self):
        onto = self._onto_with(axes=["wrong"])
        with mock.patch.object(ontology, "load_ontology",
                               return_value=onto):
            problems = check_consistency()
        self.assertIn("axes mismatch", problems[0])

    def test_continuous_axis_outside_schema_detected(self):
        onto = self._onto_with(continuous_axes={"not_an_axis": [0, 1]})
        with mock.patch.object(ontology, "load_ontology",
                               return_value=onto):
            problems = check_consistency()
        self.assertTrue(any("not in AXES" in p for p in problems))

    def test_overlapping_positive_controls_detected(self):
        onto = ontology.load_ontology()
        similar = [list(onto["similarity_positive_controls"]
                        ["known_similar"][0])]
        onto["similarity_positive_controls"] = {
            "known_similar": similar,
            "known_dissimilar": similar,
        }
        with mock.patch.object(ontology, "load_ontology",
                               return_value=onto):
            problems = check_consistency()
        self.assertTrue(any("appears in both" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
