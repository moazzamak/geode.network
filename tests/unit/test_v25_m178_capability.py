"""Unit tests for M178: the capability map v0 and its monitoring rules.

Pure stdlib + the registered map content; no data, no GPU. Covers the
four registered rules: R-cap-cluster, R-transfer-spike, R-new-axis,
R-regression.
"""
from __future__ import annotations

import unittest

from geode.core.capability import (
    CAPABILITY_MAP_V0,
    RULE_CATALOG,
    rule_r_cap_cluster,
    rule_r_new_axis,
    rule_r_regression,
    rule_r_transfer_spike,
)


class TestCapabilityMap(unittest.TestCase):

    def test_map_has_four_nodes_and_three_edges(self):
        self.assertEqual(len(CAPABILITY_MAP_V0["nodes"]), 4)
        self.assertEqual(len(CAPABILITY_MAP_V0["edges"]), 3)

    def test_rule_catalog_lists_four_rules(self):
        self.assertEqual(len(RULE_CATALOG), 4)

    def test_r_cap_cluster_flags_cross_modality_cosine(self):
        task = {"modality": "next-token-text",
                "fingerprint": [0.2786, 0.0, 0.0]}  # vision profile vector
        vision_node = {"modality": "classification-vision",
                       "fingerprint": [0.2786, 0.0, 0.0]}
        flags = rule_r_cap_cluster(task, {
            "nodes": {"vision": vision_node}})
        self.assertIn("cap_cluster:vision", flags)

    def test_r_cap_cluster_silent_within_modality(self):
        task = {"modality": "classification-vision",
                "fingerprint": [0.2786, 0.0, 0.0]}
        vision_node = {"modality": "classification-vision",
                       "fingerprint": [0.2786, 0.0, 0.0]}
        self.assertEqual(rule_r_cap_cluster(task,
                                            {"nodes": {"v": vision_node}}),
                         [])

    def test_r_transfer_spike_flags_cross_family_low_gap(self):
        self.assertEqual(rule_r_transfer_spike(
            {"same_family": False, "gap_factor": 1.04}), ["transfer_spike"])

    def test_r_transfer_spike_silent_same_family(self):
        self.assertEqual(rule_r_transfer_spike(
            {"same_family": True, "gap_factor": 1.04}), [])

    def test_r_new_axis_flags_novel_kind(self):
        self.assertIn("new_axis", rule_r_new_axis(
            {"modality": "audio-sequence"}, CAPABILITY_MAP_V0))
        self.assertEqual(rule_r_new_axis(
            {"modality": "classification-vision"}, CAPABILITY_MAP_V0), [])

    def test_r_regression_flags_drift_only(self):
        re_measured = {"sparse_frontier": 0.2786 + 0.01}
        flags = rule_r_regression("domainnet32", re_measured,
                                  CAPABILITY_MAP_V0)
        self.assertIn("regression:sparse_frontier", flags)
        flags_ok = rule_r_regression(
            "domainnet32", {"sparse_frontier": 0.2786}, CAPABILITY_MAP_V0)
        self.assertEqual(flags_ok, [])

    def test_r_regression_unknown_node(self):
        self.assertEqual(rule_r_regression("nope", {},
                                           CAPABILITY_MAP_V0),
                         ["unknown_node"])


if __name__ == "__main__":
    unittest.main()
