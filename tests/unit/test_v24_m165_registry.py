"""Unit tests for M165: the task registry, descriptor normaliser, and
frozen artifact format (v24 Phase A). Pure stdlib + the sealed artifact
helpers; no data, no GPU.
"""
from __future__ import annotations

import unittest

from geode.core.descriptor import AXES, FALLBACK_TOKEN, normalise
from geode.core.registry import TaskRegistry


class TestNormaliser(unittest.TestCase):

    def test_determinism_across_key_order(self):
        a = {"output.kind": "class", "input.modality": "image",
             "latent.label_cardinality": 345}
        b = {"latent.label_cardinality": 345, "input.modality": "image",
             "output.kind": "class"}
        self.assertEqual(normalise(a).hash(), normalise(b).hash())

    def test_quantisation_into_registered_bins(self):
        desc = normalise({"latent.label_cardinality": 345})
        self.assertEqual(desc.axes["latent.label_cardinality"], "101-1000")
        desc2 = normalise({"latent.label_cardinality": 2})
        self.assertEqual(desc2.axes["latent.label_cardinality"], "2")

    def test_above_top_bin_logged(self):
        desc = normalise({"latent.label_cardinality": 10 ** 9})
        self.assertEqual(desc.axes["latent.label_cardinality"], "1001+")
        self.assertTrue(any(e["kind"] == "above-top-bin" for e in desc.events))

    def test_oov_falls_back_and_logs(self):
        desc = normalise({"input.modality": "hologram"})
        self.assertEqual(desc.axes["input.modality"], FALLBACK_TOKEN)
        self.assertTrue(any(e["kind"] == "oov" for e in desc.events))

    def test_missing_axes_use_fallback_without_refusal(self):
        desc = normalise({})
        self.assertEqual(set(desc.axes), set(AXES))
        self.assertEqual(desc.axes["coupling"], FALLBACK_TOKEN)

    def test_canonical_is_order_fixed(self):
        desc = normalise({"output.kind": "class"})
        first_key = desc.canonical().split('"')[1]
        self.assertEqual(first_key, next(iter(AXES)))


class TestRegistry(unittest.TestCase):

    def test_same_task_same_id_idempotent(self):
        reg = TaskRegistry()
        id1, _ = reg.add({"output.kind": "class",
                          "latent.label_cardinality": 10})
        id2, _ = reg.add({"latent.label_cardinality": 10,
                          "output.kind": "class"})
        self.assertEqual(id1, id2)
        self.assertEqual(len(reg.list_ids()), 1)

    def test_adding_task_b_does_not_touch_task_a(self):
        reg = TaskRegistry()
        id_a, _ = reg.add({"output.kind": "class",
                           "input.modality": "image"})
        hash_before = reg.content_hash(id_a)
        reg.add({"output.kind": "regression",
                 "input.modality": "numeric-series"})
        self.assertEqual(hash_before, reg.content_hash(id_a))

    def test_content_hash_excludes_versions_but_catches_fit_changes(self):
        reg = TaskRegistry()
        task_id, _ = reg.add({"output.kind": "class"})
        before = reg.content_hash(task_id)
        reg.record_fit(task_id, {"arm": "spm1923", "accuracy": 0.28})
        after = reg.content_hash(task_id)
        self.assertNotEqual(before, after)

    def test_fingerprint_versions_append(self):
        reg = TaskRegistry()
        task_id, _ = reg.add({"output.kind": "class"})
        reg.set_fingerprint(task_id, [0.1, -0.2, 0.3])
        reg.set_fingerprint(task_id, [0.2, -0.1, 0.4])
        self.assertEqual(reg.get(task_id)["versions"], [1, 2, 3])

    def test_report_round_trips(self):
        import json
        reg = TaskRegistry()
        task_id, _ = reg.add({"output.kind": "class"})
        loaded = json.loads(reg.report(task_id))
        self.assertEqual(loaded["descriptor"]["axes"]["output.kind"], "class")


if __name__ == "__main__":
    unittest.main()
