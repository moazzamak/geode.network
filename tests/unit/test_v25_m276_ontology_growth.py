"""M276 — ontology growth policy tests.

Every new measured task node appends to the capability map; a novel
axis forces the extension WITH the R-new-axis flag recorded; the map
stays deterministic; duplicates are rejected (append-only).
"""
import unittest

from geode.core.capability import (
    CAPABILITY_MAP_V0,
    extend_map,
    map_content_hash,
)


class TestOntologyGrowth(unittest.TestCase):
    def test_same_axis_growth_appends_without_flag(self):
        node = {"modality": "next-token-text",
                "sealed_numbers": {"held_out_ppl": 8.0},
                "evidence": "logs/results/v25/mxxx/evidence.json"}
        updated, flags = extend_map(CAPABILITY_MAP_V0, "new_text_task",
                                    node)
        self.assertIn("new_text_task", updated["nodes"])
        self.assertEqual(flags, [])

    def test_novel_axis_forces_extension_with_flag(self):
        node = {"modality": "classification-audio",
                "sealed_numbers": {"accuracy": 0.98},
                "evidence": "ev_audio.json"}
        updated, flags = extend_map(CAPABILITY_MAP_V0, "audio_task",
                                    node)
        self.assertIn("audio_task", updated["nodes"])
        self.assertIn("new_axis", flags)

    def test_duplicate_node_rejected(self):
        node = {"modality": "next-token-text",
                "sealed_numbers": {"held_out_ppl": 8.0}}
        with self.assertRaises(ValueError):
            extend_map(CAPABILITY_MAP_V0, "wikitext103_next_token",
                       node)

    def test_map_stays_deterministic(self):
        from geode.hashing import payload_hash
        node = {"modality": "classification-vision",
                "sealed_numbers": {"top1": 0.9}}
        updated_a, _ = extend_map(CAPABILITY_MAP_V0, "t", node)
        updated_b, _ = extend_map(CAPABILITY_MAP_V0, "t", node)
        self.assertEqual(payload_hash(updated_a), payload_hash(updated_b))

    def test_growth_preserves_sealed_nodes(self):
        node = {"modality": "next-token-text",
                "sealed_numbers": {"held_out_ppl": 8.0}}
        updated, _ = extend_map(CAPABILITY_MAP_V0, "another", node)
        for sealed_id in CAPABILITY_MAP_V0["nodes"]:
            self.assertEqual(updated["nodes"][sealed_id],
                             CAPABILITY_MAP_V0["nodes"][sealed_id])


if __name__ == "__main__":
    unittest.main()
