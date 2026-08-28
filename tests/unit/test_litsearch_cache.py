"""Tests for the litsearch result cache (experiments/common/litsearch_cache.py).

The cache exists to avoid re-querying rate-limited public APIs for query strings
the programme has already run. These tests pin the two properties that matter
for the instrument's honesty: provenance is preserved, and failures are never
cached as results.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.common.litsearch_cache import LitSearchCache, cache_key


def _result(source: str, query: str, hits: int = 3, error: str | None = None) -> dict:
    return {
        "source": source,
        "query": query,
        "stage": "and",
        "status": 200 if error is None else 429,
        "error": error,
        "hits": hits,
        "records": [{"title": f"paper {i}", "year": "2024"} for i in range(hits)],
    }


def _write_evidence(directory: Path, payload: dict) -> Path:
    path = directory / "evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class CacheKeyTests(unittest.TestCase):
    def test_key_deterministic_and_sensitive(self):
        self.assertEqual(cache_key("arxiv", "scaling laws", 20), cache_key("arxiv", "scaling laws", 20))
        self.assertNotEqual(cache_key("arxiv", "scaling laws", 20), cache_key("arxiv", "scaling laws", 100))
        self.assertNotEqual(cache_key("arxiv", "scaling laws", 20), cache_key("semantic_scholar", "scaling laws", 20))
        self.assertNotEqual(cache_key("arxiv", "scaling laws", 20), cache_key("arxiv", "scaling law", 20))


class LitSearchCacheTests(unittest.TestCase):
    def test_put_get_roundtrip_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = LitSearchCache(Path(tmp) / "cache.json")
            cache.put("arxiv", "q", 20, _result("arxiv", "q"), "2026-08-13T10:00:00+00:00", "live")
            entry = cache.get("arxiv", "q", 20)
            self.assertIsNotNone(entry)
            self.assertEqual(entry["fetched_at"], "2026-08-13T10:00:00+00:00")
            self.assertEqual(entry["origin"], "live")
            self.assertEqual(entry["result"]["hits"], 3)
            # The stored result must not carry the per-run family label.
            self.assertNotIn("family", entry["result"])

    def test_newest_fetched_at_wins_on_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = LitSearchCache(Path(tmp) / "cache.json")
            cache.put("arxiv", "q", 20, _result("arxiv", "q", hits=1), "2026-01-01T00:00:00+00:00", "old")
            cache.put("arxiv", "q", 20, _result("arxiv", "q", hits=7), "2026-08-13T00:00:00+00:00", "new")
            entry = cache.get("arxiv", "q", 20)
            self.assertEqual(entry["result"]["hits"], 7)
            self.assertEqual(entry["origin"], "new")
            # An older entry must not overwrite a newer one.
            cache.put("arxiv", "q", 20, _result("arxiv", "q", hits=2), "2026-06-01T00:00:00+00:00", "older")
            self.assertEqual(cache.get("arxiv", "q", 20)["result"]["hits"], 7)

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            cache = LitSearchCache(path)
            cache.put("arxiv", "q", 20, _result("arxiv", "q"), "2026-08-13T00:00:00+00:00", "live")
            cache.save()
            reloaded = LitSearchCache(path)
            self.assertIsNotNone(reloaded.get("arxiv", "q", 20))
            self.assertIsNone(reloaded.get("arxiv", "missing", 20))

    def test_corrupt_cache_starts_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            path.write_text("{not json", encoding="utf-8")
            cache = LitSearchCache(path)
            self.assertEqual(cache.stats()["entries"], 0)

    def test_import_evidence_seeds_results_and_skips_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = _write_evidence(
                Path(tmp),
                {
                    "generated_at": "2026-08-13T00:00:00+00:00",
                    "sources": {"arxiv": {"max_results": 20}, "semantic_scholar": {"max_results": 20}},
                    "anchors": {
                        "results": [
                            _result("arxiv", "anchor query"),
                            _result("semantic_scholar", "anchor query", error="HTTPError 429"),
                        ]
                    },
                    "claims": {"D1": {"results": [_result("arxiv", "claim query")]}},
                    "recall_probes": {
                        "probes": [
                            {
                                "results": [
                                    _result("arxiv", "probe query"),
                                    _result("arxiv", "failed probe", error="HTTPError 429"),
                                ]
                            }
                        ]
                    },
                },
            )
            cache = LitSearchCache(Path(tmp) / "cache.json")
            stored = cache.import_evidence(evidence)
            # Three successful results; the two 429 failures must not be cached.
            self.assertEqual(stored, 3)
            self.assertIsNotNone(cache.get("arxiv", "anchor query", 20))
            self.assertIsNone(cache.get("semantic_scholar", "anchor query", 20))
            self.assertIsNotNone(cache.get("arxiv", "claim query", 20))
            self.assertIsNotNone(cache.get("arxiv", "probe query", 20))
            self.assertIsNone(cache.get("arxiv", "failed probe", 20))
            entry = cache.get("arxiv", "anchor query", 20)
            self.assertEqual(entry["origin"], str(evidence))
            self.assertEqual(entry["fetched_at"], "2026-08-13T00:00:00+00:00")

    def test_import_uses_per_source_max_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = _write_evidence(
                Path(tmp),
                {
                    "generated_at": "2026-08-13T00:00:00+00:00",
                    "sources": {"arxiv": {"max_results": 50}, "semantic_scholar": {"max_results": 20}},
                    "anchors": {"results": [_result("arxiv", "q")]},
                    "claims": {},
                    "recall_probes": {},
                },
            )
            cache = LitSearchCache(Path(tmp) / "cache.json")
            cache.import_evidence(evidence)
            self.assertIsNotNone(cache.get("arxiv", "q", 50))
            # A different budget is a different query: it must miss.
            self.assertIsNone(cache.get("arxiv", "q", 20))

    def test_stats_reports_origins(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = LitSearchCache(Path(tmp) / "cache.json")
            cache.put("arxiv", "a", 20, _result("arxiv", "a"), "2026-08-13T00:00:00+00:00", "live")
            cache.put("arxiv", "b", 20, _result("arxiv", "b"), "2026-08-12T00:00:00+00:00", "evidence-a.json")
            stats = cache.stats()
            self.assertEqual(stats["entries"], 2)
            self.assertEqual(stats["by_origin"], {"live": 1, "evidence-a.json": 1})


if __name__ == "__main__":
    unittest.main()
