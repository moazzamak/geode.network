"""Local query-result cache for the prior-art audit instrument.

The audit runner (``experiments/tier4/audit_v13_prior_art.py``) queries two
external indexes (arXiv, Semantic Scholar) per registered query string. Those
indexes rate-limit, and the same query strings recur across registrations
(M87/M88/M127/M129/M132). This module caches successful query results locally
so a re-run only hits the network for queries it has never seen.

Provenance is part of the cache by design, not an afterthought:

- Every cached entry stores ``fetched_at`` (the UTC time of the LIVE fetch that
  produced it) and ``origin`` (the evidence file it was seeded from, or
  ``"live"``). A cached record is a dated snapshot; it is never presented as a
  fresh result.
- When an entry is served, the runner marks the result with ``from_cache``,
  ``cache_fetched_at`` and ``cache_origin`` so the evidence payload shows a
  reader exactly how old each record is.
- Failed queries (``error`` truthy, e.g. HTTP 429) are never cached: a failure
  recorded as a hit would be indistinguishable from a query that found nothing.

Cache layout: one JSON file, a map from ``sha1(source | max_results | query)``
to the stored entry. Writes are atomic (temp file + replace) so an interrupted
run cannot corrupt the cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string (cache provenance)."""
    return datetime.now(timezone.utc).isoformat()


def cache_key(source: str, query: str, max_results: int) -> str:
    """Deterministic key for one (source, result-budget, query) triple.

    ``max_results`` is part of the key because a cached run with a different
    result budget is a different query: serving 20 records for a request that
    registered 100 would silently shrink the searched surface.
    """
    raw = f"{source}\x00{int(max_results)}\x00{query}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()  # noqa: S324


class LitSearchCache:
    """Read/write cache over one JSON file on disk."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = pathlib.Path(path)
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            try:
                self._entries = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # A corrupt cache must not abort a registered search: start
                # empty and let the run rebuild it live.
                self._entries = {}

    def get(self, source: str, query: str, max_results: int) -> dict[str, Any] | None:
        """Return the stored entry for the query, or ``None`` on a miss."""
        return self._entries.get(cache_key(source, query, max_results))

    def put(
        self,
        source: str,
        query: str,
        max_results: int,
        result: dict[str, Any],
        fetched_at: str,
        origin: str,
    ) -> None:
        """Store one result. Newer ``fetched_at`` wins on collision."""
        key = cache_key(source, query, max_results)
        existing = self._entries.get(key)
        if existing and existing.get("fetched_at", "") >= fetched_at:
            return
        self._entries[key] = {
            "source": source,
            "query": query,
            "max_results": int(max_results),
            "fetched_at": fetched_at,
            "origin": origin,
            "result": {k: v for k, v in result.items() if k != "family"},
        }

    def import_evidence(self, evidence_path: str | os.PathLike[str]) -> int:
        """Seed the cache from a previous audit's ``evidence.json``.

        Walks the three sections the runner writes (anchors, claims, recall
        probes), extracts every successful result, and stores it under the
        provenance of the run that produced it. Failed queries are skipped:
        they are failures, not results. Returns the number of entries stored.
        """
        payload = json.loads(pathlib.Path(evidence_path).read_text(encoding="utf-8"))
        sources = payload.get("sources", {})
        fetched_at = payload.get("generated_at", utc_now())
        origin = str(evidence_path)

        sections: list[list[dict[str, Any]]] = []
        sections.extend(payload.get("anchors", {}).get("results", []))
        for claim in payload.get("claims", {}).values():
            sections.extend(claim.get("results", []))
        for probe in payload.get("recall_probes", {}).get("probes", []):
            sections.extend(probe.get("results", []))

        stored = 0
        for result in sections:
            if result.get("error"):
                continue
            source = result.get("source")
            query = result.get("query")
            if not source or not query:
                continue
            max_results = int((sources.get(source) or {}).get("max_results", 20))
            self.put(source, query, max_results, result, fetched_at, origin)
            stored += 1
        return stored

    def save(self) -> None:
        """Atomically write the cache to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._entries, handle, indent=1, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_path, self.path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def stats(self) -> dict[str, Any]:
        """Size and provenance summary of the cache."""
        by_origin: dict[str, int] = {}
        for entry in self._entries.values():
            origin = entry.get("origin", "unknown")
            by_origin[origin] = by_origin.get(origin, 0) + 1
        return {"entries": len(self._entries), "by_origin": by_origin}
