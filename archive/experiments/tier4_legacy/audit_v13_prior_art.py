"""M87 — execute the registered prior-art query families and record what they return.

This module adjudicates nothing. It runs the query strings registered in
``experiments/configs/v13/m87_prior_art_audit.json`` against the arXiv and
Semantic Scholar public APIs, records every query with its hit count and the
returned records verbatim, and stops. Whether a returned record displaces a v13
claim is a judgement made by a human reading primary text (N87.6), written into
``analysis/PRIOR_ART_AUDIT_v13.md``, and it is deliberately not automated here.

Two properties are worth stating plainly.

**This evidence is not replayable** (N87.8). Every other artifact in v13 is
byte-reproducible from a sealed corpus. This one is a dated snapshot of two
external indexes. Re-running it next month will return different records, and
that is a property of the instrument rather than a defect. The run records the
exact query strings so a reader can repeat the search; it does not record them so
the bytes can be reproduced.

**Empty queries are recorded, not dropped** (N87.5). A query family that returns
nothing is a result about the searched surface, and deleting it would leave a
reader unable to tell a search that found nothing from a search never run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from datetime import datetime, timezone
from typing import Any

from experiments.common.litsearch_cache import LitSearchCache, utc_now
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
USER_AGENT = "CG-MoE-M87-prior-art-audit/1.0 (research; contact via repository)"
ALLOWED_HOSTS = {"export.arxiv.org", "api.semanticscholar.org", "api.openalex.org"}


def _resolve(path: str) -> pathlib.Path:
    """Resolve *path* inside the repository, refusing anything that escapes it."""
    resolved = (REPO_ROOT / path).resolve()
    if REPO_ROOT not in resolved.parents and resolved != REPO_ROOT:
        raise ValueError(f"path escapes the repository: {path}")
    return resolved


def _get(url: str, timeout: float = 45.0, attempts: int = 5) -> tuple[int, bytes | None, str | None]:
    """Fetch *url*, returning ``(status, body, error)`` without raising.

    A failed query must be recorded as failed rather than aborting the audit, so
    that a reader can see which parts of the searched surface were unreachable.

    Rate-limited responses are retried with a widening backoff, because a 429
    recorded as zero hits would be indistinguishable from a query that genuinely
    found nothing — and under N87.2 that difference is the whole point. If the
    retries are exhausted the failure is recorded as a failure.
    """
    host = urllib.parse.urlparse(url).hostname
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"refusing to fetch a host outside the registered set: {host}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last: tuple[int, bytes | None, str | None] = (0, None, "no attempt made")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return int(response.status), response.read(), None
        except urllib.error.HTTPError as error:
            last = (int(error.code), None, f"HTTPError {error.code}: {error.reason}")
            if error.code not in (429, 500, 502, 503, 504):
                return last
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last = (0, None, f"{type(error).__name__}: {error}")
        if attempt < attempts - 1:
            time.sleep(min(6.0 * (2**attempt), 45.0))
    return last


def _arxiv_terms(query: str) -> list[str]:
    """Split *query* into the terms arXiv will be asked about."""
    return [term for term in query.replace("'", "").split() if len(term) > 2]


def _arxiv_expression(terms: list[str], joiner: str) -> str:
    return f" {joiner} ".join(f"all:{term}" for term in terms)


def _query_arxiv(query: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Run one arXiv query under the registered two-stage rule (N87.9).

    Stage 1 ANDs the query's terms. If that returns nothing, stage 2 re-issues the
    same terms ORed and relevance-sorted. The rule is mechanical and applies to
    every query, so it cannot be aimed at the ones whose emptiness would have been
    convenient. The stage that produced the records is recorded.
    """
    terms = _arxiv_terms(query)
    last: dict[str, Any] | None = None
    for stage, joiner in (("and", "AND"), ("or", "OR")):
        url = settings["endpoint"] + "?" + urllib.parse.urlencode(
            {
                "search_query": _arxiv_expression(terms, joiner),
                "start": 0,
                "max_results": settings["max_results"],
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        result = _parse_arxiv(query, url, stage)
        last = result
        if result["hits"] > 0 or result["error"]:
            return result
        time.sleep(float(settings["delay_seconds"]))
    assert last is not None
    return last


def _parse_arxiv(query: str, url: str, stage: str) -> dict[str, Any]:
    """Fetch and parse one arXiv URL."""
    status, body, error = _get(url)
    base = {"source": "arxiv", "query": query, "stage": stage, "status": status}
    if body is None:
        return {**base, "error": error, "hits": 0, "records": []}
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as parse_error:
        return {**base, "error": f"ParseError: {parse_error}", "hits": 0, "records": []}
    records: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title = entry.findtext("atom:title", default="", namespaces=ATOM_NS)
        summary = entry.findtext("atom:summary", default="", namespaces=ATOM_NS)
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
        identifier = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
        records.append(
            {
                "title": " ".join(title.split()),
                "year": published[:4],
                "id": identifier.strip(),
                "abstract": " ".join(summary.split())[:900],
            }
        )
    return {**base, "error": None, "hits": len(records), "records": records}


def _query_semantic_scholar(query: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Run one Semantic Scholar query and return its records, or the failure reason."""
    url = settings["endpoint"] + "?" + urllib.parse.urlencode(
        {"query": query, "limit": settings["max_results"], "fields": settings["fields"]}
    )
    status, body, error = _get(url, attempts=int(settings.get("attempts", 4)))
    base = {"source": "semantic_scholar", "query": query, "stage": "relevance", "status": status}
    if body is None:
        return {**base, "error": error, "hits": 0, "records": []}
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as decode_error:
        return {**base, "error": f"JSONDecodeError: {decode_error}", "hits": 0, "records": []}
    records = []
    for paper in payload.get("data", []) or []:
        abstract = paper.get("abstract") or ""
        records.append(
            {
                "title": " ".join((paper.get("title") or "").split()),
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "citations": paper.get("citationCount"),
                "external_ids": paper.get("externalIds"),
                "abstract": " ".join(abstract.split())[:900],
            }
        )
    return {**base, "error": None, "hits": len(records), "records": records}


def _query_openalex(query: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Run one OpenAlex query (M88's third index, N88.3).

    A third index can only ever locate displacing work. It cannot establish
    absence, and no verdict may be upgraded because a claim survived one more
    place than before.
    """
    url = settings["endpoint"] + "?" + urllib.parse.urlencode(
        {"search": query, "per-page": settings["max_results"]}
    )
    status, body, error = _get(url, attempts=int(settings.get("attempts", 4)))
    base = {"source": "openalex", "query": query, "stage": "relevance", "status": status}
    if body is None:
        return {**base, "error": error, "hits": 0, "records": []}
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as decode_error:
        return {**base, "error": f"JSONDecodeError: {decode_error}", "hits": 0, "records": []}
    records = []
    for work in payload.get("results", []) or []:
        records.append(
            {
                "title": " ".join((work.get("display_name") or "").split()),
                "year": work.get("publication_year"),
                "venue": ((work.get("primary_location") or {}).get("source") or {}).get("display_name"),
                "citations": work.get("cited_by_count"),
                "id": work.get("doi") or work.get("id"),
                "abstract": "",
            }
        )
    return {**base, "error": None, "hits": len(records), "records": records}


RUNNERS = {
    "arxiv": "_query_arxiv",
    "semantic_scholar": "_query_semantic_scholar",
    "openalex": "_query_openalex",
}


def _runner(name: str):
    """Resolve the query function by name at call time.

    Binding the functions into the table at import would freeze them there, and a
    test that replaces one source to isolate another would silently keep hitting
    the original.
    """
    return getattr(sys.modules[__name__], RUNNERS[name])


def _run_family(
    queries: list[str],
    config: dict[str, Any],
    label: str,
    cache: LitSearchCache | None = None,
    cache_only: bool = False,
    stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Run every query in *queries* against both sources, recording all outcomes.

    With *cache* set, a query already stored locally (an identical query fetched
    live at an earlier date, or seeded from a previous evidence file) is served
    from the cache instead of the network, and the result carries its
    ``from_cache``/``cache_fetched_at``/``cache_origin`` provenance. With
    *cache_only* set, a cache miss is recorded as a failure instead of touching
    the network — the flag exists to prove a re-run needs zero API calls.
    """
    sources = config["sources"]
    results: list[dict[str, Any]] = []
    for query in queries:
        for name in sorted(sources):
            settings = sources[name]
            max_results = int(settings.get("max_results", 20))
            cached = cache.get(name, query, max_results) if cache is not None else None
            if cached is not None:
                result = dict(cached["result"])
                result["from_cache"] = True
                result["cache_fetched_at"] = cached["fetched_at"]
                result["cache_origin"] = cached["origin"]
                result["family"] = label
                results.append(result)
                if stats is not None:
                    stats["cache_hits"] += 1
                print(
                    f"  C [cache -> {cached['fetched_at'][:10]}] {cached['result']['hits']:3d} hits {query}",
                    flush=True,
                )
                continue
            if cache_only:
                result = {
                    "source": name,
                    "query": query,
                    "stage": "",
                    "status": 0,
                    "error": "cache miss (network forbidden by --cache-only)",
                    "hits": 0,
                    "records": [],
                    "family": label,
                }
                results.append(result)
                if stats is not None:
                    stats["cache_only_misses"] += 1
                print(f"  ? [no cache, no network]      {query}", flush=True)
                continue
            result = _runner(name)(query, settings)
            result["family"] = label
            results.append(result)
            if cache is not None and not result["error"]:
                cache.put(name, query, max_results, result, utc_now(), "live")
                if stats is not None:
                    stats["live_fetches"] += 1
            marker = "!" if result["error"] else " "
            stage = result.get("stage", "")
            print(f"  {marker} [{name:17s}] {result['hits']:3d} hits {stage:9s} {query}", flush=True)
            time.sleep(float(settings["delay_seconds"]))
    return results


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Collapsing matters: stripping punctuation from "train-test resolution" leaves
    a double space, and a probe that fails on a hyphen would be recorded as the
    index not having the paper.
    """
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _run_probes(
    config: dict[str, Any],
    cache: LitSearchCache | None = None,
    cache_only: bool = False,
    stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Run M88's recall probes (N88.2).

    Each probe pairs a paper whose existence is not in doubt with a *topic* query
    that omits the paper's title. If a topic query aimed at a paper cannot
    retrieve it, then 'found nothing' from the family that probe covers is not
    evidence of absence. Querying the title instead would establish only that the
    index has a title field.
    """
    probes: list[dict[str, Any]] = []
    for probe in config["recall_probes"]:
        needle = _normalise(probe["must_retrieve"])
        results = _run_family([probe["query"]], config, f"probe:{probe['id']}", cache, cache_only, stats)
        found_in = sorted(
            {
                result["source"]
                for result in results
                if any(needle in _normalise(record["title"]) for record in result["records"])
            }
        )
        probes.append(
            {
                "id": probe["id"],
                "must_retrieve": probe["must_retrieve"],
                "query": probe["query"],
                "covers": probe["covers"],
                "positive_control": probe.get("positive_control", False),
                "found_in": found_in,
                "retrieved": bool(found_in),
                "results": results,
            }
        )
        verdict = f"FOUND in {', '.join(found_in)}" if found_in else "NOT RETRIEVED"
        print(f"  {probe['id']}: {verdict}  <- {probe['must_retrieve'][:60]}", flush=True)
    return probes


def _note(config: dict[str, Any], suffix: str) -> str:
    """Return the registration note whose key ends with *suffix*.

    Note numbering is per-registration (M127 uses N87.x, M129 uses N88.x), so
    the evidence payload must not hardcode one registration's key prefix.
    """
    for key, value in config["registration_notes"].items():
        if str(key).endswith(suffix):
            return str(value)
    return f"<note {suffix} not registered>"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/v13/m87_prior_art_audit.json")
    parser.add_argument(
        "--claims",
        nargs="*",
        default=None,
        help="Restrict to these claim ids. Pass with no values to run the anchors only.",
    )
    parser.add_argument("--probes", action="store_true", help="Run the registered recall probes (N88.2).")
    parser.add_argument("--skip-anchors", action="store_true")
    parser.add_argument("--output", default=None, help="Override the configured output directory.")
    parser.add_argument(
        "--cache",
        default=None,
        help="Local query-result cache file. Queries already cached (or seeded via "
        "--cache-import) are served from disk instead of the network.",
    )
    parser.add_argument(
        "--cache-import",
        action="append",
        default=[],
        help="Seed the cache from a previous audit's evidence.json before running. "
        "Repeatable. Successful results only; failures are never cached.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Forbid the network: any query not in the cache is recorded as a failure. "
        "Proves that a re-run consumes zero API calls.",
    )
    args = parser.parse_args(argv)

    config_path = _resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    started = time.time()

    cache: LitSearchCache | None = None
    cache_stats: dict[str, int] = {"cache_hits": 0, "cache_only_misses": 0, "live_fetches": 0}
    if args.cache:
        cache = LitSearchCache(_resolve(args.cache))
        for evidence in args.cache_import:
            imported = cache.import_evidence(_resolve(evidence))
            print(f"Cache seeded from {evidence}: {imported} entries.")
        if args.cache_import:
            cache.save()

    print("M87 prior-art audit — executing registered query families.")
    print("This run adjudicates nothing (N87.6) and is not replayable (N87.8).")
    if cache is not None:
        print(f"Local cache active: {cache.stats()['entries']} entries in {args.cache}.")
    print()

    probe_results: list[dict[str, Any]] = []
    if args.probes:
        print("Recall probes (N88.2) — can the instrument find work known to exist?")
        probe_results = _run_probes(config, cache, args.cache_only, cache_stats)
        print()

    anchor_results: list[dict[str, Any]] = []
    if not args.skip_anchors:
        print("Anchors (their absence would mean the search itself is broken):")
        anchor_results = _run_family(config["anchors"], config, "anchor", cache, args.cache_only, cache_stats)

    selected = list(config["claims"]) if args.claims is None else args.claims
    claim_results: dict[str, list[dict[str, Any]]] = {}
    for claim_id in selected:
        claim = config["claims"][claim_id]
        print(f"\n{claim_id} — {claim['statement']}")
        claim_results[claim_id] = _run_family(claim["queries"], config, claim_id, cache, args.cache_only, cache_stats)

    anchor_hits = sum(result["hits"] for result in anchor_results)
    anchor_hits_by_source = {
        source: sum(result["hits"] for result in anchor_results if result["source"] == source)
        for source in sorted(config["sources"])
    }
    failures = [
        {"source": result["source"], "query": result["query"], "status": result["status"], "error": result["error"]}
        for result in [
            *anchor_results,
            *(r for probe in probe_results for r in probe["results"]),
            *(r for group in claim_results.values() for r in group),
        ]
        if result["error"]
    ]

    payload: dict[str, Any] = {
        "milestone": config["milestone"],
        "program": config["program"],
        "registration_notes": config["registration_notes"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "replayable": False,
        "replayable_reason": _note(config, ".8"),
        "configuration_hash": payload_hash(config),
        "sources": config["sources"],
        "anchors": {
            "queries": config["anchors"],
            "total_hits": anchor_hits,
            "hits_by_source": anchor_hits_by_source,
            "instrument_live": anchor_hits > 0,
            "live_sources": sorted(source for source, hits in anchor_hits_by_source.items() if hits > 0),
            "results": anchor_results,
        },
        "recall_probes": {
            "ran": bool(probe_results),
            "retrieved": sum(1 for probe in probe_results if probe["retrieved"]),
            "total": len(probe_results),
            "not_retrieved": [probe["id"] for probe in probe_results if not probe["retrieved"]],
            "positive_control_passed": all(
                probe["retrieved"] for probe in probe_results if probe.get("positive_control")
            ),
            "families_not_searched": sorted(
                {family for probe in probe_results if not probe["retrieved"] for family in probe["covers"]}
            ),
            "probes": probe_results,
        },
        "claims": {
            claim_id: {
                "statement": config["claims"][claim_id]["statement"],
                "queries": config["claims"][claim_id]["queries"],
                "total_hits": sum(result["hits"] for result in claim_results[claim_id]),
                "empty_queries": sorted(
                    {result["query"] for result in claim_results[claim_id] if result["hits"] == 0 and not result["error"]}
                ),
                "results": claim_results[claim_id],
            }
            for claim_id in claim_results
        },
        "failed_queries": failures,
        "cache": {
            "enabled": cache is not None,
            "path": args.cache,
            "seeded_from": args.cache_import,
            "queries_served_from_cache": cache_stats["cache_hits"],
            "queries_fetched_live": cache_stats["live_fetches"],
            "cache_only_misses": cache_stats["cache_only_misses"],
            "entries_after_run": cache.stats()["entries"] if cache is not None else 0,
        },
        "adjudication": "not performed here; see analysis/PRIOR_ART_AUDIT_v13.md (N87.6)",
        "runtime_seconds": round(time.time() - started, 2),
    }

    output_dir = _resolve(args.output or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "evidence.json"
    write_canonical_json(evidence_path, payload)
    build_artifact_index(output_dir)
    if cache is not None:
        cache.save()

    if probe_results:
        retrieved = payload["recall_probes"]["retrieved"]
        print(f"\nRecall probes: {retrieved}/{len(probe_results)} retrieved.")
        missed = payload["recall_probes"]["not_retrieved"]
        if missed:
            families = payload["recall_probes"]["families_not_searched"]
            print(f"  NOT retrieved: {', '.join(missed)}")
            print(f"  Absence from {', '.join(families)} is therefore not evidence of absence (N88.2).")
        if not payload["recall_probes"]["positive_control_passed"]:
            print("  POSITIVE CONTROL FAILED. This run has broken something rather than found something.")
    if not args.skip_anchors:
        print(f"\nAnchors returned {anchor_hits} hits; instrument live: {anchor_hits > 0}.")
    for claim_id, group in claim_results.items():
        total = sum(result["hits"] for result in group)
        print(f"  {claim_id}: {total} hits across {len(group)} queries")
    if failures:
        print(f"\n{len(failures)} queries failed and are recorded as failures, not as empty results:")
        for failure in failures[:12]:
            print(f"  {failure['source']:17s} {failure['status']} {failure['error']}  {failure['query']}")
    if cache is not None:
        print(
            f"\nCache: {cache_stats['cache_hits']} served from disk, "
            f"{cache_stats['live_fetches']} fetched live, "
            f"{cache_stats['cache_only_misses']} cache-only misses; "
            f"{cache.stats()['entries']} entries stored in {args.cache}."
        )
    print(f"\nWrote {evidence_path.relative_to(REPO_ROOT)}")

    if probe_results and not payload["recall_probes"]["positive_control_passed"]:
        return 1
    if not args.skip_anchors and anchor_hits == 0:
        print("\nAnchors returned nothing. The search surface is unreachable; no absence here means anything.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
