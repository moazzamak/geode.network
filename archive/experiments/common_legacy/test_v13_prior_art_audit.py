"""Tests for M87's prior-art search runner.

The runner adjudicates nothing, so there is no scientific result to test. What
these tests protect is the honesty of the *absence* it reports: under N87.2 the
whole audit turns on being able to tell a query that found nothing from a query
that was throttled, mis-syntaxed, or never run. Each test below breaks one of
those distinctions and asserts the runner notices.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from experiments.tier4 import audit_v13_prior_art as audit

ATOM_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>"""
ATOM_ENTRY = """
<entry>
  <id>http://arxiv.org/abs/{index}</id>
  <title>Paper {index}</title>
  <summary>Abstract {index}</summary>
  <published>2024-01-01T00:00:00Z</published>
</entry>"""


def _atom(count: int) -> bytes:
    entries = "".join(ATOM_ENTRY.format(index=index) for index in range(count))
    return ATOM_TEMPLATE.format(entries=entries).encode("utf-8")


ARXIV_SETTINGS = {"endpoint": "http://export.arxiv.org/api/query", "max_results": 20, "delay_seconds": 0.0}
S2_SETTINGS = {
    "endpoint": "https://api.semanticscholar.org/graph/v1/paper/search",
    "max_results": 20,
    "delay_seconds": 0.0,
    "attempts": 3,
    "fields": "title",
}


def test_arxiv_falls_back_to_or_only_after_and_returns_nothing(monkeypatch):
    """The two-stage rule must try AND first and reach OR only on a genuine zero."""
    calls: list[str] = []

    def fake_get(url, timeout=45.0, attempts=5):
        calls.append(url)
        return (200, _atom(0 if "AND" in url else 3), None)

    monkeypatch.setattr(audit, "_get", fake_get)
    result = audit._query_arxiv("outlier exposure degrades detection", ARXIV_SETTINGS)

    assert len(calls) == 2, "stage 2 should run exactly once after an empty stage 1"
    assert "AND" in calls[0] and "OR" in calls[1]
    assert result["stage"] == "or"
    assert result["hits"] == 3


def test_arxiv_does_not_reach_or_when_and_already_found_work(monkeypatch):
    """A query that succeeds under AND must not be widened. Widening a query that
    already worked would let a search be broadened only where it suits."""
    calls: list[str] = []

    def fake_get(url, timeout=45.0, attempts=5):
        calls.append(url)
        return (200, _atom(5), None)

    monkeypatch.setattr(audit, "_get", fake_get)
    result = audit._query_arxiv("sparse autoencoder interpretability", ARXIV_SETTINGS)

    assert len(calls) == 1
    assert result["stage"] == "and"


def test_the_two_stage_rule_is_identical_for_every_query(monkeypatch):
    """The rule must not depend on the query text, or it could be aimed (N87.9)."""
    seen: list[tuple[str, ...]] = []

    def fake_get(url, timeout=45.0, attempts=5):
        seen.append(("AND" in url, "OR" in url))
        return (200, _atom(0), None)

    monkeypatch.setattr(audit, "_get", fake_get)
    for query in ("a claim we like very much", "a claim we do not like at all"):
        seen.clear()
        audit._query_arxiv(query, ARXIV_SETTINGS)
        assert seen == [(True, False), (False, True)]


def test_rate_limited_query_is_an_error_not_an_empty_result(monkeypatch):
    """A 429 recorded as zero hits would read as 'nothing exists' (N87.2)."""

    def always_throttled(url, *args, **kwargs):
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(audit.urllib.request, "urlopen", always_throttled)
    monkeypatch.setattr(audit.time, "sleep", lambda _seconds: None)

    result = audit._query_semantic_scholar("anything at all", S2_SETTINGS)

    assert result["hits"] == 0
    assert result["error"] is not None, "a throttled query must carry an error"
    assert "429" in result["error"]


def test_throttled_query_is_retried_before_being_given_up_on(monkeypatch):
    """Backoff must actually re-issue the request, or the retry budget is decorative."""
    attempts = {"count": 0}

    def flaky(url, *args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)  # type: ignore[arg-type]
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(audit.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(audit.time, "sleep", lambda _seconds: None)

    audit._get("https://api.semanticscholar.org/graph/v1/paper/search?q=x", attempts=5)
    assert attempts["count"] == 3, "should retry the 429s and stop on the non-retryable 404"


def test_non_retryable_status_is_not_retried(monkeypatch):
    """A 404 is an answer. Retrying it would only inflate the runtime."""
    attempts = {"count": 0}

    def not_found(url, *args, **kwargs):
        attempts["count"] += 1
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(audit.urllib.request, "urlopen", not_found)
    monkeypatch.setattr(audit.time, "sleep", lambda _seconds: None)

    status, body, error = audit._get("https://api.semanticscholar.org/x", attempts=5)
    assert attempts["count"] == 1
    assert status == 404
    assert body is None
    assert error is not None


def test_hosts_outside_the_registered_set_are_refused():
    """The audit reaches the network; it may only reach the two registered indexes."""
    with pytest.raises(ValueError, match="outside the registered set"):
        audit._get("https://example.com/search?q=anything")


def test_paths_outside_the_repository_are_refused():
    with pytest.raises(ValueError, match="escapes the repository"):
        audit._resolve("../../../etc/passwd")


def test_short_terms_are_dropped_from_arxiv_expressions():
    """Stopword-length tokens make an AND query fail for uninteresting reasons."""
    terms = audit._arxiv_terms("outlier exposure is a bad idea for OOD")
    assert "is" not in terms and "a" not in terms
    assert "outlier" in terms and "exposure" in terms


def test_malformed_index_response_is_an_error_not_an_absence(monkeypatch):
    """A parse failure must not read as 'this query found nothing'."""
    monkeypatch.setattr(audit, "_get", lambda url, timeout=45.0, attempts=5: (200, b"<not xml", None))
    result = audit._parse_arxiv("query", "http://export.arxiv.org/api/query?x=1", "and")
    assert result["hits"] == 0
    assert result["error"] is not None and "ParseError" in result["error"]

    monkeypatch.setattr(audit, "_get", lambda url, timeout=45.0, attempts=5: (200, b"{not json", None))
    s2 = audit._query_semantic_scholar("query", S2_SETTINGS)
    assert s2["hits"] == 0
    assert s2["error"] is not None and "JSONDecodeError" in s2["error"]


def test_semantic_scholar_null_abstract_does_not_crash(monkeypatch):
    """Records with a null abstract are common and must survive."""
    body = json.dumps({"data": [{"title": "T", "year": 2024, "abstract": None, "venue": None}]}).encode()
    monkeypatch.setattr(audit, "_get", lambda url, timeout=45.0, attempts=5: (200, body, None))
    result = audit._query_semantic_scholar("query", S2_SETTINGS)
    assert result["hits"] == 1
    assert result["records"][0]["abstract"] == ""


def test_empty_queries_are_reported_rather_than_dropped(tmp_path, monkeypatch):
    """N87.5: a query that found nothing is a result about the searched surface."""
    monkeypatch.setattr(audit, "_get", lambda url, timeout=45.0, attempts=5: (200, _atom(0), None))
    monkeypatch.setattr(audit, "_query_semantic_scholar", lambda q, s: {
        "source": "semantic_scholar", "query": q, "stage": "relevance",
        "status": 200, "error": None, "hits": 0, "records": [],
    })
    results = audit._run_family(["a query that finds nothing"], {"sources": {"arxiv": ARXIV_SETTINGS, "semantic_scholar": S2_SETTINGS}}, "C1")
    assert len(results) == 2
    assert all(result["hits"] == 0 and result["error"] is None for result in results)
