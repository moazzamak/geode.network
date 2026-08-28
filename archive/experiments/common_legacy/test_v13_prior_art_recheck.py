"""Tests for M88's additions: a third index and the recall probes.

M87's anchors could only tell us the indexes were answering. They could not tell
us the indexes were *sensitive*, which is how the C5 family came back with none
of the forward-simulation literature and still looked like a clean search. The
probe mechanism exists to detect that, so these tests are aimed squarely at the
ways a probe could quietly stop detecting it: by matching on something the query
itself supplied, by being satisfied with a failure, or by letting a broken run
report a discovery.
"""

from __future__ import annotations

import json

from experiments.tier4 import audit_v13_prior_art as audit

OPENALEX_SETTINGS = {
    "endpoint": "https://api.openalex.org/works",
    "max_results": 20,
    "delay_seconds": 0.0,
    "attempts": 3,
}


def _openalex(titles: list[str]) -> bytes:
    return json.dumps(
        {
            "results": [
                {
                    "display_name": title,
                    "publication_year": 2022,
                    "cited_by_count": 7,
                    "doi": f"https://doi.org/10.0/{index}",
                    "primary_location": {"source": {"display_name": "Venue"}},
                }
                for index, title in enumerate(titles)
            ]
        }
    ).encode("utf-8")


def test_openalex_parses_titles_and_reports_hits(monkeypatch):
    monkeypatch.setattr(audit, "_get", lambda url, timeout=45.0, attempts=5: (200, _openalex(["A", "B"]), None))
    result = audit._query_openalex("resolution discrepancy", OPENALEX_SETTINGS)

    assert result["source"] == "openalex"
    assert result["hits"] == 2
    assert [record["title"] for record in result["records"]] == ["A", "B"]


def test_openalex_throttling_is_an_error_not_an_absence(monkeypatch):
    """The failure mode that nearly ruined M87: a refused query looking like a
    query that searched and found nothing."""
    monkeypatch.setattr(
        audit, "_get", lambda url, timeout=45.0, attempts=5: (429, None, "HTTPError 429")
    )
    result = audit._query_openalex("resolution discrepancy", OPENALEX_SETTINGS)

    assert result["hits"] == 0
    assert result["error"] is not None, "a refusal must never be recorded as an empty result"


def test_openalex_malformed_json_is_an_error_not_an_absence(monkeypatch):
    monkeypatch.setattr(audit, "_get", lambda url, timeout=45.0, attempts=5: (200, b"{not json", None))
    result = audit._query_openalex("resolution discrepancy", OPENALEX_SETTINGS)

    assert result["hits"] == 0
    assert "JSONDecodeError" in result["error"]


def test_openalex_missing_venue_does_not_crash_the_run(monkeypatch):
    body = json.dumps({"results": [{"display_name": "A", "primary_location": None}]}).encode("utf-8")
    monkeypatch.setattr(audit, "_get", lambda url, timeout=45.0, attempts=5: (200, body, None))
    result = audit._query_openalex("q", OPENALEX_SETTINGS)

    assert result["hits"] == 1
    assert result["records"][0]["venue"] is None


def test_openalex_host_is_registered():
    """A source can only be queried if its host was registered in advance."""
    assert "api.openalex.org" in audit.ALLOWED_HOSTS


def _probe_config(records_by_source: dict[str, list[str]]) -> dict:
    return {
        "sources": {
            name: {"endpoint": f"https://{name}", "max_results": 5, "delay_seconds": 0.0}
            for name in records_by_source
        },
        "recall_probes": [
            {
                "id": "P1",
                "must_retrieve": "Sanity Checks for Saliency Maps",
                "query": "randomizing model weights leaves saliency maps unchanged",
                "covers": ["C5"],
                "positive_control": False,
            }
        ],
    }


def _stub_runners(monkeypatch, records_by_source: dict[str, list[str]]) -> None:
    for name, titles in records_by_source.items():
        def fake(query, settings, titles=titles, name=name):
            return {
                "source": name,
                "query": query,
                "stage": "and",
                "status": 200,
                "error": None,
                "hits": len(titles),
                "records": [{"title": title, "abstract": ""} for title in titles],
            }

        monkeypatch.setattr(audit, audit.RUNNERS[name], fake)


def test_probe_is_retrieved_when_the_topic_query_surfaces_the_paper(monkeypatch):
    _stub_runners(monkeypatch, {"arxiv": ["Sanity Checks for Saliency Maps", "Something else"]})
    probes = audit._run_probes(_probe_config({"arxiv": []}))

    assert probes[0]["retrieved"] is True
    assert probes[0]["found_in"] == ["arxiv"]


def test_probe_fails_when_the_index_returns_only_adjacent_work(monkeypatch):
    """The whole point. An index that returns plausible-looking neighbours but not
    the paper you already know exists has not searched the topic, and its silence
    on everything else is worthless."""
    _stub_runners(monkeypatch, {"arxiv": ["A Simple Saliency Method That Passes the Sanity Checks"]})
    probes = audit._run_probes(_probe_config({"arxiv": []}))

    assert probes[0]["retrieved"] is False
    assert probes[0]["found_in"] == []


def test_probe_matching_survives_typography_but_not_absence(monkeypatch):
    """Case and punctuation must not decide a probe; content must."""
    _stub_runners(monkeypatch, {"arxiv": ["SANITY  CHECKS, FOR SALIENCY MAPS!"]})
    probes = audit._run_probes(_probe_config({"arxiv": []}))

    assert probes[0]["retrieved"] is True


def test_probe_query_does_not_contain_the_title_it_must_retrieve(monkeypatch):
    """A probe that queries the title tests only that the index has a title field.
    Every registered probe must be a genuine topic query (N88.2)."""
    config = json.loads(
        (audit.REPO_ROOT / "experiments/configs/v13/m88_prior_art_recheck.json").read_text(encoding="utf-8")
    )
    for probe in config["recall_probes"]:
        query = audit._normalise(probe["query"])
        title_words = [word for word in audit._normalise(probe["must_retrieve"]).split() if len(word) > 3]
        overlap = [word for word in title_words if word in query]
        assert len(overlap) < len(title_words), f"{probe['id']} leaks its own title into the query: {overlap}"


def test_probe_records_which_index_found_it_not_merely_that_one_did(monkeypatch):
    """Per-index recall is the measurement. Collapsing it to a boolean would hide
    that one index carries every probe while another contributes nothing."""
    _stub_runners(
        monkeypatch,
        {"arxiv": ["Sanity Checks for Saliency Maps"], "openalex": ["Unrelated"]},
    )
    probes = audit._run_probes(_probe_config({"arxiv": [], "openalex": []}))

    assert probes[0]["found_in"] == ["arxiv"], "openalex missed it and must be recorded as having missed it"


def test_registered_probes_cover_every_claim_m87_adjudicated():
    """A family with no probe has no measured sensitivity, so its verdict cannot
    be discounted honestly."""
    m88 = json.loads(
        (audit.REPO_ROOT / "experiments/configs/v13/m88_prior_art_recheck.json").read_text(encoding="utf-8")
    )
    m87 = json.loads(
        (audit.REPO_ROOT / "experiments/configs/v13/m87_prior_art_audit.json").read_text(encoding="utf-8")
    )
    covered = {family for probe in m88["recall_probes"] for family in probe["covers"]}

    assert covered == set(m87["claims"])


def test_reopened_families_are_byte_identical_to_m87():
    """N88.4: a re-run may lower a verdict and never raise one. Rewording the
    queries between runs is the easiest way to smuggle a friendlier result in."""
    m88 = json.loads(
        (audit.REPO_ROOT / "experiments/configs/v13/m88_prior_art_recheck.json").read_text(encoding="utf-8")
    )
    m87 = json.loads(
        (audit.REPO_ROOT / "experiments/configs/v13/m87_prior_art_audit.json").read_text(encoding="utf-8")
    )
    for claim_id, claim in m88["claims"].items():
        assert claim["queries"] == m87["claims"][claim_id]["queries"], claim_id
        assert claim["statement"] == m87["claims"][claim_id]["statement"], claim_id


def test_m88_does_not_reopen_the_claims_it_promised_to_leave_alone():
    """N88.1: C2's demotion is final and C4 is settled. Neither may be re-searched
    under a milestone whose stated scope excludes them."""
    m88 = json.loads(
        (audit.REPO_ROOT / "experiments/configs/v13/m88_prior_art_recheck.json").read_text(encoding="utf-8")
    )

    assert set(m88["claims"]) == {"C1", "C3", "C5"}
    assert "C2" not in m88["claims"], "a second search is not an appeal"
