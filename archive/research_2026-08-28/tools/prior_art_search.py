"""Prior-art search instrument (24 Aug 2026) — arXiv sweep for
systems similar to GEODE. Registered in
``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` (PRIOR-ART SEARCH
registered) BEFORE the queries ran.

Instrument rules applied:
- two anchors: a title query (liveness) and a topic query that must
  surface the anchor WITHOUT its title (sensitivity);
- HTTP 429 -> retry with backoff; residual failures recorded in a
  field separate from genuinely empty results;
- results written to JSON; nothing is interpreted here, only
  recorded.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests

BASE = "http://export.arxiv.org/api/query"
MAX_RESULTS = 15
RETRIES = 3
BACKOFF_SECONDS = [2.0, 4.0, 8.0]

# registered queries (order is part of the registration).
# Run 2 (24 Aug): the run-1 sensitivity anchor
# all:"decentralized machine learning network" returned 0 while the
# liveness anchor passed — the instrument was insensitive to topic
# phrasing. Repair (registered): two-stage phrasing, shorter ANDs,
# applied UNIFORMLY to every query (never only the empty ones).
QUERIES: list[tuple[str, str]] = [
    ("decentralized AI marketplace",
     'all:"decentralized AI" AND all:"marketplace"'),
    ("ml marketplace + blockchain",
     'all:"machine learning" AND all:"marketplace"'),
    ("verifiable inference",
     'all:"verifiable inference"'),
    ("incentive mechanism + model market",
     'all:"incentive" AND all:"model market"'),
    ("model marketplace",
     'all:"model marketplace"'),
    ("decentralized inference network",
     'all:"decentralized" AND all:"inference" AND all:"network"'),
    ("proof of inference",
     'all:"proof of inference"'),
    ("zkML / zero-knowledge ML",
     'all:"zkML" OR all:"zero knowledge machine learning"'),
    ("data marketplace + incentive",
     'all:"data marketplace" AND all:"incentive"'),
    ("ANCHOR liveness: Bittensor (title)",
     'all:"Bittensor"'),
    ("ANCHOR sensitivity: blockchain ML network (no title)",
     'all:"blockchain" AND all:"machine learning" AND all:"network"'),
]

NS = {"a": "http://www.w3.org/2005/Atom"}


def _query(search: str) -> tuple[int, str]:
    """One arXiv API call with backoff. Returns (status, text)."""
    url = f"{BASE}?search_query={urllib.parse.quote(search)}" \
          f"&start=0&max_results={MAX_RESULTS}"
    last_status = 0
    for attempt in range(RETRIES + 1):
        try:
            resp = requests.get(url, timeout=30)
            last_status = resp.status_code
            if resp.status_code == 200:
                return 200, resp.text
            if resp.status_code == 429:
                if attempt < RETRIES:
                    time.sleep(BACKOFF_SECONDS[min(attempt,
                                                   len(BACKOFF_SECONDS) - 1)])
                    continue
                return 429, ""
            return resp.status_code, ""
        except requests.RequestException:
            if attempt < RETRIES:
                time.sleep(BACKOFF_SECONDS[min(attempt,
                                               len(BACKOFF_SECONDS) - 1)])
                continue
            return 0, ""
    return last_status, ""


def _entries(xml_text: str) -> list[dict[str, str]]:
    root = ET.fromstring(xml_text)
    out: list[dict[str, str]] = []
    for entry in root.findall("a:entry", NS):
        title = " ".join((entry.findtext("a:title", "", NS) or "")
                         .split())
        summary = " ".join((entry.findtext("a:summary", "", NS) or "")
                           .split())
        link = ""
        for node in entry.findall("a:link", NS):
            if node.get("rel") == "alternate":
                link = node.get("href", "")
        out.append({
            "title": title,
            "summary": summary[:400],
            "link": link,
            "published": entry.findtext("a:published", "", NS)[:10],
        })
    return out


def run() -> dict[str, Any]:
    results: dict[str, Any] = {"queries": []}
    for label, search in QUERIES:
        status, text = _query(search)
        record: dict[str, Any] = {
            "label": label,
            "search": search,
            "http_status": status,
            "entries": _entries(text) if text else [],
        }
        results["queries"].append(record)
        print(f"{label:52s} status={status} hits={len(record['entries'])}")
        time.sleep(3.0)  # arXiv courtesy limit
    results["residual_failures"] = [
        q["label"] for q in results["queries"] if q["http_status"] != 200]
    results["empty_results"] = [
        q["label"] for q in results["queries"]
        if q["http_status"] == 200 and not q["entries"]]
    return results


def main() -> None:
    out_dir = (Path(__file__).resolve().parents[1] / "logs" / "results"
               / "prior_art_search_2026-08-24")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run()
    run_name = "run2" if (out_dir / "arxiv_sweep.json").exists() \
        else "run1"
    path = out_dir / f"arxiv_sweep_{run_name}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresidual_failures: {results['residual_failures']}")
    print(f"empty_results:     {results['empty_results']}")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
