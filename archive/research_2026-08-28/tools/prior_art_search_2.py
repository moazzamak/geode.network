"""Prior-art search instrument 2 (28 Aug 2026) — arXiv sweep for
the composed-codes / narrow-waist composition claim.

Registered in ``analysis/SCIENCE_LAYER_PLAN_2026-08-28.md`` §7
BEFORE the queries ran. The 24 Aug sweep (run 1) tested the
assembly/marketplace claim; this sweep tests the composed-codes
architecture claim that wave 1 + M342 made load-bearing.

Instrument rules (inherited from the validated run-2 instrument):
- two anchors: a liveness anchor (AdapterHub, the known closest
  neighbor on the composition axis) and a sensitivity anchor that
  must surface adapter-composition work WITHOUT the title;
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
QUERIES: list[tuple[str, str]] = [
    ("composable adapters",
     'all:"composable" AND all:"adapters"'),
    ("adapter composition",
     'all:"adapter" AND all:"composition"'),
    ("model stitching",
     'all:"model stitching"'),
    ("model reuse / composition",
     'all:"model composition" AND all:"reuse"'),
    ("modular deep learning marketplace",
     'all:"modular" AND all:"deep learning" AND all:"marketplace"'),
    ("LoRA marketplace / trading",
     'all:"LoRA" AND all:"marketplace"'),
    ("adapter marketplace",
     'all:"adapter" AND all:"marketplace"'),
    ("feature store",
     'all:"feature store"'),
    ("frozen backbone composition",
     'all:"frozen" AND all:"backbone" AND all:"composition"'),
    ("representation composition",
     'all:"representation" AND all:"composition"'),
    ("modular machine learning economy",
     'all:"modular" AND all:"machine learning" AND all:"economy"'),
    ("hourglass architecture ML",
     'all:"hourglass" AND all:"machine learning"'),
    ("ANCHOR liveness: AdapterHub (title)",
     'all:"AdapterHub"'),
    ("ANCHOR sensitivity: composable adapters (no title)",
     'all:"composable" AND all:"adapters"'),
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
                    time.sleep(BACKOFF_SECONDS[min(
                        attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                return 429, ""
        except requests.RequestException as exc:
            last_status = -1
            if attempt < RETRIES:
                time.sleep(BACKOFF_SECONDS[min(
                    attempt, len(BACKOFF_SECONDS) - 1)])
                continue
    return last_status, ""


def _parse(text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(text)
    out: list[dict[str, Any]] = []
    for entry in root.findall("a:entry", NS):
        out.append({
            "id": (entry.findtext("a:id", "", NS) or "").strip(),
            "title": " ".join(
                (entry.findtext("a:title", "", NS) or "").split()),
            "published": (entry.findtext("a:published", "", NS)
                          or "").strip()[:10],
            "summary": " ".join(
                (entry.findtext("a:summary", "", NS) or "").split())[
                :60],
        })
    return out


def main() -> int:
    out_dir = Path("logs/results/prior_art_search_2026-08-28")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "instrument": "prior_art_search_2 (28 Aug 2026)",
        "registered_in": "analysis/SCIENCE_LAYER_PLAN_2026-08-28.md"
                         " section 7",
        "claim_under_test": ("narrow-waist composition architecture "
                             "for ML representations: (a) frozen "
                             "artifacts composed by declaration, "
                             "(b) versioned blocks with upgrade-"
                             "without-invalidation, (c) an economic "
                             "layer paying block owners by measured "
                             "downstream use"),
        "queries": [],
    }
    for label, search in QUERIES:
        status, text = _query(search)
        if status == 200:
            hits = _parse(text)
            results["queries"].append({
                "label": label, "query": search, "status": status,
                "n_hits": len(hits), "hits": hits})
            print(f"{label}: {len(hits)} hits", flush=True)
        else:
            results["queries"].append({
                "label": label, "query": search, "status": status,
                "n_hits": None, "hits": [],
                "residual_failure": True})
            print(f"{label}: residual failure ({status})",
                  flush=True)
        time.sleep(3.0)   # the arXiv API courtesy interval
    out_path = out_dir / "arxiv_sweep.json"
    out_path.write_text(json.dumps(results, indent=2),
                        encoding="utf-8")
    print(f"written -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
