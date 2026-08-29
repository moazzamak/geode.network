"""Prior-art search instrument 3 (28 Aug 2026) — arXiv sweep for the
G13 repair: externally-verified voting weight.

Registered in ``analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md``
section G13 / M358-pa BEFORE these queries ran. Sweep 1 (24 Aug)
tested the assembly/marketplace claim; sweep 2 (28 Aug) tested the
composed-codes architecture; this sweep tests only the claim that
governance weight should be discounted by payer-payee graph linkage
so that circular payment cannot buy weight.

Instrument rules (inherited from the validated run-2 instrument):
- one liveness anchor whose absence proves the instrument is broken,
  and two sensitivity anchors that must surface known work WITHOUT
  naming its title;
- HTTP 429 -> retry with backoff; residual failures recorded in a
  field separate from genuinely empty results;
- results are written to JSON; nothing is interpreted here.

Run 1 was VOID: the liveness anchor was the phrase "Liberal
 Radicalism", but arXiv:1809.06421 has since been retitled "A
Flexible Design for Funding Public Goods", so the anchor tested a
string that no longer exists in the metadata. The instrument was
exonerated mechanically (id_list 1809.06421 resolves; all:"quadratic
funding" returns four real papers). Run 2 corrects the anchors and
re-runs EVERY query, not only the empty ones.

Run-2 addition: arXiv ``all:`` indexes title/abstract/authors/
comments only, so a multi-term AND is too strict for a concept that
appears in a paper's body. Every multi-term query is therefore run in
BOTH forms -- AND and OR -- unconditionally, so that widening is a
property of the instrument and never a reaction to a result.
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
    ("wash trading",
     'all:"wash trading"'),
    ("circular trading",
     'all:"circular trading"'),
    ("self-trading detection",
     'all:"self-trading"'),
    ("collusion-resistant reputation",
     'all:"collusion" AND all:"reputation system"'),
    ("Sybil + governance",
     'all:"Sybil" AND all:"governance"'),
    ("token voting power",
     'all:"token" AND all:"voting power"'),
    ("DAO governance attack",
     'all:"DAO" AND all:"governance" AND all:"attack"'),
    ("quadratic funding + collusion",
     'all:"quadratic funding" AND all:"collusion"'),
    ("pairwise-bounded quadratic funding",
     'all:"pairwise" AND all:"quadratic funding"'),
    ("money laundering cycle detection",
     'all:"money laundering" AND all:"cycle"'),
    ("Sybil detection on graphs",
     'all:"Sybil detection" AND all:"graph"'),
    ("verifiable contribution incentives",
     'all:"verifiable" AND all:"contribution" AND all:"incentive"'),
    ("reputation from verified work",
     'all:"reputation" AND all:"blockchain" AND all:"Sybil"'),
    ("exchange volume inflation",
     'all:"volume inflation" AND all:"exchange"'),
    ("ANCHOR liveness: retitled QF paper (current title)",
     'all:"A Flexible Design for Funding Public Goods"'),
    ("ANCHOR liveness 2: quadratic funding (topic)",
     'all:"quadratic funding"'),
    ("ANCHOR sensitivity 1: matching funds for public goods"
     " (no title)",
     'all:"public goods" AND all:"matching funds"'),
    ("ANCHOR sensitivity 2: wash trading detection (no title)",
     'all:"wash trading" AND all:"detection"'),
    ("ANCHOR decoy: retired title string (expected zero)",
     'all:"Liberal Radicalism"'),
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
        except requests.RequestException:
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
                :400],
        })
    return out


def _or_form(search: str) -> str | None:
    """The OR form of a multi-term AND query, or None if single-term."""
    parts = search.split(" AND ")
    return " OR ".join(parts) if len(parts) > 1 else None


def _run(label: str, search: str, stage: str,
         sink: list[dict[str, Any]]) -> None:
    status, text = _query(search)
    if status == 200:
        hits = _parse(text)
        sink.append({"label": label, "stage": stage, "query": search,
                     "status": status, "n_hits": len(hits),
                     "hits": hits})
        print(f"{label} [{stage}]: {len(hits)} hits", flush=True)
    else:
        sink.append({"label": label, "stage": stage, "query": search,
                     "status": status, "n_hits": None, "hits": [],
                     "residual_failure": True})
        print(f"{label} [{stage}]: residual failure ({status})",
              flush=True)
    time.sleep(3.0)   # the arXiv API courtesy interval


def main() -> int:
    out_path = Path(
        "analysis/prior_art_sweep_3_g13_2026-08-28.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "instrument": "prior_art_search_3 (28 Aug 2026)",
        "registered_in": "analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md"
                         " section G13 / M358-pa",
        "claim_under_test": (
            "governance weight in an open permissionless economic "
            "network accrues only from externally-verified revenue "
            "-- payments from counterparties with no attribution "
            "linkage to the beneficiary within a registered depth L "
            "-- so that circular payment among self-owned identities "
            "cannot convert capital into weight"),
        "displacement_criteria": {
            "D1": "weight derived from revenue AND discounted by "
                  "payer-payee graph linkage (full displacement)",
            "D2": "wash/circular-payment filter applied to voting "
                  "weight specifically (application displacement)",
            "D3": "collusion discount over a contribution graph "
                  "bounding self-dealing gain (partial; citation "
                  "becomes mandatory)",
        },
        "run": 2,
        "run_1_void_reason": (
            "liveness anchor used the retired title string 'Liberal "
            "Radicalism'; arXiv:1809.06421 was retitled 'A Flexible "
            "Design for Funding Public Goods'. Instrument exonerated "
            "mechanically; all queries re-run."),
        "queries": [],
    }
    sink: list[dict[str, Any]] = results["queries"]
    for label, search in QUERIES:
        _run(label, search, "and", sink)
        or_form = _or_form(search)
        if or_form is not None:
            _run(label, or_form, "or", sink)
    out_path.write_text(json.dumps(results, indent=2),
                        encoding="utf-8")
    print(f"written -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
