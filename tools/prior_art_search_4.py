"""Prior-art search instrument 4 (28 Aug 2026) — the M378 re-run of
sweeps 1 and 2 on TWO indexes.

Registered in ``analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md`` §G43
before these queries ran.

Why a second index. The M378 gate asked for Numerai and Rahimi &
Recht among the anchors. That is unsatisfiable on arXiv: Rahimi &
Recht is NIPS 2007 (pre-arXiv-by-default in ML) and Numerai is a
deployed system with no paper of its own. Measured before this file
was written: the arXiv title query for the Rahimi & Recht title
returns one hit, and it is a DIFFERENT paper. So sweeps 1 and 2 were
structurally blind to pre-2010 conference work and to deployed
systems -- exactly the two categories the review's G32/G33 name.
OpenAlex covers both and needs no key.

Anchor classes (all four must behave, or the run is void):
- LIVENESS: a title query that must hit (AdapterHub, Bittensor).
- SENSITIVITY: a topic query that must surface a known paper WITHOUT
  naming its title.
- COVERAGE: a paper OpenAlex must find and arXiv must MISS. The arXiv
  miss is a measurement, not a failure: it is what proves the blind
  spot is real and that the second index closes it.
- DECOY: a string that must return zero, so a false-positive index
  is detectable.

Residual failures (HTTP 429 and friends) are recorded in a field
separate from genuinely empty results -- a rate limit logged as zero
hits is indistinguishable from a query that found nothing.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests

ARXIV = "http://export.arxiv.org/api/query"
OPENALEX = "https://api.openalex.org/works"
MAX_RESULTS = 15
RETRIES = 3
BACKOFF_SECONDS = [2.0, 4.0, 8.0]
ARXIV_INTERVAL = 3.0     # the arXiv API courtesy interval
OPENALEX_INTERVAL = 1.0

NS = {"a": "http://www.w3.org/2005/Atom"}

# (label, arXiv boolean query, OpenAlex natural-language query).
# Sweep-1 and sweep-2 registered queries, unchanged, plus the anchors.
QUERIES: list[tuple[str, str, str]] = [
    # -- sweep 1: the assembly / marketplace claim -------------------
    ("s1 decentralized AI marketplace",
     'all:"decentralized AI" AND all:"marketplace"',
     "decentralized AI marketplace"),
    ("s1 ml marketplace",
     'all:"machine learning" AND all:"marketplace"',
     "machine learning model marketplace"),
    ("s1 verifiable inference",
     'all:"verifiable inference"',
     "verifiable inference"),
    ("s1 incentive + model market",
     'all:"incentive" AND all:"model market"',
     "incentive mechanism model market"),
    ("s1 model marketplace",
     'all:"model marketplace"',
     "model marketplace"),
    ("s1 decentralized inference network",
     'all:"decentralized" AND all:"inference" AND all:"network"',
     "decentralized inference network"),
    ("s1 proof of inference",
     'all:"proof of inference"',
     "proof of inference blockchain"),
    ("s1 zkML",
     'all:"zkML" OR all:"zero knowledge machine learning"',
     "zero knowledge machine learning inference proof"),
    ("s1 data marketplace + incentive",
     'all:"data marketplace" AND all:"incentive"',
     "data marketplace incentive mechanism"),
    ("s1 payment by measured held-out utility",
     'all:"held-out" AND all:"payment"',
     "paying contributors by measured held-out performance"),
    ("s1 sealed evaluation tournament",
     'all:"sealed" AND all:"evaluation" AND all:"tournament"',
     "sealed holdout evaluation prediction tournament staking"),
    # -- sweep 2: the composed-codes architecture claim --------------
    ("s2 composable adapters",
     'all:"composable" AND all:"adapters"',
     "composable adapters"),
    ("s2 adapter composition",
     'all:"adapter" AND all:"composition"',
     "adapter composition modular networks"),
    ("s2 model stitching",
     'all:"model stitching"',
     "model stitching representation compatibility"),
    ("s2 model reuse / composition",
     'all:"model composition" AND all:"reuse"',
     "model composition and reuse"),
    ("s2 modular DL marketplace",
     'all:"modular" AND all:"deep learning" AND all:"marketplace"',
     "modular deep learning marketplace"),
    ("s2 LoRA marketplace",
     'all:"LoRA" AND all:"marketplace"',
     "LoRA adapter marketplace trading"),
    ("s2 adapter marketplace",
     'all:"adapter" AND all:"marketplace"',
     "adapter marketplace registry"),
    ("s2 feature store",
     'all:"feature store"',
     "feature store versioned features machine learning"),
    ("s2 frozen backbone composition",
     'all:"frozen" AND all:"backbone" AND all:"composition"',
     "frozen backbone feature composition"),
    ("s2 representation composition",
     'all:"representation" AND all:"composition"',
     "representation composition frozen encoders"),
    ("s2 modular ML economy",
     'all:"modular" AND all:"machine learning" AND all:"economy"',
     "modular machine learning economy paying contributors"),
    ("s2 random features",
     'all:"random features" AND all:"kernel"',
     "random features kernel approximation"),
    # -- anchors -----------------------------------------------------
    ("ANCHOR liveness 1: AdapterHub (title)",
     'all:"AdapterHub"',
     "AdapterHub framework adapting transformers"),
    ("ANCHOR liveness 2: Bittensor (title)",
     'all:"Bittensor"',
     "Bittensor peer-to-peer intelligence market"),
    ("ANCHOR sensitivity: blockchain ML network (no title)",
     'all:"blockchain" AND all:"machine learning" AND all:"network"',
     "blockchain machine learning network incentive"),
    ("ANCHOR coverage 1: Rahimi & Recht (arXiv must MISS)",
     'all:"Random Features for Large-Scale Kernel Machines"',
     "Random Features for Large-Scale Kernel Machines"),
    ("ANCHOR coverage 2: Numerai (deployed, no paper)",
     'all:"Numerai"',
     "Numerai crowdsourced hedge fund staking tournament"),
    ("ANCHOR decoy: retired title string (expected zero)",
     'all:"Liberal Radicalism"',
     "zzqx nonexistent phrase decoy anchor"),
]


def _get(url: str, params: dict[str, Any] | None = None
         ) -> tuple[int, Any]:
    """One HTTP call with backoff. Returns (status, body-or-None)."""
    last_status = 0
    for attempt in range(RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            last_status = resp.status_code
            if resp.status_code == 200:
                return 200, resp
            if resp.status_code in (429, 503):
                if attempt < RETRIES:
                    time.sleep(BACKOFF_SECONDS[min(
                        attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                return resp.status_code, None
            return resp.status_code, None
        except requests.RequestException:
            last_status = -1
            if attempt < RETRIES:
                time.sleep(BACKOFF_SECONDS[min(
                    attempt, len(BACKOFF_SECONDS) - 1)])
                continue
    return last_status, None


def _arxiv(search: str) -> dict[str, Any]:
    url = (f"{ARXIV}?search_query={urllib.parse.quote(search)}"
           f"&start=0&max_results={MAX_RESULTS}")
    status, resp = _get(url)
    if status != 200 or resp is None:
        return {"status": status, "n_hits": None, "hits": [],
                "residual_failure": True}
    root = ET.fromstring(resp.text)
    hits = [{
        "id": (e.findtext("a:id", "", NS) or "").strip(),
        "title": " ".join((e.findtext("a:title", "", NS) or "").split()),
        "published": (e.findtext("a:published", "", NS)
                      or "").strip()[:10],
    } for e in root.findall("a:entry", NS)]
    return {"status": 200, "n_hits": len(hits), "hits": hits}


def _openalex(search: str) -> dict[str, Any]:
    status, resp = _get(OPENALEX, {"search": search,
                                   "per-page": MAX_RESULTS})
    if status != 200 or resp is None:
        return {"status": status, "n_hits": None, "hits": [],
                "residual_failure": True}
    body = resp.json()
    hits = [{
        "id": w.get("id"),
        "title": w.get("title"),
        "published": w.get("publication_year"),
        "venue": ((w.get("primary_location") or {}).get("source")
                  or {}).get("display_name"),
        "cited_by": w.get("cited_by_count"),
    } for w in body.get("results", [])]
    # OpenAlex ranks the whole corpus by relevance, so meta.count is
    # not a displacement signal -- only the inspected top-k is.
    return {"status": 200, "n_hits": len(hits),
            "corpus_matches": (body.get("meta") or {}).get("count"),
            "hits": hits}


def main() -> int:
    out_path = Path("analysis/prior_art_sweep_4_m378_2026-08-28.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "instrument": "prior_art_search_4 (28 Aug 2026), two-index",
        "registered_in": "analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md"
                         " section G43 / M378 (revised)",
        "indexes": ["arxiv", "openalex"],
        "claims_under_test": {
            "sweep_1": "the assembly: composable registrations of "
                       "frozen artifacts + deterministic replayable "
                       "verification + payment by measured held-out "
                       "use, settled in native ETH with epoch vesting "
                       "and burn slashing",
            "sweep_2": "the narrow-waist composition architecture: "
                       "frozen artifacts composed by declaration, "
                       "versioned blocks with upgrade-without-"
                       "invalidation, an economy paying block owners "
                       "by measured downstream use",
        },
        "anchor_contract": {
            "liveness": "must hit on arxiv",
            "sensitivity": "must surface known work without its title",
            "coverage": "openalex must hit; arxiv MUST MISS -- the "
                        "miss is the measurement of the blind spot",
            "decoy": "must return zero on both indexes",
        },
        "queries": [],
    }
    for label, ax_query, oa_query in QUERIES:
        record: dict[str, Any] = {"label": label}
        record["arxiv"] = {"query": ax_query, **_arxiv(ax_query)}
        time.sleep(ARXIV_INTERVAL)
        record["openalex"] = {"query": oa_query, **_openalex(oa_query)}
        time.sleep(OPENALEX_INTERVAL)
        results["queries"].append(record)
        print(f"{label}: arxiv={record['arxiv']['n_hits']} "
              f"openalex={record['openalex']['n_hits']}", flush=True)
    out_path.write_text(json.dumps(results, indent=2),
                        encoding="utf-8")
    print(f"written -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
