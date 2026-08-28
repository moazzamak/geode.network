"""M164 buildout-blocker literature search (the rebuilt instrument).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v23.md`` (section 4
M164; the section 6 dispatch entry, 17 Aug 2026). Runs the registered
queries against the arXiv API with the M88/M148 lessons applied:

- anchor queries are the positive control: each must hit in stage 1
  (AND of quoted abs: phrases), else stage 2 re-runs ALL queries
  uniformly with OR of the same phrases (never only the empty ones);
  an anchor that still misses voids the whole search for claims;
- HTTP 429s are retried with exponential backoff and recorded
  separately from genuinely empty results;
- role is displacement only: an unauthenticated public search cannot
  support novelty claims.

Usage:
    python -m tools.m164_buildout_blocker_search \
        --config experiments/configs/v23/m164_buildout_blockers.json \
        --output logs/results/v23/m164_buildout_blockers
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v23"
                  / "m164_buildout_blockers.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v23"
                  / "m164_buildout_blockers")

ATOM = "http://www.w3.org/2005/Atom"


def _build_query(spec: dict[str, Any], stage: str) -> str:
    phrases = spec["phrases"]
    if stage == "and":
        inner = " AND ".join(f'abs:"{p}"' for p in phrases)
    else:
        inner = " OR ".join(f'abs:"{p}"' for p in phrases)
    if spec.get("author"):
        return f'au:"{spec["author"]}" AND ({inner})'
    return inner


def _parse_hits(body: bytes) -> list[dict[str, str]]:
    root = ET.fromstring(body)
    hits = []
    for entry in root.findall(f"{{{ATOM}}}entry"):
        title_el = entry.find(f"{{{ATOM}}}title")
        id_el = entry.find(f"{{{ATOM}}}id")
        if title_el is None or id_el is None:
            continue
        hits.append({
            "title": " ".join(title_el.text.strip().split())
            if title_el.text else "",
            "id": id_el.text.strip() if id_el.text else "",
        })
    return hits


def _fetch(query: str, endpoint: str, max_results: int,
           retries: int, backoff: float) -> dict[str, Any]:
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    })
    url = f"{endpoint}?{params}"
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                status, body = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, body = error.code, b""
        except Exception as error:  # network failure, not a rate limit
            attempts.append({"attempt": attempt, "status": "error",
                             "detail": str(error)[:200]})
            return {"status": "error", "hits": [], "n_hits": 0,
                    "attempts": attempts}
        attempts.append({"attempt": attempt, "status": status})
        if status == 429 and attempt < retries:
            time.sleep(backoff ** attempt)
            continue
        if status != 200:
            return {"status": f"http_{status}", "hits": [], "n_hits": 0,
                    "attempts": attempts}
        hits = _parse_hits(body)
        return {"status": "ok", "hits": hits, "n_hits": len(hits),
                "attempts": attempts}
    return {"status": "rate_limited", "hits": [], "n_hits": 0,
            "attempts": attempts}


def run_search(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    endpoint = config["endpoint"]
    max_results = int(config["max_results_per_query"])
    retries = int(config["retries"])
    backoff = float(config["backoff_seconds_base"])

    results: dict[str, Any] = {}
    for stage in ("and", "or"):
        stage_results = {}
        for spec in config["queries"]:
            query = _build_query(spec, stage)
            res = _fetch(query, endpoint, max_results, retries, backoff)
            res["query_string"] = query
            res["stage"] = stage
            stage_results[spec["id"]] = res
            print(f"[{stage}] {spec['id']}: {res['status']} "
                  f"n={res['n_hits']}", flush=True)
            for hit in res["hits"][:3]:
                print(f"    - {hit['title'][:110]}", flush=True)
        results[stage] = stage_results
        anchors = [q for q in config["queries"] if q["kind"] == "anchor"]
        missing = [q["id"] for q in anchors
                   if stage_results[q["id"]]["n_hits"] == 0]
        if not missing:
            break
        if stage == "or":
            break
        print(f"  stage 1 anchors missing {missing} -> uniform stage-2 "
              "re-run of ALL queries (OR)", flush=True)

    anchors = [q for q in config["queries"] if q["kind"] == "anchor"]
    final_missing = [q["id"] for q in anchors
                     if all(results[s][q["id"]]["n_hits"] == 0
                            for s in ("and", "or"))]
    void = bool(final_missing)
    evidence: dict[str, Any] = {
        "milestone": "M164",
        "admissible_as_evidence": True,
        "void": void,
        "void_reason": (
            f"anchor queries with zero hits after both stages: "
            f"{final_missing} — the search must not be used for any claim"
            if void else ""),
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "question": config["question"],
        "role": config["role"],
        "gate": config["gate"],
        "results": results,
        "anchor_gate": {"missing_after_both_stages": final_missing,
                        "passed": not void},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM164 search complete -> {output_dir / 'evidence.json'}"
          f"  anchor gate passed={not void}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_search(args.config, args.output)


if __name__ == "__main__":
    main()
