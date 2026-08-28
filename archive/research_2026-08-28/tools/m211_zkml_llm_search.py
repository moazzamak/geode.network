"""M211 zkML / zk-LLM prior-art and options search.

Registered in ``RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(M211 registered 19 Aug, before the search). Reuses the M164/M203
instrument discipline: two-stage AND-then-OR applied uniformly,
anchor gate (zkLLM, ZEN, zkCNN as positive controls), 429-vs-empty
separation, displacement+learning role only. Query strings and
anchors are registered in the config before any result is read.

Usage:
    python -m tools.m211_zkml_llm_search \
        --config experiments/configs/v25/m211_zkml_llm_search.json \
        --output logs/results/v25/m211_zkml_llm_search
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from tools.m164_buildout_blocker_search import _build_query, _fetch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m211_zkml_llm_search.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m211_zkml_llm_search")


def run_m211(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    endpoint = config["endpoint"]
    max_results = int(config["max_results_per_query"])
    retries = int(config["retries"])
    backoff = float(config["backoff_seconds_base"])
    sleep_s = float(config["sleep_between_queries_seconds"])

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
            time.sleep(sleep_s)
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
    rate_limited = [
        q["id"] for q in config["queries"]
        for s in ("and", "or")
        if s in results and q["id"] in results[s]
        and any(a.get("status") == 429
                for a in results[s][q["id"]].get("attempts", []))
        and results[s][q["id"]]["status"] in ("429", "error")
    ]
    void = bool(final_missing)
    evidence: dict[str, Any] = {
        "milestone": "M211",
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
        "residual_rate_limits": sorted(set(rate_limited)),
        "reading_note": ("displacement + learning only: hits teach; "
                         "absence supports no novelty claim; any hit that "
                         "displaces a registered GEODE claim goes into the "
                         "claim ledger"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM211 search complete -> {output_dir / 'evidence.json'}"
          f"  anchor gate passed={not void}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m211(args.config, args.output)


if __name__ == "__main__":
    main()
