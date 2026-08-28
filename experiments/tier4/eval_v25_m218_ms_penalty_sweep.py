"""M218 — the ms ridge-head penalty sweep: the accuracy track's first
cell, measured on the sealed path.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(20 Aug 2026, before the build). Gates: the penalty-1.0 cell must
reproduce the sealed anchor bit-exactly (V_ms 0.24214492753623187 at
1e-9) and the M210b per-domain table (5-decimal registration, 5e-6)
before any other penalty is read; every penalty is scored on all
34,500 test rows.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v16_m109_trunk import _load_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m218_ms_penalty_sweep.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m218_ms_penalty_sweep")

CLASSES = 345
DOMAINS = 6
BLOCK = 4096
ANCHOR_V_MS = 0.24214492753623187
ANCHOR_TOL = 1e-9
TABLE_TOL = 5e-6


def _score(weights: np.ndarray, standardise, mem_test, test_labels,
           test_domains) -> tuple[float, dict[str, float]]:
    per_domain_hits = np.zeros(DOMAINS, dtype=np.int64)
    per_domain_rows = np.zeros(DOMAINS, dtype=np.int64)
    total_hits = 0
    for start in range(0, len(test_labels), BLOCK):
        stop = min(start + BLOCK, len(test_labels))
        scores = (standardise(np.asarray(mem_test[start:stop]))
                  .astype(np.float64) @ weights[:-1] + weights[-1])
        preds = np.argmax(scores, axis=1)
        hits = preds == test_labels[start:stop]
        total_hits += int(hits.sum())
        for d in range(DOMAINS):
            mask = test_domains[start:stop] == d
            per_domain_hits[d] += int(hits[mask].sum())
            per_domain_rows[d] += int(mask.sum())
    aggregate = total_hits / len(test_labels)
    per_task = {f"d{d}": float(per_domain_hits[d] / per_domain_rows[d])
                for d in range(DOMAINS)}
    return aggregate, per_task


def run_m218(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()

    configure_external_cache_environment()
    corpus, _ti, _tei = _load_corpus(config)
    test_labels = corpus["test_labels"]
    test_domains = corpus["test_domains"]

    ms_cache = data_cache_root() / config["artifacts"]["cache_relpath"]
    ms_test_cache = data_cache_root() \
        / config["artifacts"].get("test_cache_relpath",
                                  config["artifacts"]["cache_relpath"])
    labels = np.load(data_cache_root()
                     / config["artifacts"]["labels_file"])["labels"]
    mem_train = np.load(ms_cache / config["artifacts"]["train_file"],
                        mmap_mode="r")
    mem_test = np.load(ms_test_cache / config["artifacts"]["test_file"],
                       mmap_mode="r")
    width = mem_train.shape[1]
    if len(mem_train) != len(labels):
        raise SystemExit("M218 premise failure: ms train rows != labels")
    print(f"ms train {mem_train.shape}, test {mem_test.shape}",
          flush=True)

    penalties = [float(p) for p in config["penalties"]]
    if 1.0 not in penalties:
        raise SystemExit("the grid must contain penalty 1.0 (the anchor)")

    acc = RidgeAccumulator(width, CLASSES)
    for start in range(0, len(labels), BLOCK):
        stop = min(start + BLOCK, len(labels))
        acc.add(mem_train[start:stop], labels[start:stop])
    print("accumulated; solving the grid...", flush=True)
    weights_by_penalty = acc.solve_many(penalties)
    standardise = acc.standardiser()

    results: dict[str, Any] = {}
    for penalty in penalties:
        aggregate, per_task = _score(weights_by_penalty[penalty],
                                     standardise, mem_test, test_labels,
                                     test_domains)
        results[str(penalty)] = {"aggregate": aggregate,
                                 "per_task": per_task}
        print(f"  penalty {penalty:>7}: aggregate {aggregate:.15f}",
              flush=True)

    anchor_measured = results["1.0"]["aggregate"]
    anchor = {"measured": anchor_measured, "sealed": ANCHOR_V_MS,
              "delta": anchor_measured - ANCHOR_V_MS,
              "tolerance": ANCHOR_TOL,
              "ok": abs(anchor_measured - ANCHOR_V_MS) <= ANCHOR_TOL}
    registered_table = config["registered_per_domain_penalty_1"]
    table_deltas = {k: results["1.0"]["per_task"][k] - registered_table[k]
                    for k in registered_table
                    if k in results["1.0"]["per_task"]}
    table_ok = all(abs(d) <= TABLE_TOL for d in table_deltas.values())

    rows_complete = all(
        set(results[str(p)]["per_task"].keys())
        == {f"d{d}" for d in range(DOMAINS)}
        and 0.0 <= results[str(p)]["aggregate"] <= 1.0
        for p in penalties)
    best_per_domain = {}
    for d in range(DOMAINS):
        key = f"d{d}"
        best_penalty = max(penalties,
                           key=lambda p: results[str(p)]["per_task"][key])
        best_per_domain[key] = {"penalty": best_penalty,
                                "accuracy":
                                    results[str(best_penalty)]
                                    ["per_task"][key]}
    best_aggregate = max(penalties,
                         key=lambda p: results[str(p)]["aggregate"])

    gates = {
        "g1_anchor_reproduction": {"ok": anchor["ok"], "anchor": anchor},
        "g2_penalty1_table_matches_registered": {
            "ok": bool(table_ok), "deltas": table_deltas,
            "tolerance": TABLE_TOL},
        "g3_all_penalties_scored": {"ok": bool(rows_complete),
                                    "penalties": penalties,
                                    "test_rows": len(test_labels)},
    }
    gates_ok = all(g["ok"] for g in gates.values())
    if not anchor["ok"] or not table_ok:
        gates_ok = False

    evidence: dict[str, Any] = {
        "milestone": "M218",
        "cell": "ms ridge-head penalty sweep (accuracy track, first)",
        "admissible_as_evidence": not inadmissible,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "results": results,
        "best_aggregate": {"penalty": best_aggregate,
                           "accuracy":
                               results[str(best_aggregate)]["aggregate"]},
        "best_per_domain": best_per_domain,
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": (
                "measured over the sealed path with the anchor "
                "reproduced: see results; whether any penalty beats "
                "the registered penalty-1.0 accuracies is stated per "
                "domain and in aggregate"
            ) if gates_ok else "anchor or table gate failed — VOID",
        },
        "scope": "ms family only; the DINOv2-hybrid ridge is the next "
                 "accuracy cell",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok,
                      "anchor": anchor,
                      "best_aggregate": evidence["best_aggregate"],
                      "best_per_domain": best_per_domain}, indent=1),
          flush=True)
    print(f"M218 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m218(args.config, args.output)


if __name__ == "__main__":
    main()
