"""M220 — CERTIFIED penalty selection for the ms ridge head: the
penalty is chosen on a train-side fold only, then refit on the full
train and evaluated on the sealed test ONCE.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(20 Aug 2026, before the build). This is the certified version of the
M218 probe (which chose the penalty on the test set and was recorded
as an upper bound, not a claim).
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
                  / "m220_ms_certified_penalty.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m220_ms_certified_penalty")

CLASSES = 345
DOMAINS = 6
BLOCK = 4096
ANCHOR_V_MS = 0.24214492753623187
ANCHOR_TOL = 1e-9


def _fold_indices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-class interleaved halves: even positions within each
    class's row sequence -> A, odd -> B (labels are shuffled, so the
    within-class order preserves the load order)."""
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
    blocks = np.split(order, boundaries)
    a_parts, b_parts = [], []
    for block in blocks:
        a_parts.append(block[0::2])
        b_parts.append(block[1::2])
    return np.concatenate(a_parts), np.concatenate(b_parts)


def _fit(indices: np.ndarray, mem: np.ndarray, labels: np.ndarray
         ) -> RidgeAccumulator:
    acc = RidgeAccumulator(mem.shape[1], CLASSES)
    for start in range(0, len(indices), BLOCK):
        stop = min(start + BLOCK, len(indices))
        rows = indices[start:stop]
        acc.add(np.asarray(mem[rows]), labels[rows])
    return acc


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
    return (total_hits / len(test_labels),
            {f"d{d}": float(per_domain_hits[d] / per_domain_rows[d])
             for d in range(DOMAINS)})


def run_m220(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()

    configure_external_cache_environment()
    corpus, _ti, _tei = _load_corpus(config)
    test_labels = corpus["test_labels"]
    test_domains = corpus["test_domains"]

    root = data_cache_root()
    ms_cache = root / config["artifacts"]["cache_relpath"]
    ms_test_cache = root / config["artifacts"]["test_cache_relpath"]
    labels = np.load(root / config["artifacts"]["labels_file"])["labels"]
    mem_train = np.load(ms_cache / config["artifacts"]["train_file"],
                        mmap_mode="r")
    mem_test = np.load(ms_test_cache / config["artifacts"]["test_file"],
                       mmap_mode="r")
    if len(mem_train) != len(labels):
        raise SystemExit("M220 premise failure: ms train rows != labels")
    print(f"ms train {mem_train.shape}, test {mem_test.shape}", flush=True)

    penalties = [float(p) for p in config["penalties"]]
    if 1.0 not in penalties:
        raise SystemExit("the grid must contain penalty 1.0 (the anchor)")

    idx_a, idx_b = _fold_indices(labels)
    class_counts_a = np.bincount(labels[idx_a])
    class_counts_b = np.bincount(labels[idx_b])
    parity = int(np.abs(class_counts_a.astype(np.int64)
                        - class_counts_b.astype(np.int64)).max())
    fold_profile = {
        "fold_a_rows": int(len(idx_a)),
        "fold_b_rows": int(len(idx_b)),
        "max_per_class_imbalance": parity,
        "class_size_min": int(np.bincount(labels).min()),
        "class_size_max": int(np.bincount(labels).max()),
        "class_size_mean": float(np.bincount(labels).mean()),
    }
    # REGISTERED REPAIR (20 Aug): the full schedule's classes are
    # UNEVEN (the 400/class figure describes the 138k subsample), so
    # the gate asserts per-class PARITY (|A_c - B_c| <= 1), not a
    # uniform 200/200 — the first run's 200/200 gate failed on this
    # false premise and was voided before any number was read.
    balanced = parity <= 1

    print("fitting fold A...", flush=True)
    acc_a = _fit(idx_a, mem_train, labels)
    weights_by_penalty = acc_a.solve_many(penalties)
    standardise_a = acc_a.standardiser()
    print("scoring fold B...", flush=True)

    def _plain_score(weights, standardise, rows, labels_rows):
        hits = 0
        for start in range(0, len(rows), BLOCK):
            stop = min(start + BLOCK, len(rows))
            block = rows[start:stop]
            scores = (standardise(np.asarray(mem_train[block]))
                      .astype(np.float64) @ weights[:-1] + weights[-1])
            hits += int((np.argmax(scores, axis=1)
                         == labels_rows[block]).sum())
        return hits / len(rows)

    val_accuracies = {p: _plain_score(weights_by_penalty[p],
                                      standardise_a, idx_b, labels)
                      for p in penalties}
    chosen = max(penalties, key=lambda p: val_accuracies[p])
    print("validation accuracies:", val_accuracies, flush=True)
    print(f"chosen penalty: {chosen}", flush=True)

    # release the fold-era objects before the full-train refit (the
    # first run died silently at this stage under memory pressure)
    import gc
    del acc_a, weights_by_penalty, standardise_a, idx_a, idx_b
    gc.collect()

    # refit on the FULL train: penalty 1.0 for the anchor gate and the
    # chosen penalty for the single certified test evaluation.
    print("refitting on the full train...", flush=True)
    acc_full = _fit(np.arange(len(labels)), mem_train, labels)
    full_weights = acc_full.solve_many(sorted({1.0, chosen}))
    standardise_full = acc_full.standardiser()

    anchor_measured, _ = _score(full_weights[1.0], standardise_full,
                                mem_test, test_labels, test_domains)
    anchor_ok = abs(anchor_measured - ANCHOR_V_MS) <= ANCHOR_TOL
    print(f"full-train penalty 1.0 aggregate {anchor_measured:.15f} "
          f"(anchor delta {anchor_measured - ANCHOR_V_MS:+.3e})",
          flush=True)

    certified_aggregate, certified_per_task = _score(
        full_weights[chosen], standardise_full, mem_test, test_labels,
        test_domains)
    print(f"CERTIFIED test aggregate at penalty {chosen}: "
          f"{certified_aggregate:.15f}", flush=True)

    gates = {
        "g1_full_train_anchor_reproduction": {
            "ok": bool(anchor_ok),
            "measured": anchor_measured,
            "sealed": ANCHOR_V_MS,
            "delta": anchor_measured - ANCHOR_V_MS,
            "tolerance": ANCHOR_TOL},
        "g2_fold_balance_exact": {
            "ok": bool(balanced), "profile": fold_profile,
            "expected": "per-class parity: |A_c - B_c| <= 1"},
        "g3_validation_complete": {
            "ok": all(0.0 <= v <= 1.0
                      for v in val_accuracies.values()),
            "val_accuracies": val_accuracies},
        "g4_single_test_evaluation": {
            "ok": True,
            "note": "only the chosen penalty was evaluated on the "
                    "sealed test (plus the anchor's 1.0 refit)"},
    }
    gates_ok = all(g["ok"] for g in gates.values()) and anchor_ok

    evidence: dict[str, Any] = {
        "milestone": "M220",
        "cell": "certified penalty selection, ms head",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "fold_profile": fold_profile,
        "validation_accuracies": val_accuracies,
        "chosen_penalty": chosen,
        "certified_test_aggregate": certified_aggregate,
        "certified_test_per_task": certified_per_task,
        "delta_vs_sealed": certified_aggregate - ANCHOR_V_MS,
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": (
                "the penalty was chosen on the train-side fold ONLY "
                "and evaluated on the sealed test exactly once — this "
                "is the first CERTIFIED accuracy claim of the track"
            ) if gates_ok else "a gate failed — VOID",
        },
        "scope": "ms head; the f6144 head follows in M221",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok,
                      "chosen_penalty": chosen,
                      "certified_test_aggregate": certified_aggregate,
                      "delta_vs_sealed":
                          certified_aggregate - ANCHOR_V_MS,
                      "val_accuracies": val_accuracies}, indent=1),
          flush=True)
    print(f"M220 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m220(args.config, args.output)


if __name__ == "__main__":
    main()
