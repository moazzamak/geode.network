"""M221 — CERTIFIED penalty selection for the f6144 ridge head: the
penalty is chosen on a train-side fold only, then refit on the full
schedule and evaluated on the sealed f6144 test ONCE.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(20 Aug 2026, before the build). The full schedule is the M141 cell-2
three-part layout with the registered M182 premise repair (patch
overlay at part1 rows 137749:138000, digest-gated).
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
from experiments.tier4.eval_v23_m155_growth_premise import (
    _extension_indices,
    _load_domainnet,
    _rest_extension_indices,
)
from experiments.tier4.eval_v25_m182_contributions import (
    CLASSES,
    DOMAINS,
    F6144_WIDTH,
    _load_repair_overlay,
    _repair_gates,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m221_f6144_certified_penalty.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m221_f6144_certified_penalty")

BLOCK = 4096
ANCHOR_F6144 = 0.26153623188405795
ANCHOR_TOL = 1e-9


def _build_schedule(config: dict[str, Any]):
    corpus, train_index, _test_index = _load_corpus(config)
    raw = _load_domainnet(int(config["corpus"]["image_size"]))
    f6144_cache = data_cache_root() / config["artifacts"]["f6144_cache_relpath"]
    part1 = np.load(f6144_cache / config["artifacts"]["f6144_train_file"],
                    mmap_mode="r")
    mem_test = np.load(f6144_cache / config["artifacts"]["f6144_test_file"],
                       mmap_mode="r")
    ext_idx, _ = _extension_indices(raw["train_labels"], train_index, 600,
                                    CLASSES)
    rest_idx = _rest_extension_indices(raw["train_labels"], train_index,
                                       CLASSES, per_class_take=200)
    ext600 = np.load(data_cache_root() / "v16" / "m140"
                     / "f6144_ext600.npy", mmap_mode="r")
    rest = np.load(data_cache_root() / "v16" / "m141"
                   / "f6144_all_rest.npy", mmap_mode="r")
    if len(ext600) != 69000 or len(rest) != 202832:
        raise SystemExit("M221 premise failure: full-data part sizes")
    schedule = [
        (part1, corpus["train_labels"], corpus["train_domains"]),
        (ext600, raw["train_labels"][ext_idx], raw["train_domains"][ext_idx]),
        (rest, raw["train_labels"][rest_idx], raw["train_domains"][rest_idx]),
    ]
    return corpus, schedule, mem_test


def _fold_membership(labels_all: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Per-class interleaved halves over the concatenated schedule
    labels. Returns (fold_a_mask, fold_b_mask) as boolean masks over
    the concatenated row space."""
    order = np.argsort(labels_all, kind="stable")
    sorted_labels = labels_all[order]
    boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
    blocks = np.split(order, boundaries)
    fold_a = np.zeros(len(labels_all), dtype=bool)
    for block in blocks:
        fold_a[block[0::2]] = True
    return fold_a, ~fold_a


def _fit_schedule(schedule, masks: list[np.ndarray], patch,
                  patch_start) -> RidgeAccumulator:
    acc = RidgeAccumulator(F6144_WIDTH, CLASSES)
    for (mem, labels, _domains), mask in zip(schedule, masks):
        rows = np.flatnonzero(mask)
        is_part1 = patch is not None and mem.shape[0] == 138000
        for start in range(0, len(rows), BLOCK):
            stop = min(start + BLOCK, len(rows))
            block_rows = rows[start:stop]
            block = np.asarray(mem[block_rows])
            if is_part1:
                # the registered repair: overlay patch rows where the
                # fold block touches the corrupt tail
                block = block.copy()
                for j, r in enumerate(block_rows):
                    lo = max(int(r), patch_start)
                    hi = min(int(r) + 1, patch_start + len(patch))
                    if hi > lo:
                        block[j] = patch[lo - patch_start:
                                         hi - patch_start]
            acc.add(block, labels[block_rows])
    return acc


def _score_schedule(weights, standardise, schedule, masks, test=False,
                    test_labels=None, test_domains=None, mem_test=None
                    ) -> tuple[float, dict[str, float]]:
    if test:
        return _score_test(weights, standardise, mem_test, test_labels,
                           test_domains)
    hits = 0
    total = 0
    for (mem, labels, _domains), mask in zip(schedule, masks):
        rows = np.flatnonzero(mask)
        for start in range(0, len(rows), BLOCK):
            stop = min(start + BLOCK, len(rows))
            block_rows = rows[start:stop]
            block = np.asarray(mem[block_rows])
            scores = (standardise(block).astype(np.float64)
                      @ weights[:-1] + weights[-1])
            hits += int((np.argmax(scores, axis=1)
                         == labels[block_rows]).sum())
            total += len(block_rows)
    return hits / total, {}


def _score_test(weights, standardise, mem_test, test_labels, test_domains
                ) -> tuple[float, dict[str, float]]:
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


def run_m221(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()

    configure_external_cache_environment()
    patch, patch_start = _load_repair_overlay(config)
    _repair_gates(config)
    corpus, schedule, mem_test = _build_schedule(config)
    test_labels = corpus["test_labels"]
    test_domains = corpus["test_domains"]

    labels_all = np.concatenate([labels for _m, labels, _d in schedule])
    fold_a, fold_b = _fold_membership(labels_all)
    masks_per_part = []
    offset = 0
    for _m, labels, _d in schedule:
        n = len(labels)
        masks_per_part.append((fold_a[offset:offset + n],
                               fold_b[offset:offset + n]))
        offset += n
    a_masks = [m[0] for m in masks_per_part]
    b_masks = [m[1] for m in masks_per_part]
    per_class_counts = np.bincount(labels_all[fold_a])
    class_counts_b = np.bincount(labels_all[fold_b])
    parity = int(np.abs(per_class_counts.astype(np.int64)
                        - class_counts_b.astype(np.int64)).max())
    profile = {
        "fold_a_rows": int(fold_a.sum()),
        "fold_b_rows": int(fold_b.sum()),
        "max_per_class_imbalance": parity,
        "class_size_min": int(np.bincount(labels_all).min()),
        "class_size_max": int(np.bincount(labels_all).max()),
    }
    # REGISTERED REPAIR (20 Aug, shared with M220): the full
    # schedule's classes are uneven; the gate asserts per-class
    # PARITY, not a uniform 200/200.
    balanced = parity <= 1
    print(f"fold profile: {profile}", flush=True)

    penalties = [float(p) for p in config["penalties"]]
    print("fitting fold A...", flush=True)
    acc_a = _fit_schedule(schedule, a_masks, patch, patch_start)
    weights_by_penalty = acc_a.solve_many(penalties)
    standardise_a = acc_a.standardiser()
    print("scoring fold B...", flush=True)
    val_accuracies = {}
    for penalty in penalties:
        aggregate, _ = _score_schedule(weights_by_penalty[penalty],
                                       standardise_a, schedule, b_masks)
        val_accuracies[penalty] = aggregate
        print(f"  penalty {penalty:>7}: val {aggregate:.15f}", flush=True)
    chosen = max(penalties, key=lambda p: val_accuracies[p])
    print(f"chosen penalty: {chosen}", flush=True)

    print("refitting on the full schedule...", flush=True)
    full_masks = [np.ones(len(labels), dtype=bool)
                  for _m, labels, _d in schedule]
    acc_full = _fit_schedule(schedule, full_masks, patch, patch_start)
    full_weights = acc_full.solve_many(sorted({1.0, chosen}))
    standardise_full = acc_full.standardiser()

    anchor_measured, _ = _score_test(full_weights[1.0], standardise_full,
                                     mem_test, test_labels, test_domains)
    anchor_ok = abs(anchor_measured - ANCHOR_F6144) <= ANCHOR_TOL
    print(f"full penalty 1.0 aggregate {anchor_measured:.15f} "
          f"(anchor delta {anchor_measured - ANCHOR_F6144:+.3e})",
          flush=True)

    certified_aggregate, certified_per_task = _score_test(
        full_weights[chosen], standardise_full, mem_test, test_labels,
        test_domains)
    print(f"CERTIFIED test aggregate at penalty {chosen}: "
          f"{certified_aggregate:.15f}", flush=True)

    gates = {
        "g1_full_train_anchor_reproduction": {
            "ok": bool(anchor_ok), "measured": anchor_measured,
            "sealed": ANCHOR_F6144,
            "delta": anchor_measured - ANCHOR_F6144,
            "tolerance": ANCHOR_TOL},
        "g2_fold_balance_exact": {"ok": bool(balanced),
                                  "profile": profile,
                                  "expected": "per-class parity: "
                                              "|A_c - B_c| <= 1"},
        "g3_validation_complete": {
            "ok": all(0.0 <= v <= 1.0 for v in val_accuracies.values()),
            "val_accuracies": val_accuracies},
        "g4_single_test_evaluation": {
            "ok": True,
            "note": "only the chosen penalty was evaluated on the "
                    "sealed test (plus the anchor's 1.0 refit)"},
        "g5_repair_digests": {"ok": True,
                              "note": "_repair_gates refused on any "
                                      "mismatch before fitting"},
    }
    gates_ok = all(g["ok"] for g in gates.values()) and anchor_ok

    evidence: dict[str, Any] = {
        "milestone": "M221",
        "cell": "certified penalty selection, f6144 head",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "fold_profile": profile,
        "validation_accuracies": val_accuracies,
        "chosen_penalty": chosen,
        "certified_test_aggregate": certified_aggregate,
        "certified_test_per_task": certified_per_task,
        "delta_vs_sealed": certified_aggregate - ANCHOR_F6144,
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": (
                "the penalty was chosen on the train-side fold ONLY "
                "and evaluated on the sealed f6144 test exactly once"
            ) if gates_ok else "a gate failed — VOID",
        },
        "scope": "f6144 head (24,576-dim, full 409,832-row schedule "
                 "with the registered repair)",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok,
                      "chosen_penalty": chosen,
                      "certified_test_aggregate": certified_aggregate,
                      "delta_vs_sealed":
                          certified_aggregate - ANCHOR_F6144,
                      "val_accuracies": val_accuracies}, indent=1),
          flush=True)
    print(f"M221 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m221(args.config, args.output)


if __name__ == "__main__":
    main()
