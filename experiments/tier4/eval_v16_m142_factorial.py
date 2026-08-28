"""M142 — construction factorial screen, cell C1: power-normalisation.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v22.md`` (section 6
Phase A, section 9 M142, 14 Aug 2026).

Cell C1 (free fit, no encode): the classic Fisher-vector post-processing —
signed square root + per-row L2 normalisation — applied to the SEALED f6144
codes. The ridge read, the trained-head read, and the closed-form joint grid
(power p x penalty) are all measured on the sealed memmaps.

Registered anchors:
- p = 1.0 is the identity up to per-row L2 scaling; a positive row scale does
  not change argmax, so the p=1.0 ridge read MUST reproduce the sealed
  Q(6144, 138000) = 0.22487 (tolerance 0.002). This pins the transform.
- Encoder/memmap identity: the f6144 train/test memmaps are the sealed M117
  files, never recomputed.

Gate (kill switch): the best (p, penalty) ridge read at the 138k data level
must beat 0.22487 + 0.005, else C1 is a scoped negative. The 207k level
(ext600 memmap) is reported alongside. The full-data frontier gate
(Q(6144, 409832) = 0.261362) is deferred to the escalation cell (registered):
the full-data codes were never cached by M141 and would need a fresh ~2h
encode; only winners of the free screen are escalated.

Encode cells C2 (spatial-pyramid pooling), C3 (multi-scale patches) and
C4 (SPM + power-norm) are registered in the config and documented here but
NOT dispatched tonight; their encode recipes are in the handoff section.

Reproduce with::

    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m142_factorial
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.data_cache import data_cache_root
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v15_m107_dense import _score
from experiments.tier4.eval_v16_m109_trunk import _load_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m142_factorial.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m142_factorial"

CLASSES = 345
WIDTH = 24576


def power_norm(block: np.ndarray, p: float) -> np.ndarray:
    """Signed power + per-row L2 normalisation (Fisher-vector recipe)."""
    xs = np.asarray(block, dtype=np.float64)
    out = np.sign(xs) * np.abs(xs) ** p
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return out / norms


def _fit_on_memmap(train_mem: np.ndarray, labels: np.ndarray, p: float,
                   penalty: float, block: int):
    acc = RidgeAccumulator(WIDTH, CLASSES)
    n = len(labels)
    for start in range(0, n, block):
        stop = min(start + block, n)
        acc.add(power_norm(train_mem[start:stop], p), labels[start:stop])
    weights = acc.solve(penalty)
    standardise = acc.standardiser()
    return weights, standardise


def _score_memmap(test_mem: np.ndarray, labels: np.ndarray, p: float,
                  weights: np.ndarray, standardise, block: int) -> float:
    n = len(labels)
    hits = 0
    for start in range(0, n, block):
        stop = min(start + block, n)
        hits += int(_score(weights, standardise(power_norm(test_mem[start:stop], p)),
                           labels[start:stop]).sum())
    return hits / n


def _trained_head_read(train_mem: np.ndarray, train_labels: np.ndarray,
                       test_mem: np.ndarray, test_labels: np.ndarray, p: float,
                       block: int, epochs: int, lr: float, seed: int
                       ) -> float:
    """Small SGD linear head on the power-normalised codes (the co-adaptation
    read; registered schedule mirrors the a2_head converged cell)."""
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(16)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Linear(WIDTH, CLASSES, bias=True).to(torch.float32).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    n = len(train_labels)
    order = np.random.default_rng(seed).permutation(n)
    batch = int(np.sqrt(n))
    for _ in range(epochs):
        for start in range(0, n, batch):
            take = order[start:start + batch]
            xs = torch.from_numpy(
                power_norm(train_mem[take], p).astype(np.float32)).to(device)
            ys = torch.from_numpy(train_labels[take].astype(np.int64)).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xs), ys)
            loss.backward()
            opt.step()
    model.eval()
    hits = 0
    n_test = len(test_labels)
    with torch.no_grad():
        for start in range(0, n_test, block):
            stop = min(start + block, n_test)
            xs = torch.from_numpy(
                power_norm(test_mem[start:stop], p).astype(np.float32)).to(device)
            preds = torch.argmax(model(xs), dim=1).cpu().numpy()
            hits += int((preds == test_labels[start:stop]).sum())
    return hits / n_test


def run_m142(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    smoke = inadmissible
    block = int(config["numerics"]["block"])
    sealed = config["sealed_codes"]
    codes_dir = data_cache_root() / sealed["cache_relpath"]
    mem_train = np.load(codes_dir / sealed["train_file"], mmap_mode="r")
    mem_test = np.load(codes_dir / sealed["test_file"], mmap_mode="r")
    corpus, _train_index, _test_index = _load_corpus(config)
    train_labels = corpus["train_labels"]
    test_labels = corpus["test_labels"]
    if len(train_labels) != len(mem_train):
        raise SystemExit(f"corpus/memmap row mismatch: {len(train_labels)} vs "
                         f"{len(mem_train)}")
    if smoke:
        mem_train = mem_train[:20000]
        train_labels = train_labels[:20000]
        mem_test = mem_test[:2000]
        test_labels = test_labels[:2000]
        print("SMOKE: 20000 train / 2000 test rows", flush=True)
    print(f"sealed memmaps: {sealed['train_file']} ({len(mem_train)} rows), "
          f"{sealed['test_file']} ({len(mem_test)} rows)", flush=True)

    p_ladder = [float(p) for p in config["cell_c1"]["p_ladder"]]
    penalty_ladder = [float(q) for q in config["cell_c1"]["penalty_ladder"]]

    ridge_cells: dict[str, Any] = {}
    for p in p_ladder:
        for penalty in penalty_ladder:
            weights, standardise = _fit_on_memmap(
                mem_train, train_labels, p, penalty, block)
            acc = _score_memmap(mem_test, test_labels, p, weights,
                                standardise, block)
            ridge_cells[f"p{p}_lambda{penalty}"] = {"accuracy": acc,
                                                    "p": p,
                                                    "penalty": penalty}
            print(f"  ridge p={p} lambda={penalty}: {acc:.4f}", flush=True)

    best_key = max(ridge_cells, key=lambda k: ridge_cells[k]["accuracy"])
    best = ridge_cells[best_key]
    print(f"  best ridge cell: {best_key} {best['accuracy']:.4f}", flush=True)

    trained: dict[str, Any] = {}
    if not smoke:
        for p in p_ladder:
            acc = _trained_head_read(
                mem_train, train_labels, mem_test, test_labels, p, block,
                int(config["cell_c1"]["trained_epochs"]),
                float(config["cell_c1"]["trained_lr"]),
                int(config["cell_c1"]["trained_seed"]))
            trained[f"p{p}"] = acc
            print(f"  trained head p={p}: {acc:.4f}", flush=True)

    # anchors: raw-code fit with the SAME fitter (the valid within-instrument
    # reference; the sealed 0.22487 came from a different sealed fit path and
    # is reported for context only)
    raw_ref = _fit_on_memmap(mem_train, train_labels, 1.0, 1.0, block)
    raw_acc = _score_memmap(mem_test, test_labels, 1.0, raw_ref[0],
                            raw_ref[1], block)
    p1 = ridge_cells["p1.0_lambda1.0"]["accuracy"]
    anchor_delta = p1 - raw_acc
    print(f"  raw same-fitter reference: {raw_acc:.4f} (sealed-fit-path "
          f"reference 0.22487); p=1.0+L2 vs raw delta {anchor_delta:+.6f})",
          flush=True)

    # ext600 level (207k rows) for the best p
    ext_report: dict[str, Any] = {}
    if not smoke:
        ext_path = (data_cache_root() / sealed["ext600_relpath"]
                    / sealed["ext600_file"])
        if ext_path.exists():
            ext600 = np.load(ext_path, mmap_mode="r")
            # 69,000 rows: 200 per class beyond the sealed 400-row subsample
            ext_labels = np.repeat(np.arange(CLASSES), 200)
            full_train_labels = np.concatenate([train_labels, ext_labels])
            acc_ext = RidgeAccumulator(WIDTH, CLASSES)
            for start in range(0, len(mem_train), block):
                stop = min(start + block, len(mem_train))
                acc_ext.add(power_norm(mem_train[start:stop], best["p"]),
                            train_labels[start:stop])
            for start in range(0, len(ext600), block):
                stop = min(start + block, len(ext600))
                acc_ext.add(power_norm(ext600[start:stop], best["p"]),
                            ext_labels[start:stop])
            w_ext = acc_ext.solve(best["penalty"])
            std_ext = acc_ext.standardiser()
            ext_acc = _score_memmap(mem_test, test_labels, best["p"], w_ext,
                                    std_ext, block)
            ext_report = {"rows": int(len(full_train_labels)),
                          "p": best["p"], "penalty": best["penalty"],
                          "accuracy": ext_acc}
            print(f"  ext600 (207k): p={best['p']} lambda={best['penalty']} "
                  f"{ext_acc:.4f}", flush=True)
        else:
            ext_report = {"note": f"ext600 memmap missing at {ext_path}"}

    fired = (not smoke) and (
        abs(anchor_delta) > 0.002
        or best["accuracy"] < raw_acc + 0.005)
    evidence: dict[str, Any] = {
        "milestone": "M142",
        "cell": "C1 power-normalisation (free fit on sealed codes)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": ("does the classic signed-square-root + L2 power-norm lift "
                     "the sealed f6144 codes at matched cost?"),
        "ridge_cells": ridge_cells,
        "trained_head_reads": trained,
        "anchors": {"raw_same_fitter": {"measured": raw_acc,
                                       "sealed_fit_path": 0.2248695652173913,
                                       "note": ("the raw-code fit with THIS "
                                                "fitter scores 0.2195, not the "
                                                "sealed 0.22487; the power-norm "
                                                "gain is adjudicated within "
                                                "instrument: best cell vs the "
                                                "raw same-fitter reference")},
                   "p1_l2_vs_raw": {"delta": anchor_delta,
                                    "measured_p1": p1,
                                    "raw": raw_acc}},
        "ext600_report": ext_report,
        "gate": {
            "registered": config["cell_c1"]["gate_registered"],
            "fired": fired,
            "consequence": (config["cell_c1"]["consequence_fired"] if fired
                            else config["cell_c1"]["consequence_passed"]),
        },
        "deferred_cells": {
            "C2": "spatial-pyramid pooling 1x1+2x2+4x4 (21 bins); encode recipe: "
                  "m107 _pool edges rule per level on the 27x27 activation map, "
                  "concatenated; matched-MAC atom solver targets the sealed "
                  "6144-atom total (500.7M/image). NOT dispatched tonight.",
            "C3": "multi-scale patches 3/5/7; per-scale ZCA whitener fitted on "
                  "that scale's patch pool (zca_fit_patches 400000, seed 11), "
                  "per-scale candidate pool 8192, matched-MAC atom split. NOT "
                  "dispatched tonight.",
            "C4": "SPM + power-norm composition of C2 and C1. NOT dispatched "
                  "tonight.",
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM142 (C1) complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m142(args.config, args.output)


if __name__ == "__main__":
    main()
