"""M176a — the frozen-code-ceiling probe.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` (section 7
Phase D M176a; section 12 dispatch entry, 17 Aug 2026). On the C4 138k
context (the cached SPM codes, p=0.5):

- ridge: the frozen closed-form read, penalty 1.0 — the anchor
  (0.2273623188405797, tol 1e-6).
- diagonal ridge: per-column independent weights over the SAME
  standardised features the ridge sees (including the bias column),
  float64 accumulation cast at the boundary.
- kNN: exact nearest neighbours (k=1, k=5 majority vote) with cosine
  similarity on the power-normalised codes (signed sqrt + per-row L2),
  computed by chunked matmul.

Measuring stick, not a win/loss gate. Registered interpretation
(a) kNN ~ ridge: the code ceiling (L1) is confirmed for this family;
(b) diagonal << ridge and kNN ~ ridge: cross-feature structure is
load-bearing; (c) kNN >> ridge: non-linear headroom exists (would
contradict M150) and M176c priority rises. The probe covers two
alternative head families and is not a bound on all possible heads.
Smoke declares inadmissibility and refuses the sealed output directory.
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
from experiments.tier4.eval_v16_m142_c4 import _fit_power
from experiments.tier4.eval_v16_m142_factorial import power_norm
from experiments.tier4.eval_v16_m109_trunk import _load_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m176a_code_ceiling.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v24"
                  / "m176a_code_ceiling")

CLASSES = 345


def _one_hot(labels: np.ndarray, classes: int) -> np.ndarray:
    out = np.zeros((len(labels), classes), dtype=np.float64)
    out[np.arange(len(labels)), labels.astype(np.int64)] = 1.0
    return out


def _fit_diagonal(train_mem, labels, power, penalty, n_train, block, std):
    """Per-column independent ridge over the standardised features."""
    d_out = None
    diag = None
    cross = None
    count = 0
    for start in range(0, n_train, block):
        stop = min(start + block, n_train)
        xs = std(power_norm(train_mem[start:stop], power))
        ys = _one_hot(labels[start:stop], CLASSES)
        if d_out is None:
            d_out = xs.shape[1]
            diag = np.zeros(d_out, dtype=np.float64)
            cross = np.zeros((d_out, CLASSES), dtype=np.float64)
        diag += np.einsum("ij,ij->j", xs, xs)
        cross += xs.T @ ys
        count += len(ys)
    weights = (cross / (diag[:, None] + penalty)).astype(np.float32)
    return weights, d_out, count


def _score_diagonal(test_mem, labels, power, weights, std, block):
    hits = 0
    n = len(labels)
    for start in range(0, n, block):
        stop = min(start + block, n)
        xs = std(power_norm(test_mem[start:stop], power))
        hits += int(((xs @ weights).argmax(axis=1)
                     == labels[start:stop].astype(np.int64)).sum())
    return hits / n


def _knn(train_mem, test_mem, labels_train, labels_test, power, k, n_train,
         n_test, test_block, train_chunk):
    """Exact cosine kNN via chunked matmul; majority vote."""
    train_sims = []
    for start in range(0, n_train, train_chunk):
        stop = min(start + train_chunk, n_train)
        train_sims.append(power_norm(train_mem[start:stop], power)
                          .astype(np.float32))
    hits = 0
    for t0 in range(0, n_test, test_block):
        t1 = min(t0 + test_block, n_test)
        xt = power_norm(test_mem[t0:t1], power).astype(np.float32)
        sims = np.concatenate(
            [xt @ tr.T for tr in train_sims], axis=1)  # (tb, n_train)
        topk = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        votes = labels_train.astype(np.int64)[topk]
        if k == 1:
            pred = votes[:, 0]
        else:
            pred = np.array([np.bincount(v, minlength=CLASSES).argmax()
                             for v in votes])
        hits += int((pred == labels_test[t0:t1].astype(np.int64)).sum())
    return hits / n_test


def run_m176a(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    smoke = inadmissible
    skip_anchors = bool(config.get("_smoke_skip_anchors", False))
    smoke_train = int(config.get("_smoke_train_rows", 10 ** 9))
    smoke_test = int(config.get("_smoke_test_rows", 10 ** 9))

    configure_external_cache_environment()
    block = int(config["numerics"]["block"])
    corpus, _, _ = _load_corpus(config)
    cache = data_cache_root() / config["artifacts"]["cache_relpath"]
    train_mem = np.load(cache / config["artifacts"]["spm_train_file"],
                        mmap_mode="r")
    test_mem = np.load(cache / config["artifacts"]["spm_test_file"],
                       mmap_mode="r")
    labels = np.load(cache / config["artifacts"]["labels_file"])["labels"]
    test_labels = corpus["test_labels"][:smoke_test]
    n_train = min(int(config["level"]["n_train"]), smoke_train)
    power = float(config["sparse"]["power"])
    penalty = float(config["ridge"]["penalty"])
    ridge_ref = float(config["ridge"]["reference"])
    ridge_tol = float(config["ridge"]["tolerance"])
    k_values = [int(k) for k in config["knn"]["k_values"]]
    test_block = int(config["knn"]["test_block"])
    train_chunk = int(config["knn"]["train_chunk"])
    n_test = min(smoke_test, len(test_labels))
    print(f"level: {n_train} train / {n_test} test rows", flush=True)

    evidence: dict[str, Any] = {
        "milestone": "M176a",
        "cell": "frozen-code-ceiling probe (ridge vs diagonal vs kNN)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
        "interpretation": config["interpretation_registered_before_running"],
    }

    # ---- ridge (the anchor) ------------------------------------------------
    print("ridge fit (p=0.5, penalty 1.0)", flush=True)
    solved, std = _fit_power(train_mem, labels, power, [penalty],
                             n_train, block, transform=True)
    ridge_w = solved[str(penalty)]
    from experiments.tier4.eval_v15_m104_experts import _score
    hits = 0
    for start in range(0, n_test, block):
        stop = min(start + block, n_test)
        xs = std(power_norm(test_mem[start:stop], power))
        hits += int(_score(ridge_w, xs, test_labels[start:stop]).sum())
    ridge_acc = hits / n_test
    anchors = {"ridge": {"measured": ridge_acc, "sealed": ridge_ref,
                         "delta": ridge_acc - ridge_ref,
                         "tolerance": ridge_tol}}
    print(f"  ridge {ridge_acc:.6f} (delta {ridge_acc - ridge_ref:+.3e})",
          flush=True)
    if not skip_anchors and abs(ridge_acc - ridge_ref) > ridge_tol:
        evidence.update({"void": True,
                         "void_reason": "ridge anchor reproduction failed",
                         "anchors": anchors})
        _write(output_dir, evidence)
        return evidence

    # ---- diagonal ridge ----------------------------------------------------
    print("diagonal ridge fit", flush=True)
    diag_w, d_out, fit_rows = _fit_diagonal(train_mem, labels, power,
                                            penalty, n_train, block, std)
    diag_acc = _score_diagonal(test_mem, test_labels, power, diag_w, std,
                               block)
    print(f"  diagonal {diag_acc:.6f} ({d_out} columns, {fit_rows} rows)",
          flush=True)

    # ---- kNN ----------------------------------------------------------------
    knn_accs = {}
    for k in k_values:
        print(f"kNN k={k}", flush=True)
        knn_accs[str(k)] = _knn(train_mem, test_mem, labels, test_labels,
                                power, k, n_train, n_test, test_block,
                                train_chunk)
        print(f"  k={k} {knn_accs[str(k)]:.6f}", flush=True)

    evidence.update({
        "anchors": anchors,
        "ridge": {"accuracy": ridge_acc},
        "diagonal": {"accuracy": diag_acc, "columns": d_out,
                     "penalty": penalty},
        "knn": {str(k): v for k, v in knn_accs.items()},
        "runtime_seconds": round(time.time() - started, 2),
    })
    _write(output_dir, evidence)
    print(f"\nM176a complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def _write(output_dir: Path, evidence: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m176a(args.config, args.output)


if __name__ == "__main__":
    main()
