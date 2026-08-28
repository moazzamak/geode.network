"""M322e-B quantization gate harness — the FHE head arithmetic on the
real sealed ridge head.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` §8.22
M322e-B (27 Aug 2026, before any measurement). The FHE backend (BFV)
decodes to the exact integer multiply-accumulate this module
simulates, so the quantization stage settles the ONLY correctness
question the crypto cannot: how much accuracy the fixed-point
encoding costs, per artifact, against its own fp64 original (the
M91 discipline).

Registered cells (written before running):

- QG1 (16-bit, sealed ridge head, penalty 1.0): max relative score
  error vs the fp64 head <= 2^-9 and argmax agreement on the
  held-out slice >= 0.99.
- QG1b (16-bit, lambda*-ridge head, the M297 repair): same bounds,
  second artifact.
- QG2 (8-bit, sealed ridge head): argmax agreement >= 0.90,
  reported with its relative error.
- QG3: the integer-MAC path is the BFV arithmetic by construction
  (recorded identity; the BFV stage verifies bit-exact agreement).

Evidence: ``logs/results/v26/m322_fhe_quant/``.
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
from experiments.common.v5_artifacts import write_canonical_json
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v26_m297_loocv_lambda import (
    _eigh_cache_load,
    _selection_digest,
)
from geode.privacy.fhe_head import (
    SCALE_BITS_8,
    SCALE_BITS_16,
    fhe_simulated_scores,
    fhe_simulated_scores_perclass,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
                  / "m322_fhe_quant.json")
M298_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
               / "m298_lda_balanced.json")
M297_EVIDENCE = (REPO_ROOT / "logs" / "results" / "v26"
                 / "m297_loocv_lambda" / "evidence.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v26" / "m322_fhe_quant"

BLOCK = 4096
CLASSES = 345
N_ROWS = 2000  # held-out slice size (the M298 agreement_rows precedent)


def _ridge_head(centred: np.ndarray, cross: np.ndarray,
                intercept: np.ndarray, penalty: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """The ridge head: W = (centred + penalty I)^-1 cross, b = intercept."""
    a = centred.copy()
    a.flat[:: a.shape[0] + 1] += penalty
    W = np.linalg.solve(a, cross)
    return W, intercept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    m298 = json.loads(M298_CONFIG.read_text(encoding="utf-8"))
    out = DEFAULT_OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    configure_external_cache_environment()
    root = data_cache_root()

    head_cache = out / "heads_cache.npz"

    if args.smoke:
        rng = np.random.default_rng(0)
        W = rng.uniform(-0.05, 0.05, size=(768, 345))
        b = rng.uniform(-0.5, 0.5, size=(345,))
        zs = rng.uniform(-3.0, 3.0, size=(256, 768))
        labels = np.argmax(zs @ W + b, axis=1)
        heads = {"sealed_ridge": (W, b),
                 "lambda_star_ridge": (W, b)}
    elif head_cache.exists():
        cached = np.load(head_cache, allow_pickle=True)
        heads = {k: (cached[f"{k}_W"], cached[f"{k}_b"])
                 for k in ("sealed_ridge", "lambda_star_ridge")}
        zs = cached["test_slice"]
        labels = cached["test_labels_slice"]
    else:
        corpus, _train_index, _test_index = _load_corpus(m298)
        test_labels = corpus["test_labels"]
        ms_cache = root / m298["artifacts"]["cache_relpath"]
        ms_test_cache = root / m298["artifacts"]["test_cache_relpath"]
        mem_train = np.load(ms_cache / m298["artifacts"]["train_file"],
                            mmap_mode="r")
        mem_test = np.load(ms_test_cache / m298["artifacts"]["test_file"],
                           mmap_mode="r")
        labels = np.load(root / m298["artifacts"]["labels_file"])["labels"]
        train_rows = len(mem_train)

        acc = RidgeAccumulator(mem_train.shape[1], CLASSES)
        for start in range(0, train_rows, BLOCK):
            stop = min(start + BLOCK, train_rows)
            acc.add(mem_train[start:stop], labels[start:stop])
        standardiser = acc.standardiser()
        centred, cross, intercept = acc._standardised_system()

        m297 = json.loads(M297_EVIDENCE.read_text(encoding="utf-8"))
        lambda_star = float(m297["lambda_star"])
        heads = {
            "sealed_ridge": _ridge_head(centred, cross, intercept, 1.0),
            "lambda_star_ridge": _ridge_head(centred, cross, intercept,
                                             lambda_star),
        }
        # held-out slice: standardised test codes
        zs = standardiser(np.asarray(mem_test[:N_ROWS])).astype(np.float64)
        labels = test_labels[:N_ROWS]
        np.savez(head_cache,
                 sealed_ridge_W=heads["sealed_ridge"][0],
                 sealed_ridge_b=heads["sealed_ridge"][1],
                 lambda_star_ridge_W=heads["lambda_star_ridge"][0],
                 lambda_star_ridge_b=heads["lambda_star_ridge"][1],
                 test_slice=zs, test_labels_slice=labels)

    cells: dict[str, Any] = {}
    for name, (W, b) in heads.items():
        s = zs @ W + b
        preds = np.argmax(s, axis=1)
        scale = float(np.max(np.abs(s)))

        # M322e-C-D1 diagnostics: the dynamic-range diagnosis
        col_max = np.max(np.abs(W), axis=0)
        col_max_flat = np.where(col_max < 1e-300, 1e-300, col_max)
        dot_part = np.max(np.abs(zs @ W))
        cells[f"{name}_diag"] = {
            "W_abs_max": float(np.max(col_max)),
            "col_max_ratio_max_over_median": float(
                np.max(col_max_flat) / np.median(col_max_flat)),
            "b_abs_max": float(np.max(np.abs(b))),
            "s_abs_max": float(np.max(np.abs(s))),
            "Wz_abs_max": float(dot_part),
            "b_share_of_scale": float(np.max(np.abs(b)) / max(scale, 1e-300)),
        }

        for bits, label in ((SCALE_BITS_16, "q16"),
                            (SCALE_BITS_8, "q8")):
            # uniform encoding (the M322e-B negative finding, re-recorded)
            s_q = np.vstack([fhe_simulated_scores(zs[i], W, b, bits)
                             for i in range(len(zs))])
            rel_err = float(np.max(np.abs(s_q - s)) / max(scale, 1e-12))
            agreement = float(np.mean(np.argmax(s_q, axis=1) == preds))
            cells[f"{name}_uniform_{label}"] = {
                "bits": bits, "max_rel_error": rel_err,
                "argmax_agreement": agreement}

            # per-class block exponents (M322e-C, the re-registered encoding)
            s_q = np.vstack([fhe_simulated_scores_perclass(
                zs[i], W, b, bits) for i in range(len(zs))])
            rel_err = float(np.max(np.abs(s_q - s)) / max(scale, 1e-12))
            agreement = float(np.mean(np.argmax(s_q, axis=1) == preds))
            cells[f"{name}_perclass_{label}"] = {
                "bits": bits, "max_rel_error": rel_err,
                "argmax_agreement": agreement}

    # registered bounds (identical for the re-registered encoding)
    bound_q16 = float(cfg["q16_max_rel_error"])
    agreement_q16 = float(cfg["q16_agreement"])
    agreement_q8 = float(cfg["q8_agreement"])
    qg1 = (cells["sealed_ridge_perclass_q16"]["max_rel_error"] <= bound_q16
           and cells["sealed_ridge_perclass_q16"]["argmax_agreement"]
           >= agreement_q16)
    qg1b = (cells["lambda_star_ridge_perclass_q16"]["max_rel_error"]
            <= bound_q16
            and cells["lambda_star_ridge_perclass_q16"]["argmax_agreement"]
            >= agreement_q16)
    qg2 = cells["sealed_ridge_perclass_q8"]["argmax_agreement"] >= agreement_q8
    # D1: the diagnosis stands only if the range evidence supports it
    d1 = (cells["sealed_ridge_diag"]["col_max_ratio_max_over_median"] >= 8.0
          or cells["sealed_ridge_diag"]["b_share_of_scale"] >= 4.0)
    runtime = round(time.time() - started, 1)

    evidence = {
        "milestone": "M322e-C",
        "registered_in": ("analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md"
                          " §8.22 M322e-C (27 Aug 2026, after the M322e-B"
                          " negative finding)"),
        "smoke": bool(args.smoke),
        "n_rows": int(len(zs)),
        "bounds": {"q16_max_rel_error": bound_q16,
                   "q16_agreement": agreement_q16,
                   "q8_agreement": agreement_q8},
        "cells": {k: v for k, v in cells.items()},
        "verdict": ("per-class block exponents (M322e-C) measured against "
                    "the SAME bounds the uniform encoding failed"),
        "qg1": bool(qg1), "qg1b": bool(qg1b), "qg2": bool(qg2),
        "d1_dynamic_range_diagnosis": bool(d1),
        "runtime_seconds": runtime,
    }
    write_canonical_json(out / "evidence_m322e_c.json", evidence)
    write_canonical_json(out / "evidence.json", evidence)
    print(json.dumps({
        "sealed_diag": cells["sealed_ridge_diag"],
        "sealed_perclass_q16": cells["sealed_ridge_perclass_q16"],
        "sealed_perclass_q8": cells["sealed_ridge_perclass_q8"],
        "lambda_perclass_q16": cells["lambda_star_ridge_perclass_q16"],
        "sealed_uniform_q16": cells["sealed_ridge_uniform_q16"],
        "qg1": qg1, "qg1b": qg1b, "qg2": qg2, "d1": d1,
        "runtime_s": runtime,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
