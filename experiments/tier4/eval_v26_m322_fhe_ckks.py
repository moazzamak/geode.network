"""M322e-D CKKS stage harness — TenSEAL backend, noise gates and budget.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` §8.22
M322e-D. Runs AFTER the quantization stage (it consumes the heads
cached by ``eval_v26_m322_fhe_quant``). Registered cells:

- CKKS-QG3a: max |decoded*2^64 - integer_MAC| <= 2^32 (the CKKS
  noise gate at the registered parameters).
- CKKS-QG3b: argmax agreement between the decoded vector and the
  integer-path argmax >= 0.999 on n_rows = 20 real held-out rows.
- G5: per-query wall time (encrypt, evaluate, decrypt), ciphertext
  bytes — RECORDED, never asserted.

Smoke mode uses synthetic heads and 3 rows.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.common.v5_artifacts import write_canonical_json
from geode.privacy.fhe_head import (
    SCALE_BITS_16,
    quantize_head_perclass,
    quantize_input,
    quantized_scores,
)
from geode.privacy.fhe_head_ckks import (
    build_context,
    decrypt_scaled,
    encrypt_input,
    evaluate_head,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
                  / "m322_fhe_ckks.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v26" / "m322_fhe_ckks"
HEADS_CACHE = (REPO_ROOT / "logs" / "results" / "v26" / "m322_fhe_quant"
               / "heads_cache.npz")


def _run_row(context, W_q: np.ndarray, b_q: np.ndarray, q_z: np.ndarray,
             d: int, c: int, ref: np.ndarray) -> dict:
    t0 = time.time()
    cts = encrypt_input(context, q_z)
    t_enc = time.time() - t0
    ct_bytes = sum(len(ct.serialize()) for ct in cts
                   if hasattr(ct, "serialize"))

    t0 = time.time()
    ct_out = evaluate_head(cts, W_q, b_q, d, c)
    t_eval = time.time() - t0

    t0 = time.time()
    out = decrypt_scaled(ct_out, c)
    t_dec = time.time() - t0

    diff = np.abs(out - ref)
    return {"t_encrypt_s": round(t_enc, 4),
            "t_eval_s": round(t_eval, 4),
            "t_decrypt_s": round(t_dec, 4),
            "total_s": round(t_enc + t_eval + t_dec, 4),
            "ct_bytes": int(ct_bytes),
            "max_abs_diff": float(diff.max()),
            "argmax_match": bool(np.argmax(out) == np.argmax(ref))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    out = DEFAULT_OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    bits = SCALE_BITS_16
    n_rows = int(cfg["n_rows"])

    if args.smoke:
        rng = np.random.default_rng(0)
        d, c = int(cfg["smoke_d"]), int(cfg["smoke_c"])
        W = rng.uniform(-0.05, 0.05, size=(d, c))
        b = rng.uniform(-0.5, 0.5, size=(c,))
        zs = rng.uniform(-3.0, 3.0, size=(min(n_rows, 3), d))
        heads = {"sealed_ridge": (W, b)}
        row_limit = 3
    else:
        cached = np.load(HEADS_CACHE, allow_pickle=True)
        zs = cached["test_slice"][: n_rows]
        heads = {"sealed_ridge": (cached["sealed_ridge_W"],
                                  cached["sealed_ridge_b"]),
                 "lambda_star_ridge": (cached["lambda_star_ridge_W"],
                                       cached["lambda_star_ridge_b"])}
        row_limit = n_rows

    context = build_context()
    cells: dict = {}
    for name, (W, b) in heads.items():
        head = quantize_head_perclass(W, b, bits)
        rows = []
        for i in range(row_limit):
            q_z = quantize_input(zs[i], bits)
            ref = quantized_scores(q_z, head["W_q"], head["b_q"]).astype(
                np.float64)
            rows.append(_run_row(context, head["W_q"], head["b_q"],
                                 q_z, W.shape[0], W.shape[1], ref))
        max_diff = max(r["max_abs_diff"] for r in rows)
        agreement = float(np.mean([r["argmax_match"] for r in rows]))
        cells[name] = {
            "n_rows": row_limit,
            "max_abs_diff_vs_integer_path": max_diff,
            "argmax_agreement": agreement,
            "per_row": rows,
            "qg3a": bool(max_diff <= float(cfg["qg3a_bound"])),
            "qg3b": bool(agreement >= float(cfg["qg3b_agreement"])),
        }

    evidence = {
        "milestone": "M322e-D",
        "stage": "CKKS backend",
        "registered_in": ("analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md"
                          " §8.22 M322e-D"),
        "smoke": bool(args.smoke),
        "params": {"poly_degree": 8192, "coeff_bits": [60, 40, 40, 60],
                   "global_scale": 2 ** 40},
        "bounds": {"qg3a_bound": float(cfg["qg3a_bound"]),
                   "qg3b_agreement": float(cfg["qg3b_agreement"])},
        "cells": cells,
    }
    write_canonical_json(out / "evidence.json", evidence)
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk != "per_row"}
                      for k, v in cells.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
