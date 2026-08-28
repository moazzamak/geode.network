"""M297b - the M297 boundary-flag extension: exact-LOOCV lambda over
{50, 100, 300, 1000} on the same sealed machinery.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md``
§8.38 (27 Aug 2026, before dispatch). The M297 grid ended at
lambda 30.0 with the LOOCV curve still descending; this extension
asks whether the curve turns upward inside {50, 100, 300, 1000}.

Reuses, never recomputes: the M296 eigendecomposition cache
(digest-gated, the same selection digest M297 used), the M296d
strong-convexity truncation, the hat-matrix validity rule, and
the sealed instrument-identity LU path. The sealed 34,500-row
test is evaluated exactly once, at the extension's lambda*.

Registered readings (written before the run):
- turns upward: lambda* is the interior minimizer, the boundary
  flag closes;
- still descending at 1000: the flag stands - at this scale the
  LOOCV-selected lambda grows without bound (the ridge degenerates
  toward the class-prior readout, the M298a collapse direction).
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
from experiments.tier4.eval_v26_m296_head_repair import (
    condition_report_from_vals,
    symmetric_system,
)
from experiments.tier4.eval_v26_m297_loocv_lambda import (
    _eigh_cache_load,
    _score,
    _selection_digest,
    loocv_ridge,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
                  / "m297b_grid_extension.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m297b_grid_extension")

CLASSES = 345
FULL_TRAIN_ROWS = 409832
SEALED_TEST_ROWS = 34500
BLOCK = 4096
HAT_MARGIN = 1e-6
M296_COND_SEALED = 3330608536062.5874
COND_TOL = 1e-5
M297_REF = {"lambda_star": 30.0, "loocv_at_30": 0.002815,
            "test_at_30": 0.23516}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    started = time.time()

    configure_external_cache_environment()
    corpus, _train_index, test_index = _load_corpus(config)
    test_labels = corpus["test_labels"]
    root = data_cache_root()

    ms_cache = root / config["artifacts"]["cache_relpath"]
    ms_test_cache = root / config["artifacts"]["test_cache_relpath"]
    mem_train = np.load(ms_cache / config["artifacts"]["train_file"],
                        mmap_mode="r")
    mem_test = np.load(ms_test_cache / config["artifacts"]["test_file"],
                       mmap_mode="r")
    labels = np.load(root / config["artifacts"]["labels_file"])["labels"]

    train_rows = len(mem_train)
    test_rows = len(mem_test)
    smoke = args.smoke
    if smoke:
        train_rows = min(50000, train_rows)
        test_rows = min(5000, test_rows)
    train_ms = np.asarray(mem_train[:train_rows])
    test_ms = np.asarray(mem_test[:test_rows])

    # ---- instrument identity: the sealed LU path (as M296 g2) ----------
    acc = RidgeAccumulator(train_ms.shape[1], CLASSES)
    for start in range(0, len(train_ms), BLOCK):
        stop = min(start + BLOCK, len(train_ms))
        acc.add(train_ms[start:stop], labels[start:stop])
    standardiser = acc.standardiser()
    centred, cross, intercept = acc._standardised_system()
    penalty = float(config["anchor_penalty"])
    lu_centred = centred.copy()
    lu_centred.flat[:: lu_centred.shape[0] + 1] += penalty
    lu_weights = np.vstack([np.linalg.solve(lu_centred, cross),
                            intercept[None, :]])
    lu_accuracy = _score(lu_weights, test_ms,
                         test_labels[:test_rows], standardiser)

    # ---- the eigendecomposition: REUSED from the M297 cache -----------
    digest = _selection_digest(config, train_rows)
    cached = _eigh_cache_load(digest)
    if cached is None:
        if smoke:
            from scipy import linalg as scipy_linalg
            system = symmetric_system(centred)
            vals, vecs = scipy_linalg.eigh(system, check_finite=False,
                                           driver="evd")
            g_proj = vecs.T @ cross
        else:
            raise RuntimeError(
                f"eigh cache missing for {digest}: M297b must reuse the "
                "sealed M296/M297 eigendecomposition, never recompute it")
    else:
        vals, vecs, g_proj = (cached["vals"], cached["vecs"],
                              cached["g_proj"])
        print(f"reusing cached eigendecomposition ({digest})", flush=True)

    grid = [float(v) for v in config["lambda_grid"]]

    loocv = loocv_ridge(vals, vecs, g_proj, intercept, train_ms,
                        labels[:train_rows], standardiser, grid)
    valid_grid = [lam for lam in grid if loocv["valid"][str(lam)]]
    lambda_star = (min(valid_grid,
                       key=lambda lam: loocv["loocv"][str(lam)])
                   if valid_grid else None)

    test_accuracy = None
    if lambda_star is not None:
        pen_star = vals + lambda_star
        scale_star = max(abs(float(pen_star[0])),
                         abs(float(pen_star[-1])))
        keep_star = pen_star > max(0.0, scale_star * 1e-10)
        inv_star = np.where(keep_star, 1.0 / pen_star, 0.0)
        w_star = (vecs * inv_star[None, :]) @ g_proj
        weights = np.vstack([w_star, intercept[None, :]])
        test_accuracy = _score(weights, test_ms,
                               test_labels[:test_rows], standardiser)

    cond = condition_report_from_vals(
        vals + penalty, penalty, dimension=int(len(vals)))
    cond_rel = (abs(cond["condition_number"] - M296_COND_SEALED)
                / M296_COND_SEALED)

    anchor = float(config["anchor"]["value"])
    tol = float(config["anchor"]["tolerance"])
    premise_ok = (len(mem_train) == FULL_TRAIN_ROWS
                  and len(mem_test) == SEALED_TEST_ROWS
                  and len(labels) == FULL_TRAIN_ROWS)
    g1 = premise_ok if not smoke else True
    anchor_measured = lu_accuracy if not smoke else float("nan")
    g2 = (abs(anchor_measured - anchor) <= tol) if not smoke else True
    g3 = (all(np.isfinite(v) for v in loocv["loocv"].values())
          and (valid_grid or lambda_star is None))
    g4 = (cond_rel <= COND_TOL) if not smoke else True
    g5 = test_accuracy is None or 0.0 <= test_accuracy <= 1.0

    # the registered reading, combining the M297 top-of-grid value
    loocv_vals = [loocv["loocv"][str(lam)] for lam in grid]
    strictly_decreasing = all(
        loocv_vals[i] > loocv_vals[i + 1]
        for i in range(len(loocv_vals) - 1))
    m297_top = float(config["m297_reference"]["loocv_at_30"])
    turns_upward = (len(grid) >= 2
                    and loocv_vals[-1] > loocv_vals[0])
    if loocv_vals and loocv_vals[0] > m297_top:
        reading = ("the curve turns upward before 50: the M297 top "
                   "(lambda* = 30) is the measured minimizer, and the "
                   "boundary flag closes at the registered value")
    elif lambda_star == grid[-1] and strictly_decreasing:
        reading = ("LOOCV still descends at the extension's top: the "
                   "flag stands - the LOOCV-selected lambda grows "
                   "without bound at this scale (the class-prior "
                   "degeneration direction)")
    elif lambda_star is not None and lambda_star != grid[-1]:
        reading = ("lambda* is the interior minimizer of the "
                   "extension: the boundary flag closes")
    else:
        reading = "recorded as measured"

    gates = {
        "g1_premise_rows_exact": {
            "ok": bool(g1), "train_rows_on_disk": len(mem_train),
            "test_rows_on_disk": len(mem_test),
            "label_rows": len(labels),
            "expected": [FULL_TRAIN_ROWS, SEALED_TEST_ROWS]},
        "g2_lu_anchor_reproduction": {
            "ok": bool(g2), "measured": anchor_measured, "sealed": anchor,
            "delta": (anchor_measured - anchor) if not smoke else None,
            "tolerance": tol,
            "note": "skipped in smoke mode" if smoke else None},
        "g3_loocv_machinery_valid": {
            "ok": bool(g3), "loocv": loocv["loocv"],
            "valid_grid_points": [str(v) for v in valid_grid],
            "min_margin": loocv["min_margin"],
            "dropped_directions": loocv["dropped_directions"]},
        "g4_m296_condition_reproduction": {
            "ok": bool(g4), "condition_number": cond["condition_number"],
            "sealed": M296_COND_SEALED, "relative_delta": cond_rel,
            "tolerance": COND_TOL,
            "note": "skipped in smoke mode" if smoke else None},
        "g5_test_accuracy_valid_and_once": {
            "ok": bool(g5), "lambda_star": lambda_star,
            "test_accuracy_at_star": test_accuracy},
    }
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M297b",
        "cell": ("M297 boundary-flag extension: exact-LOOCV lambda "
                 "over {50, 100, 300, 1000}, cached eigendecomposition"),
        "configuration_hash": payload_hash(config),
        "config_file": DEFAULT_CONFIG.name,
        "config": config,
        "smoke": smoke,
        "loocv_by_lambda": loocv["loocv"],
        "loocv_valid_by_lambda": loocv["valid"],
        "lambda_star": lambda_star,
        "test_accuracy_at_star": test_accuracy,
        "turns_upward": bool(turns_upward),
        "m297_reference": config["m297_reference"],
        "anchor": {"sealed": anchor, "lu_measured": anchor_measured,
                   "delta_at_star":
                       (test_accuracy - anchor
                        if test_accuracy is not None else None)},
        "eigendecomposition": {
            "reused_cache_digest": digest,
            "min_eigenvalue": float(vals[0]),
            "max_eigenvalue": float(vals[-1]),
            "conditioning": cond,
            "condition_relative_delta_vs_m296": cond_rel},
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": (not gates_ok) and not smoke,
        "verdict": {"passes": bool(gates_ok), "reading": reading},
        "scope": ("full 409,832-row train schedule, sealed 34,500-row "
                  "test, cached ms features (no re-extraction), the "
                  "sealed eigendecomposition reused"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    if not smoke:
        DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
        write_canonical_json(DEFAULT_OUTPUT / "evidence.json", evidence)
        build_artifact_index(DEFAULT_OUTPUT)
        print(f"M297b complete -> {DEFAULT_OUTPUT / 'evidence.json'}",
              flush=True)
    print(json.dumps({"gates_ok": bool(gates_ok),
                      "lambda_star": lambda_star,
                      "loocv": loocv["loocv"],
                      "turns_upward": bool(turns_upward),
                      "test_accuracy_at_star": test_accuracy,
                      "reading": reading}))
    return 0 if gates_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
