"""M299 - multi-block features + per-block L2 normalization; the M228
hybrid re-run under the repaired solver.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` (M299,
26 Aug 2026, before any build). H26-2 asks whether the M228 hybrid
regression (ms-only 0.24214492753623187 -> raw-concat hybrid
0.19434782608695653 at penalty 1.0) is substantially recovered by
R-A6a plus per-block L2 normalization, WITHOUT re-extracting the
DINOv2 features at native resolution (the registered E2 confound stays
in place).

Arms, registered before running:
- ms-only, LU path, penalty 1.0: instrument anchor (M228).
- raw-concat hybrid, LU path, penalties {0.1, 1.0, 10.0}: reproduces
  the sealed M228 hybrid numbers (instrument identity).
- repaired hybrid: per-block L2 normalization (each feature column
  divided by its train-set L2 norm, per block - ms and DINOv2 are
  separate blocks), then the standard closed-form standardisation and
  the M296 symmetric solve; lambda chosen train-side by the exact
  LOOCV of the M297 machinery on the registered grid; the sealed test
  evaluated once at lambda* (and the grid readings reported).

Gate H26-2 (registered before running): the repaired hybrid at its
train-side lambda* is >= the ms-only anchor 0.24214492753623187. A
pass localises E2 to conditioning; a fail localises it to the
upscaling confound. Both are publishable.

VOID conditions: g1 premise (shapes, row counts, cache digests); g2
both LU reproductions at 1e-9; g3 LOOCV machinery finite with the
registered validity rule; g4 the repaired solve's backward instrument;
g5 accuracies valid.
"""
from __future__ import annotations

import argparse
import hashlib
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
from experiments.tier4.eval_v26_m296_head_repair import symmetric_system
from experiments.tier4.eval_v26_m297_loocv_lambda import loocv_ridge

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
                  / "m299_hybrid_blocks.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v26" / "m299_hybrid_blocks"
DINO_TRAIN_DIR = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m228_dinov2_fullscale" / "features")
DINO_TEST_DIR = (REPO_ROOT / "logs" / "results" / "v25"
                 / "m222_dinov2_hybrid_pilot" / "features")

CLASSES = 345
FULL_TRAIN_ROWS = 409832
SEALED_TEST_ROWS = 34500
BLOCK = 4096
MS_ANCHOR = 0.24214492753623187
HYBRID_ANCHORS = {"0.1": 0.1968985507246377,
                  "1.0": 0.19434782608695653,
                  "10.0": 0.18802898550724637}
ANCHOR_TOL = 1e-9
# M296d (registered): the solve keeps only strongly-convex penalised
# modes (penalised eigenvalue > max(0, scale*1e-10)); non-positive
# penalised modes are non-convex and near-zero positive modes amplify
# beyond the kept part's resolution.
SOLVE_STRONG_CONVEX_CUTOFF = 1e-10


def _digest_of(selection: np.ndarray) -> str:
    return hashlib.sha256(
        selection.astype(np.int64).tobytes()).hexdigest()


def _load_cached_dino(directory: Path, name: str,
                      expected_selection: np.ndarray) -> np.ndarray:
    """Load cached DINOv2 features iff the recorded selection digest
    matches (the standing M222 persistence contract)."""
    feat_path = directory / f"{name}_dino.npy"
    meta_path = directory / f"{name}_meta.json"
    if not (feat_path.exists() and meta_path.exists()):
        raise SystemExit(f"M299 premise failure: missing cache "
                         f"{feat_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = _digest_of(expected_selection)
    if meta.get("selection_sha256") != expected:
        raise SystemExit(f"M299 premise failure: selection digest "
                         f"mismatch for {name}")
    return np.asarray(np.load(feat_path))


def _eigen_backward(vals: np.ndarray, vecs: np.ndarray, w: np.ndarray,
                    cross: np.ndarray, lam: float) -> dict[str, Any]:
    """The backward instrument on the eigen route under M296d: raw
    residual for a full-system solve; for a truncated solve, the raw
    residual against the TRUNCATED system (the M296c instrument - the
    normal-equation form measured the eigen reconstruction error)."""
    penalised = vals + lam
    scale_pen = max(abs(float(penalised[0])), abs(float(penalised[-1])))
    cutoff_pen = max(0.0, scale_pen * SOLVE_STRONG_CONVEX_CUTOFF)
    keep = penalised > cutoff_pen
    dropped = int((~keep).sum())
    recon = vecs @ (penalised[:, None] * (vecs.T @ w)) - cross
    denom = (scale_pen * float(np.max(np.abs(w)))
             + float(np.max(np.abs(cross))))
    raw = float(np.max(np.abs(recon))) / max(denom, 1e-300)
    if dropped > 0:
        proj_w = vecs.T @ w
        proj_c = vecs.T @ cross
        trunc_residual = (vecs @ ((penalised * keep)[:, None] * proj_w)
                          - vecs @ (keep[:, None] * proj_c))
        denom_t = (scale_pen * float(np.max(np.abs(w)))
                   + float(np.max(np.abs(cross))))
        trunc_backward = (float(np.max(np.abs(trunc_residual)))
                          / max(denom_t, 1e-300))
        return {"backward_passed": trunc_backward <= 1e-10,
                "instrument": "truncated_system",
                "raw_backward": raw,
                "truncated_system_backward": trunc_backward,
                "dropped_components": dropped}
    return {"backward_passed": raw <= 1e-10, "instrument": "raw",
            "raw_backward": raw, "dropped_components": dropped}


def _column_norms(train_ms: np.ndarray, train_dino: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Per-block column L2 norms over the train rows, one streaming
    pass over both blocks (the registered per-block L2 normalization:
    ms and DINOv2 are separate blocks)."""
    n = len(train_ms)
    sq_ms = np.zeros(train_ms.shape[1], dtype=np.float64)
    sq_dino = np.zeros(train_dino.shape[1], dtype=np.float64)
    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        sq_ms += np.sum(np.square(
            np.asarray(train_ms[start:stop]).astype(np.float64)), axis=0)
        sq_dino += np.sum(np.square(
            np.asarray(train_dino[start:stop]).astype(np.float64)), axis=0)
    return np.sqrt(sq_ms) + 1e-12, np.sqrt(sq_dino) + 1e-12


def _score(weights: np.ndarray, block_fn: Any, labels: np.ndarray,
           n: int, standardiser: Any) -> float:
    hits = 0
    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        xs = standardiser(block_fn(start, stop)).astype(np.float64)
        scores = xs @ weights[:-1] + weights[-1]
        hits += int((np.argmax(scores, axis=1) == labels[start:stop]).sum())
    return hits / n


def run_m299(config_path: Path, output_dir: Path,
             smoke: tuple[int, int] | None = None) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
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
    if smoke is not None:
        smoke_train, smoke_test = smoke
        train_rows = min(smoke_train, train_rows)
        test_rows = min(smoke_test, test_rows)

    full_sel = np.arange(FULL_TRAIN_ROWS, dtype=np.int64)
    dino_train = _load_cached_dino(DINO_TRAIN_DIR, "fulltrain", full_sel)
    dino_test = _load_cached_dino(DINO_TEST_DIR, "test",
                                  test_index.astype(np.int64))
    train_ms = np.asarray(mem_train[:train_rows])
    test_ms = np.asarray(mem_test[:test_rows])
    train_dino = dino_train[:train_rows]
    test_dino = dino_test[:test_rows]

    premise_ok = (len(mem_train) == FULL_TRAIN_ROWS
                  and len(mem_test) == SEALED_TEST_ROWS
                  and len(labels) == FULL_TRAIN_ROWS
                  and len(dino_train) == FULL_TRAIN_ROWS
                  and len(dino_test) == SEALED_TEST_ROWS)
    g1 = premise_ok

    # ---- instrument identity: both LU anchors ---------------------------
    penalties = [float(p) for p in config["penalties"]]
    acc_ms = RidgeAccumulator(train_ms.shape[1], CLASSES)
    for start in range(0, len(train_ms), BLOCK):
        stop = min(start + BLOCK, len(train_ms))
        acc_ms.add(train_ms[start:stop], labels[start:stop])
    std_ms = acc_ms.standardiser()
    ms_weights = acc_ms.solve(1.0)
    ms_accuracy = _score(ms_weights, lambda s, e: np.asarray(
        test_ms[s:e]), test_labels[:test_rows], test_rows, std_ms)

    hybrid_raw = np.concatenate([train_ms, train_dino], axis=1)
    hybrid_raw_test = np.concatenate([test_ms, test_dino], axis=1)
    acc_hy = RidgeAccumulator(hybrid_raw.shape[1], CLASSES)
    for start in range(0, len(hybrid_raw), BLOCK):
        stop = min(start + BLOCK, len(hybrid_raw))
        acc_hy.add(hybrid_raw[start:stop], labels[start:stop])
    std_hy = acc_hy.standardiser()
    hy_by_penalty = acc_hy.solve_many(penalties)
    hy_accuracy: dict[str, float] = {}
    for p in penalties:
        hy_accuracy[str(p)] = _score(
            hy_by_penalty[p], lambda s, e: np.asarray(
                hybrid_raw_test[s:e]),
            test_labels[:test_rows], test_rows, std_hy)

    g2 = bool(
        abs(ms_accuracy - MS_ANCHOR) <= ANCHOR_TOL
        and all(abs(hy_accuracy[k] - HYBRID_ANCHORS[k]) <= ANCHOR_TOL
                for k in HYBRID_ANCHORS)) if smoke is None else True

    # ---- repaired hybrid: per-block L2 norm + symmetric solve ------------
    norm_ms, norm_dino = _column_norms(train_ms, train_dino)

    def train_block(start: int, stop: int) -> np.ndarray:
        return np.concatenate([
            np.asarray(train_ms[start:stop]).astype(np.float64) / norm_ms,
            np.asarray(train_dino[start:stop]).astype(np.float64) / norm_dino,
        ], axis=1)

    def test_block(start: int, stop: int) -> np.ndarray:
        return np.concatenate([
            np.asarray(test_ms[start:stop]).astype(np.float64) / norm_ms,
            np.asarray(test_dino[start:stop]).astype(np.float64) / norm_dino,
        ], axis=1)

    width = train_ms.shape[1] + train_dino.shape[1]
    acc_rep = RidgeAccumulator(width, CLASSES)
    for start in range(0, train_rows, BLOCK):
        stop = min(start + BLOCK, train_rows)
        acc_rep.add(train_block(start, stop), labels[start:stop])
    std_rep = acc_rep.standardiser()
    centred, cross, intercept = acc_rep._standardised_system()
    system = symmetric_system(centred)
    # M296b: driver pinned to evd (parallel divide-and-conquer)
    from scipy import linalg as scipy_linalg
    vals, vecs = scipy_linalg.eigh(system, check_finite=False,
                                   driver="evd")
    g_proj = vecs.T @ cross
    grid = [float(v) for v in config["lambda_grid"]]
    loocv = loocv_ridge(vals, vecs, g_proj, intercept, train_block,
                        labels[:train_rows], std_rep, grid)
    valid_grid = [lam for lam in grid if loocv["valid"][str(lam)]]
    lambda_star = (min(valid_grid, key=lambda lam: loocv["loocv"][str(lam)])
                   if valid_grid else None)

    def eigen_weights(lam: float) -> np.ndarray:
        # M296d: strongly-convex penalised modes only (non-positive
        # penalised modes are non-convex; near-zero positives amplify
        # beyond the kept part's resolution)
        penalised = vals + lam
        scale_pen = max(abs(float(penalised[0])),
                        abs(float(penalised[-1])))
        cutoff_pen = max(0.0, scale_pen * SOLVE_STRONG_CONVEX_CUTOFF)
        keep = penalised > cutoff_pen
        inv = np.where(keep, 1.0 / penalised, 0.0)
        w = vecs @ (inv[:, None] * g_proj)
        return np.vstack([w, intercept[None, :]])

    repaired_by_penalty: dict[str, float] = {}
    repaired_at_star = None
    solve_report = None
    for p in penalties:
        repaired_by_penalty[str(p)] = _score(eigen_weights(p), test_block,
                                             test_labels[:test_rows],
                                             test_rows, std_rep)
    if lambda_star is not None:
        repaired_at_star = _score(eigen_weights(lambda_star), test_block,
                                  test_labels[:test_rows], test_rows,
                                  std_rep)
    solve_report = _eigen_backward(vals, vecs, eigen_weights(1.0)[:-1],
                                   cross, 1.0)

    g3 = (all(np.isfinite(v) for v in loocv["loocv"].values())
          and (valid_grid or lambda_star is None))
    g4 = bool(solve_report and solve_report["backward_passed"])
    accs = [ms_accuracy, repaired_at_star] + list(
        repaired_by_penalty.values()) + list(hy_accuracy.values())
    g5 = all(v is None or 0.0 <= v <= 1.0 for v in accs)
    h26_2 = bool(repaired_at_star is not None
                 and repaired_at_star >= MS_ANCHOR)

    gates = {
        "g1_premise_shapes_and_digests": {
            "ok": g1, "train_rows_on_disk": len(mem_train),
            "test_rows_on_disk": len(mem_test), "label_rows": len(labels),
            "dino_train_rows": len(dino_train),
            "dino_test_rows": len(dino_test),
            "expected": [FULL_TRAIN_ROWS, SEALED_TEST_ROWS]},
        "g2_lu_anchor_reproductions": {
            "ok": g2, "ms_accuracy": ms_accuracy,
            "ms_sealed": MS_ANCHOR, "hybrid_accuracy": hy_accuracy,
            "hybrid_sealed": HYBRID_ANCHORS,
            "note": "skipped in smoke mode" if smoke is not None else None},
        "g3_loocv_machinery_valid": {
            "ok": g3, "loocv": loocv["loocv"],
            "valid_grid_points": [str(v) for v in valid_grid],
            "hat_margin": loocv["hat_margin"],
            "max_hat_diagonal": loocv["max_hat_diagonal"],
            "min_margin": loocv["min_margin"]},
        "g4_repaired_solve_backward": {
            "ok": g4, "solve_report": solve_report},
        "g5_accuracies_valid": {
            "ok": g5, "repaired_by_penalty": repaired_by_penalty,
            "repaired_at_star": repaired_at_star,
            "hybrid_lu": hy_accuracy},
    }
    gates_ok = all(g["ok"] for g in gates.values())
    h26_2_pass = h26_2 and gates_ok

    evidence: dict[str, Any] = {
        "milestone": "M299",
        "cell": "per-block L2 normalization + repaired solver on the "
                "M228 hybrid, without native-resolution re-extraction",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "smoke": (smoke is not None),
        "smoke_rows": smoke,
        "ms_anchor": MS_ANCHOR,
        "hybrid_sealed_anchors": HYBRID_ANCHORS,
        "ms_accuracy_lu": ms_accuracy,
        "hybrid_lu_by_penalty": hy_accuracy,
        "loocv_by_lambda": loocv["loocv"],
        "lambda_star": lambda_star,
        "repaired_by_penalty": repaired_by_penalty,
        "repaired_at_star": repaired_at_star,
        "repaired_delta_vs_ms_anchor":
            (repaired_at_star - MS_ANCHOR
             if repaired_at_star is not None else None),
        "h26_2_hybrid_ge_ms_anchor": bool(h26_2),
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": (not gates_ok) and smoke is None,
        "verdict": {
            "passes": bool(gates_ok),
            "h26_2": "PASS - the hybrid regression localises to "
                     "conditioning" if h26_2_pass else
                     ("FAIL - the regression localises to the upscaling "
                      "confound" if gates_ok else "a gate failed — VOID"),
        },
        "scope": "full 409,832-row train schedule, sealed 34,500-row "
                 "test, cached ms + DINOv2 (32x32 upscaled) features, "
                 "lambda train-side only",
        "runtime_seconds": round(time.time() - started, 2),
    }
    if smoke is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        print(f"M299 complete -> {output_dir / 'evidence.json'}", flush=True)
    print(json.dumps({"gates_ok": gates_ok,
                      "lambda_star": lambda_star,
                      "repaired_by_penalty": repaired_by_penalty,
                      "repaired_at_star": repaired_at_star,
                      "h26_2": h26_2}, indent=1), flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", type=int, nargs=2, default=None,
                        metavar=("TRAIN_ROWS", "TEST_ROWS"),
                        help="smoke: fit on the first TRAIN_ROWS and score "
                             "the first TEST_ROWS; no evidence is written")
    args = parser.parse_args()
    run_m299(args.config, args.output, smoke=tuple(args.smoke)
             if args.smoke else None)


if __name__ == "__main__":
    main()
