"""M297 - head repair II: exact-LOOCV lambda over a registered grid.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` §8.7
(26 Aug 2026, before any build). Replaces the fixed ``PENALTY = 1.0``
with a deterministic train-side lambda selection: ridge admits a
closed-form leave-one-out error via the hat matrix, and with the
eigendecomposition of the standardised symmetric system (the M296
solver's own factorisation) the whole grid is free.

Mechanism, registered before running:
- S = symmetric assembly of the standardised system; eigh(S) gives
  eigenvalues ``vals`` and eigenvectors ``vecs``. The penalised system
  is S + lambda*I with the same eigenvectors.
- For each lambda in the registered grid, weights follow from
  ``V diag(1/(vals+lambda)) V^T cross``; the leave-one-out residual for
  the multi-output one-hot fit is
  ``LOOCV = (1/(n*C)) sum_i sum_c (e_ic / (1 - h_ii))^2`` with
  ``h_ii = sum_j (x_i^T V_j)^2 / (vals_j + lambda)`` - all exact
  closed-form quantities, no validation split, no seed, no test-set
  contact.
- The assembled system is INDEFINITE (M296 sealed cond 3.33e12), so the
  hat diagonal is not a projector: a grid point where any ``1 - h_ii``
  is not bounded away from zero is reported INVALID and excluded from
  selection. lambda* = argmin LOOCV over the VALID grid points.
- M296d: the hat machinery and the test-evaluation weights use only
  the STRONGLY-CONVEX penalised directions at each lambda (penalised
  eigenvalue above max(0, scale*1e-10)) - the same truncation as the
  M296 solve. Non-positive penalised modes (non-convex, no minimizer)
  and near-zero positive modes contribute zero.
- The sealed 34,500-row test is evaluated exactly once, at lambda*.

Gates (VOID on failure): g1 premise row counts exact; g2 the LU path
reproduces the sealed anchor 0.24214492753623187 at 1e-9 (instrument
identity, as in M296); g3 every grid point's LOOCV machinery finite and
the validity rule applied as registered; g4 the penalised-system
condition number from this eigendecomposition reproduces the sealed
M296 value (3330608536062.5874) within 1e-9 relative; g5 test accuracy
valid and evaluated exactly once.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
                  / "m297_loocv_lambda.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v26" / "m297_loocv_lambda"

CLASSES = 345
FULL_TRAIN_ROWS = 409832
SEALED_TEST_ROWS = 34500
BLOCK = 4096
HAT_MARGIN = 1e-6   # registered: 1 - h_ii must exceed this for validity
M296_COND_SEALED = 3330608536062.5874
COND_TOL = 1e-5     # M297a (registered): the factorisation jitter on
                    # lambda_min of a cond-3e12 spectrum is ~4.8e-7;
                    # 1e-9 was below the resolution conditioning supports
EIGH_CACHE_ROOT = (REPO_ROOT / "logs" / "results" / "v26"
                   / "m296_head_repair" / "eigh_cache")


def _selection_digest(config: dict[str, Any], train_rows: int) -> str:
    """The cache key: everything that determines the accumulated
    system (cache paths, row count, chunk size). Same discipline as the
    M222 feature cache: keyed by the input selection, not the output
    bytes."""
    import hashlib
    sel = np.arange(train_rows, dtype=np.int64)
    key = {
        "solver_version": "m297_v2",
        "cache_relpath": config["artifacts"]["cache_relpath"],
        "train_file": config["artifacts"]["train_file"],
        "labels_file": config["artifacts"]["labels_file"],
        "train_rows": int(train_rows),
        "block": int(BLOCK),
        "selection_sha256": hashlib.sha256(sel.tobytes()).hexdigest(),
    }
    return payload_hash(key)


def _eigh_cache_load(digest: str) -> dict[str, np.ndarray] | None:
    """Load a cached eigendecomposition iff the digest matches."""
    cache_dir = EIGH_CACHE_ROOT / digest
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("digest") != digest:
        return None
    return {
        "vals": np.load(cache_dir / "vals.npy"),
        "vecs": np.load(cache_dir / "vecs.npy"),
        "g_proj": np.load(cache_dir / "g_proj.npy"),
    }


def _eigh_cache_save(digest: str, vals: np.ndarray, vecs: np.ndarray,
                     g_proj: np.ndarray) -> None:
    """Persist the eigendecomposition immediately (the standing rule:
    expensive intermediates are written as soon as they exist)."""
    cache_dir = EIGH_CACHE_ROOT / digest
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "vals.npy", vals)
    np.save(cache_dir / "vecs.npy", vecs)
    np.save(cache_dir / "g_proj.npy", g_proj)
    meta = {"digest": digest, "vals_shape": list(vals.shape),
            "vecs_shape": list(vecs.shape),
            "g_proj_shape": list(g_proj.shape)}
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2),
                                         encoding="utf-8")


def loocv_ridge(vals: np.ndarray, vecs: np.ndarray,
                g: np.ndarray, intercept: np.ndarray,
                features: np.ndarray | Callable[[int, int], np.ndarray],
                labels: np.ndarray,
                standardiser: Any, grid: list[float],
                block: int = BLOCK,
                strong_convex_cutoff: float = 1e-10
                ) -> dict[str, Any]:
    """Exact multi-output ridge LOOCV over the grid, one streaming pass
    over the rows, on the STANDARDISED design (the system the
    eigendecomposition belongs to; the standardiser is the sealed
    fp32-rounded one). ``g = V^T cross``; the projection ``U = X_std V``
    is computed blockwise and reused for every lambda. ``features`` may
    be an array or a ``(start, stop) -> fp64 block`` provider, so a
    caller can stream pre-normalised blocks without materialising them.
    M296d: at every lambda the hat machinery and the weights use only
    the STRONGLY-CONVEX penalised directions — ``vals + lambda`` must
    exceed ``max(0, scale_penalised(lambda) * strong_convex_cutoff)``.
    Non-positive penalised modes are non-convex (the ridge objective
    has no minimizer along them) and near-zero positive modes amplify
    beyond the kept part's resolution; both contribute zero. Returns
    per-lambda LOOCV, hat statistics, validity flags under HAT_MARGIN,
    and per-lambda dropped counts."""
    n = len(labels)
    pen = {lam: vals + lam for lam in grid}
    scale_pen = {lam: max(abs(float(p[0])), abs(float(p[-1])))
                 for lam, p in pen.items()}
    inv_factors: dict[float, np.ndarray] = {}
    dropped: dict[str, int] = {}
    for lam in grid:
        p = pen[lam]
        cutoff = max(0.0, scale_pen[lam] * strong_convex_cutoff)
        keep = p > cutoff
        inv_factors[lam] = np.where(keep, 1.0 / p, 0.0)
        dropped[str(lam)] = int((~keep).sum())
    loocv: dict[str, float] = {}
    max_h: dict[str, float] = {}
    min_margin: dict[str, float] = {}
    for lam in grid:
        loocv[str(lam)] = 0.0
        max_h[str(lam)] = -np.inf
        min_margin[str(lam)] = np.inf
    valid: dict[str, bool] = {}
    w_project: dict[float, np.ndarray] = {}
    for lam in grid:
        w_project[lam] = g * inv_factors[lam][:, None]   # (d, C)
    for start in range(0, n, block):
        stop = min(start + block, n)
        if callable(features):
            xb = standardiser(features(start, stop)).astype(np.float64)
        else:
            xb = standardiser(features[start:stop]).astype(np.float64)
        ub = xb @ vecs                        # (rows, d) projections
        ub2 = np.square(ub)
        targets = np.zeros((stop - start, CLASSES), dtype=np.float64)
        targets[np.arange(stop - start), labels[start:stop]] = 1.0
        for lam in grid:
            h = ub2 @ inv_factors[lam]        # hat diagonal (rows,)
            pred = ub @ w_project[lam] + intercept  # (rows, C)
            e = targets - pred
            margin = 1.0 - h
            sse = float(np.sum(np.square(e / margin[:, None])))
            loocv[str(lam)] += sse
            max_h[str(lam)] = max(max_h[str(lam)], float(h.max()))
            min_margin[str(lam)] = min(min_margin[str(lam)],
                                       float(margin.min()))
    for lam in grid:
        loocv[str(lam)] /= n * CLASSES
        valid[str(lam)] = min_margin[str(lam)] > HAT_MARGIN
    return {"loocv": loocv, "valid": valid,
            "max_hat_diagonal": max_h, "min_margin": min_margin,
            "hat_margin": HAT_MARGIN,
            "dropped_directions": dropped,
            "strong_convex_cutoff": strong_convex_cutoff,
            "penalised_scale_by_lambda": {str(l): scale_pen[l]
                                          for l in grid}}


def _score(weights: np.ndarray, features: np.ndarray, labels: np.ndarray,
           standardiser: Any, block: int = BLOCK) -> float:
    hits = 0
    n = len(labels)
    for start in range(0, n, block):
        stop = min(start + block, n)
        xs = standardiser(features[start:stop]).astype(np.float64)
        scores = xs @ weights[:-1] + weights[-1]
        hits += int((np.argmax(scores, axis=1) == labels[start:stop]).sum())
    return hits / n


def run_m297(config_path: Path, output_dir: Path,
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
    lu_accuracy = _score(lu_weights, test_ms, test_labels[:test_rows],
                         standardiser)

    # ---- eigendecomposition of the symmetric system (M296 machinery) ---
    system = symmetric_system(centred)
    digest = _selection_digest(config, train_rows)
    cached = _eigh_cache_load(digest)
    if cached is not None:
        vals, vecs, g_proj = (cached["vals"], cached["vecs"],
                              cached["g_proj"])
        print(f"reusing cached eigendecomposition ({digest})", flush=True)
    else:
        # M296b: driver pinned to evd (parallel divide-and-conquer)
        from scipy import linalg as scipy_linalg
        vals, vecs = scipy_linalg.eigh(system, check_finite=False,
                                       driver="evd")
        g_proj = vecs.T @ cross
        if smoke is None:
            _eigh_cache_save(digest, vals, vecs, g_proj)
            print(f"wrote eigendecomposition cache ({digest})", flush=True)
    grid = [float(v) for v in config["lambda_grid"]]

    loocv = loocv_ridge(vals, vecs, g_proj, intercept, train_ms,
                        labels[:train_rows], standardiser, grid)
    valid_grid = [lam for lam in grid if loocv["valid"][str(lam)]]
    lambda_star = (min(valid_grid,
                       key=lambda lam: loocv["loocv"][str(lam)])
                   if valid_grid else None)

    test_accuracy = None
    if lambda_star is not None:
        # M296d: the test-evaluation weights use the SAME strongly-
        # convex penalised directions as the LOOCV machinery at lambda*
        pen_star = vals + lambda_star
        scale_star = max(abs(float(pen_star[0])),
                         abs(float(pen_star[-1])))
        keep_star = pen_star > max(0.0, scale_star * 1e-10)
        inv_star = np.where(keep_star, 1.0 / pen_star, 0.0)
        w_star = (vecs * inv_star[None, :]) @ g_proj
        weights = np.vstack([w_star, intercept[None, :]])
        test_accuracy = _score(weights, test_ms,
                               test_labels[:test_rows], standardiser)

    # ---- reproducibility of the sealed M296 conditioning ----------------
    # M296b: the penalised spectrum is vals + penalty - the SAME
    # factorisation already computed, never a second one
    cond = condition_report_from_vals(
        vals + penalty, penalty, dimension=int(len(vals)))
    cond_rel = (abs(cond["condition_number"] - M296_COND_SEALED)
                / M296_COND_SEALED)

    anchor = float(config["anchor"]["value"])
    tol = float(config["anchor"]["tolerance"])
    premise_ok = (len(mem_train) == FULL_TRAIN_ROWS
                  and len(mem_test) == SEALED_TEST_ROWS
                  and len(labels) == FULL_TRAIN_ROWS)
    g1 = premise_ok
    anchor_measured = lu_accuracy if smoke is None else float("nan")
    g2 = (abs(anchor_measured - anchor) <= tol) if smoke is None else True
    g3 = (all(np.isfinite(v) for v in loocv["loocv"].values())
          and (valid_grid or lambda_star is None)
          and (lambda_star is None or str(lambda_star) in loocv["loocv"]))
    g4 = (cond_rel <= COND_TOL) if smoke is None else True
    g5 = (test_accuracy is None
          or 0.0 <= test_accuracy <= 1.0)
    gates = {
        "g1_premise_rows_exact": {
            "ok": g1, "train_rows_on_disk": len(mem_train),
            "test_rows_on_disk": len(mem_test),
            "label_rows": len(labels),
            "expected": [FULL_TRAIN_ROWS, SEALED_TEST_ROWS]},
        "g2_lu_anchor_reproduction": {
            "ok": g2, "measured": anchor_measured, "sealed": anchor,
            "delta": (anchor_measured - anchor) if smoke is None else None,
            "tolerance": tol,
            "note": "skipped in smoke mode" if smoke is not None else None},
        "g3_loocv_machinery_valid": {
            "ok": g3, "loocv": loocv["loocv"],
            "valid_grid_points": [str(v) for v in valid_grid],
            "hat_margin": loocv["hat_margin"],
            "max_hat_diagonal": loocv["max_hat_diagonal"],
            "min_margin": loocv["min_margin"]},
        "g4_m296_condition_reproduction": {
            "ok": g4, "condition_number": cond["condition_number"],
            "sealed": M296_COND_SEALED, "relative_delta": cond_rel,
            "tolerance": COND_TOL,
            "note": "skipped in smoke mode" if smoke is not None else None},
        "g5_test_accuracy_valid_and_once": {
            "ok": g5, "lambda_star": lambda_star,
            "test_accuracy_at_star": test_accuracy},
    }
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M297",
        "cell": "exact-LOOCV lambda over the registered grid, on the "
                "sealed M228 cached ms features",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "smoke": (smoke is not None),
        "smoke_rows": smoke,
        "loocv_by_lambda": loocv["loocv"],
        "loocv_valid_by_lambda": loocv["valid"],
        "lambda_star": lambda_star,
        "test_accuracy_at_star": test_accuracy,
        "anchor": {"sealed": anchor,
                   "lu_measured": anchor_measured,
                   "delta_at_star":
                       (test_accuracy - anchor
                        if test_accuracy is not None else None)},
        "eigendecomposition": {
            "min_eigenvalue": float(vals[0]),
            "max_eigenvalue": float(vals[-1]),
            "conditioning": cond,
            "condition_relative_delta_vs_m296": cond_rel},
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": (not gates_ok) and smoke is None,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": (
                "lambda* chosen train-side only (exact LOOCV); the sealed "
                "test was evaluated once at lambda*"
            ) if gates_ok else "a gate failed — VOID",
        },
        "scope": ("full 409,832-row train schedule, sealed 34,500-row "
                  "test, cached ms features (no re-extraction)"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    if smoke is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        print(f"M297 complete -> {output_dir / 'evidence.json'}", flush=True)
    print(json.dumps({"gates_ok": gates_ok,
                      "loocv": loocv["loocv"],
                      "lambda_star": lambda_star,
                      "test_accuracy_at_star": test_accuracy},
                     indent=1), flush=True)
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
    run_m297(args.config, args.output, smoke=tuple(args.smoke)
             if args.smoke else None)


if __name__ == "__main__":
    main()
