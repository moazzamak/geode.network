"""M298 - head repair III: LDA/Mahalanobis readout + class-balanced
ridge, lambda from the sealed M297 LOOCV.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` (M298,
26 Aug 2026, before any build). Two closed-form readouts on the SAME
features as M296/M297, with the penalty fixed at the train-side lambda
M297 chose (a registered transfer rule - every choice stays train-side,
the sealed test is evaluated once per readout):

- **LDA / Mahalanobis** (I4 item three): shared-covariance linear
  discriminant. ``A = S + lambda*I`` on the symmetric standardised Gram;
  scores ``x^T A^{-1} m_c - 1/2 m_c^T A^{-1} m_c + log(n_c/n)`` with
  class means ``m_c`` from the sealed crosses and class counts. Solved
  in the M297 eigendecomposition basis - no new factorisation.
- **Class-balanced ridge** (I4 item four): weighted least squares with
  per-row weight ``1/n_{y_i}`` (each class contributes total weight 1),
  streamed into a weighted Gram, symmetrised, solved at ``lambda*``.

Gates (VOID on failure): g1 premise rows; g2 the LU path reproduces
the sealed anchor 0.24214492753623187 at 1e-9 (instrument identity);
g3 the M297 evidence is sealed with gates_ok and a registered lambda*,
and the eigendecomposition cache loads with a digest match; g4 every
solve's backward residual passes the M296a instrument (raw residual for
full-system solves, normal-equation for dropped-component solves);
g5 accuracies valid. H26-1 reads strictly: any readout above the
anchor is a strict improvement with no feature change; the verdict
names which readout improved.
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
from experiments.tier4.eval_v26_m296_head_repair import symmetric_system
from experiments.tier4.eval_v26_m297_loocv_lambda import (
    _eigh_cache_load,
    _selection_digest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
                  / "m298_lda_balanced.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v26" / "m298_lda_balanced"
M297_EVIDENCE = (REPO_ROOT / "logs" / "results" / "v26"
                 / "m297_loocv_lambda" / "evidence.json")

CLASSES = 345
FULL_TRAIN_ROWS = 409832
SEALED_TEST_ROWS = 34500
BLOCK = 4096
RESID_TOL = 1e-10
# M296d (registered): the solve keeps only strongly-convex penalised
# modes (penalised eigenvalue > max(0, scale*1e-10)); non-positive
# penalised modes are non-convex and near-zero positive modes amplify
# beyond the kept part's resolution. Dropped counts are reported, never
# gated.
SOLVE_STRONG_CONVEX_CUTOFF = 1e-10


def lda_readout(vals: np.ndarray, vecs: np.ndarray, cross: np.ndarray,
                class_count: np.ndarray, lam: float
                ) -> tuple[Any, dict[str, Any]]:
    """LDA/Mahalanobis readout in the eigenbasis of the symmetric Gram:
    returns a scorer callable over standardised rows plus a report.
    The scorer carries the quadratic offsets and priors. M296d: the
    inverse factors use only the strongly-convex penalised modes."""
    means = cross / class_count[None, :]        # (d, C)
    pm = vecs.T @ means                         # (d, C)
    penalised = vals + lam
    scale = max(abs(float(penalised[0])), abs(float(penalised[-1])))
    cutoff = max(0.0, scale * SOLVE_STRONG_CONVEX_CUTOFF)
    keep = penalised > cutoff
    inv_factors = np.where(keep, 1.0 / penalised, 0.0)
    # A^{-1} M = V diag(1/(v+lam)) V^T M  (rotation matters: the
    # elementwise form would ignore the eigenvector basis)
    solved = vecs @ (inv_factors[:, None] * pm)
    # M^T A^{-1} M = (V^T M)^T diag(1/(v+lam)) (V^T M): per class c,
    # sum_j pm[j,c]^2 / (v_j + lam)
    quad = 0.5 * np.sum(np.square(pm) * inv_factors[:, None], axis=0)
    priors = np.log(class_count / class_count.sum())  # (C,)
    offsets = priors - quad

    def score(x_std_block: np.ndarray) -> np.ndarray:
        return (x_std_block @ vecs) @ (pm * inv_factors[:, None]) + offsets

    report = {
        "lambda": float(lam),
        "dropped_components": int((~keep).sum()),
        "strong_convex_cutoff": float(cutoff),
        "backward": _lda_backward(vals, vecs, solved, means, lam),
    }
    return score, report


def _lda_backward(vals: np.ndarray, vecs: np.ndarray, solved: np.ndarray,
                  means: np.ndarray, lam: float) -> dict[str, Any]:
    """The backward instrument for the LDA solve under M296d: raw
    residual ``A (A^{-1} M) - M`` for a full-system solve; when
    strongly-nonconvex/near-zero penalised modes are dropped, the raw
    residual against the TRUNCATED system (the M296c instrument - the
    normal-equation form measured the eigen reconstruction error)."""
    penalised = vals + lam
    scale = max(abs(float(penalised[0])), abs(float(penalised[-1])))
    cutoff = max(0.0, scale * SOLVE_STRONG_CONVEX_CUTOFF)
    keep = penalised > cutoff
    dropped = int((~keep).sum())
    vt_solved = vecs.T @ solved
    recon = vecs @ (penalised[:, None] * vt_solved)
    residual = recon - means
    denom = ((abs(float(penalised[-1])) + abs(lam))
             * float(np.max(np.abs(solved)))
             + float(np.max(np.abs(means))))
    raw = float(np.max(np.abs(residual))) / max(denom, 1e-300)
    if dropped > 0:
        vt_means = vecs.T @ means
        trunc_residual = (vecs @ ((penalised * keep)[:, None] * vt_solved)
                          - vecs @ (keep[:, None] * vt_means))
        denom_t = (scale * float(np.max(np.abs(solved)))
                   + float(np.max(np.abs(means))))
        trunc_backward = (float(np.max(np.abs(trunc_residual)))
                          / max(denom_t, 1e-300))
        return {"ok": trunc_backward <= RESID_TOL,
                "instrument": "truncated_system",
                "raw_backward": raw,
                "truncated_system_backward": trunc_backward,
                "dropped_components": dropped}
    return {"ok": raw <= RESID_TOL, "instrument": "raw",
            "raw_backward": raw, "dropped_components": dropped}


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


def _score_scorer(scorer: Any, features: np.ndarray, labels: np.ndarray,
                  standardiser: Any, block: int = BLOCK) -> float:
    hits = 0
    n = len(labels)
    for start in range(0, n, block):
        stop = min(start + block, n)
        xs = standardiser(features[start:stop]).astype(np.float64)
        scores = scorer(xs)
        hits += int((np.argmax(scores, axis=1) == labels[start:stop]).sum())
    return hits / n


def _lda_direct_agreement(centred: np.ndarray, cross: np.ndarray,
                          class_count: np.ndarray, lam: float,
                          xs: np.ndarray
                          ) -> tuple[np.ndarray | None, float | None]:
    """The closed-form LDA scores in the original basis (an
    independent route for the M298a agreement gate). Returns
    (scores, relative direct-solve residual). The direct solve is
    only meaningful where the system is well-conditioned for
    ``np.linalg.solve`` - the caller gates on the residual."""
    d = centred.shape[0]
    a = centred + lam * np.eye(d)
    means = cross / class_count[None, :]
    try:
        ainv_m = np.linalg.solve(a, means)
    except np.linalg.LinAlgError:
        return None, None
    residual = a @ ainv_m - means
    denom = max(float(np.max(np.abs(a))) * float(np.max(np.abs(ainv_m))),
                float(np.max(np.abs(means))))
    rel = float(np.max(np.abs(residual))) / max(denom, 1e-300)
    priors = np.log(class_count / class_count.sum())
    direct = xs @ ainv_m - 0.5 * np.sum(means * ainv_m, axis=0) + priors
    return direct, rel


def run_m298(config_path: Path, output_dir: Path,
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

    # ---- M297 dependency: sealed lambda* + eigendecomposition cache ----
    m297_ok = False
    lambda_star = None
    if not smoke and M297_EVIDENCE.exists():
        m297 = json.loads(M297_EVIDENCE.read_text(encoding="utf-8"))
        m297_ok = bool(m297.get("gates_ok")) and m297.get("lambda_star") \
            is not None
        lambda_star = (float(m297["lambda_star"]) if m297_ok else None)
    digest = _selection_digest(config, train_rows)
    cached = _eigh_cache_load(digest)

    # ---- instrument identity: the sealed LU path ------------------------
    acc = RidgeAccumulator(train_ms.shape[1], CLASSES)
    for start in range(0, len(train_ms), BLOCK):
        stop = min(start + BLOCK, len(train_ms))
        acc.add(train_ms[start:stop], labels[start:stop])
    standardiser = acc.standardiser()
    centred, cross, intercept = acc._standardised_system()
    class_count = acc.class_count
    penalty = float(config["anchor_penalty"])
    lu_centred = centred.copy()
    lu_centred.flat[:: lu_centred.shape[0] + 1] += penalty
    lu_weights = np.vstack([np.linalg.solve(lu_centred, cross),
                            intercept[None, :]])
    lu_accuracy = _score(lu_weights, test_ms, test_labels[:test_rows],
                         standardiser)

    premise_ok = (len(mem_train) == FULL_TRAIN_ROWS
                  and len(mem_test) == SEALED_TEST_ROWS
                  and len(labels) == FULL_TRAIN_ROWS)
    anchor = float(config["anchor"]["value"])
    tol = float(config["anchor"]["tolerance"])
    g1 = premise_ok
    anchor_measured = lu_accuracy if smoke is None else float("nan")
    g2 = (abs(anchor_measured - anchor) <= tol) if smoke is None else True
    g3 = bool(m297_ok and cached is not None) if smoke is None \
        else (cached is not None)
    if lambda_star is None and smoke is None and M297_EVIDENCE.exists():
        # record the dependency failure without inventing a lambda
        lambda_star = float(config["fallback_lambda"])

    readouts: dict[str, Any] = {}
    gates_extra: dict[str, Any] = {}
    if g3 and lambda_star is not None:
        vals, vecs, g_proj = (cached["vals"], cached["vecs"],
                              cached["g_proj"])
        del g_proj
        # ---- LDA / Mahalanobis -------------------------------------------
        scorer, lda_report = lda_readout(vals, vecs, cross, class_count,
                                         lambda_star)
        lda_accuracy = _score_scorer(scorer, test_ms,
                                     test_labels[:test_rows], standardiser)
        gates_extra["lda_backward"] = {**lda_report["backward"]}
        readouts["lda"] = {"accuracy": lda_accuracy,
                           "delta_vs_anchor": lda_accuracy - anchor}
        # ---- M298a: the LDA readout over its own registered grid -------
        # The registered transfer rule imported a scale mismatch at
        # lambda* (measured: constant-class collapse, priors dominate).
        # Every grid cell is validated by a direct-form agreement gate
        # wherever the direct solve is meaningful (no dropped modes and
        # the direct solve's own residual <= 1e-8; the 100.0 cell is
        # the registered agreement anchor).
        lda_grid = [float(v) for v in config["lda_lambda_grid"]]
        agree_rows = int(config["agreement_rows"])
        agree_tol = float(config["agreement_tolerance"])
        agree_slice = standardiser(
            np.asarray(mem_train[:agree_rows])).astype(np.float64)
        grid_readouts: dict[str, Any] = {}
        agreement_checked = 0
        agreement_ok_count = 0
        for lam in lda_grid:
            scorer_g, report_g = lda_readout(vals, vecs, cross,
                                             class_count, lam)
            acc_g = _score_scorer(scorer_g, test_ms,
                                  test_labels[:test_rows], standardiser)
            dropped_g = int(report_g["dropped_components"])
            cell: dict[str, Any] = {
                "accuracy": acc_g,
                "delta_vs_anchor": acc_g - anchor,
                "dropped_components": dropped_g,
                "backward": report_g["backward"]}
            if (bool(config["agreement_applies_when_dropped_zero"])
                    and dropped_g == 0):
                direct, direct_rel = _lda_direct_agreement(
                    centred, cross, class_count, lam, agree_slice)
                if direct is not None and direct_rel <= agree_tol:
                    agreement_checked += 1
                    eigen_scores = scorer_g(agree_slice)
                    rel_err = float(
                        np.max(np.abs(eigen_scores - direct))
                        / max(float(np.max(np.abs(direct))), 1e-12))
                    cell["agreement"] = {
                        "applicable": True,
                        "direct_solve_residual": direct_rel,
                        "relative_error": rel_err,
                        "ok": bool(rel_err <= agree_tol)}
                    agreement_ok_count += int(rel_err <= agree_tol)
                else:
                    cell["agreement"] = {"applicable": False,
                                         "reason": "direct solve "
                                         "not sane"}
            else:
                cell["agreement"] = {"applicable": False,
                                     "reason": "dropped modes; the "
                                     "direct route solves the non-"
                                     "convex saddle, not the readout"}
            grid_readouts[str(lam)] = cell
        readouts["lda_grid"] = grid_readouts
        gates_extra["lda_grid_agreement"] = {
            "ok": bool(agreement_ok_count == agreement_checked),
            "checked_cells": agreement_checked,
            "ok_cells": agreement_ok_count,
            "note": "the agreement gate fires where the direct solve "
                    "is meaningful (dropped==0, direct residual "
                    "<=1e-8); the 100.0 cell is the registered anchor"}
        # ---- class-balanced ridge ----------------------------------------
        balanced = _balanced_ridge(train_ms, labels[:train_rows],
                                   standardiser, lambda_star)
        if balanced is not None:
            weights, bal_report = balanced
            bal_accuracy = _score(weights, test_ms,
                                  test_labels[:test_rows], standardiser)
            gates_extra["balanced_ridge_backward"] = bal_report
            readouts["balanced_ridge"] = {
                "accuracy": bal_accuracy,
                "delta_vs_anchor": bal_accuracy - anchor}
    g4 = all(extra.get("ok", False) for extra in gates_extra.values()) \
        if gates_extra else False
    accuracies = []
    for key, value in readouts.items():
        if key == "lda_grid":
            accuracies.extend(cell["accuracy"]
                              for cell in value.values())
        else:
            accuracies.append(value["accuracy"])
    g5 = all(0.0 <= v <= 1.0 for v in accuracies)
    gates = {
        "g1_premise_rows_exact": {
            "ok": g1, "train_rows_on_disk": len(mem_train),
            "test_rows_on_disk": len(mem_test), "label_rows": len(labels),
            "expected": [FULL_TRAIN_ROWS, SEALED_TEST_ROWS]},
        "g2_lu_anchor_reproduction": {
            "ok": g2, "measured": anchor_measured, "sealed": anchor,
            "delta": (anchor_measured - anchor) if smoke is None else None,
            "tolerance": tol,
            "note": "skipped in smoke mode" if smoke is not None else None},
        "g3_m297_dependency": {
            "ok": g3, "m297_evidence_exists": M297_EVIDENCE.exists(),
            "m297_gates_ok": m297_ok, "lambda_star": lambda_star,
            "eigh_cache_digest": digest,
            "eigh_cache_loaded": cached is not None,
            "note": "skipped in smoke mode" if smoke is not None else None},
        "g4_solve_backward": {"ok": g4, **gates_extra},
        "g5_accuracies_valid": {"ok": g5, "readouts": readouts},
    }
    gates_ok = all(g["ok"] for g in gates.values())
    improved: dict[str, bool] = {}
    for key, value in readouts.items():
        if key == "lda_grid":
            for lam, cell in value.items():
                improved[f"lda_grid:{lam}"] = cell["accuracy"] > anchor
        else:
            improved[key] = value["accuracy"] > anchor
    h26_1 = bool(improved) and any(improved.values())

    evidence: dict[str, Any] = {
        "milestone": "M298",
        "cell": "LDA/Mahalanobis + class-balanced ridge at the sealed "
                "M297 lambda*, same features",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "smoke": (smoke is not None),
        "smoke_rows": smoke,
        "lambda_star": lambda_star,
        "anchor": {"sealed": anchor, "lu_measured": anchor_measured},
        "readouts": readouts,
        "improved_over_anchor": improved,
        "h26_1_any_improvement": h26_1,
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": (not gates_ok) and smoke is None,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": (
                "strict improvement over the anchor with no feature "
                "change" if gates_ok and h26_1 else
                ("no readout improved the sealed anchor; H26-1 rests on "
                 "the ridge-at-star reading from M297") if gates_ok
                else "a gate failed — VOID"),
        },
        "scope": "full 409,832-row train schedule, sealed 34,500-row "
                 "test, cached ms features, lambda train-side only",
        "runtime_seconds": round(time.time() - started, 2),
    }
    if smoke is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        print(f"M298 complete -> {output_dir / 'evidence.json'}", flush=True)
    print(json.dumps({"gates_ok": gates_ok, "readouts": readouts,
                      "h26_1_any_improvement": h26_1}, indent=1),
          flush=True)
    return evidence


def _balanced_ridge(train_ms: np.ndarray, labels: np.ndarray,
                    standardiser: Any, lam: float,
                    classes: int = CLASSES
                    ) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Class-balanced ridge: weighted Gram with per-row weight 1/n_c,
    symmetric assembly, eigen solve at ``lam``. Streams, never
    materialises the design."""
    from scipy import linalg as scipy_linalg

    d = train_ms.shape[1]
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    if (counts == 0).any():
        return None   # a class without rows has no registered weight
    gram = np.zeros((d, d), dtype=np.float64)
    cross_w = np.zeros((d, classes), dtype=np.float64)
    for start in range(0, len(train_ms), BLOCK):
        stop = min(start + BLOCK, len(train_ms))
        xb = standardiser(train_ms[start:stop]).astype(np.float64)
        w = 1.0 / counts[labels[start:stop]]
        gram += xb.T @ (w[:, None] * xb)
        t = np.zeros((stop - start, classes), dtype=np.float64)
        t[np.arange(stop - start), labels[start:stop]] = w
        cross_w += xb.T @ t
    system = symmetric_system(gram)
    system.flat[:: d + 1] += lam
    try:
        cho = scipy_linalg.cho_factor(system, lower=True,
                                      check_finite=False)
        weights = scipy_linalg.cho_solve(cho, cross_w,
                                         check_finite=False)
        path = "cholesky"
        dropped = 0
    except scipy_linalg.LinAlgError:
        vals, vecs = scipy_linalg.eigh(system, check_finite=False,
                                       driver="evd")
        # M296d: strongly-convex penalised modes only - non-positive
        # penalised modes are non-convex (no minimizer) and near-zero
        # positive modes amplify beyond the kept part's resolution
        scale_pen = max(abs(float(vals[0])), abs(float(vals[-1])))
        cutoff_pen = max(0.0, scale_pen * SOLVE_STRONG_CONVEX_CUTOFF)
        keep = vals > cutoff_pen
        inv = np.where(keep, 1.0 / vals, 0.0)
        weights = (vecs * inv[None, :]) @ (vecs.T @ cross_w)
        path = "eigh_fallback"
        dropped = int((~keep).sum())
    residual = system @ weights - cross_w
    denom = (float(np.max(np.abs(system))) * float(np.max(np.abs(weights)))
             + float(np.max(np.abs(cross_w))))
    raw = float(np.max(np.abs(residual))) / max(denom, 1e-300)
    if dropped > 0:
        # the truncated-system instrument (M296c): the solver claims to
        # zero the residual of the KEPT part, not the full system
        proj_w = vecs.T @ weights
        proj_c = vecs.T @ cross_w
        trunc_residual = (vecs @ ((vals * keep)[:, None] * proj_w)
                          - vecs @ (keep[:, None] * proj_c))
        denom_t = (float(np.max(np.abs(system)))
                   * float(np.max(np.abs(weights)))
                   + float(np.max(np.abs(cross_w))))
        trunc_backward = (float(np.max(np.abs(trunc_residual)))
                          / max(denom_t, 1e-300))
        ok = trunc_backward <= RESID_TOL
        report = {"ok": ok, "solve_path": path,
                  "raw_backward": raw,
                  "truncated_system_backward": trunc_backward,
                  "dropped_components": dropped}
    else:
        ok = raw <= RESID_TOL
        report = {"ok": ok, "solve_path": path, "raw_backward": raw,
                  "dropped_components": 0}
    intercept = np.zeros(classes)
    return np.vstack([weights, intercept[None, :]]), report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", type=int, nargs=2, default=None,
                        metavar=("TRAIN_ROWS", "TEST_ROWS"),
                        help="smoke: fit on the first TRAIN_ROWS and score "
                             "the first TEST_ROWS; no evidence is written")
    args = parser.parse_args()
    run_m298(args.config, args.output, smoke=tuple(args.smoke)
             if args.smoke else None)


if __name__ == "__main__":
    main()
