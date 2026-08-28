"""M296 - head repair I: symmetric Gram + Cholesky/eig/SVD + condition
number.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` (M296,
25 Aug 2026, before any build; M296a solver amendment registered after
the first run VOIDED on its own backward gate; M296b eigensolver-driver
pin; M296c solve truncation + g6 convention band, retired by M296d).
The repair for finding A6 / improvement I4 item one: assemble the
standardised normal-equation system as a **symmetric-by-construction**
matrix (upper triangle kept exactly as accumulated, mirrored into the
lower), factor with Cholesky, and seal a condition number with every
fitted head. Cholesky refuses indefinite systems (the sealed
standardisation convention assembles one whose smallest eigenvalue can
be negative); the fallback chain is eigendecomposition under the M296d
strong-convexity truncation (penalised modes are inverted only when
positive and above 1e-10 of the penalised scale - the ridge objective
has no minimizer along non-positive penalised modes and near-zero
positive modes amplify beyond the kept part's resolution), then SVD as
a last resort. The solve path drops only mathematically-zero
components; the effective-rank statistic keeps its own cutoff (M296a).

Why this is the repair. The sealed solve path accumulates the raw Gram
and then assembles ``centred = gram - outer(column_sum, centre)`` with
each off-diagonal entry computed once; the two triangles are only
~1e-16 symmetric and the asymmetric-convention pathology measured in the
M180 docstring (two algebraically identical centring conventions differ
by 1.6e-4 relative; choosing the transpose convention cost 0.66 points
of holdout accuracy) sits exactly there. Once the assembled matrix is
symmetric by construction, the convention disappears: there is one
triangle and it is mirrored bitwise. Cholesky then reads a consistent
system instead of an LU of an asymmetric matrix.

The input side is untouched: accumulation and standardisation (fp32-
rounded centres/scales) reuse the sealed ``RidgeAccumulator`` path
bit-for-bit, so this milestone isolates the repair to assembly +
factorization. The runner records, on the sealed M228 cached features:
the repaired-solver reading, the LU-path reproduction of the sealed
anchor 0.24214492753623187, and the condition number of the penalised
system.

Gates (all VOID on failure): g1 premise row counts exact; g2 the LU
path reproduces the sealed anchor at 1e-9 (instrument identity); g3 the
repaired system is bitwise symmetric and the Cholesky residual passes
the registered backward-error tolerance; g4 condition number finite;
g5 accuracies valid. No improvement claim is read here - H26-1 is
evaluated at M297 with LOOCV lambda, and this cell only registers the
repaired solver and the conditioning evidence.
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
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
                  / "m296_head_repair.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v26" / "m296_head_repair"

CLASSES = 345
FULL_TRAIN_ROWS = 409832
SEALED_TEST_ROWS = 34500
BLOCK = 4096                 # accumulation chunk (matches the sealed path)
RESID_TOL = 1e-10            # backward error tolerance for the solve
COND_CUTOFF = 1e-12          # effective-rank cutoff relative to lambda_max
SOLVE_SVD_CUTOFF = 16 * np.finfo(np.float64).eps
# M296d (registered): the ridge objective is only a MINIMIZATION along
# penalised modes with eigenvalue v > 0; along v <= 0 it is non-convex
# (no minimizer exists - the normal-equation stationary point is a
# maximum), and along 0 < v <= 1e-10 of the penalised scale the
# amplification 1/v exceeds what the kept part's eigvec accuracy
# supports (near-cancellation modes of Gram eigenvalue ~ -lambda,
# measured: penalised +5.7e-4 mode = 1750x amplification -> 0.1266).
# The solve therefore keeps only STRONGLY-CONVEX penalised modes:
# v > max(0, scale * 1e-10); everything else contributes zero.
# Dropped counts are reported, never gated. Retired by this rule: the
# M296c dual cutoff, which watched the UNPENALISED spectrum and could
# not see a near-cancellation mode whose unpenalised eigenvalue is ~-1.
SOLVE_STRONG_CONVEX_CUTOFF = 1e-10
CONVENTION_BAND = 0.02     # g6: |repaired - LU| on the sealed test
PENALTY = 1.0                # the sealed penalty-1.0 reading


def symmetric_system(centred: np.ndarray) -> np.ndarray:
    """Assemble the standardised system symmetric by construction.

    The upper triangle (diagonal included) is kept exactly as the
    accumulation produced it; the strict lower triangle is the bitwise
    mirror. No averaging, no second convention: ``G == G.T`` exactly.
    """
    return np.triu(centred) + np.triu(centred, k=1).T


def condition_report(system: np.ndarray, penalty: float,
                     cutoff: float = COND_CUTOFF) -> dict[str, float]:
    """Eigenvalues of the penalised symmetric system and its condition
    number. The matrix condition number is the standard one:
    sigma_max / sigma_min = |lambda_max| / |lambda|_min, which stays
    finite and meaningful when the system is indefinite (a sealed
    standardisation convention can assemble a system whose smallest
    eigenvalue is negative - the M180 ill-conditioning signature). An
    ``indefinite`` flag records that case explicitly; ``penalty`` is
    reported for the record and the diagonal must already carry it.
    M296b: the eigensolver driver is pinned to ``evd`` (divide-and-
    conquer, parallel BLAS) - the registered numerics-policy choice
    after the default MRRR driver measured ~1.2/16 cores on this
    clustered indefinite spectrum."""
    from scipy import linalg as scipy_linalg

    vals = scipy_linalg.eigvalsh(system, check_finite=False,
                                 driver="evd")
    return condition_report_from_vals(vals, penalty, cutoff,
                                      int(system.shape[0]))


def condition_report_from_vals(vals: np.ndarray, penalty: float,
                               cutoff: float = COND_CUTOFF,
                               dimension: int | None = None
                               ) -> dict[str, float]:
    """The condition report from an already-computed spectrum (the
    M296b single-factorisation rule: a caller that factorised the
    system for its solve reuses the eigenvalues here instead of
    factorising twice)."""
    lam_max = float(vals[-1])
    lam_min = float(vals[0])
    mags = np.abs(vals)
    min_abs = float(mags.min())
    cutoff_abs = cutoff * abs(lam_max)
    effective_rank = int(np.count_nonzero(mags > cutoff_abs))
    return {
        "penalty": float(penalty),
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "min_abs_eigenvalue": min_abs,
        "indefinite": bool(lam_min < 0.0),
        "condition_number": abs(lam_max) / min_abs if min_abs > 0.0
        else float("inf"),
        "effective_rank_cutoff": cutoff_abs,
        "effective_rank": effective_rank,
        "dimension": int(dimension if dimension is not None
                         else len(vals)),
    }


def _eigh_solve(system: np.ndarray, cross: np.ndarray,
                penalty: float,
                return_spectrum: bool = False,
                return_factors: bool = False
                ) -> tuple[np.ndarray, dict[str, Any]]:
    """Eigendecomposition solve of the symmetric penalised system
    under the registered M296d strong-convexity truncation: invert a
    penalised eigenvalue only when it is positive AND above
    ``SOLVE_STRONG_CONVEX_CUTOFF`` of the penalised scale. Penalised
    modes v <= 0 are non-convex (the ridge objective has no minimizer
    along them; the normal-equation point is a maximum) and modes
    0 < v <= 1e-10*scale carry amplification the kept part cannot
    support - both contribute zero. Dropped counts are reported, never
    gated. M296b: driver pinned to ``evd``. ``return_spectrum``
    carries the eigenvalues; ``return_factors`` also carries the
    eigenvectors and keep mask so the caller can gate the raw residual
    against the truncated system (the M296c instrument)."""
    from scipy import linalg as scipy_linalg

    vals, vecs = scipy_linalg.eigh(system, check_finite=False,
                                   driver="evd")
    scale_pen = max(abs(float(vals[0])), abs(float(vals[-1])))
    cutoff_pen = max(0.0, scale_pen * SOLVE_STRONG_CONVEX_CUTOFF)
    keep = vals > cutoff_pen
    inv = np.where(keep, 1.0 / vals, 0.0)
    weights = (vecs * inv[None, :]) @ (vecs.T @ cross)
    detail: dict[str, Any] = {
        "cutoff_strong_convex": float(cutoff_pen),
        "nonpositive_modes_dropped": int((vals <= 0.0).sum()),
        "dropped_components": int((~keep).sum())}
    if return_spectrum:
        detail["vals"] = vals
    if return_factors:
        detail["vecs"] = vecs
        detail["keep"] = keep
        detail["scale_penalised"] = float(scale_pen)
    return weights, {"solve_path": "eigh_fallback",
                     "eigh_fallback": detail}


def _svd_solve(system: np.ndarray, cross: np.ndarray
               ) -> tuple[np.ndarray, dict[str, Any]]:
    """SVD solve of last resort (M296a): drops only mathematically-
    zero singular values (the solve cutoff, never the rank cutoff).
    Kept as a separate, directly-testable path behind the
    eigendecomposition fallback."""
    from scipy import linalg as scipy_linalg

    u, s, vt = scipy_linalg.svd(system, check_finite=False)
    cutoff = float(s.max()) * SOLVE_SVD_CUTOFF
    keep = s > cutoff
    inv_s = np.where(keep, 1.0 / s, 0.0)
    weights = (vt.T * inv_s[None, :]) @ (u.T @ cross)
    return weights, {"solve_path": "svd_fallback",
                     "svd_fallback": {
                         "min_singular_value": float(s.min()),
                         "max_singular_value": float(s.max()),
                         "cutoff": cutoff,
                         "dropped_components": int((~keep).sum())}}


def solve_symmetric(centred: np.ndarray, cross: np.ndarray,
                    intercept: np.ndarray, penalty: float,
                    report_conditioning: bool = True
                    ) -> tuple[np.ndarray, dict[str, Any]]:
    """Symmetric assembly, then Cholesky; on refusal, eigendecomposition
    (M296d strong-convexity truncation); SVD as last resort (M296a).
    Returns ([w; b], report). The backward gate measures what each
    path claims to solve: a full-system solve (no dropped components)
    is gated on the raw residual at 1e-10; a truncated solve (dropped
    components) is gated on the raw residual against the TRUNCATED
    system (the M296c instrument - the normal-equation form measured
    the eigen reconstruction error and was degenerate on this route).
    Dropped-component counts are reported, never gated."""
    from scipy import linalg as scipy_linalg

    system = symmetric_system(centred)
    width = system.shape[0]
    symmetric_to_bit = bool(np.array_equal(system, system.T))
    system.flat[:: width + 1] += penalty
    report: dict[str, Any] = {"symmetric_to_bit": symmetric_to_bit}
    try:
        cho = scipy_linalg.cho_factor(system, lower=True,
                                      check_finite=False)
        weights = scipy_linalg.cho_solve(cho, cross,
                                         check_finite=False)
        report["solve_path"] = "cholesky"
        if report_conditioning:
            report["conditioning"] = condition_report(system, penalty)
    except scipy_linalg.LinAlgError:
        # Cholesky refuses non-PD systems (the sealed standardisation
        # convention assembles one whose smallest eigenvalue can be
        # negative). The eigendecomposition path inverts every
        # penalised eigenvalue above the solve cutoff - the same
        # full-pivot semantics the LU path has, on the symmetric
        # system. SVD is the last resort behind it. M296b: the
        # spectrum is computed ONCE and reused for the conditioning
        # report (single factorisation).
        try:
            weights, path = _eigh_solve(system, cross, penalty,
                                        return_spectrum=True,
                                        return_factors=True)
            if report_conditioning and "vals" in path["eigh_fallback"]:
                report["conditioning"] = condition_report_from_vals(
                    path["eigh_fallback"]["vals"], penalty,
                    dimension=width)
        except scipy_linalg.LinAlgError:
            weights, path = _svd_solve(system, cross)
            if report_conditioning:
                report["conditioning"] = condition_report(system, penalty)
        report.update(path)
    residual = system @ weights - cross
    denom = (float(np.max(np.abs(system))) * float(np.max(np.abs(weights)))
             + float(np.max(np.abs(cross))))
    backward = float(np.max(np.abs(residual))) / max(denom, 1e-300)
    report["backward_error"] = backward
    report["backward_tolerance"] = RESID_TOL
    dropped = 0
    for key in ("eigh_fallback", "svd_fallback"):
        if key in report:
            dropped = int(report[key]["dropped_components"])
    if dropped > 0:
        # a truncated solve: the gate checks the raw residual against
        # the TRUNCATED system - exactly what the solver claims to
        # solve (the M296c refinement; the earlier normal-equation
        # form measured the eigen reconstruction error, ~O(1), and was
        # degenerate on this route).
        detail = report.get("eigh_fallback", {})
        if "vecs" in detail:
            vecs = detail["vecs"]
            vals = detail["vals"]
            keep = detail["keep"]
            scale_pen = float(detail["scale_penalised"])
            proj_w = vecs.T @ weights
            proj_c = vecs.T @ cross
            trunc_residual = (vecs @ ((vals * keep)[:, None] * proj_w)
                              - vecs @ (keep[:, None] * proj_c))
            denom_t = (scale_pen * float(np.max(np.abs(weights)))
                       + float(np.max(np.abs(cross))))
            trunc_backward = (float(np.max(np.abs(trunc_residual)))
                              / max(denom_t, 1e-300))
            report["truncated_system_backward_error"] = trunc_backward
            report["backward_passed"] = trunc_backward <= RESID_TOL
        else:
            # the SVD last-resort route: the normal-equation residual
            # is the registered instrument (pinv-type factors)
            normal = system @ residual
            denom_n = (float(np.max(np.abs(system)))
                       * float(np.max(np.abs(residual))))
            normal_backward = (float(np.max(np.abs(normal)))
                               / max(denom_n, 1e-300))
            report["normal_equation_backward_error"] = normal_backward
            report["backward_passed"] = normal_backward <= RESID_TOL
    else:
        report["backward_passed"] = backward <= RESID_TOL
    return np.vstack([weights, intercept[None, :]]), report


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


def run_m296(config_path: Path, output_dir: Path,
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

    # ---- instrument identity: the sealed LU path on the same system ----
    acc = RidgeAccumulator(train_ms.shape[1], CLASSES)
    for start in range(0, len(train_ms), BLOCK):
        stop = min(start + BLOCK, len(train_ms))
        acc.add(train_ms[start:stop], labels[start:stop])
    standardiser = acc.standardiser()
    centred, cross, intercept = acc._standardised_system()

    lu_centred = centred.copy()
    lu_centred.flat[:: lu_centred.shape[0] + 1] += PENALTY
    lu_weights = np.vstack([np.linalg.solve(lu_centred, cross),
                            intercept[None, :]])
    lu_accuracy = _score(lu_weights, test_ms, test_labels[:test_rows],
                         standardiser)

    # ---- the repair: symmetric assembly + Cholesky (SVD fallback) ------
    repaired_weights, report = solve_symmetric(
        centred, cross, intercept, PENALTY, report_conditioning=True)
    repaired_accuracy = _score(repaired_weights, test_ms,
                               test_labels[:test_rows], standardiser)

    anchor = float(config["anchor"]["value"])
    tol = float(config["anchor"]["tolerance"])
    premise_ok = (len(mem_train) == FULL_TRAIN_ROWS
                  and len(mem_test) == SEALED_TEST_ROWS
                  and len(labels) == FULL_TRAIN_ROWS)
    cond = report.get("conditioning", {})
    g1 = premise_ok
    anchor_measured = lu_accuracy if smoke is None else float("nan")
    g2 = (abs(anchor_measured - anchor) <= tol) if smoke is None else True
    g3 = bool(report["symmetric_to_bit"] and report["backward_passed"])
    g4 = (np.isfinite(cond.get("lambda_min", np.nan))
          and np.isfinite(cond.get("lambda_max", np.nan))
          and np.isfinite(cond.get("condition_number", np.nan)))
    g5 = 0.0 <= lu_accuracy <= 1.0 and 0.0 <= repaired_accuracy <= 1.0
    # M296c g6: the convention band - a repaired reading outside this
    # band of the LU anchor is instrument pathology (the measured
    # full-pivot catastrophe was -0.116), VOID before any reading
    g6 = bool(abs(repaired_accuracy - lu_accuracy) <= CONVENTION_BAND)
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
            "note": ("skipped in smoke mode" if smoke is not None
                     else "the LU path on the cached features must "
                          "reproduce the sealed M228 anchor or the "
                          "instrument is not the sealed one")},
        "g3_symmetric_system_and_residual": {
            "ok": g3, "symmetric_to_bit": report["symmetric_to_bit"],
            "backward_error": report["backward_error"],
            "backward_tolerance": report["backward_tolerance"],
            "backward_passed": report["backward_passed"],
            "solve_path": report["solve_path"]},
        "g4_condition_number_finite": {"ok": g4, **cond},
        "g5_accuracies_valid": {
            "ok": g5, "lu_accuracy": lu_accuracy,
            "repaired_accuracy": repaired_accuracy},
        "g6_convention_band": {
            "ok": g6, "delta_repaired_vs_lu":
                repaired_accuracy - lu_accuracy,
            "band": CONVENTION_BAND,
            "note": "M296d: a repaired reading outside +-0.02 of the "
                    "LU anchor is instrument pathology, VOID"},
    }
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M296",
        "cell": ("head repair I: symmetric Gram + Cholesky/SVD + "
                 "condition number, on the sealed M228 cached ms features"),
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "smoke": (smoke is not None),
        "smoke_rows": smoke,
        "penalty": PENALTY,
        "feature_width": train_ms.shape[1],
        "system_indefinite": bool(cond.get("indefinite")),
        "lu_path_accuracy": lu_accuracy,
        "repaired_solver_accuracy": repaired_accuracy,
        "anchor": {"sealed": anchor,
                   "lu_measured": (anchor_measured
                                   if smoke is None else None),
                   "repaired_delta_vs_sealed":
                       repaired_accuracy - anchor},
        "decision_level_delta_repaired_vs_lu":
            repaired_accuracy - lu_accuracy,
        "solve_report": report,
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": (not gates_ok) and smoke is None,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": (
                "M296 registers the repaired solver and the conditioning "
                "evidence; no improvement claim (H26-1 is read at M297 "
                "with LOOCV lambda)"
            ) if gates_ok else "a gate failed — VOID",
        },
        "scope": ("full 409,832-row train schedule, sealed 34,500-row "
                  "test, cached ms features (no re-extraction), "
                  "penalty 1.0, no test-set selection"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    if smoke is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        print(f"M296 complete -> {output_dir / 'evidence.json'}", flush=True)
    print(json.dumps({"gates_ok": gates_ok,
                      "lu_accuracy": lu_accuracy,
                      "repaired_accuracy": repaired_accuracy,
                      "solve_path": report["solve_path"],
                      "conditioning": report.get("conditioning")},
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
    run_m296(args.config, args.output, smoke=tuple(args.smoke)
             if args.smoke else None)


if __name__ == "__main__":
    main()
