"""M123 — Classification-aware certificate (the M121 fix).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v19.md`` section 5.2 and
``experiments/configs/v16/m123_margin_certificate.json``.

M121's MSE proxy (a scalar E_g over all 345 channels) failed to predict the
measured accuracy crossing: accuracy is an ARGMAX phenomenon (per-class score
margins), which a scalar MSE does not track. M123 replaces the proxy with the
object accuracy actually depends on: the per-class score MARGIN distribution.

Model (first-principles, disclosed in the config). From the full-M standardised
Gram and the one-hot label projections (Canatar's modal machinery, as in M121,
applied per class):
- vals_k = raw Gram eigenvalues, a_rho,c = v_rho^T (F_std^T y_c), kappa(P) from
  the M121 fixed point.
- S_cc'(P) = sum_rho a_rho,c a_rho,c' / (vals_k + kappa(P))
- Sigma_cc'(P) = sum_rho a_rho,c a_rho,c' / (vals_k + kappa(P))^2
- score model for a class-c test point: f ~ N(mu^(c)(P), Sigma(P)) with
  mu^(c)_c'(P) = S_cc'(P) / n_c  (class-conditional modal mean = train average).
- predicted accuracy(P) = (1/C) sum_c P(f_c > max_{c'!=c} f_c'), Monte Carlo.

Gate (as registered): predicted crossing within [1/3, 3] of the measured M116
crossing midpoint (41400); gap direction at n_max must match (sparse > dense).
If it fails, the certificate is closed as a prediction tool and kept as an
explanatory diagnostic only.

Reproduce with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m123_margin_certificate
"""
from __future__ import annotations

import argparse
import json
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
from experiments.tier4.eval_v16_m121_spectrum import (
    _gram_and_targets,
    _spectrum,
    _spectrum_diagnostics,
    _standardised_ft_y,
    _standardised_gram,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m123_margin_certificate.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m123_margin_certificate"
M116_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m116_scale" / "evidence.json"
CROSS_FACTOR_LO = 1.0 / 3.0
CROSS_FACTOR_HI = 3.0
BLOCK = 8192


def _kappa(lam: float, eta: np.ndarray, P: int, iters: int = 200,
           tol: float = 1e-12) -> float:
    kappa = lam
    for _ in range(iters):
        denom = kappa + P * eta
        kappa_new = lam + float((kappa * eta / denom).sum())
        if abs(kappa_new - kappa) < tol:
            return kappa_new
        kappa = kappa_new
    return kappa


def _erho(lam: float, eta: np.ndarray, P: int, kappa: float) -> np.ndarray:
    """Canatar modal error multiplier E_rho(P) = (1/(1-gamma)) kappa^2/(kappa+P*eta)^2."""
    denom = kappa + P * eta
    gamma = float((P * eta ** 2 / denom ** 2).sum())
    return (1.0 / (1.0 - gamma)) * kappa ** 2 / denom ** 2


def _modal_matrices(a: np.ndarray, vals: np.ndarray, kappa: float,
                    erho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S (score-mean alignment Gram) and Sigma (score covariance), C x C.

    a: (kept, C) label projections; vals: (kept,) raw Gram eigenvalues.
    S_cc' = sum_rho a_rho,c a_rho,c' / (vals + kappa)      (class-conditioned
             score MEAN Gram: mean^(c)_c' = S_cc'/n_c)
    Sigma_cc' = sum_rho E_rho(P) a_rho,c a_rho,c' / vals   (Canatar's modal
             error applied per class and pairwise: the per-class learnable
             score VARIANCE; decays as P grows, so predicted accuracy rises)
    """
    S = (a / (vals + kappa)[:, None]).T @ a
    B = a / np.sqrt(vals)[:, None]
    Sigma = B.T @ (B * erho[:, None])
    return S, Sigma


def _predicted_accuracy(Sigma: np.ndarray, mu: np.ndarray, n_c: np.ndarray,
                        n_samp: int, rng: np.random.Generator) -> float:
    """Monte Carlo P(f_c > max_{c'!=c} f_c') averaged over classes (balanced).

    mu: (C, C) with mu[c] = class-c conditioned score mean; Sigma: (C, C);
    n_c: (C,) class counts (balancing handled by averaging over classes).
    """
    C = Sigma.shape[0]
    # mean shift per class: mu^(c) = S_{c,:} / n_c  -> stacked (C, C)
    mean_stack = mu / np.maximum(n_c[:, None], 1.0)
    # draw base samples N(0, Sigma) once; per-class shift is deterministic
    base = rng.multivariate_normal(np.zeros(C), Sigma, size=n_samp)
    acc_sum = 0.0
    for c in range(C):
        g = base + mean_stack[c]
        rest_max = np.maximum(
            np.maximum(g[:, :c].max(axis=1) if c > 0 else -np.inf,
                       g[:, c + 1:].max(axis=1) if c < C - 1 else -np.inf),
            -np.inf,
        )
        acc_sum += float((g[:, c] > rest_max).mean())
    return acc_sum / C


def run_m123(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    configure_external_cache_environment()
    smoke_rows = int(config.get("_smoke_rows", 0))
    smoke_skip = bool(config.get("_smoke_skip_gates", False))

    m116 = json.loads(M116_EVIDENCE.read_text(encoding="utf-8"))
    measured_sparse = {p["n"]: p["accuracy"] for p in m116["sparse"]["curve"]}
    measured_dense = {p["n"]: p["accuracy"] for p in m116["dense"]["curve"]}

    cache = data_cache_root() / "v16" / "m116"
    from experiments.tier4.eval_v16_m109_trunk import _load_corpus
    corpus_cfg = json.loads(
        (REPO_ROOT / "experiments" / "configs" / "v16" / "m116_scale.json")
        .read_text(encoding="utf-8"))
    corpus, _ti, _te = _load_corpus(corpus_cfg)
    train_labels = corpus["train_labels"]
    classes = int(train_labels.max()) + 1
    n_train = len(train_labels)
    if smoke_rows:
        n_train = min(smoke_rows, n_train)

    f_train = np.lib.format.open_memmap(cache / "f_train.npy", mode="r",
                                        dtype=np.float32)
    dense_train = np.lib.format.open_memmap(cache / "dense_train.npy", mode="r",
                                            dtype=np.float32)

    theory = config["theory"]
    ladder = [int(n) for n in theory["ladder"]]
    lambda_grid = [float(l) for l in theory["lambda_grid"]]
    n_samp = int(theory["n_mc_samples"])
    mc_seed = int(theory["mc_seed"])

    spectra: dict[str, Any] = {}
    margin: dict[str, Any] = {}
    for name, mem in (("sparse", f_train), ("dense", dense_train)):
        print(f"  streaming {name} Gram ({n_train} rows x {mem.shape[1]})",
              flush=True)
        gram, colsum, sqsum, ft_y, class_count = _gram_and_targets(
            mem, train_labels, n_train, classes)
        gstd, centre, scale = _standardised_gram(gram, colsum, sqsum, n_train)
        ft_y_std = _standardised_ft_y(ft_y, class_count, centre, scale)
        vals, vecs = _spectrum(gstd)
        keep = vals > max(float(vals.max()) * 1e-10, 1e-12)
        vals_k = vals[keep]
        vecs_k = vecs[:, keep]
        a = vecs_k.T @ ft_y_std          # (kept, classes)
        diag = _spectrum_diagnostics(vals_k / n_train, name)
        diag["n_kept_modes"] = int(keep.sum())
        spectra[name] = diag
        n_c = class_count.astype(np.float64)

        per_lambda: dict[str, Any] = {}
        for lam in lambda_grid:
            rng = np.random.default_rng(mc_seed)
            accs: dict[int, float] = {}
            per_p: dict[str, Any] = {}
            for P in ladder:
                kap = _kappa(lam, vals_k / n_train, P)
                erho = _erho(lam, vals_k / n_train, P, kap)
                S, Sigma = _modal_matrices(a, vals_k, kap, erho)
                acc = _predicted_accuracy(Sigma, S, n_c, n_samp, rng)
                accs[P] = acc
                per_p[str(P)] = {
                    "predicted_accuracy": float(acc),
                    "kappa": float(kap),
                    "mean_diag_ratio": float(
                        (S.diagonal() / np.maximum(Sigma.diagonal(), 1e-12)
                         ).mean()),
                    "sigma_trace": float(Sigma.trace()),
                    "margin_pct": {q: float(np.percentile(
                        _margin_draws(Sigma, S, n_c, 2000,
                                      np.random.default_rng(mc_seed)),
                        q)) for q in (5, 25, 50, 75, 95)},
                }
                print(f"    {name} lam={lam} P={P}: pred acc {acc:.4f}",
                      flush=True)
            per_lambda[str(lam)] = {"curve": accs, "per_p": per_p}
        margin[name] = per_lambda

    # ---- gates (primary lambda = 1.0) -----------------------------------
    lam_primary = 1.0
    pred_sparse = margin["sparse"][str(lam_primary)]["curve"]
    pred_dense = margin["dense"][str(lam_primary)]["curve"]

    def _crossing() -> int | None:
        for P in sorted(pred_sparse):
            if pred_sparse[P] > pred_dense[P]:
                return P
        return None

    pred_cross = _crossing()
    measured_mid = 41400.0
    cross_pass = (
        pred_cross is not None
        and CROSS_FACTOR_LO * measured_mid <= pred_cross
        and pred_cross <= CROSS_FACTOR_HI * measured_mid
    )
    ref_n = max((k for k in measured_sparse if k <= ladder[-1]),
                default=min(measured_sparse))
    gap_dir_pass = pred_sparse[ladder[-1]] > pred_dense[ladder[-1]]

    ks1 = {
        "registered": "predicted crossing (sparse margin-accuracy > dense) "
                      "within factor [1/3, 3] of the measured M116 crossing "
                      "midpoint (41400)",
        "predicted_crossing": pred_cross,
        "measured_midpoint": measured_mid,
        "factor_lo": CROSS_FACTOR_LO, "factor_hi": CROSS_FACTOR_HI,
        "fired": not cross_pass,
        "lambda": lam_primary,
    }
    ks2 = {
        "registered": "predicted sparse margin-accuracy > dense at n_max "
                      "(matches measured accuracy order 0.2153 > 0.1972)",
        "predicted_sparse_nmax": float(pred_sparse[ladder[-1]]),
        "predicted_dense_nmax": float(pred_dense[ladder[-1]]),
        "measured_reference_n": ref_n,
        "measured_sparse_acc_nmax": measured_sparse[ref_n],
        "measured_dense_acc_nmax": measured_dense[ref_n],
        "fired": not gap_dir_pass,
    }
    gates = {
        "kill_switch_1_crossing": ks1,
        "kill_switch_2_gap_direction": ks2,
        "lambda_grid_sensitivity": {
            "crossing_per_lambda": {
                str(lam): _crossing_lambda(margin, lam, ladder)
                for lam in lambda_grid
            },
        },
        "_smoke_skip": smoke_skip,
    }

    evidence = {
        "milestone": "M123",
        "admissible_as_evidence": not inadmissible,
        "registered_in": config.get("registered_in"),
        "question": ("does a per-class score-margin model, from the Gram "
                     "spectrum and label projections alone, predict M116's "
                     "measured Q(n) accuracy crossing?"),
        "config_file": Path(config_path).name,
        "config": config,
        "measure": {
            "rows_used": int(n_train),
            "standardise": "per-dim mean/unit-variance over the rows used",
            "m116_features": "data_cache_root()/v16/m116/{f_train,dense_train}.npy",
        },
        "spectra": spectra,
        "margin_curves": margin,
        "measured_m116": {
            "sparse_accuracy": measured_sparse,
            "dense_accuracy": measured_dense,
            "crossing_interval": [27600, 55200],
        },
        "gates": gates,
        "certificate_verdict": {
            "note": "filled by the verdict update after measurement",
        },
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM123 complete -> {output_dir / 'evidence.json'}", flush=True)
    print(f"  predicted crossing (lambda={lam_primary}): {pred_cross} "
          f"(measured midpoint 41400)", flush=True)
    print(f"  KS1 fired: {ks1['fired']}  KS2 fired: {ks2['fired']}", flush=True)
    return evidence


def _crossing_lambda(margin: dict[str, Any], lam: float,
                     ladder: list[int]) -> int | None:
    s = margin["sparse"][str(lam)]["curve"]
    d = margin["dense"][str(lam)]["curve"]
    for P in sorted(s):
        if s[P] > d[P]:
            return P
    return None


def _margin_draws(Sigma: np.ndarray, S: np.ndarray, n_c: np.ndarray,
                  n: int, rng: np.random.Generator) -> np.ndarray:
    """Margin samples (true-class score - max other) for the diagnostics."""
    C = Sigma.shape[0]
    mean_stack = S / np.maximum(n_c[:, None], 1.0)
    base = rng.multivariate_normal(np.zeros(C), Sigma, size=n)
    margins = np.empty(n)
    for c in range(C):
        g = base + mean_stack[c]
        rest_max = np.maximum(
            np.maximum(g[:, :c].max(axis=1) if c > 0 else -np.inf,
                       g[:, c + 1:].max(axis=1) if c < C - 1 else -np.inf),
            -np.inf,
        )
        margins += (g[:, c] - rest_max)
    return margins / C


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m123(args.config, args.output)


if __name__ == "__main__":
    main()
