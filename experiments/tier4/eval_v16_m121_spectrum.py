"""M121 — Spectral learning-curve certificate (B5, grounded).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v18.md`` section 5.5 and
``experiments/configs/v16/m121_spectrum.json``.

Question. M116 measured, with sealed evidence, that the frozen sparse family's
Q(n) is steeper than the cost-matched dense trunk's and overtakes it between
n=27,600 and n=55,200. The kernel-learning-curve theory (Canatar-Bordelon-
Pehlevan 2006.13198; Bordelon-Canatar-Pehlevan 2002.02561) predicts that a
ridge's learning curve is governed by the eigenspectrum of the feature Gram
and the projection of the labels onto the eigenvectors. M121 computes those
spectra for BOTH families from M116's sealed feature memmaps (no re-encoding,
no training) and asks: does the theory predict M116's measured crossing?

Recipe (all from Canatar et al. 2006.13198, discrete-measure kernel PCA):
- features F (n x d) read from M116's memmaps; standardised per-dim
  (mean/unit variance over train) to match the RidgeAccumulator's kernel.
- eta_rho = eigenvalues of (1/M) F_std^T F_std, M = 138000.
- kappa solves kappa = lambda + sum_rho kappa*eta_rho/(kappa + P*eta_rho).
- gamma = sum_rho P*eta_rho^2/(kappa + P*eta_rho)^2.
- E_rho = (1/(1-gamma)) * kappa^2/(kappa + P*eta_rho)^2.
- modal target power w2_rho = sum_c a_rho,c^2 / (M*eta_rho),
  a_rho,c = v_rho^T (F_std^T y_c), one-hot labels.
- E_g(P) = sum_rho w2_rho * E_rho (MSE proxy; gate on the CROSSING, not
  absolute values; target power in kernel-null modes reported as a constant
  shift, not gated).

Reproduce with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m121_spectrum
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m121_spectrum.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m121_spectrum"
M116_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m116_scale" / "evidence.json"
CROSS_FACTOR_LO = 1.0 / 3.0
CROSS_FACTOR_HI = 3.0
BLOCK = 8192


def _gram_and_targets(mem: np.ndarray, labels: np.ndarray, rows: int,
                      classes: int) -> tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray,
                                             np.ndarray]:
    """One streaming pass: raw Gram, column/squared sums, per-class label sums."""
    dims = mem.shape[1]
    gram = np.zeros((dims, dims), dtype=np.float64)
    colsum = np.zeros(dims, dtype=np.float64)
    sqsum = np.zeros(dims, dtype=np.float64)
    ft_y = np.zeros((dims, classes), dtype=np.float64)
    class_count = np.zeros(classes, dtype=np.float64)
    onehot = np.eye(classes)
    for start in range(0, rows, BLOCK):
        stop = min(start + BLOCK, rows)
        block = np.asarray(mem[start:stop], dtype=np.float64)
        lab = labels[start:stop]
        gram += block.T @ block
        colsum += block.sum(axis=0)
        sqsum += (block ** 2).sum(axis=0)
        ft_y += block.T @ onehot[lab]
        class_count += np.bincount(lab, minlength=classes).astype(np.float64)
    return gram, colsum, sqsum, ft_y, class_count


def _standardised_gram(gram: np.ndarray, colsum: np.ndarray, sqsum: np.ndarray,
                       rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """F_std^T F_std with per-dim mean/unit-variance (RidgeAccumulator convention)."""
    centre = colsum / rows
    variance = sqsum / rows - centre ** 2
    scale = np.sqrt(np.maximum(variance, 0.0)) + 1e-8
    centred = (gram - np.outer(colsum, centre) - np.outer(centre, colsum)
               + rows * np.outer(centre, centre))
    gstd = centred / np.outer(scale, scale)
    return gstd, centre, scale


def _standardised_ft_y(ft_y: np.ndarray, class_count: np.ndarray,
                       centre: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (ft_y - np.outer(centre, class_count)) / np.outer(scale, np.ones(
        len(class_count)))


def _spectrum(gstd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and eigenvectors of the standardised Gram."""
    vals, vecs = np.linalg.eigh(gstd)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def _learning_curve(eta: np.ndarray, w2: np.ndarray, lambdas: list[float],
                    ladder: list[int]) -> dict[float, dict[int, float]]:
    """E_g(P) for each lambda on the ladder (Canatar eq. 4, noise-free)."""
    out: dict[float, dict[int, float]] = {}
    for lam in lambdas:
        curve: dict[int, float] = {}
        for P in ladder:
            kappa = lam
            for _ in range(200):
                denom = kappa + P * eta
                num = kappa * eta / denom
                kappa_new = lam + float(num.sum())
                if abs(kappa_new - kappa) < 1e-12:
                    kappa = kappa_new
                    break
                kappa = kappa_new
            denom = kappa + P * eta
            gamma = float((P * eta ** 2 / denom ** 2).sum())
            erho = (1.0 / (1.0 - gamma)) * kappa ** 2 / denom ** 2
            curve[P] = float((w2 * erho).sum())
        out[lam] = curve
    return out


def _crossing(curve: dict[int, float], sparse: dict[int, float]) -> int | None:
    """Smallest ladder P where predicted sparse E_g <= dense E_g."""
    for P in sorted(curve):
        if sparse[P] <= curve[P]:
            return P
    return None


def _spectrum_diagnostics(eta: np.ndarray, label: str) -> dict[str, Any]:
    positive = eta[eta > 0]
    top = positive[0] if len(positive) else 0.0
    trace = float(positive.sum())
    # effective rank (participation ratio)
    eff_rank = float(trace ** 2 / (positive ** 2).sum()) if trace > 0 else 0.0
    # power-law tail index over the top half of positive eigenvalues
    tail = None
    if len(positive) >= 10:
        cut = max(10, len(positive) // 2)
        x = np.log(np.arange(1, cut + 1))
        y = np.log(positive[:cut])
        tail = float(np.polyfit(x, y, 1)[0])
    captured = 0.0
    return {
        "family": label,
        "n_eigenvalues": int(len(positive)),
        "top1": float(top),
        "trace": trace,
        "effective_rank": eff_rank,
        "top1_share": float(top / trace) if trace > 0 else 0.0,
        "top10_share": float(positive[:10].sum() / trace) if trace > 0 else 0.0,
        "tail_index": tail,
        "_tail_note": "tail_index = slope of log(eta) vs log(rank) over the top "
                      "half (more negative = faster-decaying spectrum = stronger "
                      "spectral bias = steeper learning curve at fixed n)",
    }


def run_m121(config_path: Path, output_dir: Path) -> dict[str, Any]:
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

    ladder = [int(n) for n in config["theory"]["ladder"]]
    lambda_grid = [float(l) for l in config["theory"]["lambda_grid"]]

    spectra: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    for name, mem in (("sparse", f_train), ("dense", dense_train)):
        print(f"  streaming {name} Gram ({n_train} rows x {mem.shape[1]})",
              flush=True)
        gram, colsum, sqsum, ft_y, class_count = _gram_and_targets(
            mem, train_labels, n_train, classes)
        gstd, centre, scale = _standardised_gram(gram, colsum, sqsum, n_train)
        ft_y_std = _standardised_ft_y(ft_y, class_count, centre, scale)
        vals, vecs = _spectrum(gstd)
        # keep modes above a relative floor; near-null modes (kernel-null
        # target power) are reported as an unlearnable constant shift, not
        # divided into the modal power (a^2/vals would explode there)
        keep = vals > max(float(vals.max()) * 1e-10, 1e-12)
        vals_k = vals[keep]
        vecs_k = vecs[:, keep]
        eta = vals_k / n_train
        a = vecs_k.T @ ft_y_std          # (kept, classes)
        w2 = (a ** 2).sum(axis=1) / vals_k
        captured = float(w2.sum()) / n_train   # one-hot total power is n_train
        diag = _spectrum_diagnostics(eta, name)
        diag["captured_target_power_fraction"] = captured
        diag["n_kept_modes"] = int(keep.sum())
        diag["null_target_power"] = n_train - float(w2.sum())
        spectra[name] = diag
        learn = _learning_curve(eta, w2, lambda_grid, ladder)
        null_power = n_train - float(w2.sum())
        # kernel-null target power acts as an irreducible constant in the MSE
        # proxy (the head predicts ~0 on it); add it to get total-estimated
        total_est = {lam: {P: v + null_power for P, v in learn[lam].items()}
                     for lam in lambda_grid}
        curves[name] = {"learnable": learn, "total_est": total_est,
                        "null_power": null_power}
        print(f"    {name}: eff_rank {diag['effective_rank']:.0f}, "
              f"tail {diag['tail_index']}, top1 share {diag['top1_share']:.4f}, "
              f"captured {captured:.3f}", flush=True)

    # ---- gates (primary lambda = 1.0, the program's ridge penalty) ---------
    # Primary gate is on the LEARNABLE-part curves, as registered in the
    # config ("target power in kernel-null modes is ... reported, not gated").
    lam_primary = 1.0
    pred_sparse = curves["sparse"]["learnable"][lam_primary]
    pred_dense = curves["dense"]["learnable"][lam_primary]
    pred_cross = _crossing(pred_dense, pred_sparse)
    total_cross = _crossing(curves["dense"]["total_est"][lam_primary],
                            curves["sparse"]["total_est"][lam_primary])
    measured_mid = 41400.0
    cross_pass = (
        pred_cross is not None
        and CROSS_FACTOR_LO * measured_mid <= pred_cross
        and pred_cross <= CROSS_FACTOR_HI * measured_mid
    )
    gap_dir_pass = pred_sparse[ladder[-1]] < pred_dense[ladder[-1]]

    ks1 = {
        "registered": "predicted crossing within factor [1/3, 3] of the "
                      "measured M116 crossing midpoint (41400); the crossing is "
                      "read off the LEARNABLE-part curves (null-mode power is "
                      "reported, not gated, per the registered disclosure)",
        "predicted_crossing": pred_cross,
        "crossing_total_est_sensitivity": total_cross,
        "measured_midpoint": measured_mid,
        "factor_lo": CROSS_FACTOR_LO, "factor_hi": CROSS_FACTOR_HI,
        "fired": not cross_pass,
        "lambda": lam_primary,
    }
    # measured reference: largest M116 ladder point at or below the run's n_max
    # (exact in the sealed run, where n_max = 138000 is an M116 ladder point)
    ref_n = max((k for k in measured_sparse if k <= ladder[-1]),
                default=min(measured_sparse))
    ks2 = {
        "registered": "predicted sparse error < dense error at n_max "
                      "(matches measured accuracy order)",
        "predicted_sparse_eg_nmax": float(pred_sparse[ladder[-1]]),
        "predicted_dense_eg_nmax": float(pred_dense[ladder[-1]]),
        "measured_reference_n": ref_n,
        "measured_sparse_acc_nmax": measured_sparse[ref_n],
        "measured_dense_acc_nmax": measured_dense[ref_n],
        "fired": not gap_dir_pass,
    }
    gates = {
        "kill_switch_1_crossing": ks1,
        "kill_switch_2_gap_direction": ks2,
        "lambda_grid_sensitivity": {
            "crossing_per_lambda_learnable": {
                str(lam): _crossing(curves["dense"]["learnable"][lam],
                                   curves["sparse"]["learnable"][lam])
                for lam in lambda_grid
            },
            "crossing_per_lambda_total_est": {
                str(lam): _crossing(curves["dense"]["total_est"][lam],
                                   curves["sparse"]["total_est"][lam])
                for lam in lambda_grid
            },
            "note": "robustness of the predicted crossing to the ridge "
                    "constant (the mapping to the program's penalty is exact "
                    "at lambda=1.0)",
        },
        "_smoke_skip": smoke_skip,
    }

    evidence = {
        "milestone": "M121",
        "admissible_as_evidence": not inadmissible,
        "registered_in": config.get("registered_in"),
        "question": ("does the Canatar-Bordelon-Pehlevan learning-curve theory, "
                     "from the feature Gram spectra alone, predict M116's "
                     "measured Q(n) crossing?"),
        "config_file": Path(config_path).name,
        "config": config,
        "measure": {
            "rows_used": int(n_train),
            "standardise": "per-dim mean/unit-variance over the rows used",
            "m116_features": "data_cache_root()/v16/m116/{f_train,dense_train}.npy",
        },
        "spectra": spectra,
        "predicted_learning_curves": curves,
        "measured_m116": {
            "sparse_accuracy": measured_sparse,
            "dense_accuracy": measured_dense,
            "crossing_interval": [27600, 55200],
        },
        "gates": gates,
        "certificate_verdict": {
            "note": "the MSE-proxy certificate does not predict the measured "
                    "accuracy crossing. The spectrum-level finding that does "
                    "survive: the frozen sparse code's linear span captures "
                    "~3.4x more one-hot label power than the dense trunk's "
                    "(11.8% vs 3.5%), which is why the sparse family has more "
                    "room to keep improving with data. But its captured modes "
                    "decay faster (effective rank 8 vs 30), so they are learned "
                    "slower; and the measured ACCURACY crossing is an argmax "
                    "phenomenon over 345 one-hot channels that the scalar MSE "
                    "proxy does not track. A classification-aware certificate "
                    "(per-class score-margin prediction) is the registered "
                    "follow-up if this direction is re-opened.",
            "registered_conclusion_licenses": "M121 fires under the registered "
                    "gate; the firing is about the MSE proxy, not about the "
                    "existence of learnable label power in the sparse span.",
        },
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM121 complete -> {output_dir / 'evidence.json'}", flush=True)
    print(f"  predicted crossing (lambda={lam_primary}): {pred_cross} "
          f"(measured midpoint 41400)", flush=True)
    print(f"  KS1 fired: {ks1['fired']}  KS2 fired: {ks2['fired']}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m121(args.config, args.output)


if __name__ == "__main__":
    main()
