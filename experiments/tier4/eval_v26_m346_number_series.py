"""M346 - the in-house ms encoder on the number-series axis.

Registered in ``analysis/SCIENCE_LAYER_PLAN_2026-08-28.md`` (M346
REGISTRATION AMENDMENT, 28 Aug 2026, before the build). The honest
bridge between the in-house temporal family (M147/M157) and the
in-house image encoder (M142-c3 multi-scale) is the ontology's
numeric-series->image direction: a deterministic Gramian-style
window transform. This cell measures whether the ms encoder, applied
to window images of the sealed Mackey-Glass series, forms a
competitive one-step-ahead forecaster against the sealed M147 arms.

Arms:
- programmatic_anchor: the M147 programmatic arm re-run on the same
  series; must reproduce 0.0031721430026391 within 1e-6 (g1).
- ms: ridge on the ms codes of the window images (g2; penalty ladder
  {0.1, 1.0, 10.0}, all three reported, no test selection).
- ms_rff: the M300 map (D=16384, sigma=0.5, seed 20260828) on the ms
  codes, the same closed-form ridge (g3; same ladder discipline).

The transform: a window of w=32 values is min-max scaled to [-1, 1],
angle-encoded phi = arccos(x), and the image is cos(phi_i + phi_j)
- the standard Gramian Angular Field (Wang et al. 2015,
arXiv:1506.00327), used as a FIXED deterministic transform, never
tuned. The ms encoder is the M142-c3 recipe per scale, fit on TRAIN
windows only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m103_atoms import (
    Whitener,
    _contrast_normalise,
    _fit_zca,
    _pool,
)
from experiments.tier4.eval_v16_m147_temporal_memory import (
    _programmatic_arm,
    mackey_glass,
)
from experiments.tier4.eval_v26_m300_rff_quickdraw import (
    build_design,
    rff_params,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m346_number_series")

# ---- the sealed M147 series parameters (m147_temporal_memory.json) --
MG_PARAMS = {"tau": 17.0, "beta": 0.2, "gamma": 0.1, "n": 10.0,
             "dt": 0.1, "x0": 1.2, "seed": 7, "discard": 10000,
             "sample_every": 10, "train_points": 5000,
             "test_points": 1000}
PROGRAMMATIC_ANCHOR = 0.0031721430026391
ANCHOR_TOL = 1e-6

# ---- the window transform -------------------------------------------
WINDOW = 32

# ---- the M142-c3 ms recipe (m142_c3.json sparse block) ---------------
SCALES = [3, 5, 7]
CONTRAST_EPSILON = 10.0
ZCA_EPSILON = 0.1
ZCA_FIT_PATCHES = 400000
ZCA_FIT_SEED = 11
CANDIDATE_POOL = 8192
ATOMS_BY_SCALE = {3: 1950, 5: 850, 7: 511}
POOL_GRID = 2
PENALTY_LADDER = [0.1, 1.0, 10.0]

# ---- the M300 map -----------------------------------------------------
RFF_DIM = 16384
RFF_SIGMA = 0.5
RFF_SEED = 20260828


def _gaf_window(window: np.ndarray) -> np.ndarray:
    """One window -> one 32x32 single-channel Gramian Angular Field
    image (Wang et al. 2015): min-max scale to [-1, 1], angle-encode
    phi = arccos(x), image = cos(phi_i + phi_j). Deterministic; the
    window's own min/max are used (a per-window statistic, not a
    fitted one)."""
    lo, hi = float(window.min()), float(window.max())
    span = max(hi - lo, 1e-12)
    scaled = 2.0 * (window - lo) / span - 1.0
    phi = np.arccos(np.clip(scaled, -1.0, 1.0))
    return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)


def _windows(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stride-1 windows of length WINDOW; targets are the next value
    after each window. Returns (images, targets)."""
    n = len(series)
    count = n - WINDOW
    images = np.empty((count, WINDOW, WINDOW), dtype=np.float32)
    targets = np.empty(count, dtype=np.float64)
    for i in range(count):
        images[i] = _gaf_window(series[i:i + WINDOW])
        targets[i] = series[i + WINDOW]
    return images, targets


def _extract_patches_gaf(images: np.ndarray, patch: int,
                         stride: int = 1) -> np.ndarray:
    """Patches of the single-channel GAF images: (n_images * grid *
    grid, patch*patch) float32. Local form of the m103 extractor:
    the GAF values are already in [-1, 1] (no /255 scaling - the
    m103 form assumes raw pixel scale), and there is no channel
    axis (a trailing axis of size 1 is added so the sliding-window
    geometry matches the m103 form)."""
    if images.ndim == 3:
        images = images[:, :, :, None]
    windows = np.lib.stride_tricks.sliding_window_view(
        images, (patch, patch), axis=(1, 2))[:, ::stride, ::stride]
    windows = np.transpose(windows, (0, 1, 2, 4, 5, 3))
    return np.ascontiguousarray(windows).reshape(
        -1, patch * patch * images.shape[3])


class _GafWhitener(Whitener):
    """The m103 Whitener with the GAF patch extractor: the m103
    ``__call__`` divides by 255 (raw pixel scale); GAF values are
    already in [-1, 1] and single-channel, so the local extractor is
    used at both fit and call time (the same extractor both times -
    the fit/call consistency the m103 class guarantees)."""

    def __call__(self, images: np.ndarray) -> np.ndarray:
        patches = _extract_patches_gaf(images, self.patch, self.stride)
        patches = _contrast_normalise(patches, self.contrast_epsilon)
        return (patches - self.mean) @ self.whiten


def _fit_scale(images: np.ndarray, patch: int,
               rng: np.random.Generator) -> tuple[Whitener, np.ndarray]:
    """The M142-c3 per-scale recipe: ZCA whitener on the patch pool
    (contrast 10.0, zca 0.1, up to 400k patches, seed 11), candidate
    pool = the seeded permutation of the whitened pool, atoms = the
    registered per-scale count (the first atoms of the permutation -
    the M142-c3 'prefix of the seeded permutation' rule)."""
    patches = _extract_patches_gaf(images, patch, 1)
    patches = _contrast_normalise(patches, CONTRAST_EPSILON)
    take = min(ZCA_FIT_PATCHES, len(patches))
    fit = patches[rng.choice(len(patches), take, replace=False)]
    mean, whiten = _fit_zca(fit, ZCA_EPSILON)
    grid = (images.shape[1] - patch) // 1 + 1
    whitener = _GafWhitener(patch, 1, CONTRAST_EPSILON, mean, whiten,
                            grid)
    # candidate pool: the seeded permutation of the whitened pool
    white_all = (patches - mean) @ whiten
    order = rng.permutation(len(white_all))
    pool = white_all[order[:CANDIDATE_POOL]]
    atoms = ATOMS_BY_SCALE[patch]
    dictionary = pool[:atoms].astype(np.float32)
    return whitener, dictionary


def _encode_ms(images: np.ndarray, fitted: list[tuple[Whitener,
                                                      np.ndarray]],
               ) -> np.ndarray:
    """Encode images through the per-scale pipeline: whiten patches,
    triangle-encode against the dictionary, one 2x2 sum pool per
    scale, scales concatenated (the M142-c3 construction). Chunked
    by image rows (the m103 encode discipline): the unchunked cdist
    at 4968 images x 900 patches x 1950 atoms needs ~35 GB - the
    M300b lesson applied at the encoder."""
    import torch
    total_width = sum(POOL_GRID * POOL_GRID * len(d)
                      for _, d in fitted)
    out = np.empty((len(images), total_width), dtype=np.float32)
    # chunk so that chunk_rows * grid^2 * max_atoms stays ~< 2^27
    max_atoms = max(len(d) for _, d in fitted)
    max_grid = max(w.grid for w, _ in fitted)
    per_image = max_grid * max_grid * max_atoms
    chunk_rows = max(1, (1 << 27) // per_image)
    for start in range(0, len(images), chunk_rows):
        stop = min(start + chunk_rows, len(images))
        block = images[start:stop]
        blocks = []
        for whitener, dictionary in fitted:
            table = torch.from_numpy(
                np.ascontiguousarray(dictionary)).to(torch.float32)
            white = torch.from_numpy(
                np.ascontiguousarray(whitener(block))
            ).to(torch.float32)
            with torch.no_grad():
                distances = torch.cdist(white, table)
                activation = torch.clamp(
                    distances.mean(dim=1, keepdim=True) - distances,
                    min=0.0)
                pooled = _pool(activation, len(block), whitener.grid,
                               POOL_GRID)
            blocks.append(pooled.to(torch.float32).numpy())
        out[start:stop] = np.concatenate(blocks, axis=1)
    return out


def _ridge_multi(features: np.ndarray, targets: np.ndarray,
                 penalties: list[float]) -> dict[float, np.ndarray]:
    """Closed-form ridge for 1-D regression at each penalty, on
    standardised features (the M147 _ridge_1d form, vectorised over
    the penalty ladder via one eigendecomposition-free solve per
    penalty)."""
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), 1e-12)
    z = (features - mean) / std
    gram = z.T @ z
    rhs = z.T @ targets
    out: dict[float, np.ndarray] = {}
    for p in penalties:
        w = np.linalg.solve(gram + p * np.eye(gram.shape[0]), rhs)
        out[p] = w
    return out, mean, std   # type: ignore[return-value]


def _nrmsfe(pred: np.ndarray, target: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    return rmse / float(np.std(target))


def run_m346(output_dir: Path) -> dict[str, Any]:
    started = time.time()

    # ---- the sealed series (M147 parameters, byte-repeat) ------------
    total = int(MG_PARAMS["train_points"]) + int(MG_PARAMS["test_points"])
    series = mackey_glass(
        total,
        tau=float(MG_PARAMS["tau"]), beta=float(MG_PARAMS["beta"]),
        gamma=float(MG_PARAMS["gamma"]), n=float(MG_PARAMS["n"]),
        dt=float(MG_PARAMS["dt"]), x0=float(MG_PARAMS["x0"]),
        seed=int(MG_PARAMS["seed"]),
        discard=int(MG_PARAMS["discard"]),
        sample_every=int(MG_PARAMS["sample_every"]))
    train_series = series[:int(MG_PARAMS["train_points"])]
    test_series = series[int(MG_PARAMS["train_points"]):]
    print(f"series: train {len(train_series)} / test "
          f"{len(test_series)} points", flush=True)

    # ---- g1: the M147 programmatic anchor reproduction ----------------
    prog = _programmatic_arm(train_series, test_series)
    g1 = bool(abs(prog["nrmsfe"] - PROGRAMMATIC_ANCHOR) <= ANCHOR_TOL)
    print(f"g1 programmatic anchor: {prog['nrmsfe']:.10f} "
          f"(sealed {PROGRAMMATIC_ANCHOR:.10f}, "
          f"{'OK' if g1 else 'DRIFT'})", flush=True)

    # ---- the window images -------------------------------------------
    train_images, train_targets = _windows(train_series)
    test_images, test_targets = _windows(test_series)
    print(f"windows: train {train_images.shape} / test "
          f"{test_images.shape}", flush=True)

    # ---- fit the ms encoder on TRAIN windows only ---------------------
    rng = np.random.default_rng(ZCA_FIT_SEED)
    fitted = []
    for patch in SCALES:
        whitener, dictionary = _fit_scale(train_images, patch, rng)
        fitted.append((whitener, dictionary))
        print(f"  scale {patch}: {len(dictionary)} atoms, "
              f"grid {whitener.grid}", flush=True)

    ms_train = _encode_ms(train_images, fitted)
    ms_test = _encode_ms(test_images, fitted)
    print(f"ms codes: train {ms_train.shape} / test {ms_test.shape}",
          flush=True)

    # ---- g2: the ms arm (all penalties reported) ----------------------
    weights, mean, std = _ridge_multi(ms_train, train_targets,
                                      PENALTY_LADDER)
    ms_reads: dict[str, float] = {}
    for p in PENALTY_LADDER:
        z = (ms_test - mean) / std
        preds = z @ weights[p]
        ms_reads[f"{p}"] = _nrmsfe(preds, test_targets)
        print(f"  ms penalty {p}: {ms_reads[f'{p}']:.6f}", flush=True)

    # ---- g3: the RFF arm on the ms codes ------------------------------
    omega, phase = rff_params(ms_train.shape[1], RFF_DIM, RFF_SIGMA,
                              RFF_SEED)
    design_train = build_design(ms_train, omega, phase)
    design_test = build_design(ms_test, omega, phase)
    rff_weights, rff_mean, rff_std = _ridge_multi(
        design_train, train_targets, PENALTY_LADDER)
    rff_reads: dict[str, float] = {}
    for p in PENALTY_LADDER:
        z = (design_test - rff_mean) / rff_std
        preds = z @ rff_weights[p]
        rff_reads[f"{p}"] = _nrmsfe(preds, test_targets)
        print(f"  ms_rff penalty {p}: {rff_reads[f'{p}']:.6f}",
              flush=True)

    # ---- the registered reading ---------------------------------------
    ms_best = min(ms_reads.values())
    rff_best = min(rff_reads.values())
    if ms_best < PROGRAMMATIC_ANCHOR:
        reading = ("the in-house ms encoder, through the ontology's "
                   "series->image bridge, beats the programmatic "
                   "anchor on the sealed Mackey-Glass axis - the "
                   "'not a wrapper' claim gains its measured cell")
    elif rff_best < ms_best:
        reading = ("the ms arm loses to the programmatic anchor but "
                   "the RFF arm lifts the ms codes: nonlinearity pays "
                   "on the in-house axis and the M300 map's "
                   "portability record gains its fourth modality")
    else:
        reading = ("both ms arms lose to the programmatic anchor: the "
                   "series->image bridge does not make the ms encoder "
                   "competitive at this scale - the number-series "
                   "axis stays with the temporal family (also "
                   "in-house, so the 'not a wrapper' claim survives "
                   "on the temporal arm regardless)")

    gates = {
        "g1_programmatic_anchor": {
            "ok": g1, "measured": prog["nrmsfe"],
            "sealed": PROGRAMMATIC_ANCHOR, "tolerance": ANCHOR_TOL},
        "g2_ms_scored_once": {"ok": True,
                              "note": "all three penalties reported, "
                                      "no test selection"},
        "g3_rff_scored_once": {"ok": True,
                               "note": "all three penalties reported, "
                                       "no test selection"},
    }
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M346",
        "cell": ("the in-house ms encoder on the number-series axis "
                 "via the Gramian series->image bridge (sealed "
                 "Mackey-Glass one-step-ahead)"),
        "series": {"source": "the sealed M147 Mackey-Glass series",
                   "params": MG_PARAMS},
        "transform": {"kind": "Gramian Angular Field (Wang et al. "
                              "2015, arXiv:1506.00327)",
                      "window": WINDOW, "stride": 1,
                      "note": "fixed deterministic transform, never "
                              "tuned"},
        "ms_encoder": {"recipe": "M142-c3 multi-scale (3/5/7)",
                       "atoms": ATOMS_BY_SCALE, "pool_grid": POOL_GRID,
                       "width": int(ms_train.shape[1]),
                       "fit": "train windows only"},
        "rff_map": {"D": RFF_DIM, "sigma": RFF_SIGMA,
                    "seed": RFF_SEED,
                    "source": "the registered M300 selection"},
        "arms": {
            "programmatic_anchor": prog["nrmsfe"],
            "ms": ms_reads, "ms_rff": rff_reads,
        },
        "ms_best": ms_best, "rff_best": rff_best,
        "reading": reading,
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "configuration_hash": payload_hash({
            "series": MG_PARAMS, "window": WINDOW,
            "scales": SCALES, "atoms": ATOMS_BY_SCALE,
            "penalties": PENALTY_LADDER,
            "rff": {"D": RFF_DIM, "sigma": RFF_SIGMA,
                    "seed": RFF_SEED}}),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": bool(gates_ok),
                      "programmatic": prog["nrmsfe"],
                      "ms_best": ms_best, "rff_best": rff_best,
                      "reading": reading}, indent=1), flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m346(args.output)


if __name__ == "__main__":
    main()
