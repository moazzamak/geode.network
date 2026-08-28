"""M115 — Closed-form LoFi stack: does closed-form depth raise the effective
degree (lift the single-stage ceiling) at a measured training-cost ratio vs
dense?

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v18.md`` section 5.1 and
``experiments/configs/v16/m115_lofi.json``.

Question (v18). The single-stage frozen family saturates at ~0.21 (degree-2 in
pixels). Neural LoFi (Dandi et al. 2026, arXiv 2605.13612) says deep learning is
closed-form low-degree filtering: each layer selects directions of maximal
accessible low-degree correlation to the label. A spectral estimator achieves
optimal scaling for hierarchical targets (Defilippis et al. 2026, arXiv
2602.05846); higher cumulants become cheap to learn when latent structure is
shared across layers (Bardone et al. 2026, arXiv 2603.12901). M115 instantiates
this CLOSED-FORM (no backprop, no epochs) and measures: (KS1) does depth lift
the 0.2153 ceiling, and (KS2) what is the closed-form training-cost ratio vs
dense?

Arms (shared subsample, M108-exact whitener/dictionary, closed-form ridge head
penalty 1.0, fit on all train rows, per-domain eval):
- L=0: the sealed frozen 3072-dictionary arm (re-measured, gated vs M113's
  0.21528 within 0.002). Degree 2 in pixels.
- L=1: one label-coupled spectral layer. C = F^T Y_c (F = frozen pooled codes,
  Y_c = centered one-hot); P = top-k left singular vectors of C (the directions
  of maximal accessible low-degree correlation to the label, k <= 345);
  F_proj = (F - mean) P; whiten (center + per-dim standardize); fit a mini-batch
  k-means (VQ) dictionary on the whitened projection; triangle-encode (soft
  assignment over ALL atoms); ridge. Effective degree 4.
- L=2: repeat on L=1's codes. Effective degree 8.

The layer-1/2 encode is a per-image VQ triangle (cdist in the projected space,
no spatial pooling): each layer's distance is quadratic in its input, so the
effective degree doubles per layer.

Gates (registered before measurement):
- t1: L=0 reproduces M113's sealed random-3072 (0.21528) within 0.002.
- KS1 (degree): best L does not beat L=0 by >= +0.01 overall -> closed-form
  depth does not lift the single-stage ceiling -> B1 fails.
- KS2 (training cost): closed-form stack total corpus training ops vs dense's
  ledgered corpus training ops (head ridge/solve from M109 evidence; the
  external trunk pretraining is DISCLOSED as an asymmetry, not counted in the
  measured ratio); fired (fail) if the ratio < 10x.

Reproduce with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m115_lofi
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m104_experts import (
    RidgeAccumulator,
    _chunk_rows,
    _training_macs,
)
from experiments.tier4.eval_v15_m107_dense import (
    _solve_and_score,
    _verify_pixel_identity,
)
from experiments.tier4.eval_v16_m108_dictionary import (
    _encode_block_device,
    _verify_device,
)
from experiments.tier4.eval_v16_m109_trunk import (
    _load_corpus,
    _parity_guard,
)
from experiments.tier4.eval_v16_m113_learned import (
    _build_whitener_and_candidates,
    _fit_vq,
    _m107_dense_curve,
    _random_dictionary,
)
from experiments.tier4.eval_v15_m107_dense import _dinov2_geometry

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m115_lofi.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m115_lofi"
M113_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m113_learned" / "evidence.json"
M109_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m109_trunk" / "evidence.json"
M107_EVIDENCE = REPO_ROOT / "logs" / "results" / "v15" / "m107_dense" / "evidence.json"

PATCH_DIM = 108
T1_TOLERANCE = 0.002
KS1_MARGIN = 0.01
KS2_MIN_RATIO = 10.0
BLOCK = 4096


# --------------------------------------------------------------------------
# L=0: frozen pipeline
# --------------------------------------------------------------------------
def _write_frozen_codes(corpus: dict[str, np.ndarray], dictionary: np.ndarray,
                        whitener, pool_grid: int, device: torch.device,
                        rows: np.ndarray, path: Path,
                        split: str = "train",
                        throttle_seconds: float = 0.0) -> np.ndarray:
    """Write the frozen pooled codes (n, 4*atoms) to a memmap; return it.

    ``split`` selects the corpus array the row indices refer to ("train" or
    "test"). A hardcoded train-images reference here silently turns the test
    codes into train encodings and collapses held-out accuracy to ~1/classes.

    ``throttle_seconds`` (default 0.0) inserts an idle pause between GPU
    batches. When the compute GPU is also the DISPLAY GPU (the RX 9070 XT
    drives the 2560x1440 monitor on this machine), sustained back-to-back
    compute kernels starve the display engine and the AMD driver's TDR
    watchdog resets the GPU - black screen / apparent freeze. A small pause
    per batch lets the display engine get GPU time and prevents the TDR.
    Default 0.0 leaves every existing caller bit-identical.
    """
    images = corpus[f"{split}_images"]
    n = len(rows)
    dims = pool_grid * pool_grid * len(dictionary)
    mem = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                    shape=(n, dims))
    table = torch.from_numpy(np.ascontiguousarray(dictionary)).to(
        torch.float32).to(device)
    step = min(BLOCK, _chunk_rows(len(dictionary), whitener.grid, n))
    for start in range(0, n, step):
        take = rows[start:start + step]
        mem[start:start + len(take)] = _encode_block_device(
            images[take], table, whitener, pool_grid)
        if throttle_seconds > 0.0:
            time.sleep(throttle_seconds)
    return np.lib.format.open_memmap(path, mode="r", dtype=np.float32)


def _score_accumulator(accumulator: RidgeAccumulator,
                       test_blocks: Iterator[tuple[np.ndarray, np.ndarray,
                                                   np.ndarray]]
                       ) -> dict[str, Any]:
    return _solve_and_score(accumulator, [1.0], test_blocks)


def _per_domain(result: dict[str, Any]) -> list[float]:
    pc = result["per_domain_correct"]["1.0"]
    pr = result["per_domain_rows"]["1.0"]
    return [pc[d] / pr[d] for d in range(6)]


# --------------------------------------------------------------------------
# layer rule (label-coupled spectral selection, closed-form)
# --------------------------------------------------------------------------
def _mem_gen(mem: np.ndarray, labels: np.ndarray, step: int = BLOCK
             ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    n = len(mem)
    for i in range(0, n, step):
        yield np.asarray(mem[i:i + step]), labels[i:i + step]


def _label_cross_cov(gen: Iterator[tuple[np.ndarray, np.ndarray]],
                     dims: int, classes: int
                     ) -> tuple[np.ndarray, np.ndarray, int]:
    """C = sum_i (F_i - mean) (y_i - p)^T: label cross-covariance (dims, classes)."""
    C = np.zeros((dims, classes), dtype=np.float64)
    s = np.zeros(dims, dtype=np.float64)
    counts = np.zeros(classes, dtype=np.float64)
    n = 0
    onehot = np.eye(classes)
    for block, labels in gen:
        b = block.astype(np.float64)
        n += len(block)
        s += b.sum(axis=0)
        counts += np.bincount(labels, minlength=classes).astype(np.float64)
        C += b.T @ onehot[labels]
    p = counts / max(n, 1)
    C = C - np.outer(s / max(n, 1), p) * n
    return C, s, n


def _projection(C: np.ndarray, k: int, device: torch.device) -> np.ndarray:
    """Top-k left singular vectors of C: the label-correlated directions."""
    k = min(k, C.shape[1])
    u = torch.linalg.svd(torch.from_numpy(C).to(device),
                         full_matrices=False).U[:, :k]
    return u.detach().cpu().numpy().astype(np.float64)


def _project_pass(mem: np.ndarray, P: np.ndarray, meanF: np.ndarray,
                  raw_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Project (F - meanF) @ P, write raw to memmap, return mean/std of it."""
    n, dims = mem.shape
    k = P.shape[1]
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32,
                                    shape=(n, k))
    su = np.zeros(k, dtype=np.float64)
    sq = np.zeros(k, dtype=np.float64)
    for i in range(0, n, BLOCK):
        block = np.asarray(mem[i:i + BLOCK]).astype(np.float64) - meanF
        proj = (block @ P).astype(np.float32)
        raw[i:i + len(proj)] = proj
        su += proj.astype(np.float64).sum(axis=0)
        sq += (proj.astype(np.float64) ** 2).sum(axis=0)
    mean_proj = su / n
    std_proj = np.sqrt(np.maximum(sq / n - mean_proj ** 2, 1e-12))
    return mean_proj, std_proj


def _whiten(proj: np.ndarray, mean_proj: np.ndarray, std_proj: np.ndarray
            ) -> np.ndarray:
    return ((proj - mean_proj) / std_proj).astype(np.float32)


def _fit_vq_on(raw_path: Path, mean_proj: np.ndarray, std_proj: np.ndarray,
               atoms_l: int, vq_seed: int, vq_iters: int, vq_batch: int,
               device: torch.device, fit_subsample: int = 20000
               ) -> np.ndarray:
    raw = np.lib.format.open_memmap(raw_path, mode="r", dtype=np.float32)
    n = len(raw)
    rng = np.random.default_rng(vq_seed)
    idx = rng.choice(n, min(fit_subsample, n), replace=False)
    subsample = _whiten(np.asarray(raw[idx]), mean_proj, std_proj)
    batch = min(int(vq_batch), len(subsample))   # _fit_vq draws without
    # replacement; a batch larger than the pool would raise
    centroids, _fit_macs, _dead = _fit_vq(subsample, atoms_l, vq_iters,
                                          batch, vq_seed, device)
    return centroids


def _encode_ridge_pass(raw_path: Path, mean_proj: np.ndarray,
                       std_proj: np.ndarray, centroids: np.ndarray,
                       labels: np.ndarray, classes: int,
                       device: torch.device, out_codes_path: Path | None
                       ) -> tuple[RidgeAccumulator, int]:
    """Stream raw -> whiten -> VQ triangle -> ridge; optionally write codes."""
    raw = np.lib.format.open_memmap(raw_path, mode="r", dtype=np.float32)
    n = len(raw)
    atoms_l = len(centroids)
    centroids_t = torch.from_numpy(np.ascontiguousarray(centroids)).to(
        torch.float32).to(device)
    acc = RidgeAccumulator(atoms_l, classes)
    if out_codes_path is not None:
        codes = np.lib.format.open_memmap(out_codes_path, mode="w+",
                                          dtype=np.float32, shape=(n, atoms_l))
    for i in range(0, n, BLOCK):
        white = _whiten(np.asarray(raw[i:i + BLOCK]), mean_proj, std_proj)
        p = torch.from_numpy(np.ascontiguousarray(white)).to(device)
        with torch.no_grad():
            d = torch.cdist(p, centroids_t)             # (b, atoms_l)
            act = torch.clamp(d.mean(dim=-1, keepdim=True) - d, min=0.0)
        code = act.cpu().numpy().astype(np.float32)
        acc.add(code, labels[i:i + len(code)])
        if out_codes_path is not None:
            codes[i:i + len(code)] = code
    if out_codes_path is not None:
        del codes
    return acc, atoms_l


def _test_codes(mem: np.ndarray, P: np.ndarray, meanF: np.ndarray,
                mean_proj: np.ndarray, std_proj: np.ndarray,
                centroids: np.ndarray, device: torch.device
                ) -> np.ndarray:
    """Project/whiten/VQ-encode the test features with TRAIN-fitted objects."""
    n, _dims = mem.shape
    k = P.shape[1]
    centroids_t = torch.from_numpy(np.ascontiguousarray(centroids)).to(
        torch.float32).to(device)
    out = []
    for i in range(0, n, BLOCK):
        block = np.asarray(mem[i:i + BLOCK]).astype(np.float64) - meanF
        proj = (block @ P).astype(np.float32)
        white = _whiten(proj, mean_proj, std_proj)
        p = torch.from_numpy(np.ascontiguousarray(white)).to(device)
        with torch.no_grad():
            d = torch.cdist(p, centroids_t)
            act = torch.clamp(d.mean(dim=-1, keepdim=True) - d, min=0.0)
        out.append(act.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def run_m115(config_path: Path, output_dir: Path,
             progress: bool = True) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible (_smoke_note) and would write to the SEALED output "
            f"directory {DEFAULT_OUTPUT}.")

    torch.set_num_threads(config["numerics"]["torch_threads"])
    torch.manual_seed(config["numerics"]["seed"])
    configure_external_cache_environment()
    device_report = _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("parity guard at startup", flush=True)
    parity = _parity_guard(torch, config, device)

    print("loading corpus", flush=True)
    corpus, train_index, test_index = _load_corpus(config)
    classes = int(corpus["train_labels"].max()) + 1
    size = config["corpus"]["image_size"]
    for split, idx in (("train", train_index), ("test", test_index)):
        _verify_pixel_identity(split, idx, corpus[f"{split}_images"], size,
                               config["corpus"]["pixel_identity_rows"])

    rep = config["sparse"]
    atoms = int(rep["atoms"])
    k_proj = int(rep["k_projection"])
    atoms2 = int(rep["atoms_layer2"])
    atoms3 = int(rep["atoms_layer3"])
    pool_grid = int(rep["pool_grid"])
    smoke_skip = bool(config.get("_smoke_skip_gates", False))
    gates: dict[str, Any] = {}

    print("building global whitener + random candidate pool (M108 exact)",
          flush=True)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)
    dictionary = _random_dictionary(candidates, len(candidates),
                                    int(rep["dictionary_seed"]), atoms)

    # ---- L=0: frozen arm (re-measured, gated) ------------------------------
    print("L=0: frozen arm (re-measured)", flush=True)
    acc0 = RidgeAccumulator(pool_grid * pool_grid * atoms, classes)
    fit_rows = np.arange(len(corpus["train_labels"]))
    for block, labels in _frozen_gen(corpus, dictionary, whitener, pool_grid,
                                     device, fit_rows, split="train"):
        acc0.add(block, labels)
    test_rows = np.arange(len(corpus["test_labels"]))

    def _l0_test_blocks():
        cursor = 0
        for block, labels in _frozen_gen(corpus, dictionary, whitener,
                                         pool_grid, device, test_rows,
                                         split="test"):
            yield block, labels, corpus["test_domains"][cursor:cursor + len(block)]
            cursor += len(block)
    arm0 = _score_accumulator(acc0, _l0_test_blocks())
    a0 = arm0["accuracy_by_penalty"]["1.0"]
    print(f"  L=0 accuracy: {a0:.4f}", flush=True)

    if not smoke_skip:
        m113 = json.loads(M113_EVIDENCE.read_text(encoding="utf-8"))
        ref = float(m113["arms"]["a_random"]["accuracy_by_penalty"]["1.0"])
        delta = a0 - ref
        if abs(delta) > T1_TOLERANCE:
            print(f"  t1 FAILED: {a0:.4f} vs M113 {ref:.4f} "
                  f"(delta {delta:+.5f})", flush=True)
            write_canonical_json(output_dir / "evidence.json", {
                "milestone": "M115", "admissible_as_evidence": False,
                "void": True, "void_reason": "L=0 t1 reproduction failed",
                "arm0": a0, "m113_ref": ref, "t1_delta": delta,
            })
            return {"admissible_as_evidence": False, "void": True}
        gates["t1_delta"] = delta
        print(f"  t1 reproduction delta {delta:+.5f} (<= {T1_TOLERANCE})",
              flush=True)

    cache = data_cache_root() / "v16" / "m115_lofi"
    cache.mkdir(parents=True, exist_ok=True)
    n_train = len(corpus["train_labels"])
    n_test = len(corpus["test_labels"])

    # ---- L=1: one label-coupled spectral layer -----------------------------
    print("L=1: label-coupled spectral layer (degree 4)", flush=True)
    f0_train = _write_frozen_codes(corpus, dictionary, whitener, pool_grid,
                                   device, fit_rows, cache / "f0_train.npy",
                                   split="train")
    P1, meanF1 = _project_and_fit(f0_train, corpus["train_labels"], classes,
                                  k_proj, device)
    mean_p1, std_p1 = _project_pass(f0_train, P1, meanF1,
                                    cache / "f1_raw_train.npy")
    centroids1 = _fit_vq_on(cache / "f1_raw_train.npy", mean_p1, std_p1,
                            atoms2, int(rep["vq_fit_seed"]),
                            int(rep["vq_iters"]), int(rep["vq_batch"]),
                            device)
    acc1, _w1 = _encode_ridge_pass(cache / "f1_raw_train.npy", mean_p1,
                                   std_p1, centroids1, corpus["train_labels"],
                                   classes, device, cache / "f1_train.npy")

    f0_test = _write_frozen_codes(corpus, dictionary, whitener, pool_grid,
                                  device, test_rows, cache / "f0_test.npy",
                                  split="test")
    test1 = _test_codes(f0_test, P1, meanF1, mean_p1, std_p1, centroids1,
                        device)

    def _l1_test_blocks():
        for i in range(0, n_test, BLOCK):
            yield (test1[i:i + BLOCK],
                   corpus["test_labels"][i:i + BLOCK],
                   corpus["test_domains"][i:i + BLOCK])
    arm1 = _score_accumulator(acc1, _l1_test_blocks())
    a1 = arm1["accuracy_by_penalty"]["1.0"]
    print(f"  L=1 accuracy: {a1:.4f}", flush=True)

    # ---- L=2: repeat on L=1 codes ------------------------------------------
    print("L=2: second label-coupled spectral layer (degree 8)", flush=True)
    f1_train = np.lib.format.open_memmap(cache / "f1_train.npy", mode="r",
                                         dtype=np.float32)
    P2, meanF2 = _project_and_fit(f1_train, corpus["train_labels"], classes,
                                  k_proj, device)
    mean_p2, std_p2 = _project_pass(f1_train, P2, meanF2,
                                    cache / "f2_raw_train.npy")
    centroids2 = _fit_vq_on(cache / "f2_raw_train.npy", mean_p2, std_p2,
                            atoms3, int(rep["vq_fit_seed"]) + 1,
                            int(rep["vq_iters"]), int(rep["vq_batch"]),
                            device)
    acc2, _w2 = _encode_ridge_pass(cache / "f2_raw_train.npy", mean_p2,
                                   std_p2, centroids2, corpus["train_labels"],
                                   classes, device, None)
    test2 = _test_codes(test1, P2, meanF2, mean_p2, std_p2, centroids2, device)

    def _l2_test_blocks():
        for i in range(0, n_test, BLOCK):
            yield (test2[i:i + BLOCK],
                   corpus["test_labels"][i:i + BLOCK],
                   corpus["test_domains"][i:i + BLOCK])
    arm2 = _score_accumulator(acc2, _l2_test_blocks())
    a2 = arm2["accuracy_by_penalty"]["1.0"]
    print(f"  L=2 accuracy: {a2:.4f}", flush=True)

    # ---- gates -------------------------------------------------------------
    best_l_acc = max(a0, a1, a2)
    best_l = {a0: 0, a1: 1, a2: 2}[best_l_acc]
    ks1 = {
        "registered": "best L beats L=0 by >= +0.01 -> closed-form depth "
                      "lifts the single-stage ceiling",
        "l0": a0, "l1": a1, "l2": a2, "best_layer": best_l,
        "margin": KS1_MARGIN,
        "fired": (best_l_acc - a0) < KS1_MARGIN,
        "delta": best_l_acc - a0,
        "consequence": "if fired, closed-form depth does not lift the "
                       "single-stage ceiling -> B1 fails; the long-term "
                       "buy-back thesis is dead on the quality axis",
    }

    # training-cost ledger (corpus-only, ops)
    rows = n_train
    grid = whitener.grid
    P_ = grid * grid
    ledger = {}
    ledger["l0"] = int(_training_macs(rows, atoms, grid, PATCH_DIM,
                                      pool_grid, classes))
    l1_extra = (rows * P_ * atoms * PATCH_DIM      # second frozen pass
                + rows * (4 * atoms) * k_proj      # projection
                + (4 * atoms) * (4 * atoms) * classes  # SVD of C (rough)
                + int(rep["vq_iters"]) * int(rep["vq_batch"]) * k_proj * atoms2
                + rows * k_proj * atoms2           # VQ encode
                + rows * atoms2 * atoms2           # ridge gram
                + atoms2 ** 3 // 3)                # ridge solve
    ledger["l1"] = int(ledger["l0"] + l1_extra)
    l2_extra = (rows * atoms2 * k_proj             # projection
                + (atoms2 * atoms2 * classes)      # SVD of C2
                + int(rep["vq_iters"]) * int(rep["vq_batch"]) * k_proj * atoms3
                + rows * k_proj * atoms3           # VQ encode
                + rows * atoms3 * atoms3           # ridge gram
                + atoms3 ** 3 // 3)                # ridge solve
    ledger["l2"] = int(ledger["l1"] + l2_extra)

    # ---- dense corpus training (closed-form head fit, M109 t1) ------------
    # Primary (registered): dense's head-training corpus ops = the closed-form
    # ridge fit on the frozen trunk features (M109 t1 protocol, trainable
    # parameters 0). The trunk forward over the corpus AND its LVD-142M
    # pretraining are DISCLOSED as the external asymmetry (uncounted), matching
    # the v18 registration "vs dense head-training". The sensitivity rows show
    # the gate under both alternatives.
    m109 = json.loads(M109_EVIDENCE.read_text(encoding="utf-8"))
    trunk_width = 2 * _dinov2_geometry("small")["width"]
    dense_ridge_fit = int(n_train * (trunk_width * trunk_width
                                     + trunk_width * classes)
                          + trunk_width ** 3 // 3)
    dense_t2 = m109["results"]["dense"].get("t2_r224", {})
    t2_epochs = int(dense_t2.get("training", {}).get("epochs_run", 0))
    t2_params = int(dense_t2.get("trainable_parameters", 0))
    dense_sgd_head = int(t2_epochs * n_train * 3 * t2_params)

    # sensitivity: count the dense trunk forward over the corpus too, at the
    # cheapest M107 dense point whose accuracy reaches A
    m107_curve = _m107_dense_curve()
    eligible = [p for p in m107_curve if p[1] >= best_l_acc]
    trunk_forward = None
    cheapest = None
    if eligible:
        cheapest = min(eligible, key=lambda p: p[0])
        trunk_forward = int(n_train * cheapest[0] + dense_ridge_fit)

    dense_corpus_train = dense_ridge_fit
    ratio = (ledger[f"l{best_l}"] / dense_corpus_train) if dense_corpus_train else None
    ks2 = {
        "registered": "closed-form stack total corpus training ops vs dense's "
                      "head-training corpus ops (M109 t1 closed-form ridge fit "
                      "on the frozen trunk features); fired if ratio < 10x",
        "accuracy_a": best_l_acc,
        "stack_train_ops_best": ledger[f"l{best_l}"],
        "dense_head_train_ops": dense_corpus_train,
        "ratio": ratio,
        "min_ratio": KS2_MIN_RATIO,
        "fired": ratio is None or ratio < KS2_MIN_RATIO,
        "dense_trunk_width": int(trunk_width),
        "dense_head_fit_formula": (
            "n*(w^2 + w*classes) + w^3/3 with w = 2*dinov2-small width "
            "(CLS+meanpool), measured from the ONNX graph (M107 R7 rule)"
        ),
        "dense_trunk_pretraining": (
            "DISCLOSED as external asymmetry (LVD-142M), not counted; the "
            "trunk forward over the corpus is likewise external in the "
            "primary formula, see sensitivity_trunk_forward_counted"
        ),
        "sensitivity_dense_sgd_head": {
            "formula": "epochs*n*3*trainable_parameters (M109 t2_r224: "
                        f"{t2_epochs} epochs, {t2_params} params)",
            "ops": dense_sgd_head,
            "ratio_if_used": (
                (ledger[f"l{best_l}"] / dense_sgd_head) if dense_sgd_head else None
            ),
        },
        "sensitivity_trunk_forward_counted": {
            "formula": "cheapest M107 dense point with accuracy >= A: "
                        "n*per_image_macs + head_fit (the dense trunk forward "
                        "counted as a corpus pass, symmetric with the sparse "
                        "encode pass)",
            "point": (
                None if cheapest is None else
                {"macs_per_image": int(cheapest[0]),
                 "accuracy": float(cheapest[1])}
            ),
            "ops": trunk_forward,
            "ratio_if_used": (
                (ledger[f"l{best_l}"] / trunk_forward) if trunk_forward else None
            ),
        },
        "note": "Primary counts head training on both sides: sparse = encode "
                "+ VQ fit + ridge over the corpus; dense = closed-form ridge "
                "on frozen trunk features. The dense trunk forward and its "
                "pretraining are the disclosed external asymmetry. The "
                "sensitivity rows show the gate if that asymmetry is counted "
                "(SGD head) or if the dense trunk forward is counted as a "
                "corpus pass (M107 curve).",
    }
    gates["kill_switch_1_degree"] = ks1
    gates["kill_switch_2_training_cost"] = ks2
    gates["_smoke_skip"] = smoke_skip

    evidence = {
        "milestone": "M115",
        "admissible_as_evidence": not inadmissible,
        "registered_in": config.get("registered_in"),
        "question": ("does closed-form depth raise the effective degree and "
                     "lift the single-stage ceiling, at what training-cost "
                     "ratio vs dense?"),
        "config_file": Path(config_path).name,
        "config": config,
        "device": device_report,
        "parity_guard": parity,
        "arms": {
            "l0": {"accuracy": a0, "per_domain": _per_domain(arm0)},
            "l1": {"accuracy": a1, "per_domain": _per_domain(arm1)},
            "l2": {"accuracy": a2, "per_domain": _per_domain(arm2)},
        },
        "layer_objects": {
            "l1_projection_dims": int(P1.shape[1]),
            "l1_atoms": int(atoms2),
            "l2_projection_dims": int(P2.shape[1]),
            "l2_atoms": int(atoms3),
        },
        "training_cost_ledger_ops": ledger,
        "gates": gates,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    evidence["payload_sha256"] = payload_hash(evidence)
    print(f"\nM115 complete -> {output_dir / 'evidence.json'}", flush=True)
    print(f"  L=0: {a0:.4f}  L=1: {a1:.4f}  L=2: {a2:.4f}", flush=True)
    print(f"  KS1 fired: {ks1['fired']}  KS2 fired: {ks2['fired']}", flush=True)
    return evidence


def _project_and_fit(mem: np.ndarray, labels: np.ndarray, classes: int,
                     k: int, device: torch.device
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Cross-covariance -> SVD projection; returns (P, meanF)."""
    dims = mem.shape[1]
    C, s, n_used = _label_cross_cov(_mem_gen(mem, labels), dims, classes)
    P = _projection(C, k, device)
    return P, s / max(n_used, 1)


def _frozen_gen(corpus, dictionary, whitener, pool_grid, device, rows,
                split: str = "train"):
    images = corpus[f"{split}_images"]
    labels = corpus[f"{split}_labels"]
    table = torch.from_numpy(np.ascontiguousarray(dictionary)).to(
        torch.float32).to(device)
    step = min(BLOCK, _chunk_rows(len(dictionary), whitener.grid, len(rows)))
    for start in range(0, len(rows), step):
        take = rows[start:start + step]
        yield (_encode_block_device(images[take], table, whitener, pool_grid),
               labels[take])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m115(args.config, args.output)


if __name__ == "__main__":
    main()
