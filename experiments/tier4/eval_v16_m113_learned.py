"""M113 — Learned (fitted) dictionary vs random dictionary.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v17.md`` (M113, 6 August
2026) and ``experiments/configs/v16/m113_learned.json``.

Question. Every sealed sparse figure (M107/M108/M109/M110/A2/A5) uses a
dictionary of RANDOM whitened patches — or at best a *selection* from that
random pool. M108's measured "learning" (arms c select_discriminative, e
ridge-leverage) was importance sampling, not fitting: at 3,072 atoms it scored
0.2138 / 0.2120 vs random 0.2153 — selection from a random pool does not beat
the pool. The atoms have never been *fitted* to the whitened-patch manifold.
So the ~0.21 frozen ceiling is the ceiling of a *random* basis. At matched
atoms (3,072) and matched encode MACs, does a fitted (k-means/VQ) dictionary
lift the ceiling? Is the ceiling a property of the sparse family or of the
random basis?

Arms (shared subsample, global M108 whitener, triangle encode, 2x2 pool,
closed-form ridge head penalty 1.0 = M108's chosen constant, fit on all
138,000 train rows, per-domain eval):

- (a) random-3072 — M108 arm (a) exact construction (prefix of the seeded
  permutation of the shared 8,192-whitened-patch pool), re-measured; gated to
  reproduce M108's sealed a_random_3072 (0.2153) within 0.002 or VOID.
- (b) learned-3072 — mini-batch k-means (VQ) centroids fitted on a registered
  2,000,000-whitened-patch pool (3,000 train images, seed 22), GPU. Same atom
  count and same 108-dim space -> identical encode MACs to (a). PRIMARY.
- (c) learned-topk-64 — arm (b)'s dictionary, ridge on the full-width code
  with only the top-64 nearest-atom triangle activations per patch nonzero
  (zero-padded to the full atom dimension, so the same per-image MACs as (b)).
  Reported, not a verdict: does top-k sparsity help accuracy at matched cost?
  The genuine head/encode cost cuts (compact sparse code + sparse ridge
  accumulator; approximate neighbor search) are deferred.

Kill switches (registered before measurement):

- KS1 (learned lifts the ceiling): if (b) does not beat (a) overall by
  >= +0.01 at matched atoms/MACs, the random basis is not the binding
  constraint and the learned-dictionary thesis fails at this budget.
- KS2 (vs dense at-or-below cost): (b) at 254.6M total MACs vs the best M107
  sealed dense point at-or-below that cost (dense r42: 0.1972 at 215.6M). If
  (b) < 0.1972 + 0.01, no global accuracy win is licensed. Honest note: (b)
  pays ~18% more MACs than r42, so this is accuracy-at-cost, NOT an efficiency
  claim; the efficiency regimes are per-domain (A5 KS2) and the deferred
  approximate-search top-k.

Reproduce with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m113_learned
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from experiments.common.data_cache import configure_external_cache_environment
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m103_atoms import (
    Whitener,
    _contrast_normalise,
    _extract_patches,
    _fit_zca,
    _pool,
)
from experiments.tier4.eval_v15_m104_experts import (
    RidgeAccumulator,
    _chunk_rows,
    _inference_macs,
)
from experiments.tier4.eval_v15_m107_dense import (
    _solve_and_score,
    _verify_pixel_identity,
)
from experiments.tier4.eval_v16_m108_dictionary import (
    _best_dense_at_or_below,
    _encode_block_device,
    _random_order,
    _verify_device,
)
from experiments.tier4.eval_v16_m109_trunk import (
    _load_corpus,
    _parity_guard,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m113_learned.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m113_learned"
M107_EVIDENCE = REPO_ROOT / "logs" / "results" / "v15" / "m107_dense" / "evidence.json"
M108_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m108_dictionary" / "evidence.json"

PATCH_DIM = 108          # 6 * 6 * 3
T1_TOLERANCE = 0.002
KS1_MARGIN = 0.01
KS2_MARGIN = 0.01


# --------------------------------------------------------------------------
# shared construction (M108-exact whitener + random candidate pool)
# --------------------------------------------------------------------------
def _build_whitener_and_candidates(config: dict[str, Any],
                                   corpus: dict[str, np.ndarray]
                                   ) -> tuple[Whitener, np.ndarray]:
    """M108's exact whitener + candidate pool (same seeds, sample and pool).

    Replicates eval_v16_m108_dictionary.run_m108's construction so arm (a) is
    M108's arm (a) atom-for-atom: the whitener is fit on up to 20,000 train
    images -> extract patches (patch, stride) -> contrast-normalise ->
    subsample to zca_fit_patches -> _fit_zca; the candidate pool is a
    ``candidate_pool_size`` draw from that same patch pool, whitened.
    """
    rep = config["sparse"]
    size = config["corpus"]["image_size"]
    patch, stride = int(rep["patch"]), int(rep["stride"])
    rng = np.random.default_rng(int(rep["zca_fit_seed"]))
    sample = corpus["train_images"][
        rng.choice(len(corpus["train_images"]),
                   min(len(corpus["train_images"]), 20_000), replace=False)
    ]
    patches = _extract_patches(sample, patch, stride)
    grid = (size - patch) // stride + 1
    take = min(int(rep["zca_fit_patches"]), len(patches))
    patch_pool = _contrast_normalise(
        patches[rng.choice(len(patches), take, replace=False)],
        float(rep["contrast_epsilon"]),
    )
    mean, whiten = _fit_zca(patch_pool, float(rep["zca_epsilon"]))
    whitener = Whitener(patch, stride, float(rep["contrast_epsilon"]),
                        mean, whiten, grid)
    seed_rng = np.random.default_rng(int(rep["dictionary_seed"]))
    pool_size = int(rep["candidate_pool_size"])
    candidates = ((patch_pool[
        seed_rng.choice(len(patch_pool), pool_size, replace=False)
    ] - mean) @ whiten).astype(np.float32)
    return whitener, candidates


def _random_dictionary(candidates: np.ndarray, pool_size: int, seed: int,
                       atoms: int) -> np.ndarray:
    """M108 arm (a) exact: prefix of the seeded permutation of the pool."""
    order = _random_order(candidates, pool_size, seed)
    return candidates[order[:atoms]]


# --------------------------------------------------------------------------
# learned (VQ) dictionary
# --------------------------------------------------------------------------
def _vq_pool(config: dict[str, Any], corpus: dict[str, np.ndarray],
             whitener: Whitener) -> np.ndarray:
    """A registered whitened-patch pool for the k-means fit.

    Draws ``vq_fit_images`` train images (seed ``vq_fit_seed``), extracts
    patches, contrast-normalises and whitens them (chunked), then subsamples
    to ``vq_pool_size``. The pool lives in the same whitened space as the
    random candidates, so the VQ centroids are directly comparable atoms.
    """
    rep = config["sparse"]
    rng = np.random.default_rng(int(rep["vq_fit_seed"]))
    n_img = min(int(rep["vq_fit_images"]), len(corpus["train_images"]))
    imgs_idx = rng.choice(len(corpus["train_images"]), n_img, replace=False)
    chunks: list[np.ndarray] = []
    mean, whiten = whitener.mean, whitener.whiten
    for start in range(0, n_img, 256):
        imgs = corpus["train_images"][imgs_idx[start:start + 256]]
        pn = _contrast_normalise(
            _extract_patches(imgs, whitener.patch, whitener.stride),
            whitener.contrast_epsilon,
        )
        chunks.append(((pn - mean) @ whiten).astype(np.float32))
    pool = np.concatenate(chunks, axis=0)
    cap = min(int(rep["vq_pool_size"]), len(pool))
    sel = rng.choice(len(pool), cap, replace=False)
    return np.ascontiguousarray(pool[sel])


def _fit_vq(pool: np.ndarray, atoms: int, iters: int, batch: int, seed: int,
            device: torch.device) -> tuple[np.ndarray, int, int]:
    """Mini-batch k-means on the GPU (Lloyd with a moving-average update).

    Returns (centroids float32 CPU, fit_macs, dead_units_reinitialised).
    ``fit_macs`` is the assignment term (iters * batch * atoms * dim) and is a
    disclosed one-time training cost (analogous to the dense pretraining
    disclosure), never part of per-image inference MACs. Dead centroids (never
    assigned across all iterations) are re-initialised from the pool so the
    dictionary always has ``atoms`` distinct atoms.
    """
    rng = np.random.default_rng(seed)
    dim = pool.shape[1]
    pool_t = torch.from_numpy(np.ascontiguousarray(pool)).to(torch.float32).to(device)
    init = torch.from_numpy(rng.choice(len(pool), atoms, replace=False)).to(device)
    centroids = pool_t[init].clone()
    ones = torch.ones(batch, device=device)
    total_counts = torch.zeros(atoms, device=device)
    for it in range(iters):
        idx = torch.from_numpy(rng.choice(len(pool), batch, replace=False)).to(device)
        block = pool_t[idx]
        distances = torch.cdist(block, centroids)
        assign = distances.argmin(1)
        sums = torch.zeros(atoms, dim, device=device)
        counts = torch.zeros(atoms, device=device)
        sums.index_add_(0, assign, block)
        counts.index_add_(0, assign, ones)
        total_counts += counts
        means = sums / counts.clamp(min=1)[:, None]
        lr = 1.0 / (1.0 + it)
        centroids = (1 - lr) * centroids + lr * means
    dead = int((total_counts == 0).sum().item())
    if dead:
        repl = torch.from_numpy(rng.choice(len(pool), dead, replace=False)).to(device)
        centroids[total_counts == 0] = pool_t[repl]
    fit_macs = int(iters) * batch * atoms * dim
    return centroids.detach().cpu().numpy().astype(np.float32), fit_macs, dead


# --------------------------------------------------------------------------
# encode / arms
# --------------------------------------------------------------------------
def _encode_topk_block_device(images: np.ndarray, table: torch.Tensor,
                              whitener: Whitener, pool_grid: int, k: int,
                              device: torch.device) -> np.ndarray:
    """Triangle encode keeping only the top-k nearest-atom activations/patch.

    Identical to ``_encode_block_device`` except the activation is zeroed
    outside each patch's top-``k`` atoms before pooling. The cdist is computed
    over ALL atoms (so per-image encode MACs are unchanged and disclosed);
    only the ridge head's width drops to ``k * pool_grid**2``.
    """
    white = torch.from_numpy(
        np.ascontiguousarray(whitener(images))
    ).to(torch.float32).to(table.device)
    with torch.no_grad():
        distances = torch.cdist(white, table)              # (B, P, A)
        mean_d = distances.mean(dim=-1, keepdim=True)
        act = (mean_d - distances).clamp(min=0.0)          # (B, P, A)
        vals, idx = torch.topk(act, k, dim=-1)             # (B, P, k)
        sparse_act = torch.zeros_like(act)
        sparse_act.scatter_(-1, idx, vals)
        pooled = _pool(sparse_act, len(images), whitener.grid, pool_grid)
    return pooled.to(torch.float32).cpu().numpy()


def _arm(corpus: dict[str, np.ndarray], dictionary: np.ndarray,
         whitener: Whitener, pool_grid: int, classes: int,
         device: torch.device, topk: int | None = None,
         batch: int = 4096) -> dict[str, Any]:
    """One sparse generalist arm (M108's ridge protocol, all train rows).

    Fits the closed-form ridge head on ALL train rows (M108's convention; no
    SGD, no validation carve). Returns accuracy/per-domain at penalty 1.0 and
    the M108 MAC ledger (whitening + encoding + head). With ``topk`` set, the
    head width is ``topk * pool_grid**2`` and the encode uses the top-k
    variant (cdist cost unchanged, disclosed).
    """
    atoms = len(dictionary)
    # The top-k code is zero-padded to the FULL atom dimension (pooled over
    # all atoms, zeros outside each patch's top-k), so the ridge width is
    # atoms * pool_grid**2 whether or not ``topk`` is set. Arm (c) therefore
    # costs the same as arm (b): it asks whether top-k sparsity helps accuracy
    # at matched cost. A compact sparse code + sparse ridge (the genuine
    # head-cost cut) is deferred to a follow-up.
    width = pool_grid * pool_grid * atoms
    accumulator = RidgeAccumulator(width, classes)
    table = torch.from_numpy(
        np.ascontiguousarray(dictionary)
    ).to(torch.float32).to(device)

    def _step(total: int) -> int:
        step = min(batch, _chunk_rows(atoms, whitener.grid, total))
        return max(1, step // 2) if topk is not None else step

    fit_rows = np.arange(len(corpus["train_labels"]))
    step = _step(len(fit_rows))
    for start in range(0, len(fit_rows), step):
        take = fit_rows[start:start + step]
        if topk is not None:
            block = _encode_topk_block_device(corpus["train_images"][take],
                                              table, whitener, pool_grid,
                                              topk, device)
        else:
            block = _encode_block_device(corpus["train_images"][take],
                                         table, whitener, pool_grid)
        accumulator.add(block, corpus["train_labels"][take])

    test_order = np.arange(len(corpus["test_labels"]))
    step_t = _step(len(test_order))

    def _test_blocks() -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for start in range(0, len(test_order), step_t):
            take = test_order[start:start + step_t]
            if topk is not None:
                block = _encode_topk_block_device(corpus["test_images"][take],
                                                  table, whitener, pool_grid,
                                                  topk, device)
            else:
                block = _encode_block_device(corpus["test_images"][take],
                                             table, whitener, pool_grid)
            yield block, corpus["test_labels"][take], corpus["test_domains"][take]

    result = _solve_and_score(accumulator, [1.0], _test_blocks())
    result["atoms"] = int(atoms)
    result["topk"] = None if topk is None else int(topk)
    result["width"] = int(width)
    macs = _inference_macs(atoms, whitener.grid, PATCH_DIM, pool_grid, classes)
    result["macs"] = macs
    per_correct = result["per_domain_correct"]["1.0"]
    per_rows = result["per_domain_rows"]["1.0"]
    result["per_domain_accuracy"] = [
        per_correct[d] / per_rows[d] for d in range(6)
    ]
    return result


# --------------------------------------------------------------------------
# sealed references (quoted from evidence, never hard-coded)
# --------------------------------------------------------------------------
def _m108_random_3072_reference() -> float:
    evidence = json.loads(M108_EVIDENCE.read_text(encoding="utf-8"))
    return float(evidence["arms"]["a_random_3072"]["accuracy_by_penalty"]["1.0"])


def _m107_dense_curve() -> list[list[float]]:
    evidence = json.loads(M107_EVIDENCE.read_text(encoding="utf-8"))
    curve = []
    for name, payload in evidence["arms"].items():
        if name.startswith("s_"):
            continue
        acc = payload.get("accuracy_by_penalty", {}).get("1.0")
        macs = payload.get("macs")
        if isinstance(macs, dict):
            macs = macs.get("total")
        if acc is None or macs is None:
            continue
        curve.append([float(macs), float(acc)])
    return curve


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def run_m113(config_path: Path, output_dir: Path,
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
    topk = int(rep["topk"])
    pool_grid = int(rep["pool_grid"])
    smoke_skip = bool(config.get("_smoke_skip_gates", False))
    gates: dict[str, Any] = {}

    print("building global whitener + random candidate pool (M108 exact)",
          flush=True)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)

    # ---- arm (a): random --------------------------------------------------
    print("Arm (a): random dictionary", flush=True)
    a_dict = _random_dictionary(candidates, len(candidates),
                                int(rep["dictionary_seed"]), atoms)
    arm_a = _arm(corpus, a_dict, whitener, pool_grid, classes, device)
    a_acc = arm_a["accuracy_by_penalty"]["1.0"]
    print(f"  random-{atoms}: {a_acc:.4f}", flush=True)

    if not smoke_skip:
        m108_ref = _m108_random_3072_reference()
        delta = a_acc - m108_ref
        if abs(delta) > T1_TOLERANCE:
            print(f"  t1 reproduction FAILED: measured {a_acc:.4f} vs M108 "
                  f"{m108_ref:.4f} (delta {delta:+.5f} > {T1_TOLERANCE})",
                  flush=True)
            write_canonical_json(output_dir / "evidence.json", {
                "milestone": "M113", "admissible_as_evidence": False,
                "void": True, "void_reason": "arm (a) t1 reproduction failed",
                "arm_a_measured": a_acc, "m108_reference": m108_ref,
                "t1_delta": delta, "parity_guard": parity,
            })
            return {"admissible_as_evidence": False, "void": True}
        gates["t1_delta"] = delta
        print(f"  t1 reproduction delta {delta:+.5f} (<= {T1_TOLERANCE})",
              flush=True)

    # ---- arm (b): learned (VQ) --------------------------------------------
    print("building VQ pool + fitting k-means dictionary", flush=True)
    pool = _vq_pool(config, corpus, whitener)
    learned, vq_fit_macs, dead = _fit_vq(pool, atoms, int(rep["vq_iters"]),
                                         int(rep["vq_batch"]),
                                         int(rep["vq_fit_seed"]), device)
    print(f"  VQ fit: {len(pool)} pool, {dead} dead units reinit", flush=True)
    arm_b = _arm(corpus, learned, whitener, pool_grid, classes, device)
    b_acc = arm_b["accuracy_by_penalty"]["1.0"]
    print(f"  learned-{atoms}: {b_acc:.4f}", flush=True)
    torch.cuda.empty_cache()

    # ---- arm (c): top-k sparse codes on the learned dictionary ------------
    print(f"Arm (c): top-{topk} sparse codes on learned dictionary", flush=True)
    arm_c = _arm(corpus, learned, whitener, pool_grid, classes, device,
                 topk=topk)
    c_acc = arm_c["accuracy_by_penalty"]["1.0"]
    print(f"  learned-topk{topk}: {c_acc:.4f}", flush=True)

    # ---- kill switches -----------------------------------------------------
    dense_curve = _m107_dense_curve()
    b_macs = arm_b["macs"]["total"]
    best_dense = _best_dense_at_or_below(dense_curve, b_macs)
    ks1 = {
        "registered": "learned (b) beats random (a) by >= +0.01 at matched "
                      "atoms and MACs -> learned basis lifts the ceiling",
        "random_acc": a_acc, "learned_acc": b_acc,
        "margin": KS1_MARGIN,
        "fired": (b_acc - a_acc) < KS1_MARGIN,
        "delta": b_acc - a_acc,
        "consequence": "if fired, the random basis is not the binding "
                       "constraint; the learned-dictionary thesis fails at "
                       "this budget",
    }
    ks2 = {
        "registered": "learned (b) at its MACs beats the best M107 dense "
                      "point at-or-below that cost by >= +0.01",
        "learned_acc": b_acc, "learned_macs": b_macs,
        "best_dense_at_or_below": best_dense,
        "margin": KS2_MARGIN,
        "fired": best_dense is None or (b_acc - best_dense) < KS2_MARGIN,
        "delta": None if best_dense is None else (b_acc - best_dense),
        "note": "accuracy-at-cost, not efficiency: (b) pays ~18% more MACs "
                "than dense r42; the efficiency regimes are per-domain (A5 "
                "KS2) and the deferred approximate-search top-k",
    }
    gates["kill_switch_1_learned_lifts_ceiling"] = ks1
    gates["kill_switch_2_vs_dense_at_or_below"] = ks2
    gates["_smoke_skip"] = smoke_skip

    evidence = {
        "milestone": "M113",
        "admissible_as_evidence": not inadmissible,
        "registered_in": config.get("registered_in"),
        "question": ("does a fitted (k-means/VQ) dictionary lift the frozen "
                     "sparse ceiling above the random dictionary at matched "
                     "atoms and matched encode MACs?"),
        "config_file": Path(config_path).name,
        "config": config,
        "device": device_report,
        "parity_guard": parity,
        "vq_fit": {
            "pool_size": int(len(pool)),
            "fit_macs": int(vq_fit_macs),
            "dead_units_reinitialised": int(dead),
            "note": "one-time training cost, disclosed like dense "
                    "pretraining; not part of per-image inference MACs",
        },
        "arms": {
            "a_random": arm_a,
            "b_learned": arm_b,
            "c_learned_topk": arm_c,
        },
        "gates": gates,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    evidence["payload_sha256"] = payload_hash(evidence)
    print(f"\nM113 complete -> {output_dir / 'evidence.json'}", flush=True)
    print(f"  random-{atoms}:    {a_acc:.4f}", flush=True)
    print(f"  learned-{atoms}:   {b_acc:.4f}", flush=True)
    print(f"  learned-topk{topk}: {c_acc:.4f}", flush=True)
    print(f"  KS1 fired: {ks1['fired']}  KS2 fired: {ks2['fired']}",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m113(args.config, args.output)


if __name__ == "__main__":
    main()
