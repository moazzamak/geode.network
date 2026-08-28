"""M114 — Binary soft-code encode: can Hamming (XOR/popcount) preserve the
frozen sparse accuracy at a fraction of the encode cost?

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v17.md`` (M114, 6 August
2026) and ``experiments/configs/v16/m114_binary.json``.

Question. The sparse family has never won the MAC axis: the float triangle
encode is O(P*108*A) MACs per image (241.9M at 3,072 atoms). M113 arm (c)
showed that hard top-k truncation destroys the soft triangle code (-0.031) —
the signal lives in the SOFT assignment over ALL atoms. M114 quantises the
DISTANCES instead: a learned binary hash (ITQ, Gong et al. CVPR 2011) maps
whitened patches and atoms to B-bit sign codes; the triangle kernel runs on
Hamming distances via XOR + POPCNT, each a single CPU instruction, needing no
GPU cdist GEMM. The encode cost drops ~5-8x (registered op accounting). Does
the Hamming triangle preserve the frozen random-3072 accuracy (0.2153)?

Arms (shared subsample, M108-exact whitener, the SAME random-3072 dictionary
as M113, closed-form ridge head penalty 1.0, per-domain eval):
- (a) float cdist: QUOTED from M113 sealed random-3072 = 0.2153 @ 254.6M MACs
  (M113 re-measured M108's with delta +0.00000 on 6 Aug; the dictionary is
  rebuilt by the identical deterministic construction). Not re-run.
- (b) binary RANDOM-256: seeded Gaussian projection -> 256 bits, Hamming
  triangle, ridge. Control: does hash LEARNING matter?
- (c) binary ITQ-256: learned hash (100,000-whitened-patch pool, seed 33,
  50 iters) -> 256 bits, Hamming triangle, ridge. PRIMARY.
- (d) binary ITQ-128: bit-width sensitivity (reported, not a verdict).

Kill switches (registered before measurement):
- KS1 (hash learning matters): (c) - (b) >= +0.01.
- KS2 (the MAC-axis breakthrough): (c) >= float(0.2153) - 0.01 AND binary
  total ops <= float total MACs / 3.

Cost (registered). Float = whitening (P*108^2) + encode (P*108*A) + head
(A*4*classes). Binary = whitening + projection MACs (P*108*B) + Hamming ops
(P*A*B/64 XOR + P*A*B/64 popcount, 1 op each; a 64-bit XOR and a 64-bit POPCNT
are each single CPU instructions) + head. Binary encode runs on CPU (no GPU
cdist); disclosed as part of the hardware story.

Reproduce with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m114_binary
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
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v15_m107_dense import (
    _solve_and_score,
    _verify_pixel_identity,
)
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
from experiments.tier4.eval_v16_m109_trunk import (
    _load_corpus,
    _parity_guard,
)
from experiments.tier4.eval_v16_m113_learned import (
    _build_whitener_and_candidates,
    _random_dictionary,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m114_binary.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m114_binary"
M113_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m113_learned" / "evidence.json"

PATCH_DIM = 108          # 6 * 6 * 3
KS1_MARGIN = 0.01
KS2_ACC_MARGIN = 0.01
KS2_COST_FACTOR = 3


# --------------------------------------------------------------------------
# binary hashing
# --------------------------------------------------------------------------
def _fit_itq(pool: np.ndarray, bits: int, iters: int, seed: int
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ITQ (Gong et al., CVPR 2011) on a whitened-patch pool (numpy, CPU).

    Returns (P: (108, bits) projection, bias: (bits,), center: (108,)) such
    that code(x) = sign((x - center) @ P - bias) in {+1,-1}. The pool is
    already ZCA-whitened; we still centre it for the PCA.
    """
    center = pool.mean(axis=0)
    X = pool - center
    _, _, vh = np.linalg.svd(X, full_matrices=False)   # vh: (d, d) desc
    W = vh[:bits].T                                    # (108, bits)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((bits, bits)))
    for _ in range(iters):
        b = np.sign(X @ W @ q)
        b[b == 0.0] = 1.0
        u, _, vt = np.linalg.svd(W.T @ (X.T @ b), full_matrices=False)
        q = u @ vt
    P = W @ q                                          # (108, bits)
    bias = (X @ P).mean(axis=0)
    return P, bias, center


def _random_projection(pool: np.ndarray, bits: int, seed: int
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unlearned control: seeded Gaussian projection with a centring bias."""
    center = pool.mean(axis=0)
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((pool.shape[1], bits)).astype(np.float64)
    P /= np.linalg.norm(P, axis=0, keepdims=True)
    bias = ((pool - center) @ P).mean(axis=0)
    return P, bias, center


def _hash_pool(config: dict[str, Any], corpus: dict[str, np.ndarray],
               whitener: Whitener) -> np.ndarray:
    """A registered whitened-patch pool for fitting the hash (seed 33)."""
    rep = config["sparse"]
    rng = np.random.default_rng(int(rep["hash_fit_seed"]))
    n_img = min(int(rep["hash_fit_images"]), len(corpus["train_images"]))
    idx = rng.choice(len(corpus["train_images"]), n_img, replace=False)
    chunks: list[np.ndarray] = []
    for start in range(0, n_img, 256):
        imgs = corpus["train_images"][idx[start:start + 256]]
        pn = _contrast_normalise(
            _extract_patches(imgs, whitener.patch, whitener.stride),
            whitener.contrast_epsilon,
        )
        chunks.append(((pn - whitener.mean) @ whitener.whiten).astype(np.float32))
    pool = np.concatenate(chunks, axis=0)
    cap = min(int(rep["hash_pool_size"]), len(pool))
    sel = rng.choice(len(pool), cap, replace=False)
    return np.ascontiguousarray(pool[sel])


def _project(white: np.ndarray, P: np.ndarray, bias: np.ndarray,
             center: np.ndarray) -> np.ndarray:
    """Whitened patches (n, 108) -> 0/1 bits (n, bits)."""
    vals = (white - center) @ P - bias
    return (vals >= 0).astype(np.uint8)


def _pack64(bits01: np.ndarray, words: int) -> np.ndarray:
    """0/1 bits (n, bits) -> packed uint64 (n, words), consistent bit order.

    ``bits01`` may use fewer than ``words*64`` bits; the high bits are padded
    with zeros on BOTH patches and atoms, so XOR/popcount is unaffected.
    """
    n = bits01.shape[0]
    padded = np.zeros((n, words * 64), dtype=np.uint8)
    padded[:, :bits01.shape[1]] = bits01
    b = padded.reshape(n, words, 64).astype(np.uint64)
    shifts = np.uint64(1) << np.arange(64, dtype=np.uint64)
    return (b * shifts[None, None, :]).sum(axis=2)


def _hamming(patch_words: np.ndarray, atom_words: np.ndarray,
             chunk: int = 512) -> np.ndarray:
    """Hamming (n, A) uint16 via XOR + POPCNT over uint64 words, atom-chunked."""
    out = np.empty((patch_words.shape[0], atom_words.shape[0]), dtype=np.uint16)
    for start in range(0, atom_words.shape[0], chunk):
        xor = patch_words[:, None, :] ^ atom_words[None, start:start + chunk, :]
        out[:, start:start + chunk] = np.bitwise_count(xor).sum(axis=-1)
    return out


def _pool_np(act: np.ndarray, count: int, grid: int, pool_grid: int
             ) -> np.ndarray:
    """2x2 (pool_grid x pool_grid) sum pooling, replicating m103._pool's
    uneven edges exactly (edges = [round(grid*i/pool_grid)])."""
    atoms = act.shape[-1]
    act = act.reshape(count, grid, grid, atoms)
    edges = [round(grid * i / pool_grid) for i in range(pool_grid + 1)]
    blocks = []
    for iy in range(pool_grid):
        for ix in range(pool_grid):
            blocks.append(
                act[:, edges[iy]:edges[iy + 1], edges[ix]:edges[ix + 1]]
                .sum(axis=(1, 2))
            )
    return np.concatenate(blocks, axis=1)


# --------------------------------------------------------------------------
# binary arm
# --------------------------------------------------------------------------
def _binary_arm(corpus: dict[str, np.ndarray], dictionary: np.ndarray,
                whitener: Whitener, P: np.ndarray, bias: np.ndarray,
                center: np.ndarray, bits: int, pool_grid: int, classes: int,
                batch: int = 48, backend: str = "cpu",
                device: torch.device | None = None) -> dict[str, Any]:
    """One binary-Hamming generalist arm on the same random dictionary.

    Whitening, projection and the sign code are computed IDENTICALLY on the
    CPU (numpy). The Hamming distance is then realized two ways:
    - backend="cpu": packed uint64 XOR + hardware POPCNT (np.bitwise_count)
      -- the registered algorithmic cost (2*P*A*words ops).
    - backend="gpu": the sign-GEMM identity Hamming = (B - <p,a>)/2 on the
      GPU (torch matmul). B <= 108 and every inner product is an exact
      integer in fp32, so the distances are IDENTICAL to the CPU path; the
      GPU realizes the same Hamming at n*B*A MACs (disclosed: no GPU popcount
      exists in this ROCm torch build). The accuracy verdict is identical.
    """
    atoms = len(dictionary)
    words = (bits + 63) // 64
    width = pool_grid * pool_grid * atoms
    accumulator = RidgeAccumulator(width, classes)
    grid = whitener.grid
    A = atoms
    P_ = grid * grid

    if backend == "gpu":
        # _project returns uint8 0/1 bits; convert to {+1,-1} WITHOUT a
        # redundant ">= 0" (that would be always-True on unsigned uint8 and
        # collapse every code to all-ones). The CPU path uses the uint8 bits
        # directly via _pack64, which is why it was unaffected.
        atom_sign_t = torch.from_numpy(np.ascontiguousarray(
            (_project(dictionary, P, bias, center).astype(np.float32)
             * 2.0 - 1.0).T)).to(torch.float32).to(device)   # (B, A)
        b_float = float(bits)

        def _encode_block(imgs: np.ndarray) -> np.ndarray:
            white = whitener(imgs).reshape(-1, PATCH_DIM)   # (n*P, 108)
            n = len(imgs)
            patch_sign = (_project(white, P, bias, center).astype(
                np.float32) * 2.0 - 1.0)                    # (n*P, B) +-1
            ps = torch.from_numpy(
                np.ascontiguousarray(patch_sign)).to(torch.float32).to(device)
            with torch.no_grad():
                inner = ps @ atom_sign_t              # (n*P, A) exact ints
                h = (b_float - inner) * 0.5
                act = torch.clamp(h.mean(dim=-1, keepdim=True) - h, min=0.0)
                pooled = _pool(act, n, grid, pool_grid)
            return pooled.to(torch.float32).cpu().numpy()
    else:
        atom_words = _pack64(_project(dictionary, P, bias, center), words)

        def _encode_block(imgs: np.ndarray) -> np.ndarray:
            white = whitener(imgs).reshape(-1, PATCH_DIM)   # (n*P, 108)
            n = len(imgs)
            pw = _pack64(_project(white, P, bias, center), words)
            h = _hamming(pw, atom_words)              # (n*P, A)
            act = np.maximum(
                h.mean(axis=-1, keepdims=True) - h, 0
            ).astype(np.float32)
            pooled = _pool_np(act.reshape(n, P_, -1), n, grid, pool_grid)
            return pooled

    fit_rows = np.arange(len(corpus["train_labels"]))
    for start in range(0, len(fit_rows), batch):
        take = fit_rows[start:start + batch]
        accumulator.add(_encode_block(corpus["train_images"][take]),
                        corpus["train_labels"][take])

    test_order = np.arange(len(corpus["test_labels"]))

    def _test_blocks() -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for start in range(0, len(test_order), batch):
            take = test_order[start:start + batch]
            yield (_encode_block(corpus["test_images"][take]),
                   corpus["test_labels"][take],
                   corpus["test_domains"][take])

    result = _solve_and_score(accumulator, [1.0], _test_blocks())
    result["atoms"] = int(atoms)
    result["bits"] = int(bits)
    result["width"] = int(width)
    result["backend"] = backend

    projection_macs = P_ * PATCH_DIM * bits
    hamming_ops = 2 * P_ * atoms * words   # XOR + popcount per uint64 word
    whitening_macs = P_ * PATCH_DIM * PATCH_DIM
    head_macs = pool_grid * pool_grid * atoms * classes
    result["macs"] = {
        "whitening": int(whitening_macs),
        "projection_macs": int(projection_macs),
        "hamming_ops": int(hamming_ops),
        "head": int(head_macs),
        "total_ops": int(whitening_macs + projection_macs + hamming_ops + head_macs),
        "backend": backend,
        "_note": "a 64-bit XOR and a 64-bit POPCNT are each single CPU "
                 "instructions; counted at 1 op each (conservative vs a "
                 "108-dim MAC dot). Whitening/head identical to the float arm. "
                 "backend='gpu' realizes the SAME Hamming distances via the "
                 "sign-GEMM identity at n*B*A MACs (no GPU popcount in this "
                 "ROCm torch build); the registered op accounting above is the "
                 "algorithmic cost realized by the CPU hardware-POPCNT path.",
    }
    per_correct = result["per_domain_correct"]["1.0"]
    per_rows = result["per_domain_rows"]["1.0"]
    result["per_domain_accuracy"] = [
        per_correct[d] / per_rows[d] for d in range(6)
    ]
    return result


# --------------------------------------------------------------------------
# sealed reference (quoted from M113)
# --------------------------------------------------------------------------
def _m113_float_reference() -> dict[str, Any]:
    evidence = json.loads(M113_EVIDENCE.read_text(encoding="utf-8"))
    arm = evidence["arms"]["a_random"]
    return {
        "accuracy": float(arm["accuracy_by_penalty"]["1.0"]),
        "macs_total": int(arm["macs"]["total"]),
        "admissible": bool(evidence["admissible_as_evidence"]),
    }


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def run_m114(config_path: Path, output_dir: Path,
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
    pool_grid = int(rep["pool_grid"])
    bits = [int(b) for b in rep["bits"]]
    smoke_skip = bool(config.get("_smoke_skip_gates", False))
    gates: dict[str, Any] = {}

    print("building global whitener + random candidate pool (M108 exact)",
          flush=True)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)
    dictionary = _random_dictionary(candidates, len(candidates),
                                    int(rep["dictionary_seed"]), atoms)

    ref = _m113_float_reference()
    if not ref["admissible"]:
        raise RuntimeError("M114 instrument failure: M113 evidence is not "
                           "admissible; cannot quote the float reference.")
    float_acc = ref["accuracy"]
    float_macs = ref["macs_total"]
    print(f"float reference (M113 sealed random-{atoms}): {float_acc:.4f} "
          f"@ {float_macs} MACs (quoted; M113 re-measured at +0.00000)",
          flush=True)

    print("building hash-fit pool", flush=True)
    hash_pool = _hash_pool(config, corpus, whitener)
    print(f"  hash pool: {len(hash_pool)} whitened patches", flush=True)

    print("fitting ITQ hash (learned)", flush=True)
    itq_256, itq_bias, itq_center = _fit_itq(
        hash_pool, bits[-1], int(rep["hash_iters"]), int(rep["hash_fit_seed"]))
    print("fitting random projection (control)", flush=True)
    rnd_256, rnd_bias, rnd_center = _random_projection(
        hash_pool, bits[-1], int(rep["hash_fit_seed"]))

    encode_backend = config["device"].get("encode", "cpu")
    if encode_backend not in ("cpu", "gpu"):
        raise RuntimeError(
            f"M114 instrument failure: device.encode={encode_backend!r} "
            "must be 'cpu' or 'gpu'.")
    print(f"binary encode backend: {encode_backend}", flush=True)

    arms: dict[str, Any] = {}
    arm_b = _binary_arm(corpus, dictionary, whitener, rnd_256, rnd_bias,
                        rnd_center, bits[-1], pool_grid, classes,
                        backend=encode_backend, device=device)
    arms["b_random"] = arm_b
    print(f"  binary RANDOM-{bits[-1]}: "
          f"{arm_b['accuracy_by_penalty']['1.0']:.4f}", flush=True)

    arm_c = _binary_arm(corpus, dictionary, whitener, itq_256, itq_bias,
                        itq_center, bits[-1], pool_grid, classes,
                        backend=encode_backend, device=device)
    arms["c_itq"] = arm_c
    c_acc = arm_c["accuracy_by_penalty"]["1.0"]
    print(f"  binary ITQ-{bits[-1]}: {c_acc:.4f}", flush=True)

    if len(bits) > 1:
        itq_128, itq_bias128, itq_center128 = _fit_itq(
            hash_pool, bits[0], int(rep["hash_iters"]),
            int(rep["hash_fit_seed"]) + 1)
        arm_d = _binary_arm(corpus, dictionary, whitener, itq_128, itq_bias128,
                            itq_center128, bits[0], pool_grid, classes,
                            backend=encode_backend, device=device)
        arms["d_itq_128"] = arm_d
        print(f"  binary ITQ-{bits[0]}: "
              f"{arm_d['accuracy_by_penalty']['1.0']:.4f}", flush=True)

    # ---- kill switches -----------------------------------------------------
    b_acc = arm_b["accuracy_by_penalty"]["1.0"]
    c_macs = arm_c["macs"]["total_ops"]
    ks1 = {
        "registered": "learned hash (c) beats random projection (b) by "
                      ">= +0.01",
        "random_bits_acc": b_acc, "learned_bits_acc": c_acc,
        "margin": KS1_MARGIN,
        "fired": (c_acc - b_acc) < KS1_MARGIN,
        "delta": c_acc - b_acc,
        "consequence": "if fired, the learned hash adds nothing over random "
                       "bits; the binary-axis result is about bit "
                       "quantisation, not hash learning",
    }
    ks2 = {
        "registered": "learned-bits (c) >= float - 0.01 AND binary total ops "
                      "<= float total MACs / 3 -> MAC-axis win at preserved "
                      "accuracy",
        "float_acc": float_acc, "binary_acc": c_acc,
        "acc_margin": KS2_ACC_MARGIN,
        "acc_ok": c_acc >= float_acc - KS2_ACC_MARGIN,
        "float_macs": float_macs, "binary_ops": c_macs,
        "cost_factor": KS2_COST_FACTOR,
        "cost_ok": c_macs <= float_macs / KS2_COST_FACTOR,
        "fired": not (c_acc >= float_acc - KS2_ACC_MARGIN
                      and c_macs <= float_macs / KS2_COST_FACTOR),
        "consequence": "if NOT fired, the sparse family wins the MAC axis at "
                       "preserved accuracy - the first measured time",
    }
    gates["kill_switch_1_hash_learning"] = ks1
    gates["kill_switch_2_mac_axis_breakthrough"] = ks2
    gates["_smoke_skip"] = smoke_skip

    evidence = {
        "milestone": "M114",
        "admissible_as_evidence": not inadmissible,
        "registered_in": config.get("registered_in"),
        "question": ("does a learned binary (Hamming) soft-code encode "
                     "preserve the frozen random-3072 accuracy at a fraction "
                     "of the float cdist encode cost?"),
        "config_file": Path(config_path).name,
        "config": config,
        "device": device_report,
        "parity_guard": parity,
        "float_reference": {
            "source": "M113 sealed (6 Aug, re-measured at delta +0.00000)",
            "accuracy": float_acc, "macs_total": float_macs,
            "dictionary": "identical deterministic random-3072 construction",
        },
        "hash_fit": {
            "pool_size": int(len(hash_pool)),
            "iters": int(rep["hash_iters"]),
            "seed": int(rep["hash_fit_seed"]),
            "note": "one-time CPU fit, disclosed like dense pretraining; not "
                    "part of per-image encode cost",
        },
        "arms": arms,
        "gates": gates,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    evidence["payload_sha256"] = payload_hash(evidence)
    print(f"\nM114 complete -> {output_dir / 'evidence.json'}", flush=True)
    print(f"  float random-{atoms}:        {float_acc:.4f} @ {float_macs} MACs",
          flush=True)
    print(f"  binary RANDOM-{bits[-1]}:    {b_acc:.4f} @ "
          f"{arm_b['macs']['total_ops']} ops", flush=True)
    print(f"  binary ITQ-{bits[-1]}:       {c_acc:.4f} @ {c_macs} ops",
          flush=True)
    print(f"  KS1 fired: {ks1['fired']}  KS2 fired: {ks2['fired']}",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m114(args.config, args.output)


if __name__ == "__main__":
    main()
