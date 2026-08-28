"""M126 — Push the joint frontier: atoms past the 8192-pool cap at full data.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v19.md`` section 5.5 and
``experiments/configs/v16/m126_atoms_extension.json``.

Question. M117's atoms axis was capped by the 8192-patch candidate pool. Does
the surface keep rising past 6144 atoms at full data? M126 measures Q at
atoms {6144, 8192, 12288, 16384} x n = 138000.

Construction (disclosed, registered in the config): 6144 and 8192 are prefixes
of the SEALED 8192-pool in M117's seeded-permutation order (M117 exact; the
6144 cell reuses M117's sealed code memmaps and must reproduce 0.2249). The
12288/16384-atom dictionaries APPEND 4096/8192 new atoms (drawn with
ext_pool_seed 42 from a re-derived M108 whitened-patch stream, verified
against the whitener's mean/whiten before use, ordered by a fresh seeded
permutation) after the 8192-atom prefix.

Head fit: the 12288/16384 closed-form ridges use a memmap-backed Gram (same
arithmetic, gram on D:) because the in-RAM accumulator OOMs at >= 32768
features on this 63 GB machine (measured; registered in the config).

Gate: KS fired if Q(16384, 138000) - Q(6144, 138000) < +0.005 (the atoms axis
must still pay at full data past the old pool cap). Frontier MACs reported
vs the dense curve (r70 0.3118 @ 564.2M; r98 0.4476 @ 1096M). Honest note:
dense r224 (0.54) remains far ahead in absolute accuracy; the frontier claim
is cost-matched non-domination, never dominance.

Reproduce with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m126_atoms_extension
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
from experiments.tier4.eval_v15_m103_atoms import (
    _contrast_normalise,
    _extract_patches,
    _fit_zca,
)
from experiments.tier4.eval_v15_m104_experts import _training_macs
from experiments.tier4.eval_v15_m107_dense import (
    _score,
    _solve_and_score,
    _verify_pixel_identity,
)
from experiments.tier4.eval_v16_m108_dictionary import (
    _random_order,
    _verify_device,
)
from experiments.tier4.eval_v16_m109_trunk import (
    _load_corpus,
    _parity_guard,
)
from experiments.tier4.eval_v16_m113_learned import (
    _build_whitener_and_candidates,
    _random_dictionary,
)
from experiments.tier4.eval_v16_m115_lofi import _write_frozen_codes
from experiments.tier4.eval_v16_m117_scale import _fit_and_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m126_atoms_extension.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m126_atoms_extension"
M117_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m117_scale" / "evidence.json"

T1_TOLERANCE = 0.002
KS_MARGIN = 0.005
PATCH_DIM = 108
BLOCK = 4096


def _rebuild_patch_stream(config: dict[str, Any],
                          corpus: dict[str, np.ndarray]) -> np.ndarray:
    """Re-derive the M108 contrast-normalised patch stream (same RNG sequence)."""
    rep = config["sparse"]
    size = config["corpus"]["image_size"]
    patch, stride = int(rep["patch"]), int(rep["stride"])
    rng = np.random.default_rng(int(rep["zca_fit_seed"]))
    sample = corpus["train_images"][
        rng.choice(len(corpus["train_images"]),
                   min(len(corpus["train_images"]), 20_000), replace=False)
    ]
    patches = _extract_patches(sample, patch, stride)
    take = min(int(rep["zca_fit_patches"]), len(patches))
    return _contrast_normalise(
        patches[rng.choice(len(patches), take, replace=False)],
        float(rep["contrast_epsilon"]),
    )


def _verify_stream(whitener, patch_pool: np.ndarray,
                   rep: dict[str, Any]) -> dict[str, float]:
    """Re-fit zca on the rebuilt stream and compare with the whitener."""
    mean, whiten = _fit_zca(patch_pool, float(rep["zca_epsilon"]))
    d_mean = float(np.abs(mean - whitener.mean).max())
    d_whiten = float(np.abs(whiten - whitener.whiten).max())
    return {"max_abs_mean_diff": d_mean, "max_abs_whiten_diff": d_whiten}


def _extended_dictionary(config: dict[str, Any],
                         corpus: dict[str, np.ndarray], whitener,
                         candidates: np.ndarray, atoms: int
                         ) -> tuple[np.ndarray, dict[str, Any]]:
    """Dictionary at ``atoms``: sealed-pool prefix, or appended extension.

    Returns (dictionary, construction_note). For atoms <= pool_size: M117
    exact prefix. For atoms > pool_size: the pool_size-prefix (M117 exact)
    APPENDED by (atoms - pool_size) new draws (ext_pool_seed) from the rebuilt
    M108 patch stream, whitened with the same (mean, whiten), ordered by a
    fresh seeded permutation (ext_pool_seed, 100).
    """
    rep = config["sparse"]
    pool_size = int(rep["candidate_pool_size"])
    dict_seed = int(rep["dictionary_seed"])
    if atoms <= pool_size:
        return (_random_dictionary(candidates, len(candidates), dict_seed,
                                   atoms),
                {"construction": "sealed-pool prefix (M117 exact)",
                 "atoms": atoms, "pool_size": pool_size})

    ext = rep["extension"]
    n_new = atoms - pool_size
    patch_pool = _rebuild_patch_stream(config, corpus)
    stream_check = _verify_stream(whitener, patch_pool, rep)
    if stream_check["max_abs_whiten_diff"] > 1e-6:
        raise RuntimeError(
            f"rebuilt patch stream does not match the whitener "
            f"({stream_check}); aborting the extension")
    rng = np.random.default_rng(int(ext["ext_pool_seed"]))
    sel = rng.choice(len(patch_pool), n_new, replace=False)
    new_candidates = ((patch_pool[sel] - whitener.mean) @ whitener.whiten
                      ).astype(np.float32)
    order = np.random.default_rng(
        [int(ext["ext_pool_seed"]), 100]).permutation(n_new)
    base = _random_dictionary(candidates, len(candidates), dict_seed, pool_size)
    note = {
        "construction": ("appended extension: sealed-pool prefix + "
                         f"{n_new} new draws (ext_pool_seed "
                         f"{int(ext['ext_pool_seed'])}) from the verified "
                         "rebuilt M108 patch stream"),
        "atoms": atoms, "pool_size": pool_size, "n_new": n_new,
        "stream_verification": stream_check,
    }
    return np.concatenate([base, new_candidates[order]]), note


def _fit_and_score_memgram(mem_train: np.ndarray, mem_test: np.ndarray,
                           labels: np.ndarray, test_labels: np.ndarray,
                           test_domains: np.ndarray, classes: int, n: int,
                           gram_path: Path, chunk: int = 1024
                           ) -> dict[str, Any]:
    """Closed-form ridge (penalty 1.0) with the Gram backed by a D: memmap.

    Identical arithmetic to RidgeAccumulator + _solve_and_score (the sealed
    head), but the width x width float64 Gram lives in a file-backed memmap so
    the peak committed RAM is one width^2 temp (block.T@block) during
    accumulation and one width^2 array at the solve, not two. This is a
    mechanical memory-management change (like writing the code memmaps to D:),
    NOT a change to the head: same gram values, same standardised system,
    same solve. Used only when the in-RAM accumulator would exceed the
    machine's 63 GB (width >= 32768).
    """
    width = mem_train.shape[1]
    # Pass 1: streaming stats + sum-of-squares (the latter also verifies a
    # cached gram: trace(X^T X) == sum of squared entries over all rows).
    colsum = np.zeros(width, dtype=np.float64)
    cross = np.zeros((width, classes), dtype=np.float64)
    class_count = np.zeros(classes, dtype=np.float64)
    rows = 0
    sum_sq = 0.0
    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        block = np.asarray(mem_train[start:stop], dtype=np.float64)
        targets = np.zeros((len(block), classes), dtype=np.float64)
        targets[np.arange(len(block)), labels[start:stop]] = 1.0
        colsum += block.sum(axis=0)
        cross += block.T @ targets
        class_count += targets.sum(axis=0)
        sum_sq += float((block ** 2).sum())
        rows += len(block)

    # Gram: reuse a cached, trace-verified gram when present (a run killed
    # mid-accumulation leaves a full-size but incomplete file; the trace
    # check rejects it). Otherwise accumulate with chunked-output-row GEMM:
    # scipy-openblas (numpy's bundled BLAS) abort-crashes the process on the
    # single gemm block.T @ block at width >= 49152 (M = width in the output;
    # measured C-level abort). Chunking the OUTPUT ROWS makes each gemm
    # M = 8192, which is unaffected, and produces the bit-identical Gram:
    #   gram[i0:i1] += block[:, i0:i1].T @ block
    # npy memmaps carry a ~128-byte header on top of the raw data, so the
    # size check is >= (the trace check below is the real validity gate).
    reuse = (gram_path.exists()
             and gram_path.stat().st_size >= width * width * 8)
    if reuse:
        gram = np.lib.format.open_memmap(gram_path, mode="r", dtype=np.float64)
        tr = float(np.trace(gram))
        reuse = abs(tr - sum_sq) / max(sum_sq, 1.0) < 1e-9
        if reuse:
            print(f"    memgram: reusing trace-verified cached gram "
                  f"(|trace diff| {abs(tr - sum_sq) / max(sum_sq, 1.0):.1e})",
                  flush=True)
    if not reuse:
        gram = np.lib.format.open_memmap(gram_path, mode="w+",
                                         dtype=np.float64,
                                         shape=(width, width))
        for start in range(0, n, BLOCK):
            stop = min(start + BLOCK, n)
            block = np.asarray(mem_train[start:stop], dtype=np.float64)
            for i0 in range(0, width, 8192):
                i1 = min(i0 + 8192, width)
                gram[i0:i1] += block[:, i0:i1].T @ block

    centre = colsum / rows
    variance = np.diag(gram) / rows - centre ** 2
    scale = np.sqrt(np.maximum(variance, 0.0)) + 1e-8
    inv = 1.0 / scale
    # F-order copy: numpy's linalg.solve makes an internal F-contiguous copy
    # of a C-order input (a second width^2 array, which OOM'd the 49152
    # solve on this 66 GB machine). An F-order input is solved in place.
    centred = np.array(gram, order="F")
    # Trim the working set before the solve: at this point the process holds
    # centred (width^2) plus the file page cache of the code memmap (up to
    # ~2x width^2) and the gram memmap, which together sit at the machine's
    # RAM limit and make numpy's solve allocations fail marginally. Drop the
    # gram memmap (the solve only needs centred), run a GC, and ask Windows
    # to evict file-backed pages.
    del gram
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.windll.psapi.EmptyWorkingSet(
            ctypes.windll.kernel32.GetCurrentProcess())
    except Exception:
        pass
    for i in range(0, width, chunk):  # chunked centring (no width^2 outer temp)
        centred[i:i + chunk] -= np.outer(colsum[i:i + chunk], centre)
    centred *= inv[:, None]
    centred *= inv[None, :]
    cross_s = (cross - np.outer(centre, class_count)) * inv[:, None]
    intercept = class_count / rows
    centred.flat[:: width + 1] += 1.0
    weights = np.linalg.solve(centred, cross_s)
    w = np.vstack([weights, intercept[None, :]])   # RidgeAccumulator convention

    centre32 = centre.astype(np.float32)
    scale32 = scale.astype(np.float32)
    n_test = len(test_labels)
    correct = 0
    bucket = np.zeros((2, 6), dtype=np.int64)
    for start in range(0, n_test, BLOCK):
        stop = min(start + BLOCK, n_test)
        block = np.asarray(mem_test[start:stop], dtype=np.float32)
        standardised = (block - centre32) / scale32
        hits = _score(w, standardised, test_labels[start:stop])
        correct += int(hits.sum())
        np.add.at(bucket[0], test_domains[start:stop], hits.astype(np.int64))
        np.add.at(bucket[1], test_domains[start:stop], 1)
    return {
        "accuracy": correct / n_test,
        "per_domain": [int(bucket[0][d]) / int(bucket[1][d]) for d in range(6)],
        "fit_rows": rows,
        "features": width,
        "test_rows": n_test,
        "head": "closed_form_ridge_1.0 (memmap-backed Gram, same arithmetic)",
    }


def run_m126(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

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
    pool_grid = int(rep["pool_grid"])
    atoms_ladder = [int(a) for a in rep["atoms_ladder"]]
    n_ladder = [int(n) for n in config["scaling"]["n_ladder"]]
    n_max = max(n_ladder)
    pool_size = int(rep["candidate_pool_size"])
    smoke_skip = bool(config.get("_smoke_skip_gates", False))
    gates: dict[str, Any] = {}

    print("building global whitener + candidate pool (M108 exact)", flush=True)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)
    print(f"  sealed pool: {len(candidates)} candidates", flush=True)

    cache = data_cache_root() / "v16" / "m126"
    cache.mkdir(parents=True, exist_ok=True)
    m117_cache = data_cache_root() / "v16" / "m117"
    n_train = len(corpus["train_labels"])
    n_test = len(corpus["test_labels"])
    train_rows = np.arange(n_train)
    test_rows = np.arange(n_test)

    surface: dict[str, Any] = {}
    dict_notes: dict[str, Any] = {}
    for atoms in atoms_ladder:
        # reuse cached sealed codes when present and correctly shaped (codes
        # are image functions; the t1 gate at 6144 validates the construction)
        cached = cache / f"f{atoms}_train.npy"
        if atoms == 6144:
            cached = m117_cache / "f6144_train.npy"
        reuse = (not smoke_skip and cached.exists()
                 and np.lib.format.open_memmap(
                     cached, mode="r", dtype=np.float32).shape
                 == (n_train, pool_grid * pool_grid * atoms))
        if reuse:
            mem_train = np.lib.format.open_memmap(cached, mode="r")
            mem_test = np.lib.format.open_memmap(
                (m117_cache if atoms == 6144 else cache)
                / f"f{atoms}_test.npy", mode="r")
            dict_notes[str(atoms)] = {
                "construction": (f"REUSED sealed code memmaps "
                                 f"({'M117' if atoms == 6144 else 'm126 cache'})")}
            print(f"  atoms={atoms}: REUSING sealed codes", flush=True)
        else:
            dictionary, note = _extended_dictionary(
                config, corpus, whitener, candidates, atoms)
            dict_notes[str(atoms)] = note
            throttle = float(config["numerics"].get(
                "encode_throttle_seconds", 0.0))
            print(f"  atoms={atoms}: encoding train codes "
                  f"({note['construction']})", flush=True)
            mem_train = _write_frozen_codes(corpus, dictionary, whitener,
                                            pool_grid, device, train_rows,
                                            cache / f"f{atoms}_train.npy",
                                            split="train",
                                            throttle_seconds=throttle)
            print(f"  atoms={atoms}: encoding test codes", flush=True)
            mem_test = _write_frozen_codes(corpus, dictionary, whitener,
                                           pool_grid, device, test_rows,
                                           cache / f"f{atoms}_test.npy",
                                           split="test",
                                           throttle_seconds=throttle)
        width = pool_grid * pool_grid * atoms
        cells = {}
        for n in n_ladder:
            if width > 32768:
                # memmap-backed Gram (same arithmetic): the in-RAM accumulator
                # needs two width^2 float64 arrays and OOMs >= 32768 features
                r = _fit_and_score_memgram(
                    mem_train, mem_test, corpus["train_labels"],
                    corpus["test_labels"], corpus["test_domains"], classes, n,
                    cache / f"gram{atoms}.npy")
                acc = r["accuracy"]
                pd = r["per_domain"]
            else:
                r = _fit_and_score(mem_train, mem_test, corpus["train_labels"],
                                   corpus["test_labels"], corpus["test_domains"],
                                   classes, n)
                acc = r["accuracy_by_penalty"]["1.0"]
                pc = r["per_domain_correct"]["1.0"]
                pr = r["per_domain_rows"]["1.0"]
                pd = [pc[d] / pr[d] for d in range(6)]
            cells[str(n)] = {
                "accuracy": acc,
                "per_domain": pd,
                "head": r.get("head", "closed_form_ridge_1.0"),
                "training_ops": int(_training_macs(n, atoms, whitener.grid,
                                                   PATCH_DIM, pool_grid,
                                                   classes)),
            }
            print(f"    Q({atoms}, {n}) = {acc:.4f}", flush=True)
        surface[str(atoms)] = {"width": width, "cells": cells}

    def Q(a: int, n: int) -> float:
        return float(surface[str(a)]["cells"][str(n)]["accuracy"])

    # ---- gates ------------------------------------------------------------
    a6144, a8192, a12288, a16384 = atoms_ladder
    if not smoke_skip:
        m117 = json.loads(M117_EVIDENCE.read_text(encoding="utf-8"))
        ref = float(m117["surface"]["6144"]["cells"]["138000"]["accuracy"])
        delta = Q(a6144, n_max) - ref
        if abs(delta) > T1_TOLERANCE:
            print(f"  t1 FAILED: Q(6144,{n_max}) {Q(a6144, n_max):.4f} vs "
                  f"M117 {ref:.4f} (delta {delta:+.5f})", flush=True)
            write_canonical_json(output_dir / "evidence.json", {
                "milestone": "M126", "admissible_as_evidence": False,
                "void": True, "void_reason": "t1 anchor reproduction failed",
                "measured": Q(a6144, n_max), "reference": ref,
                "t1_delta": delta,
            })
            return {"admissible_as_evidence": False, "void": True}
        gates["t1_delta"] = delta
        print(f"  t1 anchor delta {delta:+.5f} (<= {T1_TOLERANCE})", flush=True)

    gain = Q(a16384, n_max) - Q(a6144, n_max)
    ks = {
        "registered": ("Q(16384,138000) - Q(6144,138000) >= +0.005: the atoms "
                       "axis must still pay at full data past the 8192-pool cap"),
        "q_6144": float(Q(a6144, n_max)),
        "q_8192": float(Q(a8192, n_max)),
        "q_12288": float(Q(a12288, n_max)),
        "q_16384": float(Q(a16384, n_max)),
        "gain_16384_minus_6144": float(gain),
        "margin": KS_MARGIN,
        "fired": gain < KS_MARGIN,
    }
    gates["kill_switch_atoms_past_pool"] = ks
    gates["_smoke_skip"] = smoke_skip

    # frontier report (per-image encode MACs at 16384 atoms)
    grid = whitener.grid
    macs_16384 = int((grid * grid * PATCH_DIM * a16384)
                     + (grid * grid * PATCH_DIM * PATCH_DIM)
                     + (a16384 * pool_grid * pool_grid * classes))
    frontier = {
        "sparse_16384_atoms": {
            "accuracy": float(Q(a16384, n_max)),
            "per_image_encode_macs": macs_16384,
            "head_note": ("16384 head fit via the memmap-backed Gram "
                          "(identical arithmetic; the in-RAM accumulator "
                          "OOMs at 65536 features on this machine)"),
        },
        "dense_reference": {
            "r70": {"accuracy": 0.3118, "macs": 564.2e6},
            "r98": {"accuracy": 0.4476, "macs": 1096.0e6},
            "r224": {"accuracy": 0.5375, "macs": 6124.0e6},
        },
        "honest_note": ("dense r224 (0.54) remains far ahead in absolute "
                        "accuracy; the frontier claim is cost-matched "
                        "non-domination, never dominance"),
    }

    evidence = {
        "milestone": "M126",
        "admissible_as_evidence": not inadmissible,
        "registered_in": config.get("registered_in"),
        "question": ("does the atoms axis keep paying past the 8192-pool cap "
                     "at full data?"),
        "config_file": Path(config_path).name,
        "config": config,
        "device": device_report,
        "parity_guard": parity,
        "atoms_ladder": atoms_ladder,
        "n_ladder": n_ladder,
        "dictionary_constructions": dict_notes,
        "surface": surface,
        "frontier_report": frontier,
        "gates": gates,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM126 complete -> {output_dir / 'evidence.json'}", flush=True)
    for a in atoms_ladder:
        print("  atoms %d:" % a, {str(n): round(Q(a, n), 4) for n in n_ladder},
              flush=True)
    print(f"  KS fired: {ks['fired']}  (gain {gain:+.4f}, margin {KS_MARGIN})",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m126(args.config, args.output)


if __name__ == "__main__":
    main()
