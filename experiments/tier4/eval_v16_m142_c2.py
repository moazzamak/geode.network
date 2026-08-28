"""M142 cell C2 — spatial-pyramid pooling (1x1+2x2+4x4, 21 bins) vs a single
2x2 pool, at MATCHED per-image cost.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v22.md`` (section 6
Phase A, section 9 M142, and the execution log; 14 Aug 2026). RE-SCOPED
before any accuracy measurement — see the execution-log entry: the original
matched-cost cell (5,677 atoms, width 119,217) cannot be fitted with the
sealed ridge head on this machine (113.7 GB Gram needed, 63 GB installed),
and both exact subspace surrogates were REFUTED on the sealed codes before
dispatch (reduced-rank ridge r=2048 -> 0.030; Lanczos-Galerkin k=200 ->
0.086, vs direct 0.2246). The cell is therefore answered at a
width-feasible cost point; everything else is unchanged.

Question. Does replacing the single 2x2 pool with a spatial-pyramid pool
(21 bins over the 27x27 patch-activation map) lift the construction at
matched per-image cost?

Construction (both arms, same whitener + candidate pool + nested dictionary
prefixes of the seeded [11, 100] permutation):
- arm POOL: single 2x2 pool, b atoms (m107 ``_pool`` edges rule).
- arm SPM: 1x1 + 2x2 + 4x4 pyramid (21 bins), a atoms; each level uses the
  m107 edges rule ``round(grid*i/pool_grid)``; layout [1x1 | 2x2 | 4x4].

Matched-cost pair. The same ledger equation as the original registration,
with the width capped for the sealed Gram fit (peak = 3*width^2*8 bytes):
    b*(729*108 + 729 + 4*345) = a*(729*108 + 729 + 21*345)
    b*80,841 = a*86,706  ->  a = round(b*80,841/86,706)
Both arms count the pool adds (both sum the same 729-activation map).
Registered pair: b = 2,062 (pool), a = 1,923 (SPM, width 40,383). Both
~175.2M MACs/image (delta 0.02%).

Anchors:
- t1 encoders (at the sealed 6,144 atoms, bitwise vs the sealed f6144
  memmap): t1a the pool writer (the sealed m115 path), t1b the SPM encoder's
  2x2 level. The atom-prefix codes are NOT column-comparable to the sealed
  codes (the triangle activation's mean is over the atom set), which is why
  both checks run at 6,144 atoms.
- t2 environment: the direct ridge on the sealed f6144 codes at full data
  reproduces Q(6144, 409832) = 0.26136231884058 within 0.002 (context; the
  gate is the matched-pair head-to-head below, not this).
- t4 premise for the fresh baseline encode: Q(2062, 138000) must land
  within +-0.002 of the sealed atom-ladder envelope
  [Q(1536,138000)=0.19704347826086957, Q(3072,138000)=0.2152753623188406].

Gate (kill switch). Q_SPM(1923, 409832) >= Q_pool(2062, 409832) + 0.005 at
the sealed head constant (penalty 1.0), both fitted on the full 409,832-row
schedule and scored on the sealed 34,500-row test. The ridge ladder
{0.1, 1.0, 10.0} is reported for both arms; the trained-head read on the
SPM codes is the co-adaptation control; the cell closes as a scoped
negative only if BOTH the gate fires AND the trained-head read fails.

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m142_c2
"""

from __future__ import annotations

import argparse
import json
import time
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
from experiments.tier4.eval_v15_m104_experts import (
    RidgeAccumulator,
    _chunk_rows,
    _load_domainnet,
)
from experiments.tier4.eval_v15_m107_dense import (
    _solve_and_score,
    _verify_pixel_identity,
)
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m113_learned import (
    _build_whitener_and_candidates,
    _random_dictionary,
)
from experiments.tier4.eval_v16_m115_lofi import _write_frozen_codes
from experiments.tier4.eval_v16_m136_margin_head import _test_blocks
from experiments.tier4.eval_v16_m140_data_extension import _extension_indices
from experiments.tier4.eval_v16_m141_data_full import _rest_extension_indices

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m142_c2.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m142_c2"

CLASSES = 345
SPM_LEVELS = (1, 2, 4)          # -> 1 + 4 + 16 = 21 bins
POOL_PER_ATOM = 80_841          # 729*108 + 729 + 4*345 (adds counted, both arms)
SPM_PER_ATOM = 86_706           # 729*108 + 729 + 21*345
T1_TOLERANCE = 0.002
KS_MARGIN = 0.005
COST_TOLERANCE = 0.005


# --------------------------------------------------------------------------
# construction: SPM pooling and the matched-cost atom solver
# --------------------------------------------------------------------------
def _spm_pool(activation: torch.Tensor, count: int, grid: int) -> torch.Tensor:
    """1x1 + 2x2 + 4x4 sum pooling over the patch grid, m107 edges rule.

    ``activation`` is (count, grid*grid, atoms). The output layout is
    [level 1 (1 bin) | level 2 (4 bins) | level 4 (16 bins)], so the 2x2
    component is exactly columns [atoms : 5*atoms].
    """
    atoms = activation.shape[1]
    activation = activation.reshape(count, grid, grid, atoms)
    blocks: list[torch.Tensor] = []
    for level in SPM_LEVELS:
        edges = [round(grid * i / level) for i in range(level + 1)]
        for iy in range(level):
            for ix in range(level):
                blocks.append(
                    activation[:, edges[iy]:edges[iy + 1],
                               edges[ix]:edges[ix + 1]].sum(dim=(1, 2))
                )
    return torch.cat(blocks, dim=1)


def _pair_spm_atoms(pool_atoms: int) -> int:
    """SPM atom count matching the pool arm's ledger total:
    pool_atoms * POOL_PER_ATOM == spm_atoms * SPM_PER_ATOM (up to rounding)."""
    return int(round(pool_atoms * POOL_PER_ATOM / SPM_PER_ATOM))


def _pool_inference_macs(atoms: int, grid: int, dimension: int,
                         classes: int) -> dict[str, int]:
    patches = grid * grid
    whitening = patches * dimension * dimension
    encoding = patches * atoms * dimension
    pool_adds = patches * atoms
    head = 4 * atoms * classes
    return {
        "whitening": int(whitening),
        "encoding": int(encoding),
        "pool_adds": int(pool_adds),
        "head": int(head),
        "total": int(whitening + encoding + pool_adds + head),
    }


def _spm_inference_macs(atoms: int, grid: int, dimension: int, classes: int,
                        levels: tuple[int, ...] = SPM_LEVELS) -> dict[str, int]:
    """The C2 cell's per-image ledger, same exclusions as the sealed ledger."""
    patches = grid * grid
    whitening = patches * dimension * dimension
    encoding = patches * atoms * dimension
    pool_adds = patches * atoms
    bins = sum(level * level for level in levels)
    head = bins * atoms * classes
    return {
        "whitening": int(whitening),
        "encoding": int(encoding),
        "pool_adds": int(pool_adds),
        "head": int(head),
        "total": int(whitening + encoding + pool_adds + head),
        "_excluded": (
            "patch extraction and per-patch contrast normalisation are NOT "
            "counted, exactly as the sealed ledger discloses"
        ),
    }


def _spm_encode_block_device(images: np.ndarray, table: torch.Tensor,
                             whitener, grid: int) -> np.ndarray:
    """One SPM encode block: whiten on CPU, cdist/pool on the GPU.

    The whitening is numpy on the CPU (M107's exact arithmetic); only the
    cdist, activation and pooling run on the GPU, so the 2x2 level is the
    identical arithmetic to the sealed ``_encode_block_device``.
    """
    white = torch.from_numpy(
        np.ascontiguousarray(whitener(images))
    ).to(torch.float32).to(table.device)
    with torch.no_grad():
        distances = torch.cdist(white, table)
        activation = torch.clamp(
            distances.mean(dim=1, keepdim=True) - distances, min=0.0
        )
        pooled = _spm_pool(activation, len(images), grid)
    return pooled.to(torch.float32).cpu().numpy()


def _append_encode(images: np.ndarray, rows: np.ndarray, table: torch.Tensor,
                   whitener, grid: int, out: np.ndarray, offset: int,
                   throttle_seconds: float) -> int:
    """Encode ``images[rows]`` into ``out[offset:...]``; return the new offset."""
    step = _chunk_rows(table.shape[0], grid, len(rows))
    for start in range(0, len(rows), step):
        take = rows[start:start + step]
        stop = offset + len(take)
        out[offset:stop] = _spm_encode_block_device(
            images[take], table, whitener, grid)
        offset = stop
        if throttle_seconds > 0:
            time.sleep(throttle_seconds)
    return offset


# --------------------------------------------------------------------------
# fits and reads (the sealed direct path throughout)
# --------------------------------------------------------------------------


def _score_weights(test_codes: np.ndarray, test_labels: np.ndarray,
                   test_domains: np.ndarray, weights: np.ndarray,
                   standardiser=None, block: int = 4096) -> dict[str, Any]:
    """Score test codes with a stacked-weights model, standardising first.

    Blocked so the float64 standardised copy never exceeds one block.
    """
    n = len(test_labels)
    hits = 0
    per_domain_correct = np.zeros(6, dtype=np.int64)
    per_domain_rows = np.zeros(6, dtype=np.int64)
    for start in range(0, n, block):
        stop = min(start + block, n)
        if standardiser is not None:
            xs = standardiser(test_codes[start:stop]).astype(np.float64)
        else:
            xs = test_codes[start:stop].astype(np.float64)
        scores = xs @ weights[:-1] + weights[-1]
        ok = np.asarray(np.argmax(scores, axis=1) == test_labels[start:stop])
        hits += int(ok.sum())
        np.add.at(per_domain_correct, test_domains[start:stop],
                  ok.astype(np.int64))
        np.add.at(per_domain_rows, test_domains[start:stop], 1)
    return {
        "accuracy": hits / n,
        "per_domain": [
            float(per_domain_correct[d] / per_domain_rows[d] if per_domain_rows[d]
                  else 0.0) for d in range(6)
        ],
        "test_rows": n,
    }


def _fit_direct(parts: list[np.ndarray], labels: np.ndarray, width: int,
                test_blocks: Any, penalty: float) -> dict[str, Any]:
    """The sealed fit path (Gram accumulator) over the given code parts."""
    acc = RidgeAccumulator(width, CLASSES)
    offset = 0
    for part in parts:
        for start in range(0, len(part), 4096):
            stop = min(start + 4096, len(part))
            acc.add(np.asarray(part[start:stop]),
                    labels[offset + start:offset + stop])
        offset += len(part)
    result = _solve_and_score(acc, [penalty], test_blocks)
    return {
        "accuracy": result["accuracy_by_penalty"][str(penalty)],
        "per_domain": [c / r if r else 0.0 for c, r in zip(
            result["per_domain_correct"][str(penalty)],
            result["per_domain_rows"][str(penalty)])],
        "fit_rows": result["fit_rows"],
        "test_rows": result["test_rows"],
    }


def _fit_ladder(mem_train: np.ndarray, labels: np.ndarray, width: int,
                penalties: list[float], n_rows: int
                ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Direct ridge ladder on the first ``n_rows`` rows of the memmap.

    One accumulator produces every penalty; each info entry carries the
    shared standardiser for scoring."""
    acc = RidgeAccumulator(width, CLASSES)
    for start in range(0, n_rows, 4096):
        stop = min(start + 4096, n_rows)
        acc.add(np.asarray(mem_train[start:stop]), labels[start:stop])
    solved = acc.solve_many(penalties)
    std = acc.standardiser()
    info = {"fit_rows": int(acc.rows), "width": int(width),
            "standardiser": std}
    return ({str(p): w for p, w in solved.items()},
            {str(p): info for p in penalties})


def _encode_test_to_ram(test_images: np.ndarray, table: torch.Tensor,
                        whitener, grid: int, n_test: int, width: int,
                        throttle: float,
                        pool_slice) -> np.ndarray:
    """Encode the first ``n_test`` test rows to a RAM array.

    ``pool_slice`` (a column slice) restricts the output to one level of the
    SPM encoder, which is how the pool arm reuses the shared encoder (its
    2x2 level is bit-identical to the sealed single-pool encode)."""
    out = np.empty((n_test, width), dtype=np.float32)
    step = _chunk_rows(table.shape[0], grid, n_test)
    for start in range(0, n_test, step):
        stop = min(start + step, n_test)
        block = _spm_encode_block_device(test_images[start:stop], table,
                                         whitener, grid)
        if pool_slice is not None:
            block = block[:, pool_slice]
        out[start:stop] = block
        if throttle > 0:
            time.sleep(throttle)
    return out


def _trained_head_read(train_mem: np.ndarray, train_labels: np.ndarray,
                       test_codes: np.ndarray, test_labels: np.ndarray,
                       epochs: int, lr: float, seed: int,
                       device: torch.device) -> float:
    """SGD linear head on the frozen C2 codes (the co-adaptation read)."""
    torch.manual_seed(seed)
    width = train_mem.shape[1]
    model = torch.nn.Linear(width, CLASSES, bias=True).to(torch.float32).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    n = len(train_labels)
    order = np.random.default_rng(seed).permutation(n)
    batch = int(np.sqrt(n))
    for _ in range(epochs):
        for start in range(0, n, batch):
            take = order[start:start + batch]
            xs = torch.from_numpy(
                np.asarray(train_mem[take], dtype=np.float32)).to(device)
            ys = torch.from_numpy(train_labels[take].astype(np.int64)).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xs), ys)
            loss.backward()
            opt.step()
    model.eval()
    hits = 0
    n_test = len(test_labels)
    with torch.no_grad():
        for start in range(0, n_test, 4096):
            stop = min(start + 4096, n_test)
            xs = torch.from_numpy(test_codes[start:stop]).to(device)
            preds = torch.argmax(model(xs), dim=1).cpu().numpy()
            hits += int((preds == test_labels[start:stop]).sum())
    return hits / n_test


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def run_m142_c2(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    smoke = inadmissible
    smoke_skip = bool(config.get("_smoke_skip_gates", False))
    block = int(config["numerics"]["block"])
    throttle = float(config["numerics"]["encode_throttle_seconds"])
    torch.set_num_threads(int(config["numerics"]["torch_threads"]))
    torch.manual_seed(int(config["numerics"]["seed"]))
    configure_external_cache_environment()
    _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("loading corpus (subsample + raw)", flush=True)
    corpus, train_index, test_index = _load_corpus(config)
    raw = _load_domainnet(int(config["corpus"]["image_size"]))
    size = int(config["corpus"]["image_size"])
    for split, idx in (("train", train_index), ("test", test_index)):
        _verify_pixel_identity(split, idx, corpus[f"{split}_images"], size,
                               int(config["corpus"]["pixel_identity_rows"]))

    rep = config["sparse"]
    grid = (size - int(rep["patch"])) // int(rep["stride"]) + 1
    dimension = int(rep["patch"]) ** 2 * 3

    # ---- matched-pair arithmetic gate (checked at registration) -----------
    pool_atoms = int(config["sparse"]["pool_atoms"])
    spm_atoms = int(config["sparse"]["spm_atoms"])
    solved = _pair_spm_atoms(pool_atoms)
    if solved != spm_atoms:
        raise SystemExit(
            f"matched-pair arithmetic failure: solver gives a={solved}, "
            f"config registers a={spm_atoms} for b={pool_atoms}.")
    pool_macs = _pool_inference_macs(pool_atoms, grid, dimension, CLASSES)
    spm_macs = _spm_inference_macs(spm_atoms, grid, dimension, CLASSES)
    cost_delta = abs(spm_macs["total"] - pool_macs["total"]) / pool_macs["total"]
    if cost_delta > COST_TOLERANCE:
        raise SystemExit(
            f"matched-cost check failed: |SPM {spm_macs['total']} - pool "
            f"{pool_macs['total']}| = {cost_delta:.4%} > {COST_TOLERANCE:.4%}")
    spm_width = sum(level * level for level in SPM_LEVELS) * spm_atoms
    pool_width = 4 * pool_atoms
    print(f"matched pair: pool b={pool_atoms} ({pool_macs['total']}), "
          f"SPM a={spm_atoms} ({spm_macs['total']}), "
          f"delta {cost_delta:+.4%}", flush=True)

    print("building sealed whitener + nested dictionary prefixes", flush=True)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)
    dict_pool = _random_dictionary(candidates, len(candidates),
                                   int(rep["dictionary_seed"]), pool_atoms)
    dict_spm = _random_dictionary(candidates, len(candidates),
                                  int(rep["dictionary_seed"]), spm_atoms)
    table_spm = torch.from_numpy(
        np.ascontiguousarray(dict_spm)).to(torch.float32).to(device)

    evidence: dict[str, Any] = {
        "milestone": "M142",
        "cell": "C2 spatial-pyramid pooling vs single 2x2 pool at matched "
                "cost (re-scoped; see plan execution log)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "matched_cost": {
            "pool_atoms": pool_atoms,
            "spm_atoms": spm_atoms,
            "pair_equation": "b*(729*108 + 4*345) = a*(729*108 + 729 + 21*345)",
            "pool_macs": pool_macs,
            "spm_macs": spm_macs,
            "cost_delta_fraction": cost_delta,
            "rescale_note": ("the original 5,677-atom matched cell needs a "
                             "113.7 GB Gram at width 119,217 (63 GB "
                             "installed); both exact subspace surrogates were "
                             "refuted on the sealed codes before dispatch — "
                             "see the plan execution log"),
        },
    }

    # ---- t1: encoder instrument checks at the sealed 6,144 atoms ----------
    # The triangle activation's mean is over the atom set, so prefix-atom
    # codes are not column-comparable to the sealed codes; both checks run
    # at the sealed atom count.
    check_rows = int(config["anchors"]["t1_encoder_check_rows"])
    sealed_train = np.load(
        data_cache_root() / config["sealed_codes"]["cache_relpath"]
        / config["sealed_codes"]["train_file"], mmap_mode="r")
    check_dir = data_cache_root() / "v16" / "m142_c2"
    check_dir.mkdir(parents=True, exist_ok=True)

    check_pool = _write_frozen_codes(
        corpus, _random_dictionary(candidates, len(candidates),
                                   int(rep["dictionary_seed"]), 6144),
        whitener, 2, device, np.arange(check_rows),
        check_dir / "_check_pool6144.npy",
        split="train", throttle_seconds=throttle)
    d1 = float(np.abs(np.asarray(sealed_train[:check_rows], dtype=np.float64)
                      - check_pool.astype(np.float64)).max())
    t1a_ok = d1 <= float(config["anchors"]["t1_encoder_check_tolerance"])

    dict_6144 = _random_dictionary(candidates, len(candidates),
                                   int(rep["dictionary_seed"]), 6144)
    table_6144 = torch.from_numpy(
        np.ascontiguousarray(dict_6144)).to(torch.float32).to(device)
    check_full = np.empty((check_rows, 21 * 6144), dtype=np.float32)
    _append_encode(corpus["train_images"], np.arange(check_rows), table_6144,
                   whitener, grid, check_full, 0, throttle)
    d2 = float(np.abs(np.asarray(sealed_train[:check_rows], dtype=np.float64)
                      - check_full[:, 6144:5 * 6144].astype(np.float64)).max())
    t1b_ok = d2 <= float(config["anchors"]["t1_encoder_check_tolerance"])
    del table_6144, dict_6144, check_full
    torch.cuda.empty_cache()
    evidence["t1_encoder_checks"] = {
        "pool_writer_delta": d1, "pool_ok": t1a_ok,
        "spm_level2_delta": d2, "spm_ok": t1b_ok,
        "tolerance": float(config["anchors"]["t1_encoder_check_tolerance"]),
    }
    print(f"t1 encoder checks: pool {d1:.3e} (ok={t1a_ok}), "
          f"spm level2 {d2:.3e} (ok={t1b_ok})", flush=True)
    if (not t1a_ok or not t1b_ok) and not smoke_skip:
        evidence["void"] = True
        evidence["void_reason"] = "t1 encoder check failed"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        return evidence

    # ---- full train encodes (the M141 cell-2 row schedule) ----------------
    ext600_indices, _ = _extension_indices(raw["train_labels"], train_index,
                                           600, CLASSES)
    rest_indices = _rest_extension_indices(raw["train_labels"], train_index,
                                           CLASSES, per_class_take=200)
    if smoke:
        n_part1 = int(config["_smoke_train_rows"])
        schedule = [
            (corpus["train_images"], np.arange(n_part1),
             corpus["train_labels"][:n_part1]),
        ]
    else:
        schedule = [
            # part 1: the subsample corpus is ALREADY in train_index order,
            # so its rows are positions, not raw-corpus indices
            (corpus["train_images"], np.arange(len(train_index)),
             corpus["train_labels"]),
            (raw["train_images"], ext600_indices,
             raw["train_labels"][ext600_indices]),
            (raw["train_images"], rest_indices,
             raw["train_labels"][rest_indices]),
        ]
    n_train = sum(len(rows) for _, rows, _ in schedule)
    cache = data_cache_root() / "v16" / "m142_c2"
    cache.mkdir(parents=True, exist_ok=True)
    pool_path = cache / f"pool{pool_atoms}_fulltrain.npy"
    spm_path = cache / f"spm{spm_atoms}_fulltrain.npy"
    labels_path = cache / "m142_c2_fulltrain_labels.npz"

    # pool arm: encode with the SPM encoder restricted to one 2x2 level. The
    # 2x2 level of _spm_pool is bit-identical to the sealed _pool(2) path
    # (t1b), and using one shared encoder for both arms means both arms run
    # the same cdist/activation arithmetic — only the pooling differs.
    table_pool = torch.from_numpy(
        np.ascontiguousarray(dict_pool)).to(torch.float32).to(device)

    def _append_pool(images, rows, out, offset):
        step = _chunk_rows(pool_atoms, grid, len(rows))
        for start in range(0, len(rows), step):
            take = rows[start:start + step]
            stop = offset + len(take)
            block = _spm_encode_block_device(images[take], table_pool,
                                             whitener, grid)
            out[offset:stop] = block[:, pool_atoms:5 * pool_atoms]
            offset = stop
            if throttle > 0:
                time.sleep(throttle)
        return offset

    print(f"encoding pool arm ({pool_atoms} atoms, {n_train} rows)", flush=True)
    pool_mem = np.lib.format.open_memmap(
        pool_path, mode="w+", dtype=np.float32, shape=(n_train, pool_width))
    offset = 0
    for images, rows, _labels in schedule:
        offset = _append_pool(images, rows, pool_mem, offset)
    del pool_mem

    print(f"encoding SPM arm ({spm_atoms} atoms, {n_train} rows)", flush=True)
    spm_mem = np.lib.format.open_memmap(
        spm_path, mode="w+", dtype=np.float32, shape=(n_train, spm_width))
    offset = 0
    for images, rows, _labels in schedule:
        offset = _append_encode(images, rows, table_spm, whitener, grid,
                                spm_mem, offset, throttle)
    del spm_mem

    np.savez(labels_path, labels=np.concatenate([l for _, _, l in schedule]))
    full_labels = np.load(labels_path)["labels"]
    pool_mem = np.load(pool_path, mmap_mode="r")
    spm_mem = np.load(spm_path, mmap_mode="r")
    print(f"  train encodes done ({time.time() - started:.0f}s so far)",
          flush=True)

    # ---- t2: environment anchor on the sealed f6144 codes -----------------
    sealed_mem = np.load(
        data_cache_root() / config["sealed_codes"]["cache_relpath"]
        / config["sealed_codes"]["train_file"], mmap_mode="r")
    sealed_test = np.load(
        data_cache_root() / config["sealed_codes"]["cache_relpath"]
        / config["sealed_codes"]["test_file"], mmap_mode="r")
    sealed_labels = corpus["train_labels"]
    sealed_test_blocks = _test_blocks(sealed_test, corpus["test_labels"],
                                      corpus["test_domains"], block)
    print("t2: direct ridge on sealed f6144 codes (full data)", flush=True)
    sealed_parts = [sealed_mem]
    sealed_part_labels = [sealed_labels]
    if smoke:
        sealed_parts = [sealed_mem[:20000]]
        sealed_part_labels = [sealed_labels[:20000]]
    else:
        ext600_mem = np.load(
            data_cache_root() / config["sealed_codes"]["ext600_relpath"]
            / config["sealed_codes"]["ext600_file"], mmap_mode="r")
        rest_mem = np.load(
            data_cache_root() / config["sealed_codes"]["rest_relpath"]
            / config["sealed_codes"]["rest_file"], mmap_mode="r")
        sealed_parts += [ext600_mem, rest_mem]
        sealed_part_labels += [raw["train_labels"][ext600_indices],
                               raw["train_labels"][rest_indices]]
    sealed_all_labels = np.concatenate(sealed_part_labels)
    t2 = _fit_direct(sealed_parts, sealed_all_labels,
                     int(config["sealed_codes"]["width"]),
                     sealed_test_blocks, 1.0)
    t2_delta = t2["accuracy"] - float(config["anchors"]["t2_reference"])
    evidence["t2_full_data_direct"] = {
        **t2, "reference": config["anchors"]["t2_reference"],
        "delta": t2_delta}
    print(f"  t2 direct {t2['accuracy']:.4f} vs sealed "
          f"{config['anchors']['t2_reference']} (delta {t2_delta:+.6f})",
          flush=True)
    if not smoke_skip and abs(t2_delta) > T1_TOLERANCE:
        evidence["void"] = True
        evidence["void_reason"] = "t2 anchor reproduction failed"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        return evidence

    # ---- t4: premise range for the fresh pool encode ----------------------
    n_138 = len(train_index) if not smoke else len(pool_mem)
    pool_138_w, pool_138_info = _fit_ladder(
        pool_mem, full_labels, pool_width,
        [float(p) for p in config["cell_c2"]["penalty_ladder"]], n_138)
    env_lo = float(config["anchors"]["t4_envelope_lo"])
    env_hi = float(config["anchors"]["t4_envelope_hi"])
    pool_test_codes = _encode_test_to_ram(
        corpus["test_images"], table_pool, whitener, grid,
        len(corpus["test_labels"]) if not smoke else int(
            config["_smoke_test_rows"]), pool_width, throttle,
        slice(pool_atoms, 5 * pool_atoms))
    pool_138_acc = _score_weights(
        pool_test_codes, corpus["test_labels"][:len(pool_test_codes)],
        corpus["test_domains"][:len(pool_test_codes)],
        pool_138_w["1.0"], pool_138_info["1.0"]["standardiser"])["accuracy"]
    t4_ok = (env_lo - T1_TOLERANCE <= pool_138_acc
             <= env_hi + T1_TOLERANCE)
    evidence["t4_premise_range"] = {
        "measured": pool_138_acc, "envelope": [env_lo, env_hi],
        "ok": t4_ok,
        "note": "Q(2062, 138000) inside the sealed atom-ladder envelope"}
    print(f"t4 premise: Q({pool_atoms}, {n_138}) = {pool_138_acc:.4f} "
          f"envelope [{env_lo}, {env_hi}] ok={t4_ok}", flush=True)
    if not smoke_skip and not t4_ok:
        evidence["void"] = True
        evidence["void_reason"] = "t4 premise range check failed"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        return evidence

    # ---- the matched-pair fits (full data) ---------------------------------
    print("full-data ridge ladders (pool vs SPM)", flush=True)
    pool_w, pool_info = _fit_ladder(
        pool_mem, full_labels, pool_width,
        [float(p) for p in config["cell_c2"]["penalty_ladder"]], len(pool_mem))
    spm_w, spm_info = _fit_ladder(
        spm_mem, full_labels, spm_width,
        [float(p) for p in config["cell_c2"]["penalty_ladder"]], len(spm_mem))

    # 138k-level SPM read
    spm_138_w, spm_138_info = _fit_ladder(
        spm_mem, full_labels, spm_width,
        [1.0], n_138)

    print("test encodes to RAM for scoring", flush=True)
    n_test = len(corpus["test_labels"]) if not smoke else int(
        config["_smoke_test_rows"])
    test_labels = corpus["test_labels"][:n_test]
    test_domains = corpus["test_domains"][:n_test]
    spm_test_codes = _encode_test_to_ram(
        corpus["test_images"], table_spm, whitener, grid, n_test, spm_width,
        throttle, None)

    # per-level diagnostics on the SPM arm (all direct fits)
    per_level: dict[str, Any] = {}
    level_spans = {1: (0, spm_atoms), 2: (spm_atoms, 5 * spm_atoms),
                   4: (5 * spm_atoms, 21 * spm_atoms)}
    for level in (1, 2, 4):
        lo, hi = level_spans[level]
        lw = hi - lo
        acc = RidgeAccumulator(lw, CLASSES)
        for start in range(0, len(spm_mem), 4096):
            stop = min(start + 4096, len(spm_mem))
            acc.add(np.asarray(spm_mem[start:stop, lo:hi]),
                    full_labels[start:stop])
        solved_level = acc.solve(1.0)
        per_level[str(level)] = _score_weights(
            spm_test_codes[:, lo:hi], test_labels, test_domains,
            solved_level, acc.standardiser())
        print(f"  level {level}: {per_level[str(level)]['accuracy']:.4f}",
              flush=True)

    # score every ladder entry (strip the Standardiser before evidence)
    def _score_ladder(weights_dict, info_dict, test_codes):
        out = {}
        for p_str, w in weights_dict.items():
            info = {k: v for k, v in info_dict[p_str].items()
                    if k != "standardiser"}
            out[p_str] = {
                **info,
                **_score_weights(test_codes, test_labels, test_domains, w,
                                 info_dict[p_str]["standardiser"])}
        return out

    pool_ladder_acc = _score_ladder(pool_w, pool_info, pool_test_codes)
    spm_ladder_acc = _score_ladder(spm_w, spm_info, spm_test_codes)
    spm_138_acc = _score_weights(
        spm_test_codes, test_labels, test_domains, spm_138_w["1.0"],
        spm_138_info["1.0"]["standardiser"])
    del pool_test_codes

    trained_acc = None
    if not smoke:
        print("trained-head read on the SPM codes", flush=True)
        trained_acc = _trained_head_read(
            spm_mem, full_labels, spm_test_codes, test_labels,
            int(config["cell_c2"]["trained_epochs"]),
            float(config["cell_c2"]["trained_lr"]),
            int(config["cell_c2"]["trained_seed"]), device)

    # ---- gate ---------------------------------------------------------------
    spm_gate = spm_ladder_acc["1.0"]["accuracy"]
    pool_gate = pool_ladder_acc["1.0"]["accuracy"]
    gain = spm_gate - pool_gate
    fired = (not smoke) and (gain < KS_MARGIN)
    both_fail = fired and (trained_acc is not None
                           and trained_acc < pool_gate + KS_MARGIN)
    evidence["reads"] = {
        "pool_ladder": pool_ladder_acc,
        "spm_ladder": spm_ladder_acc,
        "spm_138k": spm_138_acc,
        "pool_138k": pool_138_acc,
        "per_level": per_level,
        "trained_head_read": trained_acc,
    }
    evidence["gate"] = {
        "registered": config["cell_c2"]["gate_registered"],
        "spm_penalty1_full": spm_gate,
        "pool_penalty1_full": pool_gate,
        "gain": gain,
        "required": KS_MARGIN,
        "fired": fired,
        "consequence": (
            config["cell_c2"]["consequence_fired"] if fired else
            config["cell_c2"]["consequence_passed"]),
        "closure_note": ("scoped negative requires BOTH reads to fail; "
                         f"both_fail={bool(both_fail)}"),
    }
    evidence["runtime_seconds"] = round(time.time() - started, 2)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM142 C2 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m142_c2(args.config, args.output)


if __name__ == "__main__":
    main()
