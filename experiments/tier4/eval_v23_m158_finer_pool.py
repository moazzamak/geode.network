"""M158 — finer pooling: the 8x8 level alone at matched head cost.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v23.md`` (section 4
M158; the section 6 re-scope, 17 Aug 2026). The original 85-bin pyramid
is infeasible (registered); this cell tests where pooling granularity
saturates with the 8x8 level ALONE at a = 481 atoms — the first 481
atoms of the same [11,100]-permuted pool (the C2 dictionary prefix) —
head-cost-matched to the sealed 4x4 level at 1,923 atoms
(64 x 481 x 345 == 16 x 1,923 x 345).

Construction: one streamed encode over the M141 cell-2 schedule (part 1
= the 138k subsample; parts 2-3 = the ext600/rest raw-image rows) with
the rebuilt M108 whitener + C2 dictionary; pooling = the 8x8 level only
(the m107 edges rule). The full-data fit streams into the Gram
accumulator; the 138k-part and test codes are persisted under
``v16/m158/`` for the trained-head read and test scoring.

Anchors: t1 — a fresh 21-bin C2 encode of the first 64 train rows with
the same dictionary reproduces the cached spm1923_fulltrain codes
bitwise (tol 0.0); a2 — a ridge refit on the cached 4x4-level columns
(full data, penalty 1.0) reproduces the sealed per-level read
0.26014492753623186 (tol 1e-9).

Gate: Q(8x8, 481, full data, penalty 1.0) >= 0.26014492753623186 +
0.005 at matched head cost; else the pooling-saturation point is 4x4.
The trained-head read at 138k (M109 schedule) is the dual-read
control. Smoke declares inadmissibility and refuses the sealed output
directory.
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
from experiments.tier4.eval_v15_m107_dense import _verify_pixel_identity
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
from experiments.tier4.eval_v16_m109_trunk import (
    _build_whitener_dictionary,
    _load_corpus,
    _train_with_schedule,
)
from experiments.tier4.eval_v16_m140_data_extension import _extension_indices
from experiments.tier4.eval_v16_m141_data_full import _rest_extension_indices
from experiments.tier4.eval_v16_m146_arbiter import HeadOnly

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v23"
                  / "m158_finer_pool.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v23" / "m158_finer_pool")

CLASSES = 345
LEVELS21 = (1, 2, 4)
LEVEL8 = (8,)
ATOMS_8 = 481
ATOMS_SPM = 1923
TOLERANCE_T1 = 0.0
TOLERANCE_A2 = 1e-9
MARGIN = 0.005
SEALED_4X4 = 0.26014492753623186
FLOOR = 10.0


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def _pool_levels(activation: torch.Tensor, count: int, grid: int,
                 levels: tuple[int, ...]) -> torch.Tensor:
    """Sum-pooling at the given levels (the m107 edges rule)."""
    atoms = activation.shape[1]
    activation = activation.reshape(count, grid, grid, atoms)
    blocks: list[torch.Tensor] = []
    for level in levels:
        edges = [round(grid * i / level) for i in range(level + 1)]
        for iy in range(level):
            for ix in range(level):
                blocks.append(
                    activation[:, edges[iy]:edges[iy + 1],
                               edges[ix]:edges[ix + 1]].sum(dim=(1, 2)))
    return torch.cat(blocks, dim=1)


def _encode_levels_device(images: np.ndarray, table: torch.Tensor,
                          whitener, grid: int,
                          levels: tuple[int, ...]) -> np.ndarray:
    """One encode block at the given pooling levels (whiten CPU, GPU rest)."""
    white = torch.from_numpy(
        np.ascontiguousarray(whitener(images))
    ).to(torch.float32).to(table.device)
    with torch.no_grad():
        distances = torch.cdist(white, table)
        activation = torch.clamp(
            distances.mean(dim=1, keepdim=True) - distances, min=0.0)
        pooled = _pool_levels(activation, len(images), grid, levels)
    return pooled.to(torch.float32).cpu().numpy()


def _head_macs(atoms: int, grid: int, dimension: int, levels: tuple[int, ...]
               ) -> dict[str, int]:
    """Per-image ledger at the given levels (the sealed exclusions apply)."""
    patches = grid * grid
    whitening = patches * dimension * dimension
    encoding = patches * atoms * dimension
    pool_adds = patches * atoms
    bins = sum(level * level for level in levels)
    head = bins * atoms * CLASSES
    return {
        "whitening": int(whitening),
        "encoding": int(encoding),
        "pool_adds": int(pool_adds),
        "head": int(head),
        "total": int(whitening + encoding + pool_adds + head),
    }


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run_m158(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    smoke = inadmissible
    skip_anchors = bool(config.get("_smoke_skip_anchors", False))
    smoke_train = int(config.get("_smoke_train_rows", 10 ** 9))
    smoke_test = int(config.get("_smoke_test_rows", 10 ** 9))
    block = int(config["numerics"]["block"])

    torch.set_num_threads(int(config["numerics"]["torch_threads"]))
    torch.manual_seed(109)
    configure_external_cache_environment()
    _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    batch = int(config["numerics"]["batch"])
    throttle = float(config["numerics"]["encode_throttle_seconds"])

    print("loading corpus", flush=True)
    corpus, train_index, test_index = _load_corpus(config)
    size = int(config["corpus"]["image_size"])
    for split, idx in (("train", train_index), ("test", test_index)):
        _verify_pixel_identity(split, idx, corpus[f"{split}_images"], size,
                               int(config["corpus"]
                                   ["pixel_identity_rows"]))

    evidence: dict[str, Any] = {
        "milestone": "M158",
        "cell": "finer pooling (8x8 level alone at matched head cost)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
    }

    print("building M108 whitener + C2 dictionary", flush=True)
    sparse_tmp = dict(config["sparse"])
    sparse_tmp["atoms"] = ATOMS_SPM
    config_tmp = dict(config)
    config_tmp["sparse"] = sparse_tmp
    whitener, dictionary, grid, dimension, _pg = \
        _build_whitener_dictionary(config_tmp, corpus)
    if dictionary.shape != (ATOMS_SPM, dimension):
        raise SystemExit("M158 instrument failure: dictionary shape "
                         f"{dictionary.shape}")

    spm_cache = data_cache_root() / config["artifacts"]["m142_cache_relpath"]
    spm_fulltrain = np.load(spm_cache / config["artifacts"]
                            ["spm_train_file"], mmap_mode="r")
    spm_fulltest = np.load(spm_cache / config["artifacts"]
                           ["spm_test_file"], mmap_mode="r")

    anchors: dict[str, Any] = {}

    # ---- t1: the 21-bin encode reproduces the cached codes bitwise --------
    if not skip_anchors:
        print("t1: 21-bin encode reproduction (64 rows)", flush=True)
        table21 = torch.from_numpy(
            np.ascontiguousarray(dictionary)).to(torch.float32).to(device)
        t1_encoded = _encode_levels_device(corpus["train_images"][:64],
                                           table21, whitener, grid, LEVELS21)
        t1_cached = np.asarray(spm_fulltrain[:64])
        t1_delta = float(np.abs(t1_encoded - t1_cached).max())
        anchors["t1"] = {"max_abs_delta": t1_delta,
                         "tolerance": TOLERANCE_T1}
        print(f"  t1 delta {t1_delta:.3e}", flush=True)
        if t1_delta > TOLERANCE_T1:
            evidence.update({"void": True,
                             "void_reason": "t1 encoder reproduction failed",
                             "anchors": anchors})
            _write(output_dir, evidence)
            return evidence
        del table21
        torch.cuda.empty_cache()

    # ---- a2: the sealed 4x4-level read from the cached columns -------------
    if not skip_anchors:
        print("a2: 4x4-level refit from the cached columns", flush=True)
        col4 = np.arange(5 * ATOMS_SPM, 21 * ATOMS_SPM, dtype=np.int64)
        n_train = (int(config["_smoke_train_rows"]) if smoke
                   else len(spm_fulltrain))
        acc4 = RidgeAccumulator(len(col4), CLASSES)
        labels = np.load(spm_cache / config["artifacts"]["labels_file"]
                         )["labels"][:n_train]
        for start in range(0, n_train, block):
            stop = min(start + block, n_train)
            acc4.add(np.asarray(spm_fulltrain[start:stop])[:, col4],
                     labels[start:stop])
        w4 = acc4.solve_many([1.0])[1.0]
        std4 = acc4.standardiser()
        test_labels = corpus["test_labels"][:smoke_test]
        n_test = len(test_labels)
        hits = 0
        for start in range(0, n_test, block):
            stop = min(start + block, n_test)
            xs = std4(np.asarray(spm_fulltest[start:stop])[:, col4])
            hits += int((np.argmax(xs @ w4[:-1] + w4[-1], axis=1)
                         == test_labels[start:stop]).sum())
        a2_read = hits / n_test
        anchors["a2_4x4"] = {"measured": a2_read, "sealed": SEALED_4X4,
                             "delta": a2_read - SEALED_4X4,
                             "tolerance": TOLERANCE_A2}
        print(f"  a2 4x4 {a2_read:.6f} (delta {a2_read - SEALED_4X4:+.3e})",
              flush=True)
        if abs(a2_read - SEALED_4X4) > TOLERANCE_A2:
            evidence.update({"void": True,
                             "void_reason": "a2 4x4-level reproduction "
                                            "failed",
                             "anchors": anchors})
            _write(output_dir, evidence)
            return evidence

    # ---- the 8x8 schedule ---------------------------------------------------
    print("8x8 schedule (the M141 cell-2 order)", flush=True)
    table8 = torch.from_numpy(
        np.ascontiguousarray(dictionary[:ATOMS_8])).to(torch.float32).to(device)
    raw = _load_domainnet(size)
    ext_idx, _ = _extension_indices(raw["train_labels"], train_index, 600,
                                    CLASSES)
    rest_idx = _rest_extension_indices(raw["train_labels"], train_index,
                                       CLASSES, per_class_take=200)
    if smoke:
        schedule = [(corpus["train_images"], np.arange(smoke_train),
                     corpus["train_labels"][:smoke_train])]
        n_part1 = smoke_train
    else:
        schedule = [
            (corpus["train_images"], np.arange(len(train_index)),
             corpus["train_labels"]),
            (raw["train_images"], ext_idx, raw["train_labels"][ext_idx]),
            (raw["train_images"], rest_idx, raw["train_labels"][rest_idx]),
        ]
        n_part1 = len(train_index)
        if sum(len(r) for _, r, _ in schedule) != 409832:
            raise SystemExit("M158 premise failure: schedule row count")
    n_train = sum(len(rows) for _, rows, _ in schedule)
    width8 = sum(level * level for level in LEVEL8) * ATOMS_8  # 64 x 481
    if width8 != 64 * ATOMS_8:
        raise SystemExit("M158 instrument failure: 8x8 width")

    # floor premise
    rows_per_dim = n_train // width8
    if rows_per_dim < FLOOR and not smoke:
        raise SystemExit("M158 PREMISE GATE: rows per fitted dimension "
                         f"{rows_per_dim} < {FLOOR}")
    evidence["premise"] = {"rows_per_dim": rows_per_dim, "floor": FLOOR}

    cache = data_cache_root() / "v16" / "m158"
    cache.mkdir(parents=True, exist_ok=True)

    print(f"8x8 encode: {n_train} train rows (streamed)", flush=True)
    acc8 = RidgeAccumulator(width8, CLASSES)
    part1_mem = None
    if smoke:
        part1_mem = np.empty((n_part1, width8), dtype=np.float32)
    else:
        part1_path = cache / "spm8_138k_train.npy"
        part1_mem = np.lib.format.open_memmap(
            part1_path, mode="w+", dtype=np.float32,
            shape=(n_part1, width8))
    offset = 0
    for images, rows, part_labels in schedule:
        step = _chunk_rows(ATOMS_8, grid, len(rows))
        for start in range(0, len(rows), step):
            take = rows[start:start + step]
            codes = _encode_levels_device(images[take], table8, whitener,
                                          grid, LEVEL8)
            acc8.add(codes, part_labels[start:start + len(take)])
            if offset < n_part1:
                stop = min(offset + len(take), n_part1)
                part1_mem[offset:stop] = codes[:stop - offset]
            offset += len(take)
            if throttle > 0:
                time.sleep(throttle)
    if offset != n_train:
        raise SystemExit("M158 instrument failure: encode offset mismatch")

    test_labels = corpus["test_labels"][:smoke_test]
    n_test = len(test_labels)
    print(f"8x8 test encode: {n_test} rows", flush=True)
    if smoke:
        test8 = np.empty((n_test, width8), dtype=np.float32)
    else:
        test_path = cache / "spm8_test.npy"
        test_mem = np.lib.format.open_memmap(
            test_path, mode="w+", dtype=np.float32, shape=(n_test, width8))
        test8 = test_mem
    for start in range(0, n_test, batch):
        stop = min(start + batch, n_test)
        codes = _encode_levels_device(corpus["test_images"][start:stop],
                                      table8, whitener, grid, LEVEL8)
        test8[start:stop] = codes
        if throttle > 0:
            time.sleep(throttle)
    del table8
    torch.cuda.empty_cache()

    print("8x8 ridge fit (penalty 1.0)", flush=True)
    w8 = acc8.solve_many([1.0])[1.0]
    std8 = acc8.standardiser()
    hits = 0
    for start in range(0, n_test, block):
        stop = min(start + block, n_test)
        xs = std8(np.asarray(test8[start:stop]))
        hits += int((np.argmax(xs @ w8[:-1] + w8[-1], axis=1)
                     == test_labels[start:stop]).sum())
    q8 = hits / n_test
    gain = q8 - SEALED_4X4
    passed = gain >= MARGIN
    print(f"  Q(8x8,481) = {q8:.6f} (gain vs sealed 4x4 {gain:+.6f})",
          flush=True)

    # ---- trained-head read at 138k (the dual-read control) -----------------
    print("trained-head read (138k)", flush=True)
    n_138 = min(n_part1, 138000)
    model = HeadOnly(width8, CLASSES, device)
    part1_mem_view = np.asarray(part1_mem[:n_138])
    order = np.random.default_rng(11).permutation(n_138)
    val_count = int(round(n_138 * 0.05))
    train_fit = order[val_count:]
    val_rows = order[:val_count]
    lab_138 = corpus["train_labels"][:n_138]

    def _batches(rows, labels_src, power=None):
        def gen():
            for start in range(0, len(rows), 64):
                take = rows[start:start + 64]
                block = np.asarray(part1_mem_view[take], dtype=np.float64)
                yield (torch.from_numpy(np.ascontiguousarray(
                    block.astype(np.float32))).to(device),
                    torch.from_numpy(labels_src[take]).to(device))
        return gen

    training = _train_with_schedule(
        model, _batches(train_fit, lab_138), _batches(val_rows, lab_138),
        4, 3e-4, 1e-4, device, 2)
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for start in range(0, n_test, 64):
            stop = min(start + 64, n_test)
            block = torch.from_numpy(np.ascontiguousarray(
                np.asarray(test8[start:stop], dtype=np.float32))).to(device)
            logits = model(block)
            correct += int((logits.argmax(dim=1)
                            == torch.from_numpy(test_labels[start:stop]
                                                ).to(device)).sum().item())
            total += stop - start
    trained_acc = correct / total
    print(f"  trained {trained_acc:.6f} (val "
          f"{training['best_validation_accuracy']:.6f})", flush=True)
    del model
    torch.cuda.empty_cache()

    macs8 = _head_macs(ATOMS_8, grid, dimension, LEVEL8)
    macs4 = _head_macs(ATOMS_SPM, grid, dimension, (4,))
    head_delta = macs8["head"] - macs4["head"]
    head_delta_frac = head_delta / macs4["head"]
    evidence.update({
        "anchors": anchors,
        "q8x8_481": {"accuracy": q8, "penalty": 1.0,
                     "per_image_macs": macs8,
                     "sealed_4x4_per_image_macs": macs4,
                     "head_cost_delta": head_delta,
                     "head_cost_delta_fraction": head_delta_frac,
                     "note": "64 x 481 x 345 = 10,620,480 vs "
                             "16 x 1,923 x 345 = 10,614,960: the 8x8 "
                             "arm pays +0.052%, within the family's "
                             "0.5% cost-tolerance rule (registered)."},
        "trained_head_read": trained_acc,
        "trained_val": training["best_validation_accuracy"],
        "gate": {
            "registered": config["gate"]["registered"],
            "incumbent": SEALED_4X4,
            "gain": gain,
            "required": MARGIN,
            "passed": bool(passed),
            "consequence": (config["gate"].get("consequence_passed",
                                               "passed") if passed
                            else config["gate"].get("consequence_fired",
                                                    "fired")),
        },
        "runtime_seconds": round(time.time() - started, 2),
    })
    _write(output_dir, evidence)
    print(f"\nM158 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def _write(output_dir: Path, evidence: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m158(args.config, args.output)


if __name__ == "__main__":
    main()
