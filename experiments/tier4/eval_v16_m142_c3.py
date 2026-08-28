"""M142 cell C3 — multi-scale patches (3/5/7) vs a single 6x6 scale, at
matched per-image cost.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v22.md`` (execution
log and the Remaining-milestones recipes, 14 Aug 2026, after the sealed C2
and C4 PASSes).

Question. Does a three-scale patch construction (3x3 + 5x5 + 7x7, one 2x2
pool per scale, concatenated) beat the single-scale 6x6 pool construction
at matched per-image cost?

Construction. Per scale s in {3, 5, 7}: a ZCA whitener fitted on that
scale's patch pool (up to 20,000 subsample train images, 400,000 patches,
seed 11, contrast_epsilon 10.0, zca_epsilon 0.1 — the M117/M126 exact
recipe per scale); a candidate pool of 8,192 (the seeded [11,100]
permutation of that scale's whitened patch pool); the triangle code with a
single 2x2 sum pool over that scale's grid (the m107 ``_pool`` edges rule
``round(grid_s*i/2)``); scales concatenated.

Matched-cost atoms. The same ledger, pool adds counted on every arm
(whitening_s + cdist_s + pool adds_s + head 4*a_s*345 per scale). The atom
budget = (pool2062 total 175,197,198 - per-scale whitening sum) split so
each scale gets an EQUAL MAC share (registered rule). Registered split:
a3=1950, a5=850, a7=511 (total 3,311 atoms; width 13,244, Gram-feasible);
MS total = 175,153,892 vs pool 175,197,198 (delta 0.025%).

Anchors:
- t1 encoders: (a) in-run bitwise determinism of the multi-scale encode on
  check rows; (b) the per-scale pool edges rule is pinned by unit tests
  (grids 30/28/26, pool 2). The construction reuses the sealed cdist /
  activation / pooling arithmetic per scale; there is no sealed multi-scale
  artifact to reproduce (registered limitation).
- t2 environment: the direct ridge on the sealed f6144 codes at full data
  reproduces Q(6144, 409832) = 0.26136231884058 within 0.002.
- t4 premise: the cached pool2062 codes refit at full data reproduce their
  sealed C2 read Q_pool(2062, 409832) = 0.227536 within 0.002.

Gate (kill switch). Q_MS(409832) >= Q_pool(2062, 409832) + 0.005 at the
sealed head constant (penalty 1.0), at matched cost (|MS - pool| <= 0.5%).
The ridge ladder {0.1, 1.0, 10.0} is reported; per-scale diagnostics and
the trained-head read are the co-adaptation controls; the cell closes as a
scoped negative only if BOTH the gate fires AND the trained-head read
fails. The power-norm composition (multi-scale + signed sqrt + L2) is a
registered FOLLOW-UP free cell, not part of this gate.

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m142_c3
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
from experiments.tier4.eval_v15_m103_atoms import (
    Whitener,
    _contrast_normalise,
    _extract_patches,
    _fit_zca,
)
from experiments.tier4.eval_v15_m104_experts import (
    RidgeAccumulator,
    _chunk_rows,
    _load_domainnet,
    _score,
)
from experiments.tier4.eval_v15_m107_dense import (
    _solve_and_score,
    _verify_pixel_identity,
)
from experiments.tier4.eval_v16_m108_dictionary import (
    _encode_block_device,
    _random_order,
    _verify_device,
)
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m136_margin_head import _test_blocks
from experiments.tier4.eval_v16_m140_data_extension import _extension_indices
from experiments.tier4.eval_v16_m141_data_full import _rest_extension_indices

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m142_c3.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m142_c3"

CLASSES = 345
SCALES = (3, 5, 7)
T1_TOLERANCE = 0.002
KS_MARGIN = 0.005
COST_TOLERANCE = 0.005


# --------------------------------------------------------------------------
# per-scale construction and the matched-cost split
# --------------------------------------------------------------------------
def _scale_grid(size: int, patch: int) -> int:
    return (size - patch) + 1


def _scale_macs(atoms_by_scale: dict[int, int], size: int,
                classes: int) -> dict[str, Any]:
    """The multi-scale ledger; pool adds counted on every scale."""
    total = 0
    per_scale: dict[str, Any] = {}
    for patch, atoms in sorted(atoms_by_scale.items()):
        grid = _scale_grid(size, patch)
        dimension = patch * patch * 3
        patches = grid * grid
        whitening = patches * dimension * dimension
        encoding = patches * atoms * dimension
        pool_adds = patches * atoms
        head = 4 * atoms * classes
        total += whitening + encoding + pool_adds + head
        per_scale[str(patch)] = {
            "atoms": atoms, "grid": grid, "dimension": dimension,
            "whitening": whitening, "encoding": encoding,
            "pool_adds": pool_adds, "head": head,
        }
    per_scale["total"] = total
    return per_scale


def _scale_atom_split(pool_total: int, size: int,
                      classes: int) -> dict[int, int]:
    """Equal MAC share per scale against the pool arm's total."""
    whitening_total = 0
    weights: dict[int, int] = {}
    for patch in SCALES:
        grid = _scale_grid(size, patch)
        dimension = patch * patch * 3
        patches = grid * grid
        whitening_total += patches * dimension * dimension
        weights[patch] = patches * dimension + patches + 4 * classes
    budget = pool_total - whitening_total
    share = budget / len(SCALES)
    return {patch: int(round(share / weights[patch])) for patch in SCALES}


def _build_scale_whitener(config: dict[str, Any], corpus: dict[str, Any],
                          patch: int) -> tuple[Whitener, np.ndarray]:
    """M117/M126's exact per-scale recipe: whitener + 8,192 candidate pool."""
    rep = config["sparse"]
    size = int(config["corpus"]["image_size"])
    rng = np.random.default_rng(int(rep["zca_fit_seed"]))
    sample = corpus["train_images"][
        rng.choice(len(corpus["train_images"]),
                   min(len(corpus["train_images"]), 20_000), replace=False)
    ]
    patches = _extract_patches(sample, patch, 1)
    grid = _scale_grid(size, patch)
    take = min(int(rep["zca_fit_patches"]), len(patches))
    patch_pool = _contrast_normalise(
        patches[rng.choice(len(patches), take, replace=False)],
        float(rep["contrast_epsilon"]),
    )
    mean, whiten = _fit_zca(patch_pool, float(rep["zca_epsilon"]))
    whitener = Whitener(patch, 1, float(rep["contrast_epsilon"]),
                        mean, whiten, grid)
    seed_rng = np.random.default_rng(int(rep["dictionary_seed"]))
    pool_size = int(rep["candidate_pool_size"])
    candidates = ((patch_pool[
        seed_rng.choice(len(patch_pool), pool_size, replace=False)
    ] - mean) @ whiten).astype(np.float32)
    return whitener, candidates


def _scale_dictionary(candidates: np.ndarray, pool_size: int, seed: int,
                      atoms: int) -> np.ndarray:
    order = _random_order(candidates, pool_size, seed)
    return candidates[order[:atoms]]


def _append_scale_encode(images: np.ndarray, rows: np.ndarray,
                         dictionary: np.ndarray, whitener: Whitener,
                         device: torch.device, out: np.ndarray, offset: int,
                         col_start: int, throttle: float) -> int:
    """Encode one scale's rows into ``out`` columns [col_start:col_start+w]."""
    table = torch.from_numpy(np.ascontiguousarray(dictionary)).to(
        torch.float32).to(device)
    step = _chunk_rows(len(dictionary), whitener.grid, len(rows))
    width = 4 * len(dictionary)
    for start in range(0, len(rows), step):
        take = rows[start:start + step]
        stop = offset + len(take)
        out[offset:stop, col_start:col_start + width] = _encode_block_device(
            images[take], table, whitener, 2)
        offset = stop
        if throttle > 0:
            time.sleep(throttle)
    return offset


def _fit_ladder(mem_train: np.ndarray, labels: np.ndarray, width: int,
                penalties: list[float], n_rows: int
                ) -> tuple[dict[str, np.ndarray], Any]:
    acc = RidgeAccumulator(width, CLASSES)
    for start in range(0, n_rows, 4096):
        stop = min(start + 4096, n_rows)
        acc.add(np.asarray(mem_train[start:stop]), labels[start:stop])
    solved = acc.solve_many(penalties)
    info = {"fit_rows": int(acc.rows), "width": int(width),
            "standardiser": acc.standardiser()}
    return ({str(p): w for p, w in solved.items()},
            {str(p): info for p in penalties})


def _score_weights(test_codes: np.ndarray, test_labels: np.ndarray,
                   test_domains: np.ndarray, weights: np.ndarray,
                   standardiser, block: int = 4096) -> dict[str, Any]:
    n = len(test_labels)
    hits = 0
    per_domain_correct = np.zeros(6, dtype=np.int64)
    per_domain_rows = np.zeros(6, dtype=np.int64)
    for start in range(0, n, block):
        stop = min(start + block, n)
        xs = standardiser(test_codes[start:stop]).astype(np.float64)
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


def _trained_head_read(train_mem: np.ndarray, train_labels: np.ndarray,
                       test_codes: np.ndarray, test_labels: np.ndarray,
                       epochs: int, lr: float, seed: int,
                       device: torch.device) -> float:
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
def run_m142_c3(config_path: Path, output_dir: Path) -> dict[str, Any]:
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

    pool_total = int(config["matched_cost"]["pool_total"])
    atoms_by_scale = {int(p): int(a) for p, a in config["sparse"]["atoms_by_scale"].items()}
    solved = _scale_atom_split(pool_total, size, CLASSES)
    if solved != atoms_by_scale:
        raise SystemExit(
            f"matched-MAC split failure: solver gives {solved}, config "
            f"registers {atoms_by_scale} for target {pool_total}.")
    ms_macs = _scale_macs(atoms_by_scale, size, CLASSES)
    cost_delta = abs(ms_macs["total"] - pool_total) / pool_total
    if cost_delta > COST_TOLERANCE:
        raise SystemExit(
            f"matched-cost check failed: |MS {ms_macs['total']} - pool "
            f"{pool_total}| = {cost_delta:.4%} > {COST_TOLERANCE:.4%}")
    ms_width = 4 * sum(atoms_by_scale.values())
    print(f"matched split: {atoms_by_scale} -> width {ms_width}, "
          f"MS total {ms_macs['total']} vs pool {pool_total} "
          f"({cost_delta:+.4%})", flush=True)

    print("building per-scale whiteners + dictionaries", flush=True)
    whiteners: dict[int, Whitener] = {}
    dictionaries: dict[int, np.ndarray] = {}
    for patch in SCALES:
        whitener, candidates = _build_scale_whitener(config, corpus, patch)
        atoms = atoms_by_scale[patch]
        whiteners[patch] = whitener
        dictionaries[patch] = _scale_dictionary(
            candidates, len(candidates), int(config["sparse"]["dictionary_seed"]),
            atoms)
        print(f"  scale {patch}: whitener grid {whitener.grid}, "
              f"{atoms} atoms", flush=True)

    evidence: dict[str, Any] = {
        "milestone": "M142",
        "cell": "C3 multi-scale patches 3/5/7 vs single 6x6 pool at matched "
                "cost",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "matched_cost": {"atoms_by_scale": atoms_by_scale,
                         "ms_macs": ms_macs,
                         "pool_total": pool_total,
                         "cost_delta_fraction": cost_delta},
    }

    # ---- t1: encoder determinism + edges (the edges rule is unit-tested) ---
    check_rows = int(config["anchors"]["t1_check_rows"])
    first = np.empty((check_rows, ms_width), dtype=np.float32)
    second = np.empty((check_rows, ms_width), dtype=np.float32)
    col = 0
    for patch in SCALES:
        rows = np.arange(check_rows)
        _append_scale_encode(corpus["train_images"], rows,
                             dictionaries[patch], whiteners[patch], device,
                             first, 0, col, throttle)
        _append_scale_encode(corpus["train_images"], rows,
                             dictionaries[patch], whiteners[patch], device,
                             second, 0, col, throttle)
        col += 4 * atoms_by_scale[patch]
    t1_delta = float(np.abs(first.astype(np.float64)
                            - second.astype(np.float64)).max())
    t1_ok = t1_delta <= float(config["anchors"]["t1_tolerance"])
    evidence["t1_determinism"] = {"rows": check_rows, "max_abs_delta": t1_delta,
                                  "ok": t1_ok}
    print(f"t1 determinism: max-abs delta {t1_delta:.3e} (ok={t1_ok})",
          flush=True)
    if not t1_ok and not smoke_skip:
        evidence["void"] = True
        evidence["void_reason"] = "t1 encoder determinism failed"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        return evidence
    del first, second

    # ---- train encode (the M141 cell-2 schedule) ---------------------------
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
            (corpus["train_images"], np.arange(len(train_index)),
             corpus["train_labels"]),
            (raw["train_images"], ext600_indices,
             raw["train_labels"][ext600_indices]),
            (raw["train_images"], rest_indices,
             raw["train_labels"][rest_indices]),
        ]
    n_train = sum(len(rows) for _, rows, _ in schedule)
    cache = data_cache_root() / "v16" / "m142_c3"
    cache.mkdir(parents=True, exist_ok=True)
    ms_path = cache / "ms357_fulltrain.npy"
    col_start: dict[int, int] = {}
    col = 0
    for patch in SCALES:
        col_start[patch] = col
        col += 4 * atoms_by_scale[patch]
    print(f"encoding multi-scale train ({n_train} rows)", flush=True)
    ms_mem = np.lib.format.open_memmap(
        ms_path, mode="w+", dtype=np.float32, shape=(n_train, ms_width))
    offset = 0
    for images, rows, _labels in schedule:
        for patch in SCALES:
            _append_scale_encode(images, rows, dictionaries[patch],
                                 whiteners[patch], device, ms_mem, offset,
                                 col_start[patch], throttle)
        offset += len(rows)
    del ms_mem
    ms_mem = np.load(ms_path, mmap_mode="r")
    labels_path = (data_cache_root() / "v16" / "m142_c2"
                   / "m142_c2_fulltrain_labels.npz")
    full_labels = np.load(labels_path)["labels"][:n_train]
    scheduled = np.concatenate([l for _, _, l in schedule])
    if len(full_labels) != n_train or not np.array_equal(full_labels,
                                                         scheduled):
        raise SystemExit("cached labels do not match the encode schedule")
    print(f"  train encode done ({time.time() - started:.0f}s so far)",
          flush=True)

    # ---- t2: environment anchor on the sealed f6144 codes -----------------
    sealed_mem = np.load(
        data_cache_root() / config["sealed_codes"]["cache_relpath"]
        / config["sealed_codes"]["train_file"], mmap_mode="r")
    sealed_test = np.load(
        data_cache_root() / config["sealed_codes"]["cache_relpath"]
        / config["sealed_codes"]["test_file"], mmap_mode="r")
    sealed_test_blocks = _test_blocks(sealed_test, corpus["test_labels"],
                                      corpus["test_domains"], block)
    print("t2: direct ridge on sealed f6144 codes (full data)", flush=True)
    sealed_parts = [sealed_mem]
    sealed_part_labels = [corpus["train_labels"]]
    if smoke:
        sealed_parts = [sealed_mem[:20000]]
        sealed_part_labels = [corpus["train_labels"][:20000]]
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
    acc_t2 = RidgeAccumulator(int(config["sealed_codes"]["width"]), CLASSES)
    off = 0
    for part in sealed_parts:
        for start in range(0, len(part), 4096):
            stop = min(start + 4096, len(part))
            acc_t2.add(np.asarray(part[start:stop]),
                       sealed_all_labels[off + start:off + stop])
        off += len(part)
    t2_result = _solve_and_score(acc_t2, [1.0], sealed_test_blocks)
    t2_acc = t2_result["accuracy_by_penalty"]["1.0"]
    t2_delta = t2_acc - float(config["anchors"]["t2_reference"])
    evidence["t2_full_data_direct"] = {
        "accuracy": t2_acc, "reference": config["anchors"]["t2_reference"],
        "delta": t2_delta}
    print(f"  t2 direct {t2_acc:.4f} vs sealed "
          f"{config['anchors']['t2_reference']} (delta {t2_delta:+.6f})",
          flush=True)
    if not smoke_skip and abs(t2_delta) > T1_TOLERANCE:
        evidence["void"] = True
        evidence["void_reason"] = "t2 anchor reproduction failed"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        return evidence

    # ---- t4: the cached pool2062 arm refit --------------------------------
    print("t4: pool2062 refit (the matched baseline)", flush=True)
    pool_mem = np.load(
        data_cache_root() / "v16" / "m142_c2" / "pool2062_fulltrain.npy",
        mmap_mode="r")
    if len(pool_mem) < n_train:
        raise SystemExit(f"pool memmap rows {len(pool_mem)} < {n_train}")
    # pool test codes: the full run persists them once; the smoke encodes
    # its tiny slice to RAM and never touches the persisted artifact
    n_test = len(corpus["test_labels"]) if not smoke else int(
        config["_smoke_test_rows"])
    pool_test_path = (data_cache_root() / "v16" / "m142_c2"
                      / "pool2062_fulltest.npy")
    pool_width = pool_mem.shape[1]
    if smoke:
        pool_whitener, pool_candidates = _build_scale_whitener(config, corpus,
                                                               6)
        dict_pool = _scale_dictionary(
            pool_candidates, 8192,
            int(config["sparse"]["dictionary_seed"]),
            int(config["sparse"]["pool_atoms"]))
        pool_test = np.empty((n_test, pool_width), dtype=np.float32)
        _append_scale_encode(corpus["test_images"], np.arange(n_test),
                             dict_pool, pool_whitener, device, pool_test, 0,
                             0, throttle)
    elif pool_test_path.exists() and np.load(
            pool_test_path, mmap_mode="r").shape[0] == n_test:
        pool_test = np.load(pool_test_path, mmap_mode="r")
    else:
        pool_whitener, pool_candidates = _build_scale_whitener(config, corpus,
                                                               6)
        dict_pool = _scale_dictionary(
            pool_candidates, 8192,
            int(config["sparse"]["dictionary_seed"]),
            int(config["sparse"]["pool_atoms"]))
        pool_test = np.lib.format.open_memmap(
            pool_test_path, mode="w+", dtype=np.float32,
            shape=(n_test, pool_width))
        _append_scale_encode(corpus["test_images"], np.arange(n_test),
                             dict_pool, pool_whitener, device, pool_test, 0,
                             0, throttle)
        del pool_test
        pool_test = np.load(pool_test_path, mmap_mode="r")
    test_labels = corpus["test_labels"][:n_test]
    test_domains = corpus["test_domains"][:n_test]
    pool_w, pool_std = _fit_ladder(pool_mem, full_labels,
                                   pool_mem.shape[1], [1.0], n_train)
    pool_acc = _score_weights(pool_test, test_labels, test_domains,
                              pool_w["1.0"],
                              pool_std["1.0"]["standardiser"])["accuracy"]
    t4_delta = pool_acc - float(config["anchors"]["t4_reference"])
    evidence["t4_pool_refit"] = {"accuracy": pool_acc,
                                 "reference": config["anchors"]["t4_reference"],
                                 "delta": t4_delta}
    print(f"  t4 pool refit {pool_acc:.4f} vs sealed C2 "
          f"{config['anchors']['t4_reference']} (delta {t4_delta:+.6f})",
          flush=True)
    if not smoke_skip and abs(t4_delta) > T1_TOLERANCE:
        evidence["void"] = True
        evidence["void_reason"] = "t4 pool refit failed"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        return evidence

    # ---- the multi-scale reads --------------------------------------------
    print("multi-scale ridge ladder (full data)", flush=True)
    ms_w, ms_info = _fit_ladder(
        ms_mem, full_labels, ms_width,
        [float(p) for p in config["cell_c3"]["penalty_ladder"]], n_train)
    n_138 = len(train_index) if not smoke else n_train
    ms_138_w, ms_138_std = _fit_ladder(ms_mem, full_labels, ms_width,
                                       [1.0], n_138)

    print("test encode to RAM for scoring", flush=True)
    ms_test = np.empty((n_test, ms_width), dtype=np.float32)
    for patch in SCALES:
        _append_scale_encode(corpus["test_images"], np.arange(n_test),
                             dictionaries[patch], whiteners[patch], device,
                             ms_test, 0, col_start[patch], throttle)

    per_scale: dict[str, Any] = {}
    for patch in SCALES:
        lo, hi = col_start[patch], col_start[patch] + 4 * atoms_by_scale[patch]
        w_s, std_s = _fit_ladder(ms_mem[:, lo:hi], full_labels, hi - lo,
                                 [1.0], n_train)
        per_scale[str(patch)] = _score_weights(
            ms_test[:, lo:hi], test_labels, test_domains, w_s["1.0"],
            std_s["1.0"]["standardiser"])
        print(f"  scale {patch}: {per_scale[str(patch)]['accuracy']:.4f}",
              flush=True)

    ms_ladder_acc = {}
    for p_str, w in ms_w.items():
        ms_ladder_acc[p_str] = _score_weights(
            ms_test, test_labels, test_domains, w,
            ms_info[p_str]["standardiser"])
    ms_138_acc = _score_weights(ms_test, test_labels, test_domains,
                                ms_138_w["1.0"], ms_138_std["1.0"]
                                ["standardiser"])

    trained_acc = None
    if not smoke:
        print("trained-head read on the multi-scale codes", flush=True)
        trained_acc = _trained_head_read(
            ms_mem, full_labels, ms_test, test_labels,
            int(config["cell_c3"]["trained_epochs"]),
            float(config["cell_c3"]["trained_lr"]),
            int(config["cell_c3"]["trained_seed"]), device)

    # ---- gate ---------------------------------------------------------------
    ms_gate = ms_ladder_acc["1.0"]["accuracy"]
    gain = ms_gate - pool_acc
    fired = (not smoke) and (gain < KS_MARGIN)
    both_fail = fired and (trained_acc is not None
                           and trained_acc < pool_acc + KS_MARGIN)
    evidence["reads"] = {
        "ms_ladder": ms_ladder_acc,
        "ms_138k": ms_138_acc,
        "per_scale": per_scale,
        "trained_head_read": trained_acc,
    }
    evidence["gate"] = {
        "registered": config["cell_c3"]["gate_registered"],
        "ms_penalty1_full": ms_gate,
        "pool_penalty1_full": pool_acc,
        "gain": gain,
        "required": KS_MARGIN,
        "fired": fired,
        "consequence": (config["cell_c3"]["consequence_fired"] if fired
                        else config["cell_c3"]["consequence_passed"]),
        "closure_note": ("scoped negative requires BOTH reads to fail; "
                         f"both_fail={bool(both_fail)}"),
    }
    evidence["runtime_seconds"] = round(time.time() - started, 2)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM142 C3 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m142_c3(args.config, args.output)


if __name__ == "__main__":
    main()
