"""M141 — the data-axis escalation: uniform cap 612 and the all-available corpus.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v21.md`` section 5 and
``experiments/configs/v16/m141_data_full.json``.

Question. M140 PASSED (+50% rows -> +0.0161). Does the data axis keep paying to
the corpus's full extent?

Cells (same sealed M117 construction, rows only):
1. n = 211,140 (uniform cap 612: every class gets 212 more rows).
2. n = 409,832 (ALL available raw rows per class, non-uniform schedule,
   disclosed). The first 69,000 extension rows are exactly the sealed M140
   ext600 selection, REUSED from the sealed ext600 memmap behind a t2 anchor.

Anchors: t1 Q(6144, 138000) = 0.22487 (0.002); t2 the reused-ext600 fit
reproduces the sealed M140 Q(6144, 207000) = 0.240667 (0.002); encoder check
bit-exact (<= 1e-5) on 256 subsample rows.

Kill switch: Q(6144, 409832) - 0.240667 >= +0.005 required.

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m141_data_full
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
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator, _load_domainnet
from experiments.tier4.eval_v15_m107_dense import _class_subsample, _solve_and_score, _verify_pixel_identity
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m113_learned import (
    _build_whitener_and_candidates,
    _random_dictionary,
)
from experiments.tier4.eval_v16_m115_lofi import _write_frozen_codes
from experiments.tier4.eval_v16_m136_margin_head import _test_blocks
from experiments.tier4.eval_v16_m140_data_extension import _extension_indices

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m141_data_full.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m141_data_full"

T1_TOLERANCE = 0.002
KS_MARGIN = 0.005
T1_REFERENCE = 0.2248695652173913
T2_REFERENCE = 0.24066666666666667
DENSE_R42 = 0.1972
ATOMS = 6144
CLASSES = 345
SUBSAMPLE_PER_CLASS = 400


def _all_extension_indices(raw_train_labels: np.ndarray,
                           subsample_indices: np.ndarray,
                           classes: int) -> np.ndarray:
    """Per class, ALL raw rows (raw array order) not in the subsample."""
    used = set(int(i) for i in subsample_indices)
    extension: list[int] = []
    for c in range(classes):
        for i in np.flatnonzero(raw_train_labels == c):
            if int(i) not in used:
                extension.append(int(i))
                used.add(int(i))
    return np.array(extension, dtype=np.int64)


def _rest_extension_indices(raw_train_labels: np.ndarray,
                            subsample_indices: np.ndarray,
                            classes: int, per_class_take: int) -> np.ndarray:
    """Per class, the unused raw rows AFTER the first *per_class_take*, in raw
    order — the rows the M140 ext600 selection did not use."""
    used = set(int(i) for i in subsample_indices)
    rest: list[int] = []
    for c in range(classes):
        taken = 0
        for i in np.flatnonzero(raw_train_labels == c):
            if int(i) not in used:
                if taken < per_class_take:
                    taken += 1
                else:
                    rest.append(int(i))
    return np.array(rest, dtype=np.int64)


def _fit_cell(acc: RidgeAccumulator, mems: list[np.ndarray],
              labels_list: list[np.ndarray], block: int,
              mem_test: np.ndarray, test_labels: np.ndarray,
              test_domains: np.ndarray) -> dict[str, Any]:
    for mem, labels in zip(mems, labels_list):
        for start in range(0, len(mem), block):
            stop = min(start + block, len(mem))
            acc.add(np.asarray(mem[start:stop]), labels[start:stop])
    result = _solve_and_score(
        acc, [1.0],
        _test_blocks(mem_test, test_labels, test_domains, block))
    return {
        "accuracy": result["accuracy_by_penalty"]["1.0"],
        "per_domain": [c / r if r else 0.0 for c, r in zip(
            result["per_domain_correct"]["1.0"],
            result["per_domain_rows"]["1.0"])],
    }


def run_m141(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if "_smoke_note" in config and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    torch.set_num_threads(int(config["numerics"]["torch_threads"]))
    torch.manual_seed(int(config["numerics"]["seed"]))
    configure_external_cache_environment()
    _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    smoke = bool(config.get("_smoke_skip_gates", False))
    block = int(config["numerics"]["block"])
    throttle = float(config["numerics"]["encode_throttle_seconds"])
    started = time.time()

    print("loading raw domainnet + subsample corpus", flush=True)
    raw = _load_domainnet(int(config["corpus"]["image_size"]))
    corpus, train_index, test_index = _load_corpus(config)
    size = int(config["corpus"]["image_size"])
    for split, idx in (("train", train_index), ("test", test_index)):
        _verify_pixel_identity(split, idx, corpus[f"{split}_images"], size,
                               int(config["corpus"]["pixel_identity_rows"]))

    print("building whitener + 6144-atom dictionary (M117 exact)", flush=True)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)
    dictionary = _random_dictionary(candidates, len(candidates),
                                    int(config["sparse"]["dictionary_seed"]), ATOMS)
    pool_grid = int(config["sparse"]["pool_grid"])

    codes_dir = data_cache_root() / config["sealed_codes"]["cache_relpath"]
    mem_train = np.load(codes_dir / config["sealed_codes"]["train_file"], mmap_mode="r")
    mem_test = np.load(codes_dir / config["sealed_codes"]["test_file"], mmap_mode="r")
    width = int(config["sealed_codes"]["width"])

    evidence: dict[str, Any] = {
        "milestone": "M141",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "cells": {},
    }

    # ---- encoder instrument check -------------------------------------------
    check_rows = int(config["extension"]["encoder_check_rows"])
    print(f"encoder instrument check: {check_rows} subsample rows", flush=True)
    check_block = _write_frozen_codes(
        corpus, dictionary, whitener, pool_grid, device,
        np.arange(check_rows), codes_dir / "m141_encoder_check.npy",
        split="train", throttle_seconds=throttle)
    delta = float(np.abs(
        np.asarray(mem_train[:check_rows], dtype=np.float64)
        - np.asarray(check_block, dtype=np.float64)).max())
    encoder_ok = delta <= float(config["extension"]["encoder_check_tolerance"])
    evidence["encoder_check"] = {"rows": check_rows, "max_abs_delta": delta,
                                 "ok": encoder_ok}
    print(f"  encoder check max-abs delta {delta:.3e} (ok={encoder_ok})", flush=True)
    if not encoder_ok and not smoke:
        evidence["void"] = True
        evidence["void_reason"] = "encoder instrument check failed"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        return evidence

    # ---- t1: sealed codes at n=138000 ----------------------------------------
    print("t1: sealed f6144 codes at n=138000", flush=True)
    acc_t1 = RidgeAccumulator(width, CLASSES)
    for start in range(0, len(mem_train), block):
        stop = min(start + block, len(mem_train))
        acc_t1.add(np.asarray(mem_train[start:stop]),
                   corpus["train_labels"][start:stop])
    t1_result = _solve_and_score(
        acc_t1, [1.0],
        _test_blocks(mem_test, corpus["test_labels"], corpus["test_domains"], block))
    q_138 = t1_result["accuracy_by_penalty"]["1.0"]
    if not smoke:
        t1_delta = q_138 - T1_REFERENCE
        evidence["t1"] = {"measured": q_138, "delta": t1_delta,
                          "tolerance": T1_TOLERANCE}
        if abs(t1_delta) > T1_TOLERANCE:
            evidence["void"] = True
            evidence["void_reason"] = "t1 anchor reproduction failed"
            output_dir.mkdir(parents=True, exist_ok=True)
            write_canonical_json(output_dir / "evidence.json", evidence)
            build_artifact_index(output_dir)
            return evidence
        print(f"  t1 delta {t1_delta:+.6f} (<= {T1_TOLERANCE})", flush=True)

    cache = data_cache_root() / "v16" / "m141"
    cache.mkdir(parents=True, exist_ok=True)
    all_indices = _all_extension_indices(raw["train_labels"], train_index, CLASSES)
    evidence["premise"] = {
        "total_raw_train_rows": int(len(raw["train_labels"])),
        "all_extension_rows": int(len(all_indices)),
        "per_class_raw_min": int(np.bincount(raw["train_labels"],
                                             minlength=CLASSES).min()),
        "schedule_note": config["extension"]["all_available_note"],
    }

    # ---- cell 1: uniform cap (smoke: _smoke_cap) ------------------------------
    if smoke:
        cap = int(config["_smoke_cap"])
    else:
        cap = int(config["extension"]["uniform_cap"])
    n_cell1 = CLASSES * cap
    need1 = n_cell1 - len(mem_train)
    # The uniform cell takes (cap - 400) rows PER CLASS (the M140 rule), not the
    # first `need1` rows of the per-class-concatenated all-extension order.
    cell1_indices, _ = _extension_indices(raw["train_labels"], train_index, cap,
                                          CLASSES)
    cell1_indices = cell1_indices[:need1]
    print(f"cell 1: uniform cap {cap} -> n={n_cell1}, encoding {len(cell1_indices)} rows",
          flush=True)
    cell1_mem = _write_frozen_codes(
        {"train_images": raw["train_images"]}, dictionary, whitener, pool_grid,
        device, cell1_indices, cache / f"f6144_cap{cap}.npy",
        split="train", throttle_seconds=throttle)
    cell1 = _fit_cell(
        RidgeAccumulator(width, CLASSES),
        [mem_train, cell1_mem],
        [corpus["train_labels"], raw["train_labels"][cell1_indices]],
        block, mem_test, corpus["test_labels"], corpus["test_domains"])
    cell1["gain_vs_138000"] = cell1["accuracy"] - q_138
    cell1["crossover_vs_dense_r42"] = cell1["accuracy"] - DENSE_R42
    evidence["cells"][str(n_cell1)] = cell1
    print(f"  Q({ATOMS}, {n_cell1}) = {cell1['accuracy']:.4f} "
          f"(gain {cell1['gain_vs_138000']:+.4f})", flush=True)

    if smoke:
        evidence["runtime_seconds"] = round(time.time() - started, 2)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        print(f"wrote {output_dir / 'evidence.json'}", flush=True)
        return evidence

    # ---- t2: reused sealed ext600 -> Q(6144, 207000) --------------------------
    print("t2: reused sealed ext600 memmap -> n=207000", flush=True)
    ext600_path = (data_cache_root() / config["sealed_codes"]["ext600_relpath"]
                   / config["sealed_codes"]["ext600_file"])
    if not ext600_path.exists():
        raise SystemExit(f"sealed ext600 memmap missing: {ext600_path}")
    ext600 = np.load(ext600_path, mmap_mode="r")
    if len(ext600) != 69000:
        raise SystemExit(f"ext600 rows {len(ext600)} != 69000")
    # The ext600 selection is exactly (600 - 400) = 200 rows PER CLASS in raw
    # order (the M140 rule) — the same rule, not the front-loaded prefix.
    ext600_indices, _ = _extension_indices(raw["train_labels"], train_index, 600,
                                           CLASSES)
    if len(ext600_indices) != 69000:
        raise SystemExit(f"ext600 selection {len(ext600_indices)} != 69000")
    t2 = _fit_cell(
        RidgeAccumulator(width, CLASSES),
        [mem_train, ext600],
        [corpus["train_labels"], raw["train_labels"][ext600_indices]],
        block, mem_test, corpus["test_labels"], corpus["test_domains"])
    t2_delta = t2["accuracy"] - T2_REFERENCE
    evidence["t2"] = {"measured": t2["accuracy"], "reference": T2_REFERENCE,
                      "delta": t2_delta, "tolerance": T1_TOLERANCE}
    if abs(t2_delta) > T1_TOLERANCE:
        evidence["void"] = True
        evidence["void_reason"] = "t2 anchor reproduction failed"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        return evidence
    print(f"  t2 delta {t2_delta:+.6f} (<= {T1_TOLERANCE})", flush=True)

    # ---- cell 2: all available (reuse ext600 + fresh rest) --------------------
    rest_indices = _rest_extension_indices(raw["train_labels"], train_index,
                                           CLASSES, per_class_take=200)
    n_cell2 = len(mem_train) + len(ext600_indices) + len(rest_indices)
    print(f"cell 2: all available n={n_cell2}; reusing 69000, encoding "
          f"{len(rest_indices)}", flush=True)
    rest_mem = _write_frozen_codes(
        {"train_images": raw["train_images"]}, dictionary, whitener, pool_grid,
        device, rest_indices, cache / "f6144_all_rest.npy",
        split="train", throttle_seconds=throttle)
    cell2 = _fit_cell(
        RidgeAccumulator(width, CLASSES),
        [mem_train, ext600, rest_mem],
        [corpus["train_labels"], raw["train_labels"][ext600_indices],
         raw["train_labels"][rest_indices]],
        block, mem_test, corpus["test_labels"], corpus["test_domains"])
    cell2["gain_vs_207000"] = cell2["accuracy"] - T2_REFERENCE
    cell2["gain_vs_138000"] = cell2["accuracy"] - q_138
    cell2["crossover_vs_dense_r42"] = cell2["accuracy"] - DENSE_R42
    evidence["cells"][str(n_cell2)] = cell2
    print(f"  Q({ATOMS}, {n_cell2}) = {cell2['accuracy']:.4f} "
          f"(gain vs 207000 {cell2['gain_vs_207000']:+.4f})", flush=True)

    fired = cell2["gain_vs_207000"] < KS_MARGIN
    evidence["gate"] = {
        "registered": config["gate"]["kill_switch_data_full"],
        "gain_vs_207000": cell2["gain_vs_207000"],
        "required": KS_MARGIN,
        "fired": fired,
        "consequence": ("the data axis saturates between 207k and the full "
                        "corpus (negative sealed)" if fired else
                        "the data lever holds to the corpus's full extent"),
    }
    evidence["runtime_seconds"] = round(time.time() - started, 2)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"wrote {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    run_m141(Path(args.config), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
