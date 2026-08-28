"""M140 — the data-axis extension past 138,000 rows.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v21.md`` (M140, amended
13 Aug 2026 before measurement) and
``experiments/configs/v16/m140_data_extension.json``.

Question. Data is the steep lever (M116). The subsample caps at 400 train rows
per class. Does Q(6144, n) keep rising past 138,000 on the same class schedule,
and does the crossover vs dense r42 hold or widen?

Extension rule (registered): per class, the first (cap - 400) raw-train rows of
that class (raw array order) not in the sealed 400-row subsample; caps
{450, 600} -> n in {172,500, 207,000}. Whitener and dictionary are the SEALED
M117 construction, unchanged - only rows are added.

Premise gates (GATING, ahead of every operand): (a) per-class raw availability
must cover the cap - infeasible cells are recorded void, never silently
dropped; (b) the fresh encoder must reproduce the sealed f6144 memmaps on
encoder_check_rows subsample rows (max-abs delta <= 1e-5) before any new-row
encode is trusted. t1: the sealed memmaps fit at n = 138,000 reproduces 0.22487.

Kill switch: Q(6144, 207000) - 0.22487 >= +0.005 required.

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m140_data_extension
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m140_data_extension.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m140_data_extension"

T1_TOLERANCE = 0.002
KS_MARGIN = 0.005
T1_REFERENCE = 0.2248695652173913
DENSE_R42 = 0.1972
ATOMS = 6144
CLASSES = 345
SUBSAMPLE_PER_CLASS = 400


def _extension_indices(raw_train_labels: np.ndarray,
                       subsample_indices: np.ndarray, cap: int,
                       classes: int) -> tuple[np.ndarray, dict[int, int]]:
    """Per class, the first (cap - 400) raw rows not in the subsample.

    Returns (extension raw indices, per-class shortfall counts).
    """
    used = set(int(i) for i in subsample_indices)
    per_class_used = {c: 0 for c in range(classes)}
    for i in subsample_indices:
        per_class_used[int(raw_train_labels[i])] += 1
    extension: list[int] = []
    shortfall: dict[int, int] = {}
    for c in range(classes):
        raw_idx = np.flatnonzero(raw_train_labels == c)
        added = 0
        need = cap - per_class_used[c]
        for i in raw_idx:
            if int(i) not in used:
                extension.append(int(i))
                used.add(int(i))
                added += 1
                if added >= need:
                    break
        if added < need:
            shortfall[c] = need - added
    return np.array(extension, dtype=np.int64), shortfall


def run_m140(config_path: Path, output_dir: Path) -> dict[str, Any]:
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

    caps = ([int(c) for c in config.get("_smoke_caps", [])]
            or [int(c) for c in config["extension"]["class_caps"]])

    print("building whitener + 6144-atom dictionary (M117 exact)", flush=True)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)
    dictionary = _random_dictionary(candidates, len(candidates),
                                    int(config["corpus"]["shuffle_seed"]), ATOMS)
    pool_grid = 2

    # sealed codes
    codes_dir = data_cache_root() / config["sealed_codes"]["cache_relpath"]
    mem_train = np.load(codes_dir / config["sealed_codes"]["train_file"], mmap_mode="r")
    mem_test = np.load(codes_dir / config["sealed_codes"]["test_file"], mmap_mode="r")
    width = int(config["sealed_codes"]["width"])
    if mem_train.shape != (len(corpus["train_labels"]), width):
        raise SystemExit(f"sealed train memmap shape {mem_train.shape} != expected")

    # ---- premise gate (b): the fresh encoder reproduces the sealed codes ----
    check_rows = int(config["extension"]["encoder_check_rows"])
    print(f"encoder instrument check: {check_rows} subsample rows", flush=True)
    check_block = _write_frozen_codes(
        corpus, dictionary, whitener, pool_grid, device,
        np.arange(check_rows), codes_dir / "m140_encoder_check.npy",
        split="train", throttle_seconds=throttle)
    delta = float(np.abs(
        np.asarray(mem_train[:check_rows], dtype=np.float64)
        - np.asarray(check_block, dtype=np.float64)).max())
    encoder_ok = delta <= float(config["extension"]["encoder_check_tolerance"])
    print(f"  encoder check max-abs delta {delta:.3e} (ok={encoder_ok})", flush=True)
    if not encoder_ok and not smoke:
        evidence = {"milestone": "M140", "admissible_as_evidence": False,
                    "void": True,
                    "void_reason": "encoder instrument check failed (fresh encode "
                                   "does not reproduce the sealed memmaps)",
                    "encoder_check_delta": delta}
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
    evidence: dict[str, Any] = {
        "milestone": "M140",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "encoder_check": {"rows": check_rows, "max_abs_delta": delta,
                          "ok": encoder_ok},
        "cells": {},
        "premise": {},
    }
    if not smoke:
        t1_delta = q_138 - T1_REFERENCE
        evidence["t1"] = {"measured": q_138, "reference": T1_REFERENCE,
                          "delta": t1_delta, "tolerance": T1_TOLERANCE}
        if abs(t1_delta) > T1_TOLERANCE:
            evidence["void"] = True
            evidence["void_reason"] = "t1 anchor reproduction failed"
            output_dir.mkdir(parents=True, exist_ok=True)
            write_canonical_json(output_dir / "evidence.json", evidence)
            build_artifact_index(output_dir)
            return evidence
        print(f"  t1 delta {t1_delta:+.6f} (<= {T1_TOLERANCE})", flush=True)

    # ---- extension cells ------------------------------------------------------
    cache = data_cache_root() / "v16" / "m140"
    cache.mkdir(parents=True, exist_ok=True)
    for cap in caps:
        n_cell = CLASSES * cap
        print(f"cap {cap} -> n={n_cell}", flush=True)
        ext_rows, shortfall = _extension_indices(
            raw["train_labels"], train_index, cap, CLASSES)
        if shortfall:
            evidence["premise"][str(cap)] = {
                "infeasible": True,
                "shortfall_classes": shortfall,
                "note": "premise gate fired: the cell is recorded VOID, never silently dropped",
            }
            evidence["cells"][str(n_cell)] = {"void": True,
                                              "void_reason": "per-class raw availability below cap"}
            print(f"  VOID: {len(shortfall)} classes short of {cap} rows", flush=True)
            continue
        need = n_cell - len(mem_train)
        print(f"  encoding {len(ext_rows)} extension rows (need {need})", flush=True)
        raw_corpus = {"train_images": raw["train_images"]}
        ext_path = cache / f"f6144_ext{cap}.npy"
        ext_mem = _write_frozen_codes(
            raw_corpus, dictionary, whitener, pool_grid, device,
            ext_rows[:need], ext_path, split="train",
            throttle_seconds=throttle)
        acc = RidgeAccumulator(width, CLASSES)
        for start in range(0, len(mem_train), block):
            stop = min(start + block, len(mem_train))
            acc.add(np.asarray(mem_train[start:stop]),
                    corpus["train_labels"][start:stop])
        for start in range(0, len(ext_mem), block):
            stop = min(start + block, len(ext_mem))
            acc.add(np.asarray(ext_mem[start:stop]),
                    raw["train_labels"][ext_rows[:need]][start:stop])
        result = _solve_and_score(
            acc, [1.0],
            _test_blocks(mem_test, corpus["test_labels"], corpus["test_domains"], block))
        q_ext = result["accuracy_by_penalty"]["1.0"]
        evidence["cells"][str(n_cell)] = {
            "cap": cap,
            "extension_rows": int(len(ext_mem)),
            "accuracy": q_ext,
            "gain_vs_138000": q_ext - q_138,
            "crossover_vs_dense_r42": q_ext - DENSE_R42,
            "per_domain": [
                c / r for c, r in zip(
                    result["per_domain_correct"]["1.0"],
                    result["per_domain_rows"]["1.0"])],
        }
        print(f"  Q({ATOMS}, {n_cell}) = {q_ext:.4f} "
              f"(gain {q_ext - q_138:+.4f}, crossover {q_ext - DENSE_R42:+.4f})",
              flush=True)

    if not smoke:
        final_n = CLASSES * max(caps)
        final_cell = evidence["cells"].get(str(final_n))
        if final_cell and not final_cell.get("void"):
            gain = final_cell["gain_vs_138000"]
            fired = gain < KS_MARGIN
            evidence["gate"] = {
                "registered": config["gate"]["kill_switch_data"],
                "gain": gain,
                "required": KS_MARGIN,
                "fired": fired,
                "consequence": ("data axis saturates at the corpus cap (negative "
                                "sealed)" if fired else
                                "data extension is a measured lever; escalate"),
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
    run_m140(Path(args.config), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
