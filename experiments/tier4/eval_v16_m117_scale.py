"""M117 — 2D scaling surface Q_S(C, n).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v18.md`` section 5.5 and
``experiments/configs/v16/m117_scale.json``.

Question. M108 measured the atoms axis at fixed n (24x more atoms buys ~+10
points, flattening); M116 measured the n axis at fixed atoms (steep; overtakes
the cost-matched dense trunk). If the two axes interact positively — more
atoms use more data productively — the joint surface Q_S(C, n) rises
super-additively and the 0.2153 "ceiling" was a slice artifact of holding one
axis fixed. If atoms and data are separable resources, the ceiling is real.

Arms. The joint 3x3 surface: atoms in {1536, 3072, 6144} (prefixes of the
seeded permutation of the shared 8192-whitened-patch pool, so all are slices
of the SAME random dictionary) x n in {34500, 69000, 138000} (first-n-rows of
the M107-shuffled train order, nested). Codes are image functions: train/test
codes are encoded ONCE per atom count into D: memmaps, then a closed-form
ridge (penalty 1.0) is fitted at each n from the first n rows. All cells score
the SAME full 34500-row test set. M120 (head-width mechanism) is folded in:
the atoms axis IS the head-width axis (width = 4*atoms), and the per-atom
Q(n) steepness is reported.

Gates:
- t1: (atoms=3072, n=138000) reproduces M113's sealed random-3072 (0.2153)
  within 0.002, or the run voids.
- KS (super-additivity) at cell (1536, 34500): joint gain
  Q(3072,69000) - Q(1536,34500) must exceed
  [Q(3072,34500) - Q(1536,34500)] + [Q(1536,69000) - Q(1536,34500)] by +0.005,
  else fired (atoms and data are separable; the ceiling is a real slice).
  Reported: the same test at cell (3072, 69000).

Reproduce with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m117_scale
"""
from __future__ import annotations

import argparse
import json
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
    _training_macs,
)
from experiments.tier4.eval_v15_m107_dense import (
    _solve_and_score,
    _verify_pixel_identity,
)
from experiments.tier4.eval_v16_m108_dictionary import (
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m117_scale.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m117_scale"
M113_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m113_learned" / "evidence.json"

PATCH_DIM = 108
T1_TOLERANCE = 0.002
KS_MARGIN = 0.005
BLOCK = 4096


def _fit_and_score(mem_train: np.ndarray, mem_test: np.ndarray,
                   labels: np.ndarray, test_labels: np.ndarray,
                   test_domains: np.ndarray, classes: int,
                   n: int) -> dict[str, Any]:
    acc = RidgeAccumulator(mem_train.shape[1], classes)
    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        acc.add(np.asarray(mem_train[start:stop]), labels[start:stop])
    n_test = len(test_labels)

    def _blocks() -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for start in range(0, n_test, BLOCK):
            stop = min(start + BLOCK, n_test)
            yield (np.asarray(mem_test[start:stop]),
                   test_labels[start:stop], test_domains[start:stop])

    return _solve_and_score(acc, [1.0], _blocks())


def run_m117(config_path: Path, output_dir: Path) -> dict[str, Any]:
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
    smoke_skip = bool(config.get("_smoke_skip_gates", False))
    gates: dict[str, Any] = {}

    print("building global whitener + candidate pool (M108 exact)", flush=True)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)

    cache = data_cache_root() / "v16" / "m117"
    cache.mkdir(parents=True, exist_ok=True)
    n_train = len(corpus["train_labels"])
    n_test = len(corpus["test_labels"])
    train_rows = np.arange(n_train)
    test_rows = np.arange(n_test)

    surface: dict[str, Any] = {}
    for atoms in atoms_ladder:
        dictionary = _random_dictionary(candidates, len(candidates),
                                        int(rep["dictionary_seed"]), atoms)
        print(f"  atoms={atoms}: encoding train codes", flush=True)
        mem_train = _write_frozen_codes(corpus, dictionary, whitener, pool_grid,
                                        device, train_rows,
                                        cache / f"f{atoms}_train.npy",
                                        split="train")
        print(f"  atoms={atoms}: encoding test codes", flush=True)
        mem_test = _write_frozen_codes(corpus, dictionary, whitener, pool_grid,
                                       device, test_rows,
                                       cache / f"f{atoms}_test.npy",
                                       split="test")
        cells = {}
        for n in n_ladder:
            r = _fit_and_score(mem_train, mem_test, corpus["train_labels"],
                               corpus["test_labels"], corpus["test_domains"],
                               classes, n)
            acc = r["accuracy_by_penalty"]["1.0"]
            pc = r["per_domain_correct"]["1.0"]
            pr = r["per_domain_rows"]["1.0"]
            cells[str(n)] = {
                "accuracy": acc,
                "per_domain": [pc[d] / pr[d] for d in range(6)],
                "training_ops": int(_training_macs(n, atoms, whitener.grid,
                                                   PATCH_DIM, pool_grid,
                                                   classes)),
            }
            print(f"    Q({atoms}, {n}) = {acc:.4f}", flush=True)
        surface[str(atoms)] = {"width": int(pool_grid * pool_grid * atoms),
                               "cells": cells}

    # ---- gates ------------------------------------------------------------
    def Q(a: int, n: int) -> float:
        return float(surface[str(a)]["cells"][str(n)]["accuracy"])

    a0, a1, a2 = atoms_ladder[0], atoms_ladder[1], atoms_ladder[2]
    n0, n1, n2 = n_ladder[0], n_ladder[1], n_ladder[2]

    if not smoke_skip:
        m113 = json.loads(M113_EVIDENCE.read_text(encoding="utf-8"))
        ref = float(m113["arms"]["a_random"]["accuracy_by_penalty"]["1.0"])
        delta = Q(a1, n2) - ref
        if abs(delta) > T1_TOLERANCE:
            print(f"  t1 FAILED: Q({a1},{n2}) {Q(a1,n2):.4f} vs M113 {ref:.4f} "
                  f"(delta {delta:+.5f})", flush=True)
            write_canonical_json(output_dir / "evidence.json", {
                "milestone": "M117", "admissible_as_evidence": False,
                "void": True, "void_reason": "t1 anchor reproduction failed",
                "measured": Q(a1, n2), "reference": ref, "t1_delta": delta,
            })
            return {"admissible_as_evidence": False, "void": True}
        gates["t1_delta"] = delta
        print(f"  t1 anchor delta {delta:+.5f} (<= {T1_TOLERANCE})", flush=True)

    joint1 = Q(a1, n1) - Q(a0, n0)
    axis_c1 = Q(a1, n0) - Q(a0, n0)
    axis_n1 = Q(a0, n1) - Q(a0, n0)
    super_add1 = joint1 - (axis_c1 + axis_n1)

    joint2 = Q(a2, n2) - Q(a1, n1)
    axis_c2 = Q(a2, n1) - Q(a1, n1)
    axis_n2 = Q(a1, n2) - Q(a1, n1)
    super_add2 = joint2 - (axis_c2 + axis_n2)

    ks = {
        "registered": "at cell (a0, n0): joint gain > sum of single-axis gains "
                      "+ 0.005, else atoms and data are separable resources "
                      "and the ceiling is a real slice (not an artifact of "
                      "holding n fixed)",
        "cell_a0_n0": [a0, n0],
        "joint": float(joint1),
        "axis_atoms": float(axis_c1),
        "axis_data": float(axis_n1),
        "sum_axes": float(axis_c1 + axis_n1),
        "excess": float(super_add1),
        "margin": KS_MARGIN,
        "fired": super_add1 <= KS_MARGIN,
        "second_cell": {
            "cell_a1_n1": [a1, n1],
            "joint": float(joint2),
            "sum_axes": float(axis_c2 + axis_n2),
            "excess": float(super_add2),
        },
        "consequence": "if fired, the joint axis is no better than the sum of "
                       "the single axes: scaling atoms and data together does "
                       "not beat scaling either alone -> the 0.2153 ceiling at "
                       "fixed n is real and the B-condition is not rescued by "
                       "the joint surface.",
    }
    gates["kill_switch_superadditive"] = ks
    gates["_smoke_skip"] = smoke_skip

    # per-atom Q(n) steepness (M120 mechanism, folded in)
    steepness = {
        str(a): {
            "n": n_ladder,
            "accuracy": [surface[str(a)]["cells"][str(n)]["accuracy"]
                         for n in n_ladder],
            "delta": float(surface[str(a)]["cells"][str(n2)]["accuracy"]
                           - surface[str(a)]["cells"][str(n0)]["accuracy"]),
        }
        for a in atoms_ladder
    }

    evidence = {
        "milestone": "M117",
        "admissible_as_evidence": not inadmissible,
        "registered_in": config.get("registered_in"),
        "question": ("does Q_S(C, n) rise super-additively in the joint "
                     "atoms x data surface, i.e. was the 0.2153 ceiling a "
                     "slice artifact?"),
        "config_file": Path(config_path).name,
        "config": config,
        "device": device_report,
        "parity_guard": parity,
        "surface": surface,
        "atoms_ladder": atoms_ladder,
        "n_ladder": n_ladder,
        "steepness_per_atoms_m120": steepness,
        "gates": gates,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM117 complete -> {output_dir / 'evidence.json'}", flush=True)
    for a in atoms_ladder:
        print("  atoms %d:" % a, {str(n): round(Q(a, n), 4) for n in n_ladder},
              flush=True)
    print(f"  KS super-additivity fired: {ks['fired']}  "
          f"(excess {super_add1:+.4f} vs margin {KS_MARGIN})", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m117(args.config, args.output)


if __name__ == "__main__":
    main()
