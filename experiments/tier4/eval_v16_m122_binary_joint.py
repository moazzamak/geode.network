"""M122 — Binary joint surface (atoms x data x bits).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v19.md`` section 5.1 and
``experiments/configs/v16/m122_binary_joint.json``.

Question. M114's 108-bit loss (3.3 pts) is intrinsic at fixed atoms (M118).
Does the loss shrink when atoms, bits and data rise JOINTLY? M117 showed atoms
help at full data. M122 measures the 2x2x2 surface: atoms {3072, 6144} x
n {69000, 138000} x bits {108, 216}, where 216 bits = two INDEPENDENT 108-bit
projections concatenated (seeds 33 and 34; disclosed: NOT a linear-cap
violation, the code is genuinely 216 bits and costs ~2x an 108-bit encode).

Cells:
- (3072, 108): REUSED from M118's sealed code memmaps (data_cache_root/v16/m118/
  {arm}_{split}.npy); t1 verifies the reuse by re-fitting b_random 0.1842 and
  c_itq 0.1820 at n=138000 within 0.002.
- (3072, 216), (6144, 108), (6144, 216): freshly encoded (GPU sign-GEMM
  Hamming, M118 exact) for both arms.
All cells fit a closed-form ridge (penalty 1.0) at each n from the first n
rows and score the SAME full 34500-row test set. Float references: M117 sealed
surface (0.2153 at 3072/138000, 0.2249 at 6144/138000).

Gate:
- t1 (reuse reproduction), then
- KS1 (joint buyback): fired if for EITHER arm
  gap(6144,138000,216) > gap(3072,138000,108) - 0.01, where
  gap(a,n,b) = float(a,n) - binary(a,b,n). The joint budget must narrow the
  bit loss by >= 0.01 to re-open the binary axis as a quality route.
Reported decomposition (not gates): head-only (108 bits at 6144) vs bits-only
(216 bits at 3072) contributions to the narrowing.

Reproduce with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m122_binary_joint
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
from experiments.tier4.eval_v15_m103_atoms import (
    _contrast_normalise,
    _extract_patches,
    _pool,
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
from experiments.tier4.eval_v16_m114_binary import (
    _fit_itq,
    _hash_pool,
    _project,
    _random_projection,
)
from experiments.tier4.eval_v16_m118_binary_scale import (
    _binary_codes_to_memmap,
    _fit_and_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m122_binary_joint.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m122_binary_joint"
M117_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m117_scale" / "evidence.json"
M118_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m118_binary_scale" / "evidence.json"

T1_TOLERANCE = 0.002
KS_NARROWING = 0.01
BLOCK = 4096
REUSE_ATOMS = 3072
REUSE_BITS = 108


def _fit_projection(pool: np.ndarray, bits: int, arm: str,
                    iters: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P, bias, center) for a (bits, arm) cell.

    108 bits: the single M114/M118 projection (seed 33).
    216 bits: two independent 108-bit projections concatenated (seeds 33, 34).
    """
    if bits == 108:
        if arm == "c_itq":
            return _fit_itq(pool, 108, iters, 33)
        return _random_projection(pool, 108, 33)
    if arm == "c_itq":
        P1, b1, c1 = _fit_itq(pool, 108, iters, 33)
        P2, b2, _ = _fit_itq(pool, 108, iters, 34)
    else:
        P1, b1, c1 = _random_projection(pool, 108, 33)
        P2, b2, _ = _random_projection(pool, 108, 34)
    return np.concatenate([P1, P2], axis=1), np.concatenate([b1, b2]), c1


def run_m122(config_path: Path, output_dir: Path) -> dict[str, Any]:
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
    atoms_ladder = [int(a) for a in rep["atoms_ladder"]]
    bits_ladder = [int(b) for b in rep["bits_ladder"]]
    n_ladder = [int(n) for n in config["scaling"]["n_ladder"]]
    arms = list(rep["arms"])
    pool_grid = int(rep["pool_grid"])
    smoke_skip = bool(config.get("_smoke_skip_gates", False))
    gates: dict[str, Any] = {}

    print("building global whitener + candidates (M108 exact)", flush=True)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)
    print("building hash-fit pool", flush=True)
    pool = _hash_pool(config, corpus, whitener)

    cache = data_cache_root() / "v16" / "m122"
    cache.mkdir(parents=True, exist_ok=True)
    m118_cache = data_cache_root() / "v16" / "m118"
    n_train = len(corpus["train_labels"])
    n_test = len(corpus["test_labels"])
    train_rows = np.arange(n_train)
    test_rows = np.arange(n_test)

    # projections depend only on (bits, arm); fit once
    proj = {}
    for bits in bits_ladder:
        for arm in arms:
            proj[(bits, arm)] = _fit_projection(
                pool, bits, arm, int(rep["hash_iters"]))

    surface: dict[str, Any] = {}
    for atoms in atoms_ladder:
        print(f"atoms {atoms}: building dictionary (prefix, seed 11)",
              flush=True)
        dictionary = _random_dictionary(candidates, len(candidates),
                                        int(rep["dictionary_seed"]), atoms)
        surface[str(atoms)] = {}
        for bits in bits_ladder:
            surface[str(atoms)][str(bits)] = {}
            for arm in arms:
                reuse = (atoms == REUSE_ATOMS and bits == REUSE_BITS)
                if reuse:
                    mem_train = np.lib.format.open_memmap(
                        m118_cache / f"{arm}_train.npy", mode="r")
                    mem_test = np.lib.format.open_memmap(
                        m118_cache / f"{arm}_test.npy", mode="r")
                    print(f"  cell ({atoms},{bits},{arm}): REUSING M118 "
                          f"sealed codes", flush=True)
                else:
                    P, bias, center = proj[(bits, arm)]
                    print(f"  cell ({atoms},{bits},{arm}): encoding train",
                          flush=True)
                    mem_train = _binary_codes_to_memmap(
                        corpus, dictionary, whitener, P, bias, center, bits,
                        pool_grid, device, train_rows,
                        cache / f"a{atoms}_b{bits}_{arm}_train.npy",
                        split="train")
                    print(f"  cell ({atoms},{bits},{arm}): encoding test",
                          flush=True)
                    mem_test = _binary_codes_to_memmap(
                        corpus, dictionary, whitener, P, bias, center, bits,
                        pool_grid, device, test_rows,
                        cache / f"a{atoms}_b{bits}_{arm}_test.npy",
                        split="test")
                curve = {}
                for n in n_ladder:
                    r = _fit_and_score(mem_train, mem_test,
                                       corpus["train_labels"],
                                       corpus["test_labels"],
                                       corpus["test_domains"], classes, n)
                    acc = r["accuracy_by_penalty"]["1.0"]
                    pc = r["per_domain_correct"]["1.0"]
                    pr = r["per_domain_rows"]["1.0"]
                    curve[str(n)] = {
                        "accuracy": acc,
                        "per_domain": [pc[d] / pr[d] for d in range(6)],
                    }
                    print(f"    ({atoms},{bits},{arm}) n={n}: {acc:.4f}",
                          flush=True)
                surface[str(atoms)][str(bits)][arm] = {
                    "curve": curve,
                    "reused_m118_codes": reuse,
                }

    # float references from M117 sealed surface
    m117 = json.loads(M117_EVIDENCE.read_text(encoding="utf-8"))
    float_q: dict[tuple[int, int], float] = {}
    for a in m117["surface"]:
        for n, cell in m117["surface"][a]["cells"].items():
            float_q[(int(a), int(n))] = float(cell["accuracy"])

    def Q(atoms: int, bits: int, arm: str, n: int) -> float:
        return float(surface[str(atoms)][str(bits)][arm]["curve"][str(n)]["accuracy"])

    if not smoke_skip:
        m118 = json.loads(M118_EVIDENCE.read_text(encoding="utf-8"))
        for arm in arms:
            ref = float(m118["arm_curves"][arm]["138000"]["accuracy"])
            measured = Q(REUSE_ATOMS, REUSE_BITS, arm, 138000)
            delta = measured - ref
            gates[f"t1_delta_{arm}"] = delta
            if abs(delta) > T1_TOLERANCE:
                print(f"  t1 FAILED ({arm}): reused {measured:.4f} vs M118 "
                      f"{ref:.4f} (delta {delta:+.5f})", flush=True)
                write_canonical_json(output_dir / "evidence.json", {
                    "milestone": "M122", "admissible_as_evidence": False,
                    "void": True, "void_reason": "t1 reuse reproduction failed",
                    "arm": arm, "measured": measured, "reference": ref,
                    "t1_delta": delta,
                })
                return {"admissible_as_evidence": False, "void": True}
            print(f"  t1 ({arm}) reuse delta {delta:+.5f} "
                  f"(<= {T1_TOLERANCE})", flush=True)
        gates["t1_registered"] = ("reused (3072,108) cells reproduce M118 "
                                  "b_random 0.1842 / c_itq 0.1820 at n=138000")

    # KS1: joint buyback per arm (only meaningful on the sealed ladder, whose
    # cells include 3072/6144 atoms at n=138000; bypassed in smoke)
    ks1: dict[str, Any] = {"registered": (
        "gap(6144,138000,216) <= gap(3072,138000,108) - 0.01 per arm "
        "(M117 floats 0.2153 / 0.2249)")}
    if not smoke_skip:
        for arm in arms:
            gap_ref = float_q[(REUSE_ATOMS, 138000)] - Q(REUSE_ATOMS, REUSE_BITS, arm, 138000)
            gap_new = float_q[(6144, 138000)] - Q(6144, 216, arm, 138000)
            narrowing = gap_ref - gap_new
            ks1[arm] = {
                "gap_at_3072_108": float(gap_ref),
                "gap_at_6144_216": float(gap_new),
                "narrowing": float(narrowing),
                "min_narrowing": KS_NARROWING,
                "fired": narrowing < KS_NARROWING,
            }
        ks1["fired_any"] = any(v["fired"] for k, v in ks1.items()
                               if k != "registered")
    else:
        ks1["fired_any"] = False
        ks1["note"] = "bypassed in smoke (cells absent from the smoke ladder)"
    gates["kill_switch_1_joint_buyback"] = ks1
    gates["_smoke_skip"] = smoke_skip

    # reported decomposition (not gates): head-only vs bits-only at full data
    decomp: dict[str, Any] = {}
    if not smoke_skip:
        for arm in arms:
            gap_head = float_q[(6144, 138000)] - Q(6144, 108, arm, 138000)
            gap_bits = float_q[(3072, 138000)] - Q(3072, 216, arm, 138000)
            gap_base = float_q[(3072, 138000)] - Q(3072, 108, arm, 138000)
            decomp[arm] = {
                "gap_3072_108": float(gap_base),
                "gap_6144_108_head_only": float(gap_head),
                "gap_3072_216_bits_only": float(gap_bits),
                "note": "head-only = wider head at 108 bits; bits-only = 216 "
                        "bits at 3072 atoms; both at n=138000",
            }
        decomp["note"] = "reported, not a gate"
    gates["reported_head_vs_bits_decomposition"] = decomp

    evidence = {
        "milestone": "M122",
        "admissible_as_evidence": not inadmissible,
        "registered_in": config.get("registered_in"),
        "question": ("does the joint budget (wider head + more bits + more "
                     "data) narrow the binary-vs-float gap by >= 0.01, "
                     "re-opening the binary axis as a quality route?"),
        "config_file": Path(config_path).name,
        "config": config,
        "device": device_report,
        "parity_guard": parity,
        "atoms_ladder": atoms_ladder,
        "bits_ladder": bits_ladder,
        "n_ladder": n_ladder,
        "arms": arms,
        "float_reference_m117": {f"{a}/{n}": v for (a, n), v in float_q.items()},
        "surface": surface,
        "gates": gates,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM122 complete -> {output_dir / 'evidence.json'}", flush=True)
    for atoms in atoms_ladder:
        for bits in bits_ladder:
            for arm in arms:
                print(f"  ({atoms},{bits},{arm}): "
                      + {str(n): round(Q(atoms, bits, arm, n), 4)
                         for n in n_ladder}.__str__(), flush=True)
    print(f"  KS1 fired: {ks1['fired_any']}", flush=True)
    if not smoke_skip:
        for arm in arms:
            k = ks1[arm]
            print(f"    {arm}: gap {k['gap_at_3072_108']:.4f} -> "
                  f"{k['gap_at_6144_216']:.4f} (narrowing {k['narrowing']:+.4f}, "
                  f"min {KS_NARROWING}, fired {k['fired']})", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m122(args.config, args.output)


if __name__ == "__main__":
    main()
