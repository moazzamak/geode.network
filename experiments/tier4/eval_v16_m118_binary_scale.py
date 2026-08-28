"""M118 — Binary x data scaling.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v18.md`` section 5.5 and
``experiments/configs/v16/m118_binary_scale.json``.

Question. M114's sealed binary Hamming encode cut the per-image ops ~8.4x
(30.2M vs 254.6M) but lost 3.3 points at full data (ITQ-108 0.1820 vs float
0.2153); hash learning added nothing (KS1 fired). M116/M117 then showed the
frozen family's Q(n) is steep and its steepness is a head-width property.
M118 asks: does the 108-bit code inherit that steep Q(n) (same 12288-wide
head on binary features), and does DATA buy back the bit loss at scale?

Arms (M114's exact construction: atoms 3072, dictionary seed 11, ITQ seed 33,
50 iters, 108 bits; sign-GEMM Hamming, bit-identical to CPU POPCNT):
- b_random: seeded Gaussian projection (hash control), 108 bits.
- c_itq: learned ITQ projection, 108 bits.
Each arm's codes are image functions: encoded ONCE into D: memmaps (train
138000 x 12288, test 34500 x 12288), then a closed-form ridge (penalty 1.0)
is fitted at each n in M116's ladder from the first n rows. All points score
the same full test set. The float reference curve is M116's sealed sparse
Q(n) (gain 0.1668).

Gates:
- t1: b_random at n=138000 reproduces M114's sealed b_random (0.1842) within
  0.002, or the run voids.
- KS1 (binary steepness): binary gain across the ladder >= 0.5 * float gain.
- KS2 (bit-loss buyback): binary-vs-float gap at n_max <= gap at n_min + 0.01.

Reproduce with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m118_binary_scale
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m118_binary_scale.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m118_binary_scale"
M114_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m114_binary" / "evidence.json"
M116_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m116_scale" / "evidence.json"

PATCH_DIM = 108
T1_TOLERANCE = 0.002
KS_STEEPNESS = 0.5
KS_GAP = 0.01
BLOCK = 4096


def _binary_codes_to_memmap(corpus: dict[str, np.ndarray], dictionary: np.ndarray,
                            whitener, P: np.ndarray, bias: np.ndarray,
                            center: np.ndarray, bits: int, pool_grid: int,
                            device: torch.device, rows: np.ndarray, path: Path,
                            split: str) -> np.ndarray:
    """Encode binary triangle codes (n, 4*atoms) once into a D: memmap."""
    atoms = len(dictionary)
    dims = pool_grid * pool_grid * atoms
    mem = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                    shape=(len(rows), dims))
    images = corpus[f"{split}_images"]
    grid = whitener.grid
    atom_sign_t = torch.from_numpy(np.ascontiguousarray(
        (_project(dictionary, P, bias, center).astype(np.float32)
         * 2.0 - 1.0).T)).to(torch.float32).to(device)
    b_float = float(bits)

    step = 48
    for start in range(0, len(rows), step):
        take = rows[start:start + step]
        imgs = images[take]
        white = whitener(imgs).reshape(-1, PATCH_DIM)
        n = len(imgs)
        patch_sign = (_project(white, P, bias, center).astype(
            np.float32) * 2.0 - 1.0)
        ps = torch.from_numpy(np.ascontiguousarray(patch_sign)).to(
            torch.float32).to(device)
        with torch.no_grad():
            inner = ps @ atom_sign_t
            h = (b_float - inner) * 0.5
            act = torch.clamp(h.mean(dim=-1, keepdim=True) - h, min=0.0)
            pooled = _pool(act, n, grid, pool_grid)
        mem[start:start + n] = pooled.to(torch.float32).cpu().numpy()
    return np.lib.format.open_memmap(path, mode="r", dtype=np.float32)


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


def run_m118(config_path: Path, output_dir: Path) -> dict[str, Any]:
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
    atoms = int(rep["atoms"])
    bits = int(rep["bits"])
    pool_grid = int(rep["pool_grid"])
    ladder = [int(n) for n in config["scaling"]["n_ladder"]]
    n_max = ladder[-1]
    arms = list(rep["arms"])
    smoke_skip = bool(config.get("_smoke_skip_gates", False))
    gates: dict[str, Any] = {}

    print("building global whitener + dictionary (M114 exact)", flush=True)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)
    dictionary = _random_dictionary(candidates, len(candidates),
                                    int(rep["dictionary_seed"]), atoms)

    print("building hash-fit pool", flush=True)
    pool = _hash_pool(config, corpus, whitener)

    cache = data_cache_root() / "v16" / "m118"
    cache.mkdir(parents=True, exist_ok=True)
    n_train = len(corpus["train_labels"])
    n_test = len(corpus["test_labels"])
    train_rows = np.arange(n_train)
    test_rows = np.arange(n_test)

    arm_curves: dict[str, Any] = {}
    for arm in arms:
        if arm == "c_itq":
            P, bias, center = _fit_itq(pool, bits, int(rep["hash_iters"]),
                                       int(rep["hash_fit_seed"]))
        else:  # b_random
            P, bias, center = _random_projection(pool, bits,
                                                 int(rep["hash_fit_seed"]))
        print(f"  arm {arm}: encoding train codes", flush=True)
        mem_train = _binary_codes_to_memmap(
            corpus, dictionary, whitener, P, bias, center, bits, pool_grid,
            device, train_rows, cache / f"{arm}_train.npy", split="train")
        print(f"  arm {arm}: encoding test codes", flush=True)
        mem_test = _binary_codes_to_memmap(
            corpus, dictionary, whitener, P, bias, center, bits, pool_grid,
            device, test_rows, cache / f"{arm}_test.npy", split="test")
        curve = {}
        for n in ladder:
            r = _fit_and_score(mem_train, mem_test, corpus["train_labels"],
                               corpus["test_labels"], corpus["test_domains"],
                               classes, n)
            acc = r["accuracy_by_penalty"]["1.0"]
            pc = r["per_domain_correct"]["1.0"]
            pr = r["per_domain_rows"]["1.0"]
            curve[str(n)] = {
                "accuracy": acc,
                "per_domain": [pc[d] / pr[d] for d in range(6)],
            }
            print(f"    {arm} n={n}: {acc:.4f}", flush=True)
        arm_curves[arm] = curve

    # float reference (M116 sealed sparse Q(n))
    m116 = json.loads(M116_EVIDENCE.read_text(encoding="utf-8"))
    float_q = {p["n"]: p["accuracy"] for p in m116["sparse"]["curve"]}

    def Q(arm: str, n: int) -> float:
        return float(arm_curves[arm][str(n)]["accuracy"])

    if not smoke_skip:
        m114 = json.loads(M114_EVIDENCE.read_text(encoding="utf-8"))
        ref = float(m114["arms"]["b_random"]["accuracy_by_penalty"]["1.0"])
        delta = Q("b_random", n_max) - ref
        if abs(delta) > T1_TOLERANCE:
            print(f"  t1 FAILED: b_random@{n_max} {Q('b_random', n_max):.4f} "
                  f"vs M114 {ref:.4f} (delta {delta:+.5f})", flush=True)
            write_canonical_json(output_dir / "evidence.json", {
                "milestone": "M118", "admissible_as_evidence": False,
                "void": True, "void_reason": "t1 binary reproduction failed",
                "measured": Q("b_random", n_max), "reference": ref,
                "t1_delta": delta,
            })
            return {"admissible_as_evidence": False, "void": True}
        gates["t1_delta"] = delta
        print(f"  t1 binary delta {delta:+.5f} (<= {T1_TOLERANCE})", flush=True)

    n0 = ladder[0]
    # float reference at the largest M116 ladder point at or below n (exact in
    # the sealed run, where the ladder IS M116's)
    def _float_at(n: int) -> float:
        keys = sorted(float_q)
        return float(float_q[max((k for k in keys if k <= n), default=keys[0])])
    float_gain = _float_at(n_max) - _float_at(n0)
    ks1 = {}
    ks2 = {}
    for arm in arms:
        gain = Q(arm, n_max) - Q(arm, n0)
        ks1[arm] = {
            "gain": float(gain),
            "float_gain": float(float_gain),
            "ratio": float(gain / float_gain) if float_gain else None,
            "min_ratio": KS_STEEPNESS,
            "fired": gain < KS_STEEPNESS * float_gain,
        }
        gap_n0 = _float_at(n0) - Q(arm, n0)
        gap_nm = _float_at(n_max) - Q(arm, n_max)
        ks2[arm] = {
            "gap_n_min": float(gap_n0),
            "gap_n_max": float(gap_nm),
            "gap_change": float(gap_nm - gap_n0),
            "tolerance": KS_GAP,
            "fired": (gap_nm - gap_n0) > KS_GAP,
        }
    ks1["registered"] = "binary gain >= 0.5 x float gain (M116: 0.1668)"
    ks1["fired_any"] = any(v["fired"] for k, v in ks1.items()
                           if k != "registered")
    ks2["registered"] = ("binary-vs-float gap at n_max <= gap at n_min + 0.01 "
                         "(data must narrow or hold the M114 bit loss)")
    ks2["fired_any"] = any(v["fired"] for k, v in ks2.items()
                           if k != "registered")
    gates["kill_switch_1_binary_steepness"] = ks1
    gates["kill_switch_2_bit_loss_buyback"] = ks2
    gates["_smoke_skip"] = smoke_skip

    evidence = {
        "milestone": "M118",
        "admissible_as_evidence": not inadmissible,
        "registered_in": config.get("registered_in"),
        "question": ("does the 108-bit Hamming code inherit the frozen "
                     "family's steep Q(n), and does data buy back M114's "
                     "3.3-point bit loss?"),
        "config_file": Path(config_path).name,
        "config": config,
        "device": device_report,
        "parity_guard": parity,
        "ladder": ladder,
        "float_reference_curve_m116": float_q,
        "arm_curves": arm_curves,
        "gates": gates,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM118 complete -> {output_dir / 'evidence.json'}", flush=True)
    for arm in arms:
        print("  %s:" % arm, {str(n): round(Q(arm, n), 4) for n in ladder},
              flush=True)
    print(f"  KS1 fired: {ks1['fired_any']}  KS2 fired: {ks2['fired_any']}",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m118(args.config, args.output)


if __name__ == "__main__":
    main()
