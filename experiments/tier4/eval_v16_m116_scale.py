"""M116 — Data scaling Q(n): does the frozen family's accuracy scale with
training rows comparably to a cost-matched dense trunk?

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v18.md`` section 5.2 and
``experiments/configs/v16/m116_scale.json``.

Question (v18 5.2). Fit the best arm from M115 (which was L=0, the frozen
single-stage 3072-dictionary arm — the depth arms lost KS1) and the
cost-matched dense trunk at subsample sizes n of the shared corpus; compare
Q(n) curves. Does the frozen family's accuracy improve with data at a rate
comparable to dense's? Test the Bardone-Goldt prediction (2603.12901):
with shared latent structure, higher-order statistics become learnable at
linear sample complexity. Independent of B1.

Protocol. The sparse codes and the dense trunk features are image functions,
not n functions: each is computed ONCE over the full corpus (memmaps on D:
under ``data_cache_root()/v16/m116``), then a closed-form ridge head (penalty
1.0, M108's constant) is fitted at every ladder point from the first n rows of
the M107-shuffled train order (nested, deterministic). Every ladder point and
both families score the SAME full 34500-row test set. Dense = M109 t1 protocol
(frozen DINOv2-small, r42 = the best M107/M109 dense point at-or-below the
sparse family's 254.6M per-image MACs; trunk forward on GPU).

Gates (registered before measurement):
- t1_sparse: sparse at n_max reproduces M113's sealed random-3072 (0.2153)
  within 0.002, or the run voids.
- t1_dense: dense at n_max reproduces M109's sealed t1_r42 (0.1971) within
  0.002, or the run voids.
- KS (data-scaling parity): Delta_S = Q_S(n_max) - Q_S(n_min) >= 0.5 * Delta_T
  with Delta_T = Q_T(n_max) - Q_T(n_min) (and Delta_S > 0); fired (fail) if
  the frozen family improves with data less than half as fast as the
  cost-matched dense trunk.

Reproduce with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m116_scale
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
    IMAGENET_MEAN,
    IMAGENET_STD,
    _solve_and_score,
    _verify_pixel_identity,
)
from experiments.tier4.eval_v16_m108_dictionary import (
    _encode_block_device,
    _verify_device,
)
from experiments.tier4.eval_v16_m109_trunk import (
    DenseModel,
    _dense_pixels,
    _load_corpus,
    _parity_guard,
)
from experiments.tier4.eval_v16_m113_learned import (
    _build_whitener_and_candidates,
    _random_dictionary,
)
from experiments.tier4.eval_v16_m115_lofi import _write_frozen_codes

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m116_scale.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m116_scale"
M113_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m113_learned" / "evidence.json"
M109_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m109_trunk" / "evidence.json"

PATCH_DIM = 108
T1_TOLERANCE = 0.002
KS_GRAIN = 0.5
BLOCK = 4096


def _fit_and_score_from_memmap(mem_train: np.ndarray, mem_test: np.ndarray,
                               labels: np.ndarray, test_labels: np.ndarray,
                               test_domains: np.ndarray, classes: int,
                               n: int) -> dict[str, Any]:
    """Fit a closed-form ridge on the first n rows of ``mem_train`` and score
    the full test memmap. ``mem_train``/``mem_test`` are read-only memmaps."""
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


def _dense_features(pixels, res: int, n_train: int, n_test: int, classes: int,
                    device: torch.device, cache: Path
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Frozen DINOv2-small features (CLS ++ meanpool, M109 t1 protocol) for all
    train and test rows; written once to memmaps on D: and returned read-only.
    """
    model = DenseModel("small", classes, device)
    width = model.width
    mem_train = np.lib.format.open_memmap(
        cache / "dense_train.npy", mode="w+", dtype=np.float32,
        shape=(n_train, width))
    mem_test = np.lib.format.open_memmap(
        cache / "dense_test.npy", mode="w+", dtype=np.float32,
        shape=(n_test, width))
    pixels_train = np.load(pixels["train"][res], mmap_mode="r")
    pixels_test = np.load(pixels["test"][res], mmap_mode="r")

    def _run(src: np.ndarray, dst: np.ndarray) -> None:
        total = len(src)
        for start in range(0, total, 256):
            stop = min(start + 256, total)
            block = np.asarray(src[start:stop], dtype=np.float32) / 255.0
            block = (block - IMAGENET_MEAN) / IMAGENET_STD
            block = np.ascontiguousarray(block.transpose(0, 3, 1, 2))
            with torch.no_grad():
                feat = model.features(
                    torch.from_numpy(block).to(device)).cpu().numpy()
            dst[start:stop] = feat
            if start % 25600 == 0:
                print(f"    trunk {start}/{total}", flush=True)

    print("  dense trunk features: train", flush=True)
    _run(pixels_train, mem_train)
    print("  dense trunk features: test", flush=True)
    _run(pixels_test, mem_test)
    del model
    torch.cuda.empty_cache()
    return (np.lib.format.open_memmap(cache / "dense_train.npy", mode="r",
                                      dtype=np.float32),
            np.lib.format.open_memmap(cache / "dense_test.npy", mode="r",
                                      dtype=np.float32))


def run_m116(config_path: Path, output_dir: Path,
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
    ladder = [int(n) for n in config["scaling"]["n_ladder"]]
    n_max = ladder[-1]
    res = int(config["dense"]["resolution"])
    smoke_skip = bool(config.get("_smoke_skip_gates", False))
    gates: dict[str, Any] = {}

    print("building global whitener + random candidate pool (M108 exact)",
          flush=True)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)
    dictionary = _random_dictionary(candidates, len(candidates),
                                    int(rep["dictionary_seed"]), atoms)
    width = pool_grid * pool_grid * atoms

    cache = data_cache_root() / "v16" / "m116"
    cache.mkdir(parents=True, exist_ok=True)
    n_train = len(corpus["train_labels"])
    n_test = len(corpus["test_labels"])
    train_rows = np.arange(n_train)
    test_rows = np.arange(n_test)

    # ---- sparse frozen codes (once, memmap) ------------------------------
    print("sparse frozen codes: train", flush=True)
    f_train = _write_frozen_codes(corpus, dictionary, whitener, pool_grid,
                                  device, train_rows, cache / "f_train.npy",
                                  split="train")
    print("sparse frozen codes: test", flush=True)
    f_test = _write_frozen_codes(corpus, dictionary, whitener, pool_grid,
                                 device, test_rows, cache / "f_test.npy",
                                 split="test")

    # ---- dense trunk features (once, memmap) ------------------------------
    print("dense pixels (r42, M109 cache)", flush=True)
    dense_cfg = {**config, "dense": {**config["dense"],
                                     "resolutions": [res]}}
    pixels = _dense_pixels(dense_cfg, train_index, test_index)
    g_train, g_test = _dense_features(pixels, res, n_train, n_test, classes,
                                      device, cache)
    dense_width = g_train.shape[1]

    # ---- Q(n) ladders -----------------------------------------------------
    sparse_curve: list[dict[str, Any]] = []
    dense_curve: list[dict[str, Any]] = []
    for n in ladder:
        print(f"  sparse n={n}", flush=True)
        r = _fit_and_score_from_memmap(f_train, f_test,
                                       corpus["train_labels"],
                                       corpus["test_labels"],
                                       corpus["test_domains"], classes, n)
        acc = r["accuracy_by_penalty"]["1.0"]
        pc = r["per_domain_correct"]["1.0"]
        pr = r["per_domain_rows"]["1.0"]
        sparse_curve.append({
            "n": int(n),
            "accuracy": acc,
            "per_domain": [pc[d] / pr[d] for d in range(6)],
            "training_ops": int(_training_macs(n, atoms, whitener.grid,
                                               PATCH_DIM, pool_grid, classes)),
        })
        print(f"    sparse n={n}: {acc:.4f}", flush=True)
        print(f"  dense n={n}", flush=True)
        r = _fit_and_score_from_memmap(g_train, g_test,
                                       corpus["train_labels"],
                                       corpus["test_labels"],
                                       corpus["test_domains"], classes, n)
        acc = r["accuracy_by_penalty"]["1.0"]
        pc = r["per_domain_correct"]["1.0"]
        pr = r["per_domain_rows"]["1.0"]
        dense_curve.append({
            "n": int(n),
            "accuracy": acc,
            "per_domain": [pc[d] / pr[d] for d in range(6)],
            # ridge fit only; the trunk forward (n * per_image_macs) is added
            # below once the per-image figure is measured from the ONNX graph
            "training_ops": int(n * (dense_width * dense_width
                                     + dense_width * classes)
                                + dense_width ** 3 // 3),
        })
        print(f"    dense n={n}: {acc:.4f}", flush=True)

    # dense trunk forward is a corpus cost: add n * per-image MACs
    from experiments.tier4.eval_v15_m107_dense import _dinov2_geometry
    from experiments.tier4.eval_v15_m107_dense import _transformer_macs
    per_image = _transformer_macs(_dinov2_geometry("small"), res, classes)["total"]
    for point in dense_curve:
        point["training_ops"] += int(point["n"] * per_image)

    # ---- gates ------------------------------------------------------------
    qs = {p["n"]: p["accuracy"] for p in sparse_curve}
    qt = {p["n"]: p["accuracy"] for p in dense_curve}
    n_min = ladder[0]

    if not smoke_skip:
        m113 = json.loads(M113_EVIDENCE.read_text(encoding="utf-8"))
        ref_s = float(m113["arms"]["a_random"]["accuracy_by_penalty"]["1.0"])
        m109 = json.loads(M109_EVIDENCE.read_text(encoding="utf-8"))
        ref_t = float(m109["results"]["dense"][f"t1_r{res}"]["accuracy"])
        d_s = qs[n_max] - ref_s
        d_t = qt[n_max] - ref_t
        if abs(d_s) > T1_TOLERANCE or abs(d_t) > T1_TOLERANCE:
            print(f"  t1 FAILED: sparse {qs[n_max]:.4f} vs {ref_s:.4f} "
                  f"(delta {d_s:+.5f}); dense {qt[n_max]:.4f} vs {ref_t:.4f} "
                  f"(delta {d_t:+.5f})", flush=True)
            write_canonical_json(output_dir / "evidence.json", {
                "milestone": "M116", "admissible_as_evidence": False,
                "void": True, "void_reason": "t1 reproduction failed",
                "sparse_n_max": qs[n_max], "sparse_ref": ref_s,
                "dense_n_max": qt[n_max], "dense_ref": ref_t,
                "sparse_t1_delta": d_s, "dense_t1_delta": d_t,
            })
            return {"admissible_as_evidence": False, "void": True}
        gates["t1_sparse_delta"] = d_s
        gates["t1_dense_delta"] = d_t
        print(f"  t1 sparse delta {d_s:+.5f}, dense delta {d_t:+.5f} "
              f"(<= {T1_TOLERANCE})", flush=True)

    delta_s = qs[n_max] - qs[n_min]
    delta_t = qt[n_max] - qt[n_min]
    ratio = (delta_s / delta_t) if delta_t > 0 else None
    gap = {n: qt[n] - qs[n] for n in ladder}
    ks = {
        "registered": "Delta_S >= 0.5 * Delta_T (frozen family improves with "
                      "data at least half as fast as the cost-matched dense "
                      "trunk), with Delta_S > 0",
        "n_ladder": ladder,
        "sparse": qs,
        "dense": qt,
        "gap_dense_minus_sparse": gap,
        "delta_s": delta_s,
        "delta_t": delta_t,
        "gain_ratio": ratio,
        "min_gain_ratio": KS_GRAIN,
        "fired": not (delta_s > 0 and ratio is not None and ratio >= KS_GRAIN),
        "note": "the gap at each n must not diverge for a competitive "
                "data-scaling claim; reported alongside the gain ratio.",
    }
    gates["kill_switch_data_scaling"] = ks
    gates["_smoke_skip"] = smoke_skip

    evidence = {
        "milestone": "M116",
        "admissible_as_evidence": not inadmissible,
        "registered_in": config.get("registered_in"),
        "question": ("does the frozen family's Q_S(n) scale comparably to the "
                     "cost-matched dense trunk's Q_T(n)?"),
        "config_file": Path(config_path).name,
        "config": config,
        "device": device_report,
        "parity_guard": parity,
        "ladder": {
            "n_ladder": ladder,
            "subsample_rule": config["scaling"]["subsample_rule"],
            "test_set": config["scaling"]["test_set"],
        },
        "sparse": {
            "atoms": atoms, "width": width, "dictionary": "M113 random-3072",
            "curve": sparse_curve,
        },
        "dense": {
            "model": "small", "resolution": res, "width": int(dense_width),
            "per_image_macs": int(per_image),
            "curve": dense_curve,
        },
        "gates": gates,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM116 complete -> {output_dir / 'evidence.json'}", flush=True)
    print("  sparse:", {n: round(a, 4) for n, a in qs.items()}, flush=True)
    print("  dense: ", {n: round(a, 4) for n, a in qt.items()}, flush=True)
    print(f"  KS fired: {ks['fired']}  (Delta_S {delta_s:+.4f}, "
          f"Delta_T {delta_t:+.4f}, ratio {ratio if ratio is None else round(ratio, 3)})",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m116(args.config, args.output)


if __name__ == "__main__":
    main()
