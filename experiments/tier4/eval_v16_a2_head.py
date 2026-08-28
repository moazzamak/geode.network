"""A2 — head-vs-representation decomposition.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v17.md`` section 3 (A2).

At M109 (t2) the same frozen sparse codes scored 0.2148 with the closed-form
ridge head but 0.0554 with the 4-epoch SGD head — a 4x collapse — while the
dense head *improved*. A2 asks whether that collapse is a head-underfit
artefact or a property of the representation: it encodes the frozen
representations once (sparse codes at 3,072 atoms; DINOv2 features at
resolutions {42, 224}), then fits heads on the frozen codes:

1. **Ridge reference** — closed-form float64 ridge at penalty 1.0 (M107's
   head; the converged linear-head optimum). This must reproduce M109's t1
   (sparse 0.2148, dense r42 0.1971, r224 0.5368) within the registered
   tolerance, or A2 is VOID and the instrument is at fault.
2. **SGD sweep** — AdamW linear head over a registered grid of {epochs, lr}
   with early stopping on validation, until converged. The exact M109 t2
   schedule (4 epochs, lr 3e-4) is a reproduction control: it must reproduce
   M109's t2 (sparse 0.0554, dense r42 0.2212, r224 0.6441) within the
   registered tolerance.

No trunk gradients run anywhere in A2. Device placement is registered in the
config; the sparse whitening stays on numpy CPU (M107's arithmetic) and all
head training runs on the GPU ("cuda" = ROCm/HIP in this interpreter).

Reproduce with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_a2_head
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

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
    _load_domainnet,
    _score,
)
from experiments.tier4.eval_v15_m107_dense import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    _class_subsample,
    _index_digest,
    _materialise_original,
    _transformer_macs,
    _verify_pixel_identity,
)
from experiments.tier4.eval_v16_m108_dictionary import (
    _encode_block_device,
    _verify_device,
)
from experiments.tier4.eval_v16_m109_trunk import (
    DenseModel,
    _build_whitener_dictionary,
    _dense_pixels,
    _load_corpus,
    _parity_guard,
    _train_with_schedule,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "a2_head.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "a2_head"
M109_EVIDENCE = REPO_ROOT / "logs" / "results" / "v16" / "m109_trunk" / "evidence.json"

T1_TOLERANCE = 0.002   # ridge reference must reproduce M109 t1
T2_TOLERANCE = 0.01    # 4-epoch SGD cell must reproduce M109 t2


class LinearHead(torch.nn.Module):
    """A linear head operating on precomputed frozen features/codes."""

    def __init__(self, dim: int, classes: int, device: torch.device):
        super().__init__()
        self.head = torch.nn.Linear(dim, classes)
        torch.nn.init.normal_(self.head.weight, std=0.01)
        torch.nn.init.zeros_(self.head.bias)
        self.head.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def _head_factory(feat: np.ndarray, labels: np.ndarray, rows: np.ndarray,
                  batch: int, device: torch.device) -> Callable[[], Iterator]:
    """Factory of (feature_block, labels) generators over a frozen code matrix."""

    def gen():
        for start in range(0, len(rows), batch):
            take = rows[start:start + batch]
            yield (torch.from_numpy(np.ascontiguousarray(feat[take])).to(device),
                   torch.from_numpy(labels[take]).to(device))
    return gen


def _encode_sparse(images: np.ndarray, rows: np.ndarray, table: torch.Tensor,
                   whitener, pool_grid: int, atoms: int, device: torch.device,
                   cache_path: Path) -> Path:
    """Encode frozen sparse codes and write them to a disk memmap (bounded RAM)."""
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    feats = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float32,
        shape=(len(rows), pool_grid * pool_grid * atoms))
    for start in range(0, len(rows), 512):
        take = rows[start:start + 512]
        feats[start:start + len(take)] = _encode_block_device(
            images[take], table, whitener, pool_grid)
    feats.flush()
    return cache_path


def _encode_dense(mem, rows: np.ndarray, model: DenseModel, device: torch.device,
                  cache_path: Path) -> Path:
    """Encode frozen DINOv2 features and write them to a disk memmap."""
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    feats = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32,
                                      shape=(len(rows), model.width))
    for start in range(0, len(rows), 256):
        take = rows[start:start + 256]
        block = np.asarray(mem[take], dtype=np.float32) / 255.0
        block = (block - IMAGENET_MEAN) / IMAGENET_STD
        block = np.ascontiguousarray(block.transpose(0, 3, 1, 2))
        with torch.no_grad():
            feats[start:start + len(take)] = model.features(
                torch.from_numpy(block).to(device)).cpu().numpy()
    feats.flush()
    return cache_path


def _fit_ridge(feat, labels: np.ndarray, test_feat, test_labels: np.ndarray,
               classes: int, penalty: float) -> float:
    """Closed-form float64 ridge on a (possibly mmapped) frozen-code matrix."""
    dim = feat.shape[1]
    acc = RidgeAccumulator(dim, classes)
    for start in range(0, len(feat), 2048):
        block = np.asarray(feat[start:start + 2048])
        acc.add(block, labels[start:start + len(block)])
    solutions = acc.solve_many([penalty])
    standardise = acc.standardiser()
    correct = 0
    for start in range(0, len(test_feat), 2048):
        block = np.asarray(test_feat[start:start + 2048])
        correct += int(_score(solutions[penalty], standardise(block),
                              test_labels[start:start + len(block)]).sum())
    return correct / len(test_labels)


def _fit_sgd(feat: np.ndarray, labels: np.ndarray, val_feat: np.ndarray,
             val_labels: np.ndarray, test_feat: np.ndarray,
             test_labels: np.ndarray, classes: int, epochs: int, lr: float,
             wd: float, batch: int, device: torch.device, patience: int,
             seed: int) -> dict[str, Any]:
    dim = feat.shape[1]
    train_rows = np.arange(len(feat))
    val_rows = np.arange(len(val_feat))
    test_rows = np.arange(len(test_feat))
    model = LinearHead(dim, classes, device)
    torch.manual_seed(seed)
    train_out = _train_with_schedule(
        model,
        _head_factory(feat, labels, train_rows, batch, device),
        _head_factory(val_feat, val_labels, val_rows, batch, device),
        epochs, lr, wd, device, patience)
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(test_rows), batch):
            take = test_rows[start:start + batch]
            logits = model(torch.from_numpy(
                np.ascontiguousarray(test_feat[take])).to(device))
            correct += int((logits.argmax(dim=1) == torch.from_numpy(
                test_labels[take]).to(device)).sum().item())
    return {**train_out, "test_accuracy": correct / len(test_labels)}


def run_a2(config_path: Path, output_dir: Path, progress: bool = True
           ) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    torch.set_num_threads(config["numerics"]["torch_threads"])
    torch.manual_seed(config["numerics"]["seed"])
    configure_external_cache_environment()
    _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("parity guard at startup", flush=True)
    parity = _parity_guard(torch, config, device)

    print("loading corpus", flush=True)
    corpus, train_index, test_index = _load_corpus(config)
    test_seq = np.arange(len(test_index))
    classes = int(corpus["train_labels"].max()) + 1
    size = config["corpus"]["image_size"]
    for split, idx in (("train", train_index), ("test", test_index)):
        _verify_pixel_identity(split, idx, corpus[f"{split}_images"], size,
                               config["corpus"]["pixel_identity_rows"])

    print("building whitener and dictionary (M108 arm (a) at 3072)", flush=True)
    whitener, dictionary, grid, dimension, pool_grid = _build_whitener_dictionary(
        config, corpus)
    atoms = int(config["sparse"]["atoms"])
    sparse_cfg = config["sparse"]

    order = np.random.default_rng(config["corpus"]["shuffle_seed"]).permutation(
        len(train_index))
    val_count = int(round(len(train_index) * config["schedule"]["validation_fraction"]))
    train_fit = order[val_count:]
    val_rows = order[:val_count]
    train_labels = corpus["train_labels"]
    test_labels = corpus["test_labels"]

    ridge_penalty = float(config["ridge"]["penalty"])
    batch = int(config["schedule"]["batch"])
    wd = float(config["schedule"]["weight_decay"])
    patience = int(config["schedule"]["early_stopping"]["patience"])
    grid = config["schedule"]["head_grid"]
    epochs_list = [int(e) for e in grid["epochs"]]
    lr_list = [float(l) for l in grid["learning_rate"]]
    seed = int(config["numerics"]["seed"])

    m109 = json.loads(M109_EVIDENCE.read_text(encoding="utf-8"))
    m109_t1 = m109["t1_reproduction"]
    m109_t2 = {  # measured in M109 evidence
        "sparse": m109["results"]["sparse"]["t2"]["accuracy"],
        "dense_42": m109["results"]["dense"]["t2_r42"]["accuracy"],
        "dense_224": m109["results"]["dense"]["t2_r224"]["accuracy"],
    }

    results: dict[str, Any] = {}
    reproductions: dict[str, Any] = {}
    smoke_skip = bool(config.get("_smoke_skip_gates", False))

    # ---- encode frozen representations once, to disk memmaps ----------------
    table = torch.from_numpy(np.ascontiguousarray(dictionary)).to(torch.float32)
    table = table.to(device)
    # The frozen-code matrices are large (sparse train ~6.8 GB); the workspace
    # drive is nearly full, so they live under the data-cache root (D:) instead.
    cache_dir = data_cache_root() / "v16" / "a2_head"
    reps: dict[str, dict[str, Path]] = {}

    print("encoding sparse codes (frozen)", flush=True)
    reps["sparse"] = {
        "train": _encode_sparse(corpus["train_images"], train_fit, table,
                                whitener, pool_grid, atoms, device,
                                cache_dir / "sparse_train.npy"),
        "val": _encode_sparse(corpus["train_images"], val_rows, table,
                              whitener, pool_grid, atoms, device,
                              cache_dir / "sparse_val.npy"),
        "test": _encode_sparse(corpus["test_images"], test_seq, table,
                               whitener, pool_grid, atoms, device,
                               cache_dir / "sparse_test.npy"),
    }
    torch.cuda.empty_cache()

    dense_pixels = _dense_pixels(config, train_index, test_index)
    resolutions = [int(r) for r in config["dense"]["resolutions"]]
    dense_model = DenseModel("small", classes, device)
    for r in resolutions:
        print(f"encoding dense features r{r} (frozen)", flush=True)
        mem = np.load(dense_pixels["train"][r], mmap_mode="r")
        mem_test = np.load(dense_pixels["test"][r], mmap_mode="r")
        reps[f"dense_{r}"] = {
            "train": _encode_dense(mem, train_fit, dense_model, device,
                                   cache_dir / f"dense{r}_train.npy"),
            "val": _encode_dense(mem, val_rows, dense_model, device,
                                 cache_dir / f"dense{r}_val.npy"),
            "test": _encode_dense(mem_test, test_seq, dense_model, device,
                                  cache_dir / f"dense{r}_test.npy"),
        }
    del dense_model
    torch.cuda.empty_cache()

    # open the frozen-code matrices read-only (bounded RAM; streamed per block)
    reps = {name: {k: np.load(v, mmap_mode="r") for k, v in rep.items()}
            for name, rep in reps.items()}

    # ---- per representation: ridge reference + SGD sweep -------------------
    for name, rep in reps.items():
        print(f"  {name}: ridge reference + SGD head grid", flush=True)
        ridge_acc = _fit_ridge(rep["train"], train_labels[train_fit],
                               rep["test"], test_labels[test_seq],
                               classes, ridge_penalty)
        cells = {}
        for epochs in epochs_list:
            for lr in lr_list:
                key = f"e{epochs}_lr{lr:g}"
                print(f"    {key} ...", flush=True)
                out = _fit_sgd(rep["train"], train_labels[train_fit],
                               rep["val"], train_labels[val_rows],
                               rep["test"], test_labels[test_seq],
                               classes, epochs, lr, wd, batch, device,
                               patience, seed)
                cells[key] = {
                    "test_accuracy": out["test_accuracy"],
                    "epochs_run": out["epochs_run"],
                    "best_validation_accuracy": out["best_validation_accuracy"],
                }
        best_cell = max(cells, key=lambda k: cells[k]["best_validation_accuracy"])
        results[name] = {
            "feature_dim": int(rep["train"].shape[1]),
            "ridge_reference_penalty_1.0": ridge_acc,
            "sgd_cells": cells,
            "best_converged_cell": best_cell,
            "best_converged_test_accuracy": cells[best_cell]["test_accuracy"],
            "gap_converged_minus_ridge": cells[best_cell]["test_accuracy"] - ridge_acc,
        }
        print(f"    {name}: ridge {ridge_acc:.4f} best_sgd "
              f"{cells[best_cell]['test_accuracy']:.4f} @ {best_cell}", flush=True)

    # ---- instrument gates ---------------------------------------------------
    # 1. ridge reference must reproduce M109 t1 (same frozen codes, same head)
    t1_targets = {
        "sparse": float(m109_t1["sparse_t1"]["m108"]),
        "dense_42": float(m109_t1["dense_t1_r42"]["m107"]),
        "dense_224": float(m109_t1["dense_t1_r224"]["m107"]),
    }
    for name, target in t1_targets.items():
        if name in results:
            reproductions[f"t1_{name}"] = {
                "m109": target,
                "measured": results[name]["ridge_reference_penalty_1.0"],
                "delta": results[name]["ridge_reference_penalty_1.0"] - target,
            }
    # 2. the exact M109 t2 schedule (4 epochs, lr 3e-4) must reproduce M109 t2
    t2_key = f"e4_lr{3e-4:g}"
    for name, target in m109_t2.items():
        if name in results and t2_key in results[name]["sgd_cells"]:
            reproductions[f"t2_{name}"] = {
                "m109": target,
                "measured": results[name]["sgd_cells"][t2_key]["test_accuracy"],
                "delta": results[name]["sgd_cells"][t2_key]["test_accuracy"] - target,
            }

    gate = {}
    if not smoke_skip:
        t1_deltas = [abs(v["delta"]) for k, v in reproductions.items()
                     if k.startswith("t1_")]
        t2_deltas = [abs(v["delta"]) for k, v in reproductions.items()
                     if k.startswith("t2_")]
        if max(t1_deltas, default=0.0) > T1_TOLERANCE:
            gate["_verdict"] = ("A2 VOID: ridge reference does not reproduce "
                                "M109 t1 within tolerance; instrument at fault.")
            print("  A2 VOID: t1 reproduction gate failed", flush=True)
            write_canonical_json(output_dir / "evidence.json", {
                "milestone": "A2", "admissible_as_evidence": False, "void": True,
                "void_reason": "t1 reproduction gate failed",
                "reproductions": reproductions, "parity_guard": parity})
            return {"admissible_as_evidence": False, "void": True,
                    "reproductions": reproductions}
        if max(t2_deltas, default=0.0) > T2_TOLERANCE:
            gate["_verdict"] = ("A2 VOID: the 4-epoch SGD cell does not "
                                "reproduce M109 t2 within tolerance.")
            print("  A2 VOID: t2 reproduction gate failed", flush=True)
            write_canonical_json(output_dir / "evidence.json", {
                "milestone": "A2", "admissible_as_evidence": False, "void": True,
                "void_reason": "t2 reproduction gate failed",
                "reproductions": reproductions, "parity_guard": parity})
            return {"admissible_as_evidence": False, "void": True,
                    "reproductions": reproductions}
        gate["t1_max_delta"] = max(t1_deltas, default=None)
        gate["t2_max_delta"] = max(t2_deltas, default=None)
        gate["_verdict"] = "reproduction gates passed"
        print(f"  t1 max delta {gate['t1_max_delta']:.5f} (<= {T1_TOLERANCE}), "
              f"t2 max delta {gate['t2_max_delta']:.5f} (<= {T2_TOLERANCE})",
              flush=True)
    else:
        gate["_skipped_by_smoke"] = "SMOKE ONLY: gates bypassed so the head paths execute."

    # ---- A2 kill switch -----------------------------------------------------
    sparse = results.get("sparse", {})
    ks = {
        "registered_prediction": (
            "the converged SGD head on frozen sparse codes approaches the ridge "
            "reference (0.2148); the t2 collapse is a schedule artefact"),
        "margin": 0.02,
    }
    if sparse:
        ks["sparse_ridge"] = sparse["ridge_reference_penalty_1.0"]
        ks["sparse_best_converged"] = sparse["best_converged_test_accuracy"]
        ks["fired"] = (sparse["ridge_reference_penalty_1.0"]
                       - sparse["best_converged_test_accuracy"]) > 0.02
        ks["consequence"] = ("if fired, the sparse codes are not linearly "
                             "separable by SGD at a practical budget; A5's "
                             "Arm P uses the ridge head only, and the t2 "
                             "reading reopens")
    else:
        ks["fired"] = None
    results["gate"] = gate
    results["kill_switch"] = ks

    evidence = {
        "milestone": "A2",
        "admissible_as_evidence": not inadmissible,
        "registered_in": config["registered_in"],
        "question": ("is the sparse side's M109 loss a head-underfit artefact, "
                     "or a property of the representation?"),
        "config_file": Path(config_path).name,
        "config": config,
        "corpus": {"train_rows": len(train_index), "test_rows": len(test_index),
                   "classes": classes},
        "device": _verify_device(torch),
        "parity_guard": parity,
        "reproductions": reproductions,
        "results": {k: v for k, v in results.items() if k not in ("gate", "kill_switch")},
        "gate": gate,
        "kill_switch": ks,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    evidence["payload_sha256"] = payload_hash(evidence)
    print(f"\nA2 complete -> {output_dir / 'evidence.json'}", flush=True)
    print(f"  kill switch fired: {ks['fired']}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_a2(args.config, args.output)


if __name__ == "__main__":
    main()
