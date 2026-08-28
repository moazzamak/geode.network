"""M234 - dinov2_vitb14 native-resolution extraction + per-domain
probes and ridges (the bigger-backbone lever).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` (M234
REGISTERED, 20 Aug). Reuses the M230 streaming extraction with the
768-d ViT-B backbone, then the winning M233 recipe (dino per-domain
probes, fixed Adam) plus the closed-form per-domain ridges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

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
)
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m140_data_extension import _extension_indices
from experiments.tier4.eval_v16_m141_data_full import _rest_extension_indices
from experiments.tier4.eval_v25_m230_native_res_dinov2 import _stream_extract
from experiments.tier4.eval_v25_m233_trained_probes import (
    _probe_score,
    _train_probe,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m234_vitb14.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25" / "m234_vitb14")

FULL_TRAIN_ROWS = 409832
CLASSES = 345
BLOCK = 4096


def _ridge_domain(rows: np.ndarray, feat: np.ndarray, labels: np.ndarray,
                  penalties: list[float]):
    acc = RidgeAccumulator(feat.shape[1], CLASSES)
    for s in range(0, len(rows), BLOCK):
        e = min(s + BLOCK, len(rows))
        acc.add(np.asarray(feat[rows[s:e]]), labels[rows[s:e]])
    weights = acc.solve_many(penalties)
    std = acc.standardiser()
    return std, weights


def _ridge_score(std, w, tfeat: np.ndarray, tlabels: np.ndarray) -> float:
    hits = 0
    for s in range(0, len(tfeat), BLOCK):
        e = min(s + BLOCK, len(tfeat))
        scores = (std(np.asarray(tfeat[s:e])).astype(np.float64)
                  @ w[:-1] + w[-1])
        hits += int((np.argmax(scores, axis=1) == tlabels[s:e]).sum())
    return hits / len(tfeat)


def run_m234(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    root = data_cache_root()

    corpus, train_index, test_index = _load_corpus(config)
    raw = _load_domainnet(32)
    ext600_indices, _ = _extension_indices(raw["train_labels"], train_index,
                                           600, CLASSES)
    rest_indices = _rest_extension_indices(raw["train_labels"], train_index,
                                           CLASSES, per_class_take=200)
    perm = np.concatenate([train_index, ext600_indices, rest_indices])
    labels = np.load(root / config["artifacts"]["labels_file"])["labels"]
    g2_ok = (len(perm) == FULL_TRAIN_ROWS
             and np.array_equal(raw["train_labels"][perm], labels))
    train_domains = raw["train_domains"][perm]
    test_labels = corpus["test_labels"]
    test_domains = corpus["test_domains"]

    import torch
    import torchvision.transforms as T
    device = torch.device("cuda")
    model = torch.hub.load(config["model"]["repo"],
                           config["model"]["name"]).to(device)
    print(f"device: {torch.cuda.get_device_name(0)} "
          f"model {config['model']['name']}", flush=True)
    transform = T.Compose([
        T.Resize(config["extraction"]["resize"]),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = output_dir / "features"
    feat_dir.mkdir(exist_ok=True)
    source_dir = root / "domainnet" / "repository" / "data"

    print("streaming train extraction (vitb14)", flush=True)
    train_feat_raw, train_labels_raw, _td, extract_train_s = _stream_extract(
        source_dir, "train", model, transform, device,
        int(config["extraction"]["resize"]),
        int(config["extraction"]["batch"]),
        int(config["extraction"]["threads"]),
        feat_dir / "vitb14_train_dino.npy")
    print("streaming test extraction (vitb14)", flush=True)
    test_feat_raw, _tl, _td2, extract_test_s = _stream_extract(
        source_dir, "test", model, transform, device,
        int(config["extraction"]["resize"]),
        int(config["extraction"]["batch"]),
        int(config["extraction"]["threads"]),
        feat_dir / "vitb14_test_dino.npy")
    del model
    torch.cuda.empty_cache()

    train_dino = np.ascontiguousarray(train_feat_raw[perm])
    test_dino = np.ascontiguousarray(test_feat_raw[test_index])
    del train_feat_raw, test_feat_raw, raw
    import gc
    gc.collect()

    # g1: ms anchor (unchanged)
    ms_cache = root / config["artifacts"]["cache_relpath"]
    ms_test_cache = root / config["artifacts"]["test_cache_relpath"]
    train_ms = np.asarray(np.load(
        ms_cache / config["artifacts"]["train_file"], mmap_mode="r"))
    test_ms = np.asarray(np.load(
        ms_test_cache / config["artifacts"]["test_file"], mmap_mode="r"))
    std_ms, w_ms = _ridge_domain(np.arange(FULL_TRAIN_ROWS), train_ms,
                                 labels, [1.0])
    anchor_measured = _ridge_score(std_ms, w_ms[1.0], test_ms, test_labels)
    anchor = float(config["anchor"]["value"])
    tol = float(config["anchor"]["tolerance"])
    g1_ok = abs(anchor_measured - anchor) <= tol
    print(f"g1: {anchor_measured:.17f} ok={g1_ok}", flush=True)
    del train_ms, test_ms, std_ms, w_ms
    gc.collect()

    t = config["training"]
    epochs, lr, wd, batch, seed = (int(t["epochs"]), float(t["lr"]),
                                   float(t["weight_decay"]),
                                   int(t["batch"]), int(t["seed"]))
    penalties = [float(p) for p in config["penalties"]]

    probe_table: dict[str, float] = {}
    ridge_table: dict[str, dict[str, float]] = {}
    for d, name in enumerate(config["domain_names"]):
        rows = np.flatnonzero(train_domains == d)
        trows = np.flatnonzero(test_domains == d)
        w, b = _train_probe(train_dino[rows], labels[rows], epochs, lr, wd,
                            batch, seed, device)
        probe_table[name] = _probe_score(w, b, test_dino[trows],
                                         test_labels[trows])
        std, weights = _ridge_domain(rows, train_dino, labels, penalties)
        ridge_table[name] = {
            str(p): _ridge_score(std, weights[p], test_dino[trows],
                                 test_labels[trows]) for p in penalties}
        print(f"{name}: probe {probe_table[name]:.4f} ridge "
              f"{ {k: round(v, 4) for k, v in ridge_table[name].items()} }",
              flush=True)

    # g3 reproducibility (clipart probe)
    rows = np.flatnonzero(train_domains == 0)
    trows = np.flatnonzero(test_domains == 0)
    w1, b1 = _train_probe(train_dino[rows], labels[rows], epochs, lr, wd,
                          batch, seed, device)
    w2, b2 = _train_probe(train_dino[rows], labels[rows], epochs, lr, wd,
                          batch, seed, device)
    g3_ok = bool(np.array_equal(w1, w2) and np.array_equal(b1, b2))
    print(f"g3: {g3_ok}", flush=True)

    ladder = config["ladder"]
    verdicts: dict[str, dict[str, Any]] = {}
    for name in config["domain_names"]:
        v = probe_table[name]
        if name in ladder["easy"]:
            verdicts[name] = {"best": round(v, 4), "bar": ladder["easy_bar"],
                              "met": v >= ladder["easy_bar"]}
        elif name in ladder["middle"]:
            verdicts[name] = {"best": round(v, 4),
                              "bar": ladder["middle_bar"],
                              "met": ladder["middle_bar"][0] <= v}
        else:
            verdicts[name] = {"best": round(v, 4),
                              "bar": ladder["hard_first_bar"],
                              "met": ladder["hard_first_bar"][0] <= v}

    gates = {
        "g1_ms_global_anchor": {"ok": bool(g1_ok),
                                "measured": anchor_measured,
                                "sealed": anchor,
                                "delta": anchor_measured - anchor,
                                "tolerance": tol},
        "g2_schedule_alignment": {"ok": bool(g2_ok)},
        "g3_reproducibility": {"ok": bool(g3_ok)},
    }
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M234",
        "cell": "dinov2_vitb14 native-res + per-domain probes/ridges",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "per_domain_probe_accuracies": probe_table,
        "per_domain_ridge_accuracies": ridge_table,
        "ladder_verdicts": verdicts,
        "extraction": {"train_seconds": extract_train_s,
                       "test_seconds": extract_test_s},
        "feature_digests": {
            "train": hashlib.sha256(
                np.ascontiguousarray(train_dino).tobytes()).hexdigest(),
            "test": hashlib.sha256(
                np.ascontiguousarray(test_dino).tobytes()).hexdigest()},
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": ("vitb14 per-domain probes scored against the "
                        "M229 ladder") if gates_ok else "a gate failed — VOID",
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok, "probes": probe_table,
                      "verdicts": verdicts}, indent=1), flush=True)
    print(f"M234 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m234(args.config, args.output)


if __name__ == "__main__":
    main()
