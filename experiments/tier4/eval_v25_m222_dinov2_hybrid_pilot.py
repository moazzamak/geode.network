"""M222 — DINOv2-hybrid ridge, BOUNDED PILOT: on a registered train
subset, measure whether concatenating DINOv2-small features with the
ms codes helps the closed-form ridge on the sealed test selection.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(20 Aug 2026, before the build). GPU extraction (the .venv-rocm torch
build exposes the RX 9070 XT). PILOT ONLY: the train subset is a
per-class prefix (58/class = 20,010 rows); the ms-only fit on the same
subset is the fair baseline. The sealed full-data accuracies are NOT
comparable to this pilot.
"""
from __future__ import annotations

import argparse
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
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v16_m109_trunk import _load_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m222_dinov2_hybrid_pilot.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m222_dinov2_hybrid_pilot")

CLASSES = 345
DOMAINS = 6
BLOCK = 4096
BATCH = 256


def _pilot_indices(labels: np.ndarray, per_class: int) -> np.ndarray:
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
    blocks = np.split(order, boundaries)
    return np.concatenate([block[:per_class] for block in blocks])


def _extract(model, transform, images: np.ndarray, device
             ) -> tuple[np.ndarray, float]:
    import torch
    model.eval()
    features = []
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(images), BATCH):
            stop = min(start + BATCH, len(images))
            batch = transform(
                torch.from_numpy(images[start:stop]).permute(0, 3, 1, 2)
                .float().div(255.0).to(device))
            out = model.forward_features(batch)
            tok = out["x_norm_clstoken"]
            if tok.ndim == 3:      # older repo: (B, tokens, D)
                tok = tok[:, 0]
            elif tok.ndim != 2:    # current repo: (B, D)
                raise RuntimeError(
                    f"M222 premise failure: clstoken shape {tuple(tok.shape)}")
            features.append(tok.float().cpu().numpy())
            if (start // BATCH) % 25 == 0:
                print(f"  extract batch {start // BATCH + 1}: "
                      f"{round(time.time() - started, 1)}s elapsed",
                      flush=True)
    return (np.concatenate(features),
            round(time.time() - started, 2))


def _features_or_extract(model, transform, images: np.ndarray, device,
                         output_dir: Path, name: str,
                         selection: np.ndarray) -> tuple[np.ndarray, float]:
    """Extract features once and persist them keyed by the row selection.

    The first M222 build lost a completed multi-hour extraction to a
    downstream crash; features are now written to disk immediately and
    reused iff the recorded selection digest matches (GPU inference is
    not bitwise reproducible run-to-run, so the KEY is the input
    selection, not the feature bytes).
    """
    import hashlib
    sel_bytes = selection.astype(np.int64).tobytes()
    meta = {"selection_sha256": hashlib.sha256(sel_bytes).hexdigest()}
    feat_dir = output_dir / "features"
    feat_path = feat_dir / f"{name}_dino.npy"
    meta_path = feat_dir / f"{name}_meta.json"
    if feat_path.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("selection_sha256") == meta["selection_sha256"]:
            print(f"reusing cached {name} DINOv2 features "
                  f"({feat_path.name})", flush=True)
            return np.load(feat_path), float(old.get("extract_seconds", 0.0))
    features, seconds = _extract(model, transform, images, device)
    feat_dir.mkdir(parents=True, exist_ok=True)
    np.save(feat_path, features)
    meta["extract_seconds"] = seconds
    meta["feature_sha256"] = payload_hash(features)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return features, seconds


def _fit_and_score(features: np.ndarray, labels: np.ndarray,
                   test_features: np.ndarray, test_labels: np.ndarray,
                   penalties: list[float]) -> dict[str, float]:
    acc = RidgeAccumulator(features.shape[1], CLASSES)
    for start in range(0, len(features), BLOCK):
        stop = min(start + BLOCK, len(features))
        acc.add(features[start:stop], labels[start:stop])
    weights_by_penalty = acc.solve_many(penalties)
    standardise = acc.standardiser()
    out: dict[str, float] = {}
    for penalty in penalties:
        weights = weights_by_penalty[penalty]
        hits = 0
        for start in range(0, len(test_features), BLOCK):
            stop = min(start + BLOCK, len(test_features))
            scores = (standardise(test_features[start:stop])
                      .astype(np.float64) @ weights[:-1] + weights[-1])
            hits += int((np.argmax(scores, axis=1)
                         == test_labels[start:stop]).sum())
        out[str(penalty)] = hits / len(test_features)
    return out


def run_m222(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()

    configure_external_cache_environment()
    corpus, _train_index, test_index = _load_corpus(config)
    test_labels = corpus["test_labels"]
    root = data_cache_root()

    decoded = np.load(root / "domainnet_decoded" / "size32.npz",
                      mmap_mode="r")
    train_images = decoded["train_images"]
    test_images = decoded["test_images"][test_index]
    train_labels = decoded["train_labels"]
    if len(train_images) != 409832 or len(test_images) != 34500:
        raise SystemExit("M222 premise failure: decoded shapes")

    ms_cache = root / config["artifacts"]["cache_relpath"]
    ms_test_cache = root / config["artifacts"]["test_cache_relpath"]
    mem_train = np.load(ms_cache / config["artifacts"]["train_file"],
                        mmap_mode="r")
    mem_test = np.load(ms_test_cache / config["artifacts"]["test_file"],
                       mmap_mode="r")

    pilot_rows = _pilot_indices(train_labels,
                                int(config["pilot"]["train_rows_per_class"]))
    pilot_labels = train_labels[pilot_rows]
    pilot_ms = np.asarray(mem_train[pilot_rows])
    test_ms = np.asarray(mem_test)

    # ---- GPU extraction
    import torch
    import torchvision.transforms as T
    device = torch.device("cuda")
    model = torch.hub.load(config["model"]["repo"],
                           config["model"]["name"]).to(device)
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    transform = T.Compose([
        T.Resize(config["model"]["input_size"]),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])
    pilot_dino, extract_train_s = _features_or_extract(
        model, transform, np.asarray(train_images[pilot_rows]), device,
        output_dir, "train", pilot_rows)
    test_dino, extract_test_s = _features_or_extract(
        model, transform, np.asarray(test_images), device,
        output_dir, "test", test_index)
    print(f"extraction: train {extract_train_s}s, test {extract_test_s}s",
          flush=True)

    penalties = [float(p) for p in config["pilot"]["penalties"]]
    ms_only = _fit_and_score(pilot_ms, pilot_labels, test_ms, test_labels,
                             penalties)
    hybrid_features = np.concatenate([pilot_ms, pilot_dino], axis=1)
    hybrid_test = np.concatenate([test_ms, test_dino], axis=1)
    hybrid = _fit_and_score(hybrid_features, pilot_labels, hybrid_test,
                            test_labels, penalties)
    for p in penalties:
        print(f"penalty {p}: ms-only {ms_only[str(p)]:.6f}  hybrid "
              f"{hybrid[str(p)]:.6f}", flush=True)

    gates = {
        "g1_row_counts_exact": {
            "ok": len(pilot_rows) == 345 * int(
                config["pilot"]["train_rows_per_class"])
            and len(test_images) == 34500,
            "pilot_rows": len(pilot_rows), "test_rows": len(test_images)},
        "g2_features_recorded": {
            "ok": True,
            "test_feature_hash": payload_hash(test_dino),
            "extract_train_seconds": extract_train_s,
            "extract_test_seconds": extract_test_s},
        "g3_valid_accuracies": {
            "ok": all(0.0 <= v <= 1.0 for d in (ms_only, hybrid)
                      for v in d.values()),
            "ms_only": ms_only, "hybrid": hybrid},
        "g4_pilot_scoped": {
            "ok": True,
            "note": "pilot only: per-class-prefix 58 subset; the "
                    "sealed full-data numbers are NOT comparable"},
    }
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M222",
        "cell": "DINOv2-hybrid ridge, bounded pilot",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "ms_only_test_accuracies": ms_only,
        "hybrid_test_accuracies": hybrid,
        "best_penalty": {"ms_only": max(penalties,
                                        key=lambda p: ms_only[str(p)]),
                         "hybrid": max(penalties,
                                       key=lambda p: hybrid[str(p)])},
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": (
                "pilot only: the hybrid's direction on the registered "
                "subset, NOT a full-scale claim"
            ) if gates_ok else "a gate failed — VOID",
        },
        "scope": "bounded pilot; full-scale extraction is a separate "
                 "registered compute decision",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok,
                      "ms_only": ms_only, "hybrid": hybrid}, indent=1),
          flush=True)
    print(f"M222 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m222(args.config, args.output)


if __name__ == "__main__":
    main()
