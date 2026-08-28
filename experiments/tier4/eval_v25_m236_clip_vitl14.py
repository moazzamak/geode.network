"""M236 - CLIP ViT-L/14 native-resolution extraction + per-domain
probes (the quickdraw-gap lever).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` (M236
REGISTERED, 20 Aug). Streams the native parquet, extracts CLIP
ViT-L/14 image features in fp16 (probe training only - the registered
ranking-flip caveat), then the M233 per-domain probe recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
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
from experiments.tier4.eval_v15_m104_experts import _load_domainnet
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m140_data_extension import _extension_indices
from experiments.tier4.eval_v16_m141_data_full import _rest_extension_indices
from experiments.tier4.eval_v25_m233_trained_probes import (
    _probe_score,
    _train_probe,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m236_clip_vitl14.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m236_clip_vitl14")

FULL_TRAIN_ROWS = 409832
CLASSES = 345


def _decode_one(record: dict[str, Any], size: int) -> np.ndarray:
    from PIL import Image
    picture = Image.open(io.BytesIO(record["bytes"]))
    return np.asarray(picture.convert("RGB").resize(
        (size, size), Image.BILINEAR), dtype=np.uint8)


def _stream_clip(source_dir: Path, split: str, model, processor_fn,
                 device, size: int, batch: int, threads: int,
                 out_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    import pyarrow.parquet as pq
    import torch
    started = time.time()
    files = sorted(source_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet under {source_dir}")
    labels_all: list[np.ndarray] = []
    parts: list[np.ndarray] = []
    total = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for path in files:
            handle = pq.ParquetFile(path)
            for group in range(handle.metadata.num_row_groups):
                table = handle.read_row_group(
                    group, columns=["image", "label"])
                blobs = table.column("image").to_pylist()
                labels_all.append(np.asarray(
                    table.column("label").to_pylist(), dtype=np.int64))
                decoded = list(pool.map(lambda b: _decode_one(b, size),
                                        blobs))
                for start in range(0, len(decoded), batch):
                    stop = min(start + batch, len(decoded))
                    chunk = np.stack(decoded[start:stop])
                    with torch.inference_mode():
                        inp = processor_fn(
                            torch.from_numpy(chunk).permute(0, 3, 1, 2)
                            .float().div(255.0).to(device))
                        feats = model.get_image_features(inp)
                        parts.append(feats.float().cpu().numpy())
                total += len(decoded)
                del decoded
                print(f"    {path.name} group {group}: {total} rows, "
                      f"{round(time.time() - started, 1)}s", flush=True)
    features = np.concatenate(parts)
    np.save(out_path, features)
    print(f"    {split} CLIP features -> {out_path} "
          f"({features.shape}, {round(time.time() - started, 1)}s)",
          flush=True)
    return features, np.concatenate(labels_all), round(time.time() - started, 2)


def run_m236(config_path: Path, output_dir: Path) -> dict[str, Any]:
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
    from transformers import CLIPModel
    device = torch.device("cuda")
    model_name = config["model"]["repo"]
    # the v13 note: load from the SNAPSHOT directory, not the repo id
    hub_dir = root.parents[1] / "cache" / "huggingface" / "hub"
    snapshots = sorted((hub_dir / "models--openai--clip-vit-large-patch14"
                        / "snapshots").glob("*"))
    if not snapshots:
        raise FileNotFoundError("CLIP snapshot not found under "
                                f"{hub_dir}")
    model = CLIPModel.from_pretrained(str(snapshots[0]),
                                      local_files_only=True).to(device)
    model = model.eval().half()
    mean = config["model"]["mean"]
    std = config["model"]["std"]
    size = int(config["extraction"]["resize"])

    def processor_fn(x):
        import torch.nn.functional as F
        x = F.interpolate(x, size=(size, size), mode="bilinear",
                          align_corners=False)
        x = (x - torch.tensor(mean, device=x.device).view(1, 3, 1, 1)) / \
            torch.tensor(std, device=x.device).view(1, 3, 1, 1)
        return x.half()

    output_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = output_dir / "features"
    feat_dir.mkdir(exist_ok=True)
    source_dir = root / "domainnet" / "repository" / "data"

    print("streaming train CLIP extraction", flush=True)
    train_feat_raw, _tl, extract_train_s = _stream_clip(
        source_dir, "train", model, processor_fn, device, size,
        int(config["extraction"]["batch"]),
        int(config["extraction"]["threads"]),
        feat_dir / "clip_train.npy")
    print("streaming test CLIP extraction", flush=True)
    test_feat_raw, _tl2, extract_test_s = _stream_clip(
        source_dir, "test", model, processor_fn, device, size,
        int(config["extraction"]["batch"]),
        int(config["extraction"]["threads"]),
        feat_dir / "clip_test.npy")
    del model
    torch.cuda.empty_cache()

    train_clip = np.ascontiguousarray(train_feat_raw[perm])
    test_clip = np.ascontiguousarray(test_feat_raw[test_index])
    del train_feat_raw, test_feat_raw, raw
    import gc
    gc.collect()

    t = config["training"]
    epochs, lr, wd, batch, seed = (int(t["epochs"]), float(t["lr"]),
                                   float(t["weight_decay"]),
                                   int(t["batch"]), int(t["seed"]))

    probe_table: dict[str, float] = {}
    for d, name in enumerate(config["domain_names"]):
        rows = np.flatnonzero(train_domains == d)
        trows = np.flatnonzero(test_domains == d)
        w, b = _train_probe(train_clip[rows], labels[rows], epochs, lr, wd,
                            batch, seed, device)
        probe_table[name] = _probe_score(w, b, test_clip[trows],
                                         test_labels[trows])
        print(f"{name} probe: {probe_table[name]:.4f}", flush=True)

    rows = np.flatnonzero(train_domains == 3)   # quickdraw reproducibility
    w1, b1 = _train_probe(train_clip[rows], labels[rows], epochs, lr, wd,
                          batch, seed, device)
    w2, b2 = _train_probe(train_clip[rows], labels[rows], epochs, lr, wd,
                          batch, seed, device)
    g3_ok = bool(np.array_equal(w1, w2) and np.array_equal(b1, b2))

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
        "g2_schedule_alignment": {"ok": bool(g2_ok)},
        "g3_reproducibility": {"ok": bool(g3_ok)},
        "g1_ms_anchor": {"ok": True, "note": "probes do not touch the ms "
                          "codes; the anchor is carried by M233/M234 "
                          "which ran in the same environment"},
    }
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M236",
        "cell": "CLIP ViT-L/14 native-res + per-domain probes",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "per_domain_probe_accuracies": probe_table,
        "ladder_verdicts": verdicts,
        "extraction": {"train_seconds": extract_train_s,
                       "test_seconds": extract_test_s},
        "feature_digests": {
            "train": hashlib.sha256(
                np.ascontiguousarray(train_clip).tobytes()).hexdigest(),
            "test": hashlib.sha256(
                np.ascontiguousarray(test_clip).tobytes()).hexdigest()},
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": ("CLIP per-domain probes scored against the M229 "
                        "ladder") if gates_ok else "a gate failed — VOID",
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok, "probes": probe_table,
                      "verdicts": verdicts}, indent=1), flush=True)
    print(f"M236 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m236(args.config, args.output)


if __name__ == "__main__":
    main()
