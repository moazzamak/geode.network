"""M230 - native-resolution streaming DINOv2 extraction + ridge fits.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` (M230
REGISTERED, 20 Aug). The 32x32 feature path was closed by M228's
measured negative; this cell streams the NATIVE-resolution parquet
(the registered streaming rule - no full-resolution image caches),
resizes each image to 224, extracts DINOv2-small features on the GPU,
and persists ONLY the features (digest-only evidence).

Alignment: features are extracted in RAW parquet file order, then
reindexed to the M142 cell-2 SCHEDULE order with the registered
permutation (part1 subsample positions + ext600 + rest). Gate g2
reconstructs the schedule labels from the streamed raw labels and
requires byte-exact equality with the rebuilt labels file - the
same alignment the ms codes use.
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
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m140_data_extension import _extension_indices
from experiments.tier4.eval_v16_m141_data_full import _rest_extension_indices
from experiments.tier4.eval_v25_m222_dinov2_hybrid_pilot import _fit_and_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m230_native_res_dinov2.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m230_native_res_dinov2")

FULL_TRAIN_ROWS = 409832
SEALED_TEST_ROWS = 34500
CLASSES = 345


def _decode_one(record: dict[str, Any], size: int) -> np.ndarray:
    from PIL import Image
    picture = Image.open(io.BytesIO(record["bytes"]))
    return np.asarray(picture.convert("RGB").resize(
        (size, size), Image.BILINEAR), dtype=np.uint8)


def _stream_extract(source_dir: Path, split: str, model, transform,
                    device, size: int, batch: int, threads: int,
                    out_path: Path) -> tuple[np.ndarray, np.ndarray,
                                             np.ndarray, float]:
    """Stream one parquet split, extract features, keep raw-order
    labels/domains. Returns (features, labels, domains, seconds)."""
    import pyarrow.parquet as pq
    import torch
    started = time.time()
    files = sorted(source_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet under {source_dir}")

    labels_all: list[np.ndarray] = []
    domains_all: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    total = 0

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for path in files:
            handle = pq.ParquetFile(path)
            for group in range(handle.metadata.num_row_groups):
                table = handle.read_row_group(
                    group, columns=["image", "label", "domain"])
                blobs = table.column("image").to_pylist()
                group_labels = np.asarray(
                    table.column("label").to_pylist(), dtype=np.int64)
                group_domains = np.asarray(
                    table.column("domain").to_pylist(), dtype=np.int64)
                labels_all.append(group_labels)
                domains_all.append(group_domains)

                decoded = list(pool.map(lambda b: _decode_one(b, size),
                                        blobs))
                for start in range(0, len(decoded), batch):
                    stop = min(start + batch, len(decoded))
                    chunk = np.stack(decoded[start:stop])
                    with torch.inference_mode():
                        inp = transform(
                            torch.from_numpy(chunk).permute(0, 3, 1, 2)
                            .float().div(255.0).to(device))
                        out = model.forward_features(inp)
                        tok = out["x_norm_clstoken"]
                        if tok.ndim == 3:
                            tok = tok[:, 0]
                        feature_parts.append(tok.float().cpu().numpy())
                total += len(decoded)
                del decoded
                print(f"    {path.name} group {group}: {total} rows, "
                      f"{round(time.time() - started, 1)}s", flush=True)

    features = np.concatenate(feature_parts)
    np.save(out_path, features)
    print(f"    {split} features -> {out_path} "
          f"({features.shape}, {round(time.time() - started, 1)}s)",
          flush=True)
    return (features, np.concatenate(labels_all),
            np.concatenate(domains_all), round(time.time() - started, 2))


def run_m230(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    root = data_cache_root()

    print("sealed corpus + schedule indices", flush=True)
    corpus, train_index, test_index = _load_corpus(config)
    from experiments.tier4.eval_v15_m104_experts import _load_domainnet
    raw = _load_domainnet(int(config["corpus"]["image_size"]))
    ext600_indices, _ = _extension_indices(raw["train_labels"], train_index,
                                           600, CLASSES)
    rest_indices = _rest_extension_indices(raw["train_labels"], train_index,
                                           CLASSES, per_class_take=200)
    perm = np.concatenate([train_index, ext600_indices, rest_indices])
    labels_schedule = np.load(root / config["artifacts"]["labels_file"])[
        "labels"]

    print("gpu + model", flush=True)
    import torch
    import torchvision.transforms as T
    device = torch.device("cuda")
    model = torch.hub.load(config["model"]["repo"],
                           config["model"]["name"]).to(device)
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    transform = T.Compose([
        T.Resize(config["extraction"]["resize"]),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = output_dir / "features"
    feat_dir.mkdir(exist_ok=True)
    source_dir = root / "domainnet" / "repository" / "data"

    print("streaming train extraction (native -> 224)", flush=True)
    train_feat, train_labels_raw, _train_domains, extract_train_s = \
        _stream_extract(source_dir, "train", model, transform, device,
                        int(config["extraction"]["resize"]),
                        int(config["extraction"]["batch"]),
                        int(config["extraction"]["threads"]),
                        feat_dir / "native224_train_dino.npy")
    print("streaming test extraction (native -> 224)", flush=True)
    test_feat, test_labels_raw, _test_domains, extract_test_s = \
        _stream_extract(source_dir, "test", model, transform, device,
                        int(config["extraction"]["resize"]),
                        int(config["extraction"]["batch"]),
                        int(config["extraction"]["threads"]),
                        feat_dir / "native224_test_dino.npy")
    del model
    torch.cuda.empty_cache()

    # ---- g2: the schedule permutation must reconstruct the labels ----
    g2_ok = (len(perm) == FULL_TRAIN_ROWS
             and np.array_equal(train_labels_raw[perm], labels_schedule))
    if not g2_ok:
        print("g2 FAILED - the schedule permutation does not reproduce "
              "the labels file", flush=True)

    train_dino = np.ascontiguousarray(train_feat[perm])   # schedule order
    test_dino = np.ascontiguousarray(test_feat[test_index])
    test_labels = corpus["test_labels"]
    test_domains = corpus["test_domains"]
    del train_feat, test_feat, raw
    import gc
    gc.collect()

    # ---- fits (fixed penalties, no test-set selection) ----------------
    ms_cache = root / config["artifacts"]["cache_relpath"]
    ms_test_cache = root / config["artifacts"]["test_cache_relpath"]
    train_ms = np.asarray(np.load(
        ms_cache / config["artifacts"]["train_file"], mmap_mode="r"))
    test_ms = np.asarray(np.load(
        ms_test_cache / config["artifacts"]["test_file"], mmap_mode="r"))
    penalties = [float(p) for p in config["penalties"]]

    print("fits", flush=True)
    ms_only = _fit_and_score(train_ms, labels_schedule, test_ms,
                             test_labels, penalties)
    dino_only = _fit_and_score(train_dino, labels_schedule, test_dino,
                               test_labels, penalties)
    hybrid_train = np.concatenate([train_ms, train_dino], axis=1)
    hybrid_test = np.concatenate([test_ms, test_dino], axis=1)
    hybrid = _fit_and_score(hybrid_train, labels_schedule, hybrid_test,
                            test_labels, penalties)

    # per-domain scoring with the best-of-fixed penalties? No: report the
    # per-domain table for EVERY penalty of every arm (no selection)
    def _per_domain(features: np.ndarray, train_feat_arr: np.ndarray,
                    labels: np.ndarray) -> dict[str, Any]:
        from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
        acc = RidgeAccumulator(train_feat_arr.shape[1], CLASSES)
        for s in range(0, len(train_feat_arr), 4096):
            e = min(s + 4096, len(train_feat_arr))
            acc.add(train_feat_arr[s:e], labels[s:e])
        weights_by_p = acc.solve_many(penalties)
        std = acc.standardiser()
        out: dict[str, Any] = {}
        for p in penalties:
            w = weights_by_p[p]
            per: dict[str, float] = {}
            for d, name in enumerate(config["domain_names"]):
                rows = np.flatnonzero(test_domains == d)
                scores = (std(features[rows]).astype(np.float64)
                          @ w[:-1] + w[-1])
                per[name] = float((np.argmax(scores, axis=1)
                                   == test_labels[rows]).mean())
            out[str(p)] = per
        return out

    ms_domain = _per_domain(test_ms, train_ms, labels_schedule)
    dino_domain = _per_domain(test_dino, train_dino, labels_schedule)
    hybrid_domain = _per_domain(hybrid_test, hybrid_train, labels_schedule)

    anchor = float(config["anchor"]["value"])
    tol = float(config["anchor"]["tolerance"])
    anchor_measured = ms_only["1.0"]
    g1_ok = abs(anchor_measured - anchor) <= tol
    gates = {
        "g1_ms_anchor_reproduction": {
            "ok": g1_ok, "measured": anchor_measured, "sealed": anchor,
            "delta": anchor_measured - anchor, "tolerance": tol},
        "g2_schedule_alignment": {
            "ok": bool(g2_ok), "perm_rows": len(perm),
            "note": "streamed raw labels reindexed by the schedule "
                    "permutation must equal the labels file byte-exactly"},
        "g3_features_recorded": {
            "ok": True,
            "train_feature_sha256": hashlib.sha256(
                np.ascontiguousarray(train_dino).tobytes()).hexdigest(),
            "test_feature_sha256": hashlib.sha256(
                np.ascontiguousarray(test_dino).tobytes()).hexdigest()},
        "g4_fixed_penalties": {
            "ok": True,
            "note": "all penalties reported for every arm; none selected "
                    "on the test set"},
    }
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M230",
        "cell": "native-resolution streaming DINOv2 extraction + fits",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "ms_only_test_accuracies": ms_only,
        "dino_only_test_accuracies": dino_only,
        "hybrid_test_accuracies": hybrid,
        "per_domain": {"ms_only": ms_domain, "dino_only": dino_domain,
                       "hybrid": hybrid_domain},
        "ladder": {"easy_domains": ["quickdraw", "real", "clipart"],
                   "middle": ["sketch", "painting"],
                   "hard": ["infograph"],
                   "bars": {"easy": 0.8, "middle": [0.5, 0.6],
                            "hard_first": [0.3, 0.4]}},
        "extraction": {"train_seconds": extract_train_s,
                       "test_seconds": extract_test_s,
                       "resize": int(config["extraction"]["resize"])},
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": (
                "native-resolution fixed-penalty comparison scored "
                "against the M229 ladder"
            ) if gates_ok else "a gate failed — VOID",
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok, "ms_only": ms_only,
                      "dino_only": dino_only, "hybrid": hybrid},
                     indent=1), flush=True)
    print(f"M230 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m230(args.config, args.output)


if __name__ == "__main__":
    main()
