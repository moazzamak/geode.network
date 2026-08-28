"""M238 - the quickdraw-specific arm: Sobel edge histograms + CLIP.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` (M238,
the quickdraw-specific representation cell, dispatched 20 Aug). The
wall sits at ~0.63 for frozen general backbones; this cell adds a
stroke representation (Sobel gradient magnitude/orientation
histograms on grayscale) to the quickdraw rows only, concatenates
with the cached CLIP features, and trains the fixed-recipe probe.
"""
from __future__ import annotations

import argparse
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
                  / "m238_quickdraw_edges.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m238_quickdraw_edges")

FULL_TRAIN_ROWS = 409832
CLASSES = 345
QUICKDRAW = 3


def _edge_hist(record: dict[str, Any], size: int, bins: int,
               cells: int) -> np.ndarray:
    from PIL import Image
    picture = Image.open(io.BytesIO(record["bytes"])).convert("L")
    gray = np.asarray(picture.resize((size, size), Image.BILINEAR),
                      dtype=np.float32) / 255.0
    gx, gy = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    orient = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)  # [0, 1)
    obin = np.minimum((orient * bins).astype(np.int64), bins - 1)
    cell_h, cell_w = size // cells, size // cells
    feat = np.zeros((cells * cells * bins,), dtype=np.float32)
    for cy in range(cells):
        for cx in range(cells):
            block = slice(cy * cell_h, (cy + 1) * cell_h), \
                slice(cx * cell_w, (cx + 1) * cell_w)
            hist = np.bincount(obin[block].ravel(), weights=mag[block].ravel(),
                               minlength=bins)
            feat[(cy * cells + cx) * bins:(cy * cells + cx) * bins + bins] \
                = hist
    norm = np.linalg.norm(feat) + 1e-12
    return feat / norm


def _stream_quickdraw_edges(source_dir: Path, split: str, rows_out: list,
                            size: int, bins: int, cells: int, threads: int
                            ) -> np.ndarray:
    """Edge features for the quickdraw rows of one split, in file order;
    ``rows_out`` receives the list of file-order row indices processed."""
    import pyarrow.parquet as pq
    started = time.time()
    files = sorted(source_dir.glob(f"{split}-*.parquet"))
    parts: list[np.ndarray] = []
    total = 0
    offset = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for path in files:
            handle = pq.ParquetFile(path)
            for group in range(handle.metadata.num_row_groups):
                table = handle.read_row_group(
                    group, columns=["image", "domain"])
                domains = np.asarray(table.column("domain").to_pylist(),
                                     dtype=np.int64)
                blobs = table.column("image").to_pylist()
                keep = [i for i, d in enumerate(domains) if d == QUICKDRAW]
                if not keep:
                    offset += len(domains)
                    continue
                selected = [blobs[i] for i in keep]
                feats = list(pool.map(
                    lambda b: _edge_hist(b, size, bins, cells), selected))
                parts.append(np.stack(feats))
                rows_out.extend(offset + np.asarray(keep, dtype=np.int64))
                total += len(keep)
                offset += len(domains)
                print(f"    {path.name} group {group}: {total} quickdraw "
                      f"rows, {round(time.time() - started, 1)}s", flush=True)
    return np.concatenate(parts)


def run_m238(config_path: Path, output_dir: Path) -> dict[str, Any]:
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
    del raw
    import gc
    gc.collect()

    e = config["edge"]
    source_dir = root / "domainnet" / "repository" / "data"
    print("streaming quickdraw edge features (train)", flush=True)
    train_file_rows: list[int] = []
    edge_tr_file = _stream_quickdraw_edges(
        source_dir, "train", train_file_rows, int(e["size"]),
        int(e["bins"]), int(e["cells"]), int(e["threads"]))
    print("streaming quickdraw edge features (test)", flush=True)
    test_file_rows: list[int] = []
    edge_te_file = _stream_quickdraw_edges(
        source_dir, "test", test_file_rows, int(e["size"]),
        int(e["bins"]), int(e["cells"]), int(e["threads"]))

    # map file-order quickdraw rows to schedule/test order
    train_pos = {int(i): p for p, i in enumerate(perm)}
    test_pos = {int(i): p for p, i in enumerate(test_index)}
    edge_tr = np.zeros((FULL_TRAIN_ROWS, edge_tr_file.shape[1]),
                       dtype=np.float32)
    for row, feat in zip(train_file_rows, edge_tr_file):
        edge_tr[train_pos[int(row)]] = feat
    edge_te = np.zeros((len(test_index), edge_te_file.shape[1]),
                       dtype=np.float32)
    for row, feat in zip(test_file_rows, edge_te_file):
        if int(row) in test_pos:
            edge_te[test_pos[int(row)]] = feat

    f = config["features"]
    clip_tr = np.asarray(np.load(REPO_ROOT / f["clip_train"], mmap_mode="r"))
    clip_te = np.asarray(np.load(REPO_ROOT / f["clip_test"], mmap_mode="r"))
    tr = np.concatenate([clip_tr[perm],
                         edge_tr.astype(np.float32)], axis=1)
    te = np.concatenate([clip_te[test_index],
                         edge_te.astype(np.float32)], axis=1)
    del clip_tr, clip_te, edge_tr_file, edge_te_file, edge_tr, edge_te
    gc.collect()

    import torch
    device = torch.device("cuda")
    t = config["training"]
    epochs, lr, wd, batch, seed = (int(t["epochs"]), float(t["lr"]),
                                   float(t["weight_decay"]),
                                   int(t["batch"]), int(t["seed"]))

    rows = np.flatnonzero(train_domains == QUICKDRAW)
    trows = np.flatnonzero(test_domains == QUICKDRAW)
    w, b = _train_probe(tr[rows], labels[rows], epochs, lr, wd, batch,
                        seed, device)
    score = _probe_score(w, b, te[trows], test_labels[trows])
    print(f"quickdraw edge+CLIP probe: {score:.4f}", flush=True)

    w2, b2 = _train_probe(tr[rows], labels[rows], epochs, lr, wd, batch,
                          seed, device)
    score2 = _probe_score(w2, b2, te[trows], test_labels[trows])
    g3_ok = bool(np.array_equal(w, w2) and np.array_equal(b, b2)
                 and score == score2)

    met = score >= 0.8
    gates = {"g2_schedule_alignment": {"ok": bool(g2_ok)},
             "g3_reproducibility": {"ok": bool(g3_ok)}}
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M238",
        "cell": "quickdraw edge+CLIP probe (stroke representation)",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "quickdraw_probe_accuracy": score,
        "wall_reference": {"dino_s": 0.6040, "dino_b": 0.6302,
                           "clip_l": 0.6267, "mlp_concat": 0.6335,
                           "edge_clip": score},
        "ladder_verdict": {"quickdraw": {"best": round(score, 4),
                                         "bar": 0.8, "met": bool(met)}},
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": ("quickdraw >= 0.8 closes the ladder; any gain over "
                        "0.6335 is recorded; otherwise the wall stands with "
                        "the stroke evidence") if gates_ok
            else "a gate failed — VOID",
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok, "quickdraw": score,
                      "met": met}, indent=1), flush=True)
    print(f"M238 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m238(args.config, args.output)


if __name__ == "__main__":
    main()
