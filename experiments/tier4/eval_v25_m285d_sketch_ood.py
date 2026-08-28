"""M285d — the sketch planted set: the same candidate grid (c1-c4)
re-run with a planted set that is genuinely outside the photo
manifold — QuickDraw sketches (CC BY 4.0), rasterized to 224x224.

The M285c finding is the premise: frozen DINOv2 maps synthetic
noise to the CENTER of the feature distribution, so noise is not
OOD for a photo encoder. This cell keeps every detector and gate
identical and swaps only the planted set.

Gates (unchanged): train-p99 operating point per candidate
(train-side only); planted flag rate >= 0.5; in-distribution flag
rate <= 0.05 on the first 10k cached test features.

CPU-only. Evidence:
logs/results/v25/m285d_sketch_ood/evidence.json.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v25_m285c_ood_candidates import (
    N_ANCHORS,
    N_TEST,
    N_TRAIN_CAL,
    SEED,
    _pca_subspace,
    _rates,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m285d_sketch_ood")
CACHE = Path(r"F:\geode-ml\data\cache\quickdraw")
SKETCH_CLASSES = ["cat", "airplane", "flower", "car", "house"]
N_PER_CLASS = 200
TRAIN_FEATURES = Path(r"F:\geode-ml\data\cache\v25\m261c_oid_vision"
                      r"\oid_train_137149_feat.npy")
TEST_FEATURES = Path(r"F:\geode-ml\data\cache\v25\m261c_oid_vision"
                     r"\oid_test_245723_feat.npy")
PLANTED_FEATURES = CACHE / "sketch_feats.npy"
_URL = ("https://storage.googleapis.com/quickdraw_dataset/full/"
        "numpy_bitmap/{cls}.npy")


def _sketch_images() -> list[np.ndarray]:
    """200 sketches per class, 28x28 -> 224x224 RGB."""
    CACHE.mkdir(parents=True, exist_ok=True)
    images: list[np.ndarray] = []
    for cls in SKETCH_CLASSES:
        path = CACHE / f"{cls}.npy"
        if not path.exists():
            urllib.request.urlretrieve(_URL.format(cls=cls), path)
        arr = np.load(path)[:N_PER_CLASS]  # the first 200 rows
        for row in arr:
            img = row.reshape(28, 28).astype(np.uint8)
            big = np.kron(img, np.ones((8, 8), dtype=np.uint8))
            images.append(np.stack([big] * 3, axis=-1))
    return images


def _planted_features() -> np.ndarray:
    if PLANTED_FEATURES.exists():
        return np.load(PLANTED_FEATURES)
    import torch
    torch.backends.cudnn.enabled = False
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    ckpt = "F:/geode-ml/data/cache/huggingface/dinov2-base"
    proc = AutoImageProcessor.from_pretrained(ckpt,
                                              local_files_only=True)
    model = AutoModel.from_pretrained(ckpt,
                                      local_files_only=True).eval()
    images = _sketch_images()
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), 64):
            chunk = [Image.fromarray(a)
                     for a in images[start:start + 64]]
            enc = proc(images=chunk, return_tensors="pt")
            out = model(**enc).last_hidden_state[:, 0].numpy()
            feats.extend(list(out))
    arr = np.asarray(feats, dtype=np.float64)
    np.save(PLANTED_FEATURES, arr)
    return arr


def run_m285d(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    train = np.asarray(np.load(TRAIN_FEATURES, mmap_mode="r"),
                       dtype=np.float64)
    test = np.asarray(np.load(TEST_FEATURES, mmap_mode="r"),
                      dtype=np.float64)[:N_TEST]
    planted = _planted_features()

    rng = np.random.default_rng(SEED)
    cal_idx = rng.choice(len(train), size=N_TRAIN_CAL, replace=False)
    cal = np.asarray(train[cal_idx], dtype=np.float64)
    mu = cal.mean(axis=0)
    centered = cal - mu

    cells: list[dict[str, Any]] = []
    for ridge, name in ((0.1, "c1_fullcov_ridge_0.1"),
                        (1.0, "c2_fullcov_ridge_1.0")):
        sigma = (centered.T @ centered) / len(cal)
        inv = np.linalg.inv(sigma + ridge * np.eye(sigma.shape[0]))
        delta = cal - mu
        cal_scores = np.sqrt(np.einsum("ij,jk,ik->i", delta, inv, delta))
        cells.append({
            "candidate": name,
            "train_p99": float(np.quantile(cal_scores, 0.99)),
            "id_scores_quantiles": [float(q) for q in np.quantile(
                np.sqrt(np.einsum("ij,jk,ik->i", test - mu, inv,
                                  test - mu)),
                [0, .5, .9, .99, 1.0])],
            "ood_scores_quantiles": [float(q) for q in np.quantile(
                np.sqrt(np.einsum("ij,jk,ik->i", planted - mu, inv,
                                  planted - mu)),
                [0, .5, .9, .99, 1.0])],
        })

    vals, vecs = _pca_subspace(centered, 256)
    proj = centered @ vecs
    proj_mu = proj.mean(axis=0)
    proj_centered = proj - proj_mu
    test_proj = (test - mu) @ vecs
    planted_proj = (planted - mu) @ vecs

    sigma_s = (proj_centered.T @ proj_centered) / len(proj)
    inv_s = np.linalg.inv(sigma_s + 1e-3 * np.eye(256))
    def _sm(X: np.ndarray) -> np.ndarray:
        d = X - proj_mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, inv_s, d))
    cells.append({
        "candidate": "c4_spectral_256_ridge_1e-3",
        "train_p99": float(np.quantile(_sm(proj), 0.99)),
        "id_scores_quantiles": [float(q) for q in
                                np.quantile(_sm(test_proj),
                                            [0, .5, .9, .99, 1.0])],
        "ood_scores_quantiles": [float(q) for q in
                                 np.quantile(_sm(planted_proj),
                                             [0, .5, .9, .99, 1.0])],
    })

    from sklearn.neighbors import NearestNeighbors
    anchor_idx = rng.choice(len(proj), size=N_ANCHORS, replace=False)
    anchors = np.asarray(proj[anchor_idx], dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=6, algorithm="brute",
                          metric="euclidean").fit(anchors)
    def _knn_dist(X: np.ndarray) -> np.ndarray:
        out: list[np.ndarray] = []
        for start in range(0, len(X), 512):
            d, _ = nn.kneighbors(X[start:start + 512])
            out.append(d[:, 1:].mean(axis=1))
        return np.concatenate(out)
    cal_knn = _knn_dist(proj[:N_TRAIN_CAL // 4])
    cells.append({
        "candidate": "c3_knn5_subspace_2k_anchors",
        "train_p99": float(np.quantile(cal_knn, 0.99)),
        "id_scores_quantiles": [float(q) for q in
                                np.quantile(_knn_dist(test_proj),
                                            [0, .5, .9, .99, 1.0])],
        "ood_scores_quantiles": [float(q) for q in
                                 np.quantile(_knn_dist(planted_proj),
                                             [0, .5, .9, .99, 1.0])],
    })

    verdicts: list[dict[str, Any]] = []
    for c in cells:
        op = c["train_p99"]
        id_rate, ood_rate = _rates(c, test, planted, mu, vecs,
                                   centered, op)
        passed = ood_rate >= 0.5 and id_rate <= 0.05
        verdicts.append({
            "candidate": c["candidate"],
            "operating_point_train_p99": op,
            "id_flag_rate_at_op": round(id_rate, 4),
            "ood_flag_rate_at_op": round(ood_rate, 4),
            "passed": bool(passed),
        })

    evidence: dict[str, Any] = {
        "milestone": "M285d",
        "cell": "sketch planted set — the same candidate grid",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "planted_set": ("1,000 QuickDraw sketches (cat/airplane/"
                            "flower/car/house, first 200 rows each), "
                            "28x28 -> 224x224, CC BY 4.0"),
            "candidates": "c1-c4 (identical to M285c)",
            "gates": "ood >= 0.5 AND id <= 0.05 at the train-p99 "
                     "operating point",
            "seed": SEED,
        }),
        "results": {
            "cells": cells,
            "verdicts": verdicts,
            "any_pass": any(v["passed"] for v in verdicts),
        },
        "scope_note": ("the planted set is the only change from "
                       "M285c; contamination declared: sketches may "
                       "appear in LVD-142M"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": {"verdicts": verdicts,
                                  "any_pass": evidence["results"][
                                      "any_pass"]}}, indent=1), flush=True)
    print(f"M285d complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


if __name__ == "__main__":
    run_m285d(DEFAULT_OUTPUT)
