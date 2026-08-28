"""M285c — the OOD-detector candidate grid, registered BEFORE any
cell runs.

Four candidates, all measured against the SAME gates and ALL
reported (no cherry-picking):
- c1 full-covariance Mahalanobis, ridge 0.1
- c2 full-covariance Mahalanobis, ridge 1.0
- c3 kNN distance (k=5) on the top-256 PCA projection, 2k seeded
  train anchors
- c4 spectral Mahalanobis in the top-256 PCA subspace, ridge 1e-3

Operating point (registered BEFORE any cell runs): the 99th
percentile of TRAIN scores per candidate — computed on train only
(no test leakage). The naive fixed threshold 3.0 is also reported.
GATES at the train-p99 operating point: planted-OOD flag rate
>= 0.5 AND in-distribution flag rate <= 0.05 (first 10k cached
test features). A candidate passes only with both gates.

CPU-only. Evidence:
logs/results/v25/m285c_ood_candidates/evidence.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m285c_ood_candidates")
TRAIN_FEATURES = Path(r"F:\geode-ml\data\cache\v25\m261c_oid_vision"
                      r"\oid_train_137149_feat.npy")
TEST_FEATURES = Path(r"F:\geode-ml\data\cache\v25\m261c_oid_vision"
                     r"\oid_test_245723_feat.npy")
PLANTED_FEATURES = Path(r"F:\geode-ml\data\cache\v25\m285_ood_repair"
                        r"\planted_feats.npy")
SEED = 20260831
N_TRAIN_CAL = 20000
N_TEST = 10000
N_ANCHORS = 2000


def _pca_subspace(centered: np.ndarray, k: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Top-k PCA from the covariance (768x768 eigendecomposition)."""
    sigma = (centered.T @ centered) / len(centered)
    vals, vecs = np.linalg.eigh(sigma)
    idx = np.argsort(vals)[::-1][:k]
    return vals[idx], vecs[:, idx]


def run_m285c(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    train = np.asarray(np.load(TRAIN_FEATURES, mmap_mode="r"),
                       dtype=np.float64)
    test = np.asarray(np.load(TEST_FEATURES, mmap_mode="r"),
                      dtype=np.float64)[:N_TEST]
    planted = np.asarray(np.load(PLANTED_FEATURES), dtype=np.float64)

    rng = np.random.default_rng(SEED)
    cal_idx = rng.choice(len(train), size=N_TRAIN_CAL, replace=False)
    cal = np.asarray(train[cal_idx], dtype=np.float64)
    mu = cal.mean(axis=0)
    centered = cal - mu

    cells: list[dict[str, Any]] = []

    # ---- c1 / c2: full-covariance with strong ridges -----------------
    for ridge, name in ((0.1, "c1_fullcov_ridge_0.1"),
                        (1.0, "c2_fullcov_ridge_1.0")):
        sigma = (centered.T @ centered) / len(cal)
        inv = np.linalg.inv(sigma + ridge * np.eye(sigma.shape[0]))
        delta = cal - mu
        cal_scores = np.sqrt(np.einsum("ij,jk,ik->i", delta, inv, delta))
        t_delta = test - mu
        p_delta = planted - mu
        cells.append({
            "candidate": name,
            "train_p99": float(np.quantile(cal_scores, 0.99)),
            "id_scores_quantiles": [float(q) for q in np.quantile(
                np.sqrt(np.einsum("ij,jk,ik->i", t_delta, inv, t_delta)),
                [0, .5, .9, .99, 1.0])],
            "ood_scores_quantiles": [float(q) for q in np.quantile(
                np.sqrt(np.einsum("ij,jk,ik->i", p_delta, inv, p_delta)),
                [0, .5, .9, .99, 1.0])],
        })

    # ---- shared PCA subspace (256) for c3 / c4 -----------------------
    vals, vecs = _pca_subspace(centered, 256)
    proj = centered @ vecs
    proj_mu = proj.mean(axis=0)
    proj_centered = proj - proj_mu
    test_proj = (test - mu) @ vecs
    planted_proj = (planted - mu) @ vecs

    # c4: spectral Mahalanobis in the subspace
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

    # c3: kNN distance (k=5) on the subspace, 2k anchors
    from sklearn.neighbors import NearestNeighbors
    anchor_idx = rng.choice(len(proj), size=N_ANCHORS, replace=False)
    anchors = np.asarray(proj[anchor_idx], dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=6, algorithm="brute",
                          metric="euclidean").fit(anchors)
    def _knn_dist(X: np.ndarray) -> np.ndarray:
        out: list[np.ndarray] = []
        for start in range(0, len(X), 512):
            d, _ = nn.kneighbors(X[start:start + 512])
            out.append(d[:, 1:].mean(axis=1))  # exclude self (k=5 others)
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

    # ---- the pre-registered gate at the train-p99 operating point ----
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
        "milestone": "M285c",
        "cell": "OOD-detector candidate grid (pre-registered)",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "candidates": ["c1 ridge 0.1", "c2 ridge 1.0",
                           "c3 knn5 subspace 2k anchors",
                           "c4 spectral 256 ridge 1e-3"],
            "operating_point": "train p99 per candidate (train only)",
            "gates": "ood >= 0.5 AND id <= 0.05 at the operating point",
            "seed": SEED,
        }),
        "results": {
            "cells": cells,
            "verdicts": verdicts,
            "any_pass": any(v["passed"] for v in verdicts),
        },
        "scope_note": ("all candidates measured and reported, no "
                       "cherry-picking; the operating point is "
                       "train-side only; the naive threshold 3.0 is "
                       "implied by the quantiles reported"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": {"verdicts": verdicts,
                                  "any_pass": evidence["results"][
                                      "any_pass"]}}, indent=1), flush=True)
    print(f"M285c complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def _rates(c: dict[str, Any], test: np.ndarray, planted: np.ndarray,
           mu: np.ndarray, vecs: np.ndarray, centered: np.ndarray,
           op: float) -> tuple[float, float]:
    """Recompute id/ood flag rates at the operating point for the
    given candidate (re-derives scores, never reuses quantiles)."""
    name = c["candidate"]
    if name.startswith("c1") or name.startswith("c2"):
        ridge = 0.1 if name.startswith("c1") else 1.0
        sigma = (centered.T @ centered) / len(centered)
        inv = np.linalg.inv(sigma + ridge * np.eye(sigma.shape[0]))
        id_s = np.sqrt(np.einsum("ij,jk,ik->i", test - mu, inv,
                                 test - mu))
        ood_s = np.sqrt(np.einsum("ij,jk,ik->i", planted - mu, inv,
                                  planted - mu))
    elif name.startswith("c4"):
        proj = centered @ vecs
        pmu = proj.mean(axis=0)
        pc = proj - pmu
        sigma_s = (pc.T @ pc) / len(pc)
        inv_s = np.linalg.inv(sigma_s + 1e-3 * np.eye(256))
        def _sm(X: np.ndarray) -> np.ndarray:
            d = (X - mu) @ vecs - pmu
            return np.sqrt(np.einsum("ij,jk,ik->i", d, inv_s, d))
        id_s = _sm(test)
        ood_s = _sm(planted)
    else:  # c3
        from sklearn.neighbors import NearestNeighbors
        rng = np.random.default_rng(SEED)
        proj = centered @ vecs
        anchor_idx = rng.choice(len(proj), size=N_ANCHORS, replace=False)
        anchors = np.asarray(proj[anchor_idx], dtype=np.float64)
        nn = NearestNeighbors(n_neighbors=6, algorithm="brute",
                              metric="euclidean").fit(anchors)
        def _knn(X: np.ndarray) -> np.ndarray:
            out = []
            for start in range(0, len(X), 512):
                d, _ = nn.kneighbors(((X - mu) @ vecs)[start:start + 512])
                out.append(d[:, 1:].mean(axis=1))
            return np.concatenate(out)
        id_s = _knn(test)
        ood_s = _knn(planted)
    return (float(np.mean(id_s > op)), float(np.mean(ood_s > op)))


if __name__ == "__main__":
    run_m285c(DEFAULT_OUTPUT)
