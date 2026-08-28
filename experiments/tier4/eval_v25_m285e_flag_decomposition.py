"""M285e — the class-conditioned flag decomposition for c1.

Pre-registered question: do the c1 flags on the first 10k test
rows concentrate in classes with few or no train rows (the
legitimate train-profile outliers), or are they spread across
common classes?

Gates (pre-registered): g1 >= 70% of the flags land in buckets
A∪B (A = 0 train rows — unseen classes; B = 1-50 train rows);
g2 the flag rate in bucket C (51+ train rows) is <= 0.05.

CPU-only. Evidence:
logs/results/v25/m285e_flag_decomposition/evidence.json.
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
                  / "m285e_flag_decomposition")
TRAIN_FEATURES = Path(r"F:\geode-ml\data\cache\v25\m261c_oid_vision"
                      r"\oid_train_137149_feat.npy")
TEST_FEATURES = Path(r"F:\geode-ml\data\cache\v25\m261c_oid_vision"
                     r"\oid_test_245723_feat.npy")
TRAIN_MANIFEST = Path(r"F:\geode-ml\data\cache\oid\manifests"
                      r"\train_manifest.json")
TEST_MANIFEST = Path(r"F:\geode-ml\data\cache\oid\manifests"
                     r"\test_manifest.json")
SEED = 20260831
N_CAL = 20000
N_TEST = 10000
RIDGE = 0.1


def run_m285e(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    train = np.asarray(np.load(TRAIN_FEATURES, mmap_mode="r"),
                       dtype=np.float64)
    test = np.asarray(np.load(TEST_FEATURES, mmap_mode="r"),
                      dtype=np.float64)[:N_TEST]
    train_rows = json.loads(TRAIN_MANIFEST.read_text(
        encoding="utf-8"))["rows"]
    test_rows = json.loads(TEST_MANIFEST.read_text(
        encoding="utf-8"))["rows"][:N_TEST]

    # train-row count per class (mid)
    train_count: dict[str, int] = {}
    for r in train_rows:
        mid = r["label_mid"]
        train_count[mid] = train_count.get(mid, 0) + 1

    def bucket(mid: str) -> str:
        n = train_count.get(mid, 0)
        if n == 0:
            return "A"
        if n <= 50:
            return "B"
        return "C"

    rng = np.random.default_rng(SEED)
    cal_idx = rng.choice(len(train), size=N_CAL, replace=False)
    cal = np.asarray(train[cal_idx], dtype=np.float64)
    mu = cal.mean(axis=0)
    centered = cal - mu
    sigma = (centered.T @ centered) / len(cal)
    inv = np.linalg.inv(sigma + RIDGE * np.eye(sigma.shape[0]))
    cal_scores = np.sqrt(np.einsum("ij,jk,ik->i", cal - mu, inv,
                                   cal - mu))
    op = float(np.quantile(cal_scores, 0.99))
    test_scores = np.sqrt(np.einsum("ij,jk,ik->i", test - mu, inv,
                                    test - mu))
    flags = test_scores > op

    n_flags = int(flags.sum())
    buckets = [bucket(r["label_mid"]) for r in test_rows]
    counts: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    flag_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    for b, f in zip(buckets, flags):
        counts[b] += 1
        if f:
            flag_counts[b] += 1
    share_ab = (flag_counts["A"] + flag_counts["B"]) / n_flags \
        if n_flags else 0.0
    rate_c = flag_counts["C"] / counts["C"] if counts["C"] else 0.0

    g1_ok = share_ab >= 0.70
    g2_ok = rate_c <= 0.05
    evidence: dict[str, Any] = {
        "milestone": "M285e",
        "cell": "class-conditioned flag decomposition (c1)",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "detector": "c1 full-covariance Mahalanobis, ridge 0.1, "
                        "train-p99 operating point",
            "buckets": "A = 0 train rows; B = 1-50; C = 51+",
            "gates": "flags in A∪B >= 70%; flag rate in C <= 0.05",
            "seed": SEED,
        }),
        "results": {
            "operating_point_train_p99": op,
            "n_test_rows": N_TEST,
            "n_flags": n_flags,
            "bucket_row_counts": counts,
            "bucket_flag_counts": flag_counts,
            "bucket_flag_rates": {b: (flag_counts[b] / counts[b]
                                      if counts[b] else 0.0)
                                  for b in ("A", "B", "C")},
            "share_of_flags_in_A_or_B": round(share_ab, 4),
            "flag_rate_in_C": round(rate_c, 4),
            "g1_ok": bool(g1_ok),
            "g2_ok": bool(g2_ok),
            "verdict": ("M285e PASS — the flags are the unseen/"
                        "rare-class rows: the guard is functioning"
                        if (g1_ok and g2_ok) else
                        "M285e FAIL — the flags are spread across "
                        "common classes: c1 is a weak detector"),
        },
        "scope_note": ("train profile = capped class-balanced rows "
                       "(576 of 601 classes); the test split is the "
                       "full long tail"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evidence["results"]}, indent=1),
          flush=True)
    print(f"M285e complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


if __name__ == "__main__":
    run_m285e(DEFAULT_OUTPUT)
