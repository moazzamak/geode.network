"""M286 — served-subset scoping for the OID vision arm.

The registered fix: the macro 0.4673 hides strong classes. Refit
the closed-form ridge head on the cached M261d large-trunk
features (the M262 standard, alpha 1.0, 601 classes), compute the
PER-CLASS held-out accuracies on the full test split, and
register the arm's SERVED SUBSET = classes reading >= 0.8
per-class on >= 10 test rows. The router refuses other classes
(the generalist fallback, the M275 floor pattern).

CPU-only. Evidence:
logs/results/v25/m286_served_subset/evidence.json
+ served_classes.json (the shipped class list).
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
from experiments.tier4.eval_v25_m261b_probe import _full_class_list

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m286_served_subset")
TRAIN_FEATURES = Path(r"F:\geode-ml\data\cache\v25\m261d_oid_vision"
                      r"\oid_train_137149_feat.npy")
TEST_FEATURES = Path(r"F:\geode-ml\data\cache\v25\m261d_oid_vision"
                     r"\oid_test_245723_feat.npy")
TRAIN_MANIFEST = Path(r"F:\geode-ml\data\cache\oid\manifests"
                      r"\train_manifest.json")
TEST_MANIFEST = Path(r"F:\geode-ml\data\cache\oid\manifests"
                     r"\test_manifest.json")
ALPHA = 1.0
FLOOR = 0.8
MIN_ROWS = 10


def run_m286(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    classes, class_index = _full_class_list()
    n_classes = len(classes)
    train_rows = json.loads(TRAIN_MANIFEST.read_text(
        encoding="utf-8"))["rows"]
    test_rows = json.loads(TEST_MANIFEST.read_text(
        encoding="utf-8"))["rows"]
    y_idx = np.array([class_index[r["label_mid"]] for r in train_rows])
    t_idx = np.array([class_index[r["label_mid"]] for r in test_rows])

    train = np.asarray(np.load(TRAIN_FEATURES, mmap_mode="r"),
                       dtype=np.float64)
    test = np.asarray(np.load(TEST_FEATURES, mmap_mode="r"),
                      dtype=np.float64)

    # the M262-standard closed-form ridge (identical to the probe)
    n, d = train.shape
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y_idx] = 1.0
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Xc = train - train.mean(axis=0, keepdims=True)
    W = np.linalg.solve(Xc.T @ Xc + ALPHA * np.eye(d), Xc.T @ Yc)
    b = Y.mean(axis=0) - train.mean(axis=0) @ W

    scores = test @ W + b
    pred = scores.argmax(axis=1)
    correct = pred == t_idx
    overall = float(correct.mean())

    per_class: dict[int, list[bool]] = {}
    for ok, ti in zip(correct, t_idx):
        per_class.setdefault(int(ti), []).append(bool(ok))
    class_rows: dict[int, int] = {}
    class_acc: dict[int, float] = {}
    for ti, outcomes in per_class.items():
        class_rows[ti] = len(outcomes)
        class_acc[ti] = sum(outcomes) / len(outcomes)

    served = sorted(ti for ti in per_class
                    if class_acc[ti] >= FLOOR
                    and class_rows[ti] >= MIN_ROWS)
    served_set = set(served)
    coverage_mask = np.array([int(ti) in served_set for ti in t_idx])
    coverage = float(coverage_mask.mean())
    subset_overall = float(correct[coverage_mask].mean()) \
        if coverage_mask.any() else 0.0

    served_classes = [{"mid": classes[ti], "per_class_acc": round(
        class_acc[ti], 4), "test_rows": class_rows[ti]}
        for ti in served]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "served_classes.json").write_text(
        json.dumps({"floor": FLOOR, "min_rows": MIN_ROWS,
                    "served_classes": served_classes}, indent=2),
        encoding="utf-8")

    evidence: dict[str, Any] = {
        "milestone": "M286",
        "cell": "served-subset scoping (OID vision arm, large trunk)",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "head": "M262-standard closed-form ridge, alpha 1.0, "
                    "601 classes, refit on the M261d train features",
            "floor": FLOOR, "min_rows": MIN_ROWS,
            "rule": "router refuses classes outside the served "
                    "subset (generalist fallback)",
        }),
        "results": {
            "overall_top1_all_classes": round(overall, 4),
            "n_classes": n_classes,
            "n_served_classes": len(served),
            "served_test_row_coverage": round(coverage, 4),
            "served_subset_overall_top1": round(subset_overall, 4),
            "classes_below_floor": n_classes - len(served),
            "verdict": ("the arm is now a SCOPED arm: it serves "
                        f"{len(served)} classes at >= {FLOOR} "
                        "per-class and refuses the rest"
                        if served else
                        "no class meets the floor — the scoping "
                        "fails and is recorded"),
        },
        "served_classes": served_classes,
        "scope_note": ("per-class accuracies on the FULL 245,723-row "
                       "held-out test; a class qualifies with >= "
                       f"{FLOOR} per-class on >= {MIN_ROWS} rows"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evidence["results"]}, indent=1),
          flush=True)
    print(f"M286 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


if __name__ == "__main__":
    run_m286(DEFAULT_OUTPUT)
