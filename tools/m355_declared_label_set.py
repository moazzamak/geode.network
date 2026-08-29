"""M355 declared-label-set scoring (28 Aug 2026).

Registered in ``analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md`` under
G20, whose gate reads: "On the sealed Open Images evidence, show the
scoped arm outranks the full-coverage arm for a buyer declaring the
129 served classes, and is _not_ qualified for a buyer declaring all
601. Both directions required."

G20 is that ``s_a = metric x coverage`` inverts quality: it ranks the
0.901-accurate scoped arm at 0.044 and a 0.164-accurate
full-coverage arm at 0.164, so best-quality mode returns the worse
arm. The proposed replacement conditions the score on a buyer's
DECLARED label set instead of multiplying by coverage.

Order of operations matters here and is not negotiable. The sealed
M286 evidence file did not survive the public-release squash of
``logs/``, so the head is refit from the same cached features. A
refit object is NOT the sealed object until it reproduces the
sealed object's registered values. Clause one of the gate is
therefore an anchor reproduction against the two numbers the
whitepaper publishes -- 0.1643 overall on 601 classes and 0.901 on
the served subset at 4.9% row coverage. Nothing else is read off
this head unless those match.

CPU by construction: this reproduces a sealed CPU contract path and
must not be silently moved to a different arithmetic.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

CACHE = Path(r"F:\geode-ml\data\cache")
TRAIN_FEATURES = CACHE / "v25/m261d_oid_vision/oid_train_137149_feat.npy"
TEST_FEATURES = CACHE / "v25/m261d_oid_vision/oid_test_245723_feat.npy"
TRAIN_MANIFEST = CACHE / "oid/manifests/train_manifest.json"
TEST_MANIFEST = CACHE / "oid/manifests/test_manifest.json"

ALPHA = 1.0        # the M262 standard
FLOOR = 0.8        # per-class accuracy floor for the served subset
MIN_ROWS = 10      # minimum held-out rows for a class to qualify

# Registered anchors from the whitepaper's measured table.
ANCHOR_OVERALL_601 = 0.1643
ANCHOR_SUBSET = 0.901
ANCHOR_COVERAGE = 0.049
ANCHOR_N_SERVED = 129
ANCHOR_TOL = 5e-4


def _labels(manifest: Path, index: dict[str, int]) -> np.ndarray:
    rows = json.loads(manifest.read_text(encoding="utf-8"))["rows"]
    return np.array([index[r["label_mid"]] for r in rows])


def _fit_head() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    train_rows = json.loads(
        TRAIN_MANIFEST.read_text(encoding="utf-8"))["rows"]
    test_rows = json.loads(
        TEST_MANIFEST.read_text(encoding="utf-8"))["rows"]
    classes = sorted({r["label_mid"] for r in train_rows}
                     | {r["label_mid"] for r in test_rows})
    index = {m: i for i, m in enumerate(classes)}
    y_idx = np.array([index[r["label_mid"]] for r in train_rows])
    t_idx = np.array([index[r["label_mid"]] for r in test_rows])
    n_classes = len(classes)

    train = np.asarray(np.load(TRAIN_FEATURES, mmap_mode="r"),
                       dtype=np.float64)
    n, d = train.shape
    y_onehot = np.zeros((n, n_classes), dtype=np.float64)
    y_onehot[np.arange(n), y_idx] = 1.0
    yc = y_onehot - y_onehot.mean(axis=0, keepdims=True)
    xc = train - train.mean(axis=0, keepdims=True)
    w = np.linalg.solve(xc.T @ xc + ALPHA * np.eye(d), xc.T @ yc)
    b = y_onehot.mean(axis=0) - train.mean(axis=0) @ w
    del train, y_onehot, yc, xc

    test = np.asarray(np.load(TEST_FEATURES, mmap_mode="r"),
                      dtype=np.float64)
    scores = test @ w + b
    return scores, t_idx, np.array(classes), n_classes


def main() -> int:
    started = time.time()
    scores, t_idx, classes, n_classes = _fit_head()

    pred_full = scores.argmax(axis=1)
    correct_full = pred_full == t_idx
    overall = float(correct_full.mean())

    per_class_rows = np.bincount(t_idx, minlength=n_classes)
    per_class_hits = np.bincount(t_idx, weights=correct_full,
                                 minlength=n_classes)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class_acc = np.where(per_class_rows > 0,
                                 per_class_hits / per_class_rows, 0.0)
    served = np.flatnonzero((per_class_acc >= FLOOR)
                            & (per_class_rows >= MIN_ROWS))
    served_mask_rows = np.isin(t_idx, served)
    coverage = float(served_mask_rows.mean())
    subset_overall = float(correct_full[served_mask_rows].mean())

    # --- gate clause 1: anchor reproduction ------------------------
    anchors = {
        "overall_601": {"measured": round(overall, 4),
                        "registered": ANCHOR_OVERALL_601},
        "served_subset": {"measured": round(subset_overall, 4),
                          "registered": ANCHOR_SUBSET},
        "row_coverage": {"measured": round(coverage, 4),
                         "registered": ANCHOR_COVERAGE},
        "n_served_classes": {"measured": int(served.size),
                             "registered": ANCHOR_N_SERVED},
    }
    reproduced = (
        abs(overall - ANCHOR_OVERALL_601) <= ANCHOR_TOL
        and abs(subset_overall - ANCHOR_SUBSET) <= ANCHOR_TOL
        and abs(coverage - ANCHOR_COVERAGE) <= ANCHOR_TOL
        and int(served.size) == ANCHOR_N_SERVED
    )
    print("anchor reproduction:", json.dumps(anchors, indent=1))
    print("reproduced:", reproduced)

    payload: dict[str, Any] = {
        "milestone": "M355",
        "finding": "G20 -- coverage multiplication inverts quality",
        "registered_in": "analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md",
        "gate_clause_1_anchor_reproduction": {
            "anchors": anchors, "reproduced": reproduced,
            "tolerance": ANCHOR_TOL,
            "note": "the sealed M286 evidence did not survive the "
                    "logs/ squash; the head is refit from the same "
                    "cached features and must reproduce the "
                    "published values before anything is read off it",
        },
    }
    out = Path("analysis/m355_declared_label_set.json")

    if not reproduced:
        payload["verdict"] = ("VOID -- the refit head does not "
                              "reproduce the registered anchors, so "
                              "no declared-set figure may be read "
                              "off it")
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"VOID -> {out}")
        return 1

    # --- gate clause 2: the two declared-set directions ------------
    # Declared set D = the 129 served classes. Both arms are scored
    # on the rows whose true class lies in D. The scoped arm argmaxes
    # only over the classes it serves; the full-coverage arm argmaxes
    # over all 601 and so keeps 472 distractors in play.
    rows_d = served_mask_rows
    scoped_pred = served[scores[np.ix_(rows_d, served)].argmax(axis=1)]
    scoped_acc = float((scoped_pred == t_idx[rows_d]).mean())
    full_acc_on_d = float(correct_full[rows_d].mean())

    old_scoped = subset_overall * coverage
    old_full = overall * 1.0

    payload["gate_clause_2_declared_129"] = {
        "declared_classes": int(served.size),
        "test_rows_in_declared_set": int(rows_d.sum()),
        "scoped_arm": {"qualified": True, "s_a": round(scoped_acc, 4),
                       "coverage_within_declared_set": 1.0},
        "full_coverage_arm": {"qualified": True,
                              "s_a": round(full_acc_on_d, 4),
                              "coverage_within_declared_set": 1.0},
        "scoped_outranks_full": scoped_acc > full_acc_on_d,
    }
    payload["gate_clause_3_declared_601"] = {
        "declared_classes": int(n_classes),
        "scoped_arm": {
            "qualified": False,
            "reason": f"covers {served.size} of {n_classes} declared "
                      "classes; qualification is a membership "
                      "question, not a ranking penalty",
        },
        "full_coverage_arm": {"qualified": True,
                              "s_a": round(overall, 4)},
    }
    payload["old_rule_for_contrast"] = {
        "rule": "s_a = metric x coverage",
        "scoped": round(old_scoped, 4),
        "full_coverage": round(old_full, 4),
        "inverted": old_scoped < old_full,
    }
    passed = (payload["gate_clause_2_declared_129"]
              ["scoped_outranks_full"])
    payload["verdict"] = "PASS" if passed else "FAIL"
    payload["runtime_seconds"] = round(time.time() - started, 1)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\ndeclared D=129: scoped {scoped_acc:.4f} vs "
          f"full-coverage {full_acc_on_d:.4f} -> "
          f"scoped wins: {scoped_acc > full_acc_on_d}")
    print(f"declared D=601: scoped NOT qualified; full-coverage "
          f"{overall:.4f}")
    print(f"old rule: scoped {old_scoped:.4f} vs full {old_full:.4f} "
          f"-> inverted: {old_scoped < old_full}")
    print(f"verdict: {payload['verdict']} -> {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
