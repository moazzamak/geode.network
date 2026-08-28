"""M285f — the energy-score OOD detector (the final guard cell of
the wave).

Score = -logsumexp over the REGISTERED ridge head's 601 logits
(the free energy, Liu et al. 2020, arXiv:2010.03759). The head is
refit identically to the M261c protocol (closed-form ridge,
alpha 1.0, centered one-hot) on the same train features.

Gates (pre-registered): the SAME train-p99 operating point on a
seeded 20k train subsample; sketch planted flag rate >= 0.5;
in-distribution flag rate <= 0.05 on the first 10k cached test
rows. Whatever the result, the guard finding closes as a user
decision point after this cell.

CPU-only. Evidence:
logs/results/v25/m285f_energy_score/evidence.json.
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
                  / "m285f_energy_score")
TRAIN_FEATURES = Path(r"F:\geode-ml\data\cache\v25\m261c_oid_vision"
                      r"\oid_train_137149_feat.npy")
TEST_FEATURES = Path(r"F:\geode-ml\data\cache\v25\m261c_oid_vision"
                     r"\oid_test_245723_feat.npy")
PLANTED_FEATURES = Path(r"F:\geode-ml\data\cache\quickdraw"
                        r"\sketch_feats.npy")
TRAIN_MANIFEST = Path(r"F:\geode-ml\data\cache\oid\manifests"
                      r"\train_manifest.json")
SEED = 20260831
N_CAL = 20000
N_TEST = 10000
ALPHA = 1.0


def _fit_head(train: np.ndarray, y_idx: np.ndarray, n_classes: int
              ) -> tuple[np.ndarray, np.ndarray]:
    n, d = train.shape
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y_idx] = 1.0
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Xc = train - train.mean(axis=0, keepdims=True)
    A = Xc.T @ Xc + ALPHA * np.eye(d)
    W = np.linalg.solve(A, Xc.T @ Yc)
    b = Y.mean(axis=0) - train.mean(axis=0) @ W
    return W, b


def _energy(logits: np.ndarray) -> np.ndarray:
    return -np.log(np.exp(logits - logits.max(axis=1, keepdims=True))
                   .sum(axis=1)) - logits.max(axis=1)


def run_m285f(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    classes, class_index = _full_class_list()
    n_classes = len(classes)
    train_rows = json.loads(TRAIN_MANIFEST.read_text(
        encoding="utf-8"))["rows"]
    y_idx = np.array([class_index[r["label_mid"]] for r in train_rows])

    train = np.asarray(np.load(TRAIN_FEATURES, mmap_mode="r"),
                       dtype=np.float64)
    test = np.asarray(np.load(TEST_FEATURES, mmap_mode="r"),
                      dtype=np.float64)[:N_TEST]
    planted = np.asarray(np.load(PLANTED_FEATURES), dtype=np.float64)

    W, b = _fit_head(train, y_idx, n_classes)
    rng = np.random.default_rng(SEED)
    cal_idx = rng.choice(len(train), size=N_CAL, replace=False)
    cal = np.asarray(train[cal_idx], dtype=np.float64)
    cal_energy = _energy(cal @ W + b)
    op = float(np.quantile(cal_energy, 0.99))
    test_energy = _energy(test @ W + b)
    planted_energy = _energy(planted @ W + b)

    ood_rate = float(np.mean(planted_energy > op))
    id_rate = float(np.mean(test_energy > op))
    g1_ok = ood_rate >= 0.5
    g2_ok = id_rate <= 0.05
    evidence: dict[str, Any] = {
        "milestone": "M285f",
        "cell": "energy-score OOD detector (ridge-head free energy)",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "head": "M261c-identical closed-form ridge, alpha 1.0, "
                    "601 classes",
            "score": "free energy over the 601 logits "
                     "(-logsumexp, Liu et al. 2020)",
            "operating_point": "train p99 on a seeded 20k subsample",
            "gates": "sketch flag rate >= 0.5; id flag rate <= 0.05 "
                     "(first 10k test rows)",
            "seed": SEED,
        }),
        "results": {
            "operating_point_train_p99": op,
            "sketch_flag_rate": round(ood_rate, 4),
            "id_flag_rate": round(id_rate, 4),
            "sketch_energy_quantiles": [float(q) for q in np.quantile(
                planted_energy, [0, .5, .9, .99, 1.0])],
            "id_energy_quantiles": [float(q) for q in np.quantile(
                test_energy, [0, .5, .9, .99, 1.0])],
            "g1_ok": bool(g1_ok),
            "g2_ok": bool(g2_ok),
            "verdict": ("M285f PASS — the energy score separates "
                        "sketches from photos at the pre-registered "
                        "gates" if (g1_ok and g2_ok) else
                        "M285f FAIL — recorded; the guard finding "
                        "closes as a user decision point"),
        },
        "scope_note": ("the final guard cell of the wave; the "
                       "sketch planted set is the registered OOD "
                       "class; contamination declared (LVD-142M "
                       "may include sketches)"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evidence["results"]}, indent=1),
          flush=True)
    print(f"M285f complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


if __name__ == "__main__":
    run_m285f(DEFAULT_OUTPUT)
