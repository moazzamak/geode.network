"""M285b — the guard repair: full-covariance Mahalanobis.

The M285 finding: the diagonal Mahalanobis OodGate is dead on
planted-OOD (synthetic noise scores in the same band as real
images, flag rate 0.0). Repair: the classic full-covariance
Mahalanobis distance (x-mu)^T Sigma^-1 (x-mu) with a small ridge
on the covariance, fit on the SAME M261c train features.

Gates: g1 the planted-OOD flag rate at the SAME threshold 3.0 is
measured (no post-hoc tuning); g2 the in-distribution flag rate
(the cached M261b test features) stays near 0. A repair that
fails g1 is recorded as failed — no threshold dialing after the
fact.

CPU-only. Evidence:
logs/results/v25/m285b_ood_repair/evidence.json.
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
from experiments.tier4.eval_v25_m285_ood_calibration import (
    _synthetic_images,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m285b_ood_repair")
TRAIN_FEATURES = Path(r"F:\geode-ml\data\cache\v25\m261c_oid_vision"
                      r"\oid_train_137149_feat.npy")
TEST_FEATURES = Path(r"F:\geode-ml\data\cache\v25\m261c_oid_vision"
                     r"\oid_test_245723_feat.npy")
PLANTED_FEATURES = Path(r"F:\geode-ml\data\cache\v25\m285_ood_repair"
                        r"\planted_feats.npy")
RIDGE = 1e-3
THRESHOLD = 3.0


def _planted_features() -> np.ndarray:
    """Extract (once) the planted-OOD features with the registered
    DINOv2-base trunk on CPU."""
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
    images = _synthetic_images()
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), 64):
            chunk = [Image.fromarray(a)
                     for a in images[start:start + 64]]
            enc = proc(images=chunk, return_tensors="pt")
            out = model(**enc).last_hidden_state[:, 0].numpy()
            feats.extend(list(out))
    arr = np.asarray(feats, dtype=np.float64)
    PLANTED_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    np.save(PLANTED_FEATURES, arr)
    return arr


def run_m285b(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    train = np.asarray(np.load(TRAIN_FEATURES, mmap_mode="r"),
                       dtype=np.float64)
    test = np.asarray(np.load(TEST_FEATURES, mmap_mode="r"),
                      dtype=np.float64)
    planted = _planted_features()

    mu = train.mean(axis=0)
    centered = train - mu
    sigma = (centered.T @ centered) / len(train)
    d = sigma.shape[0]
    sigma_r = sigma + RIDGE * np.eye(d)
    inv = np.linalg.inv(sigma_r)

    def score(X: np.ndarray) -> np.ndarray:
        delta = X - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", delta, inv, delta))

    # g2 first (cheap read on the registered in-distribution test)
    test_scores = score(test[:50000])
    id_flag_rate = float(np.mean(test_scores > THRESHOLD))
    # g1: planted-OOD
    planted_scores = score(planted)
    ood_flag_rate = float(np.mean(planted_scores > THRESHOLD))
    planted_q = [float(q) for q in
                 np.quantile(planted_scores, [0, .5, .9, .99, 1.0])]
    test_q = [float(q) for q in
              np.quantile(test_scores, [0, .5, .9, .99, 1.0])]

    g1_ok = ood_flag_rate >= 0.5
    g2_ok = id_flag_rate <= 0.05
    evidence: dict[str, Any] = {
        "milestone": "M285b",
        "cell": "guard repair — full-covariance Mahalanobis",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "profile": "full-covariance Mahalanobis, ridge 1e-3, "
                       "fit on the M261c train features",
            "threshold": THRESHOLD,
            "g1": "planted-OOD flag rate >= 0.5 (measured, no "
                  "post-hoc tuning)",
            "g2": "in-distribution flag rate <= 0.05 on the first "
                  "50k cached M261c test features",
        }),
        "results": {
            "ood_flag_rate": round(ood_flag_rate, 4),
            "id_flag_rate": round(id_flag_rate, 4),
            "planted_score_quantiles": planted_q,
            "id_score_quantiles": test_q,
            "g1_ok": bool(g1_ok),
            "g2_ok": bool(g2_ok),
            "verdict": ("M285b PASS — the full-covariance guard "
                        "separates planted-OOD from in-distribution"
                        if (g1_ok and g2_ok) else
                        "M285b FAIL — the repair does not separate; "
                        "recorded, no threshold dialing"),
        },
        "scope_note": ("same threshold and same fit features as the "
                       "dead diagonal gate; device caveat: planted "
                       "features extracted on CPU"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evidence["results"]}, indent=1),
          flush=True)
    print(f"M285b complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


if __name__ == "__main__":
    run_m285b(DEFAULT_OUTPUT)
