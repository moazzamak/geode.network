"""M285 — planted-OOD guard calibration.

The OodGate's firing behavior on genuinely out-of-distribution
inputs was untested (the M261b seal caveat: the 0.0 flag rate is
test==train distribution plus a threshold far above the observed
range). This cell plants a deterministic synthetic OOD set —
uniform noise, checkerboards, and ramps, seeded — extracts its
features with the REGISTERED DINOv2-base trunk, and scores them
with the M261c-fitted OodGate (threshold 3.0).

Gates: the flag rate is measured and recorded — no a-priori bar;
a ~0 rate means the guard is dead on planted-OOD (the recorded
risk), a high rate closes the caveat. Device caveat recorded:
CPU extraction here vs the GPU-fit train profile.

CPU-only (the GPU is running the ladder cells). Evidence:
logs/results/v25/m285_ood_calibration/evidence.json.
"""
from __future__ import annotations

import json
import time
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m285_ood_calibration")
TRAIN_FEATURES = Path(r"F:\geode-ml\data\cache\v25\m261c_oid_vision"
                      r"\oid_train_137149_feat.npy")
SEED = 20260830
N_PER_KIND = 334
SIZE = 224


def _synthetic_images() -> list[np.ndarray]:
    """1,000 seeded synthetic images: uniform noise, checkerboards,
    ramps — none of which is an OID photograph."""
    rng = np.random.default_rng(SEED)
    out: list[np.ndarray] = []
    for _ in range(N_PER_KIND):
        out.append(rng.uniform(0, 255, size=(SIZE, SIZE, 3))
                  .astype(np.uint8))
    for _ in range(N_PER_KIND):
        base = rng.uniform(0, 255, size=(SIZE // 8, SIZE // 8, 3))
        out.append(np.kron(base, np.ones((8, 8, 1))).astype(np.uint8))
    for _ in range(1000 - 2 * N_PER_KIND):
        x = np.linspace(0, 255, SIZE, dtype=np.float32)
        y = np.linspace(0, 255, SIZE, dtype=np.float32)
        g = (x[:, None] + y[None, :]) / 2.0
        arr = np.stack([g] * 3, axis=-1).astype(np.uint8)
        out.append(arr)
    return out


def run_m285(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    configure_external_cache_environment()

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
    features: list[np.ndarray] = []
    batch = 64
    with torch.no_grad():
        for start in range(0, len(images), batch):
            chunk = [Image.fromarray(a) for a in images[start:start + batch]]
            enc = proc(images=chunk, return_tensors="pt")
            out = model(**enc).last_hidden_state[:, 0].numpy()
            features.extend(list(out))
    feats = np.asarray(features, dtype=np.float64)

    from geode.core.ood import OodGate
    train_feats = np.load(TRAIN_FEATURES, mmap_mode="r")
    gate = OodGate(threshold=3.0)
    gate.fit_profile(np.asarray(train_feats, dtype=np.float64).tolist())
    scores = np.array([gate.score(v.tolist()) for v in feats])
    flag_rate = float(np.mean(scores > gate.threshold))
    quantiles = [float(q) for q in
                 np.quantile(scores, [0, .5, .9, .99, 1.0])]

    evidence: dict[str, Any] = {
        "milestone": "M285",
        "cell": "planted-OOD guard calibration",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "planted_set": "1,000 seeded synthetic images (noise / "
                           "checkerboards / ramps)",
            "seed": SEED,
            "trunk": "DINOv2-base (the registered vision trunk)",
            "gate": "OodGate, diagonal Mahalanobis, threshold 3.0, "
                    "profile fit on the M261c train features",
        }),
        "results": {
            "n_planted": len(images),
            "flag_rate": round(flag_rate, 4),
            "score_quantiles_min_p50_p90_p99_max": quantiles,
            "verdict": ("the guard FIRES on planted-OOD — the "
                        "calibration caveat is closed"
                        if flag_rate >= 0.5 else
                        "the guard is WEAK/DEAD on planted-OOD — "
                        "recorded, a threshold/profile repair is "
                        "registered as the finding"),
        },
        "scope_note": ("device caveat: planted features extracted on "
                       "CPU, the train profile was fit on GPU-extracted "
                       "features (the registered arm pipeline)"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evidence["results"]}, indent=1),
          flush=True)
    print(f"M285 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


if __name__ == "__main__":
    run_m285(DEFAULT_OUTPUT)
