"""M345 - the RFF reading on the sealed audio axis (M266b).

Registered in ``analysis/SCIENCE_LAYER_PLAN_2026-08-28.md`` (M345
REGISTRATION AMENDMENT, 28 Aug 2026, before the build). M266b sealed
the audio recipe's base reading (frozen wav2vec2-base mean-pooled
features + closed-form ridge, alpha=1.0: Speech Commands v2 test
0.878691503861881, 35 classes). M345 asks whether the M300 RFF map
(D=16384, sigma=0.5, seed 20260828) lifts that reading - the audio
twin of M344.

Arms, both on the SAME cached features (the M266b feature caches ARE
retained on F:):
- linear_reproduction: the closed-form ridge re-fit on the cached
  train features; its weights+bias sha256 must equal the sealed
  evidence weights_hash and its test accuracy must equal the sealed
  0.878691503861881 (the g1 instrument-identity gate, the M344 g1
  amendment form).
- rff: [features, phi(features)] with the same closed-form ridge
  head and the same alpha=1.0, scored once on the same test split
  (the g2 reading).

CPU-only: the features are cached, so no encoder is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
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
from experiments.tier4.eval_v26_m344_text_rff import (
    _ridge_probe,
    _ridge_probe_design,
    _score_design,
)
from experiments.tier4.eval_v26_m300_rff_quickdraw import rff_params

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m345_audio_rff")
CACHE_REL = "v25/m266_audio_arm"

# the registered M300 map, applied to the audio axis
RFF_DIM = 16384
RFF_SIGMA = 0.5
RFF_SEED = 20260828

RIDGE_ALPHA = 1.0          # the M266b sealed probe's alpha
REPRO_TOL = 1e-9

# the sealed M266b reading (logs/results/v25/m266_audio_arm/
# evidence_m266b.json, 21 Aug 2026) - the g1 targets
M266B_SEALED_ACC = 0.878691503861881
M266B_WEIGHTS_HASH = ("e19b69f53160ad163c35d8da4494252b"
                      "9e8bf1c82b692b91ecd0ab94014f3612")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_m345(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    configure_external_cache_environment()
    cache_root = data_cache_root() / CACHE_REL

    tr_feat = np.load(cache_root / "scv2_train_84843_feat.npy",
                      mmap_mode="r")
    te_feat = np.load(cache_root / "scv2_test_11005_feat.npy",
                      mmap_mode="r")
    tr_labels = np.load(cache_root
                        / "scv2_train_84843_labels.npy").tolist()
    te_labels = np.load(cache_root
                        / "scv2_test_11005_labels.npy").tolist()
    print(f"features: train {tr_feat.shape}, test {te_feat.shape}",
          flush=True)

    # ---- g1: the sealed ridge, re-fit and re-scored end-to-end -----
    refit = _ridge_probe(np.asarray(tr_feat), tr_labels, RIDGE_ALPHA)
    weights_hash = _sha256_hex(
        refit["weights"].astype(np.float32).tobytes()
        + np.asarray(refit["bias"], dtype=np.float32).tobytes())
    hash_ok = weights_hash == M266B_WEIGHTS_HASH
    norm = (np.asarray(te_feat) - refit["mean"]) / refit["std"]
    scores = norm @ refit["weights"] + refit["bias"]
    preds = np.asarray([refit["classes"][int(i)]
                        for i in scores.argmax(axis=1)],
                       dtype=np.int64)
    linear_acc = float((preds == np.asarray(te_labels)).mean())
    acc_ok = abs(linear_acc - M266B_SEALED_ACC) <= REPRO_TOL
    g1 = bool(hash_ok and acc_ok)
    print(f"g1: refit hash {'MATCH' if hash_ok else 'MISMATCH'} "
          f"({weights_hash[:16]}...); linear repro {linear_acc:.6f} "
          f"(sealed {M266B_SEALED_ACC:.6f}, "
          f"{'OK' if acc_ok else 'DRIFT'})", flush=True)

    # ---- g2: the RFF arm, scored once --------------------------------
    omega, phase = rff_params(tr_feat.shape[1], RFF_DIM,
                              RFF_SIGMA, RFF_SEED)
    rff_probe = _ridge_probe_design(np.asarray(tr_feat), tr_labels,
                                    RIDGE_ALPHA, omega, phase)
    rff_acc = _score_design(rff_probe, np.asarray(te_feat), omega,
                            phase, te_labels)
    delta = rff_acc - linear_acc
    print(f"g2: rff {rff_acc:.6f} (linear {linear_acc:.6f}, "
          f"delta {delta:+.6f})", flush=True)

    # ---- the registered reading --------------------------------------
    if delta >= 0.01:
        reading = ("RFF lifts the sealed audio reading by >= 0.01: "
                   "the breadth claim gains its third modality and "
                   "the M300 map is recorded as modality-portable "
                   "across vision, text, and audio")
    else:
        reading = ("a null: the audio features are already "
                   "linear-sufficient at this scale - the breadth "
                   "claim rests on the recipe, not the map")

    gates = {
        "g1_sealed_ridge_reproduction": {
            "ok": g1, "tolerance": REPRO_TOL,
            "form": ("re-fit on the cached features; weights+bias "
                     "sha256 must equal the sealed evidence hash "
                     "and the accuracy must equal the sealed "
                     "reading"),
            "weights_hash_ok": bool(hash_ok),
            "accuracy_ok": bool(acc_ok),
            "sealed": M266B_SEALED_ACC, "measured": linear_acc},
        "g2_rff_scored_once": {"ok": True},
    }
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M345",
        "cell": ("the RFF reading on the sealed audio axis: does the "
                 "M300 map (D=16384, sigma=0.5) lift the M266b "
                 "reading?"),
        "rff_map": {"D": RFF_DIM, "sigma": RFF_SIGMA,
                    "seed": RFF_SEED,
                    "source": "the registered M300 selection"},
        "ridge_alpha": RIDGE_ALPHA,
        "linear_reproduction": linear_acc,
        "rff": rff_acc,
        "delta": delta,
        "reading": reading,
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "configuration_hash": payload_hash({
            "features": ["scv2_train_84843_feat.npy",
                         "scv2_test_11005_feat.npy"],
            "rff": {"D": RFF_DIM, "sigma": RFF_SIGMA,
                    "seed": RFF_SEED},
            "ridge_alpha": RIDGE_ALPHA}),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": bool(gates_ok),
                      "linear": linear_acc, "rff": rff_acc,
                      "delta": delta, "reading": reading},
                     indent=1), flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m345(args.output)


if __name__ == "__main__":
    main()
