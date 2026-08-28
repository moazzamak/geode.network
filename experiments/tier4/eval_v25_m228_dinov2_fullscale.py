"""M228 - full-scale DINOv2 extraction + hybrid ridge at full data.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` (M228
REGISTERED, 20 Aug 2026). The M222 follow-up, finally dispatched:
extract DINOv2-small features for the FULL 409,832-row train schedule
on the GPU (the pilot measured ~38.5s per 20k rows), REUSE the cached
sealed 34,500-row test features from M222 (same row-selection key),
and fit the closed-form ridge on ms alone and on ms+DINOv2 at the
SAME penalties (0.1, 1.0, 10.0) - a fixed-penalty comparison with no
test-set lambda selection.

Gates: g1 the ms-only penalty-1.0 full-train refit reproduces the
sealed anchor 0.24214492753623187 at 1e-9 (VOID on failure); g2 exact
row counts (VOID on failure); g3 accuracies valid; g4 scope note.
"""
from __future__ import annotations

import argparse
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
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v25_m222_dinov2_hybrid_pilot import (
    _features_or_extract,
    _fit_and_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m228_dinov2_fullscale.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m228_dinov2_fullscale")

FULL_TRAIN_ROWS = 409832
SEALED_TEST_ROWS = 34500


def run_m228(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()

    configure_external_cache_environment()
    corpus, _train_index, test_index = _load_corpus(config)
    test_labels = corpus["test_labels"]
    root = data_cache_root()

    decoded = np.load(root / "domainnet_decoded" / "size32.npz",
                      mmap_mode="r")
    train_images = decoded["train_images"]
    test_images = decoded["test_images"][test_index]
    if len(train_images) != FULL_TRAIN_ROWS \
            or len(test_images) != SEALED_TEST_ROWS:
        raise SystemExit("M228 premise failure: decoded shapes")

    ms_cache = root / config["artifacts"]["cache_relpath"]
    ms_test_cache = root / config["artifacts"]["test_cache_relpath"]
    mem_train = np.load(ms_cache / config["artifacts"]["train_file"],
                        mmap_mode="r")
    mem_test = np.load(ms_test_cache / config["artifacts"]["test_file"],
                       mmap_mode="r")
    labels = np.load(root / config["artifacts"]["labels_file"])["labels"]
    if len(labels) != FULL_TRAIN_ROWS:
        raise SystemExit("M228 premise failure: schedule labels")
    print(f"labels from the M142 cell-2 schedule file: {len(labels)} rows",
          flush=True)

    train_ms = np.asarray(mem_train)
    test_ms = np.asarray(mem_test)
    print(f"ms shapes: train {train_ms.shape}, test {test_ms.shape}",
          flush=True)

    # ---- GPU extraction (train: full schedule; test: reuse the M222
    # cache keyed by the sealed test selection)
    import torch
    import torchvision.transforms as T
    device = torch.device("cuda")
    model = torch.hub.load(config["model"]["repo"],
                           config["model"]["name"]).to(device)
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    transform = T.Compose([
        T.Resize(config["model"]["input_size"]),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    train_sel = np.arange(FULL_TRAIN_ROWS, dtype=np.int64)
    train_dino, extract_train_s = _features_or_extract(
        model, transform, np.asarray(train_images), device,
        output_dir, "fulltrain", train_sel)
    test_feature_dir = (REPO_ROOT
                        / config["reuse_test_features_from"])
    test_dino, extract_test_s = _features_or_extract(
        model, transform, np.asarray(test_images), device,
        test_feature_dir, "test", test_index.astype(np.int64))
    print(f"extraction: train {extract_train_s}s, test {extract_test_s}s",
          flush=True)
    del train_images, test_images, decoded
    import gc
    gc.collect()

    penalties = [float(p) for p in config["penalties"]]
    ms_only = _fit_and_score(train_ms, labels, test_ms, test_labels,
                             penalties)
    hybrid_train = np.concatenate([train_ms, train_dino], axis=1)
    hybrid_test = np.concatenate([test_ms, test_dino], axis=1)
    hybrid = _fit_and_score(hybrid_train, labels, hybrid_test,
                            test_labels, penalties)
    for p in penalties:
        print(f"penalty {p}: ms-only {ms_only[str(p)]:.6f}  hybrid "
              f"{hybrid[str(p)]:.6f}", flush=True)

    anchor = float(config["anchor"]["value"])
    tol = float(config["anchor"]["tolerance"])
    anchor_measured = ms_only["1.0"]
    g1 = abs(anchor_measured - anchor) <= tol
    g2 = len(train_ms) == FULL_TRAIN_ROWS and len(test_ms) == SEALED_TEST_ROWS
    gates = {
        "g1_ms_anchor_reproduction": {
            "ok": g1, "measured": anchor_measured, "sealed": anchor,
            "delta": anchor_measured - anchor, "tolerance": tol},
        "g2_row_counts_exact": {
            "ok": g2, "train_rows": len(train_ms),
            "test_rows": len(test_ms)},
        "g3_valid_accuracies": {
            "ok": all(0.0 <= v <= 1.0 for d in (ms_only, hybrid)
                      for v in d.values()),
            "ms_only": ms_only, "hybrid": hybrid},
        "g4_fixed_penalty_comparison": {
            "ok": True,
            "note": "all three penalties reported; none selected on the "
                    "test set (the M218 caveat avoided)"},
    }
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M228",
        "cell": "full-scale DINOv2 extraction + hybrid ridge at full data",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "ms_only_test_accuracies": ms_only,
        "hybrid_test_accuracies": hybrid,
        "anchor": {"sealed": anchor, "measured_at_penalty_1": anchor_measured,
                   "delta": anchor_measured - anchor},
        "extraction": {"train_seconds": extract_train_s,
                       "test_seconds": extract_test_s,
                       "test_features_reused_from":
                           config["reuse_test_features_from"],
                       "train_feature_rows": len(train_dino),
                       "test_feature_rows": len(test_dino)},
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": (
                "fixed-penalty comparison: the hybrid vs ms-only at "
                "matched penalties on the full 409,832-row schedule"
            ) if gates_ok else "a gate failed — VOID",
        },
        "scope": "full-data train, sealed 34,500-row test, no test-set "
                 "lambda selection",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok,
                      "ms_only": ms_only, "hybrid": hybrid}, indent=1),
          flush=True)
    print(f"M228 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m228(args.config, args.output)


if __name__ == "__main__":
    main()
