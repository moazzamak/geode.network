"""M261b — the Open Images vision arm: frozen DINOv2-small trunk +
closed-form ridge head over the 601 boxable classes.

Registered and dispatched 22 Aug 2026 (plan v25, the M261 switch —
ImageNet stays data-blocked; Open Images V7 annotations CC BY 4.0 /
images listed CC BY 2.0 permit commercial use with attribution).
The head is the repo's closed-form ridge (the M262 post-lbfgs
standard); the trunk is frozen; the guard is fit on train features;
the test rows are read once. Feature cache keys include the row
count (the registered smoke-poisoning lesson).

Caveats, declared: DINOv2's LVD-142M corpus is web-scale
(contamination declared — product quality, never a novelty claim);
OID images are multi-label, so images with several positive labels
contribute several rows (correlated test rows — per-class mean top-1
reported alongside the overall).

Evidence: logs/results/v25/m261b_oid_vision/evidence.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
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
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m261b_probe.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m261b_oid_vision")
BOXABLE_CSV = Path("F:/geode-ml/data/cache/oid/meta/"
                   "class-descriptions-boxable.csv")


def _full_class_list() -> tuple[list[str], dict[str, int]]:
    """The released 601-class list, fixed order (the operational
    class set — recorded 600-vs-601 discrepancy)."""
    classes: list[str] = []
    with open(BOXABLE_CSV, encoding="utf-8") as fh:
        for mid, _name in csv.reader(fh):
            classes.append(mid)
    return classes, {c: i for i, c in enumerate(classes)}


def extract_features(rows: list[dict[str, Any]], split: str,
                     config: dict[str, Any], cache_root: Path,
                     smoke: bool) -> np.ndarray:
    """Frozen DINOv2-small features, cached keyed by (split, n_rows).
    Multi-row images are extracted ONCE (the row-level manifest may
    repeat an image across labels)."""
    n = len(rows)
    cache_path = cache_root / f"oid_{split}_{n}_feat.npy"
    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r").copy()

    import torch
    torch.backends.cudnn.enabled = False  # registered M267 env note
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = config["trunk"]["checkpoint"]
    proc = AutoImageProcessor.from_pretrained(ckpt,
                                              local_files_only=True)
    model = AutoModel.from_pretrained(ckpt,
                                      local_files_only=True).to(device).eval()
    dim = int(config["trunk"]["feature_dim"])
    paths = [r["image_path"] for r in rows]
    unique = sorted(set(paths))
    path_feat: dict[str, np.ndarray] = {}
    batch = 64
    with torch.no_grad():
        for start in range(0, len(unique), batch):
            chunk_paths = unique[start:start + batch]
            images = []
            for p in chunk_paths:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            enc = proc(images=images, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc).last_hidden_state[:, 0].cpu().numpy()
            for p, vec in zip(chunk_paths, out):
                path_feat[p] = vec.astype(np.float32)
            if (start // batch) % 20 == 0:
                print(f"  {split}: {start}/{len(unique)} unique images",
                      flush=True)
    feats = np.stack([path_feat[p] for p in paths])
    np.save(cache_path, feats)
    return feats


def run_m261b_probe(config_path: Path, output_dir: Path,
                    smoke: bool = False) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    cache_root = data_cache_root() / config["feature_cache_relpath"]
    cache_root.mkdir(parents=True, exist_ok=True)

    train_rows = json.loads(Path(
        config["data"]["train_manifest"]).read_text(
            encoding="utf-8"))["rows"]
    test_rows = json.loads(Path(
        config["data"]["test_manifest"]).read_text(
            encoding="utf-8"))["rows"]
    if smoke:
        train_rows = train_rows[:config["smoke"]["max_rows"]]
        test_rows = test_rows[:config["smoke"]["max_rows"]]

    print(f"  train rows {len(train_rows)}, test rows {len(test_rows)}",
          flush=True)
    train_feats = extract_features(train_rows, "train", config,
                                   cache_root, smoke)
    test_feats = extract_features(test_rows, "test", config,
                                  cache_root, smoke)

    # ---- closed-form ridge head (M262 standard) ----------------------
    alpha = float(config["head"]["alpha"])
    classes, class_index = _full_class_list()
    n_classes = len(classes)
    assert all(r["label_mid"] in class_index for r in train_rows)
    assert all(r["label_mid"] in class_index for r in test_rows), \
        "test label outside the released class set"
    y_idx = [class_index[r["label_mid"]] for r in train_rows]
    t_idx = [class_index[r["label_mid"]] for r in test_rows]
    n_train_classes = len({i for i in y_idx})
    n, d = train_feats.shape
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y_idx] = 1.0
    # center the one-hot (rank n_classes-1) and solve exactly
    Yc = Y - Y.mean(axis=0, keepdims=True)
    X = np.asarray(train_feats, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    A = Xc.T @ Xc + alpha * np.eye(d)
    W = np.linalg.solve(A, Xc.T @ Yc)
    b = Y.mean(axis=0) - X.mean(axis=0) @ W

    scores = np.asarray(test_feats, dtype=np.float64) @ W + b
    pred = scores.argmax(axis=1)
    t_arr = np.array(t_idx)
    correct = pred == t_arr
    overall = float(correct.mean())

    accs: dict[str, list[float]] = defaultdict(list)
    for ok, row in zip(correct, test_rows):
        accs[row["label_mid"]].append(float(ok))
    per_class_accs = {m: sum(v) / len(v) for m, v in accs.items()}
    per_class_mean = float(np.mean(list(per_class_accs.values())))

    # top-5
    top5 = scores.argsort(axis=1)[:, -5:]
    top5_ok = float(np.mean([int(t in row)
                             for t, row in zip(t_arr, top5)]))

    # ---- guard (report, not gate) ------------------------------------
    from geode.core.ood import OodGate
    gate = OodGate(threshold=float(config["guard"]["threshold"]))
    gate.fit_profile(np.asarray(train_feats,
                                dtype=np.float64).tolist())
    test_scores = [gate.score(v.tolist())
                   for v in np.asarray(test_feats, dtype=np.float64)]
    flagged = float(np.mean([s > gate.threshold
                             for s in test_scores]))

    evidence: dict[str, Any] = {
        "milestone": "M261b",
        "cell": "Open Images vision arm (frozen DINOv2-small + ridge)",
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "results": {
            "n_classes": n_classes,
            "n_classes_with_train_rows": n_train_classes,
            "n_train_rows": int(n),
            "n_test_rows": int(len(test_rows)),
            "overall_top1": round(overall, 4),
            "per_class_mean_top1": round(per_class_mean, 4),
            "top5": round(top5_ok, 4),
            "guard_flag_rate_test": round(flagged, 4),
        },
        "declarations": {
            "contamination": config["trunk"]["contamination_declared"],
            "multi_label": config["data"]["multi_row_caveat"],
            "license": config["data"]["license_recorded"],
            "anchors": ("no canonical published OID-601 linear-probe "
                        "number exists; the published DINOv2 ImageNet "
                        "probe figures are cited context, never gated on"),
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / config["evidence_filename"], evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evidence["results"]}, indent=1),
          flush=True)
    print(f"M261b probe complete -> "
          f"{output_dir / config['evidence_filename']}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output
    if args.smoke and output == DEFAULT_OUTPUT:
        output = DEFAULT_OUTPUT.parent / (DEFAULT_OUTPUT.name + "_smoke")
    run_m261b_probe(args.config, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
