"""M175 cell B diagnostic — what blocks the vision transfer?

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` section 12
(18 Aug 2026) before running. All inputs sealed; the probes are new
fixed-function reads only.

R2 (reference): torch DINOv2-small CLS on the FULL-RES bounded
flowers (the same 1020 rows the M19 split pins) — must land near the
sealed ONNX baseline 0.9902 to license reading R1.
R1 (information-matched control): the SAME torch CLS reader on the
SAME 32x32 images the SPM arm read.

Registered reading: R1 >= 0.8 -> the 32x32 input still carries the
species signal; the blocker is the CONSTRUCTION (then R3 follows:
flowers-fitted SPM at 32x32). R1 <= 0.4 -> 32x32 downsampling
destroys the fine-grained signal; resolution is the primary blocker.
Same ridge ladder {0.1, 1.0, 10.0} on 510 train / 306 test.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.common.data_cache import configure_external_cache_environment
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
from experiments.tier4.eval_v24_m175_cell_b import _load_flowers
from experiments.tier4.eval_v24_m176c_c1 import _load_backbone

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m175_cell_b_diag.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v24" / "m175_cell_b_diag"

CLASSES = 102
WIDTH = 384


def _cls_features(model, processor, device: torch.device,
                  images: np.ndarray, batch: int) -> np.ndarray:
    """DINOv2-small CLS token, float64 on CPU."""
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch):
            stop = min(start + batch, len(images))
            inputs = processor(images=list(images[start:stop]),
                               return_tensors="pt")
            pixels = inputs["pixel_values"].to(device)
            hidden = model(pixels).last_hidden_state
            out.append(hidden[:, 0, :].to(torch.float64).cpu().numpy())
    return np.concatenate(out, axis=0)


def _fit_and_score(train: np.ndarray, train_labels: np.ndarray,
                   test: np.ndarray, test_labels: np.ndarray,
                   penalties: list[float]) -> dict[str, Any]:
    acc = RidgeAccumulator(WIDTH, CLASSES)
    acc.add(train, train_labels)
    solved = acc.solve_many(penalties)
    std = acc.standardiser()
    result: dict[str, Any] = {}
    for p in penalties:
        weights = solved[p]
        hits = 0
        n = len(test_labels)
        for start in range(0, n, 4096):
            stop = min(start + 4096, n)
            xs = std(test[start:stop]).astype(np.float64)
            scores = xs @ weights[:-1] + weights[-1]
            hits += int((np.argmax(scores, axis=1)
                         == test_labels[start:stop]).sum())
        result[str(p)] = hits / n
    return result


def run_m175_cell_b_diag(config_path: Path,
                         output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible

    configure_external_cache_environment()
    _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    flowers = _load_flowers(config)
    penalties = [float(p) for p in config["cell"]["penalty_ladder"]]
    batch = int(config["numerics"]["batch"])

    print("loading DINOv2-small (torch)", flush=True)
    processor, model = _load_backbone(config)
    model = model.to(device).eval()

    # ---- R2: full-resolution reference -------------------------------------
    # Reconstruct full-res images from the cached M19 ids (the split pins
    # the ids; the original full-res extraction used these same rows).
    from PIL import Image
    fl = config["flowers"]
    root = REPO_ROOT / fl["dataset_root"]
    folder = root / fl["jpg_dir"]
    full_res: dict[str, list[np.ndarray]] = {}
    for split, f in flowers.items():
        images = []
        for image_id in f["ids"]:
            with Image.open(folder / f"image_{int(image_id):05d}.jpg") as im:
                images.append(np.asarray(im.convert("RGB")))
        full_res[split] = images  # variable native sizes; processor resizes
    r2 = {}
    for split in ("train", "test"):
        feats = _cls_features(model, processor, device, full_res[split],
                              batch)
        r2[split] = feats
    r2_acc = _fit_and_score(r2["train"], flowers["train"]["labels"],
                            r2["test"], flowers["test"]["labels"], penalties)
    print(f"R2 full-res torch CLS: {r2_acc}", flush=True)
    del full_res
    torch.cuda.empty_cache()

    # ---- R1: the SPM arm's exact 32x32 input --------------------------------
    r1 = {}
    for split in ("train", "test"):
        r1[split] = _cls_features(model, processor, device,
                                  flowers[split]["images"], batch)
    r1_acc = _fit_and_score(r1["train"], flowers["train"]["labels"],
                            r1["test"], flowers["test"]["labels"], penalties)
    print(f"R1 32x32 torch CLS: {r1_acc}", flush=True)

    r2_best = max(r2_acc.values())
    r1_best = max(r1_acc.values())
    r2_reference_ok = bool(r2_best >= float(config["probes"]["r2_reference_min"]))
    reading = config["probes"]["reading_high"] if r1_best >= 0.8 else (
        config["probes"]["reading_low"] if r1_best <= 0.4 else
        config["probes"]["reading_between"])

    evidence: dict[str, Any] = {
        "milestone": "M175",
        "cell": "B diagnostic — what blocks the vision transfer",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "reference_sealed": {
            "full_res_onxx_cls_test_acc": 0.9901960784313726,
            "spm_arm_test_acc": 0.16666666666666666,
        },
        "probes": {
            "r2_full_res_torch_cls": r2_acc,
            "r1_32x32_torch_cls": r1_acc,
        },
        "reading": {
            "r2_reference_ok": r2_reference_ok,
            "r1_best_test_acc": r1_best,
            "conclusion": reading,
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"r2": r2_acc, "r1": r1_acc,
                      "reading": reading}, indent=1), flush=True)
    print(f"M175 cell B diagnostic complete -> "
          f"{output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m175_cell_b_diag(args.config, args.output)


if __name__ == "__main__":
    main()
