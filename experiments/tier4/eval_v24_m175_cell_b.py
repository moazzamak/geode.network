"""M175 cell B — vision->vision transfer: the frozen DomainNet-32 SPM
encoder applied to Oxford Flowers-102 (bounded S1, five-shot).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` section 12
(18 Aug 2026) and ``analysis/v24_m175_transfer_cells.md`` (17 Aug 2026).

Question. Does the frozen DomainNet-32 SPM code construction transfer to
Flowers-102? The encoder is the sealed M142 C2 construction: 6x6 patches,
ZCA whitener, 1,923 atoms, 1x1+2x2+4x4 pyramid pooling (21 bins, width
40,383). The target is the M19 bounded Flowers-102 split (5/2/3 samples
per class, seeds 11/12/13 -> 510 train / 204 dev / 306 test).

Arms (the SAME exact ridge read on both: accumulator + standardiser +
intercept, penalty ladder {0.1, 1.0, 10.0}):

- baseline: the cached DINOv2-small CLS features (384-d, float64, extracted
  from full-resolution images by the M19 pipeline).
- SPM: flowers images -> RGB -> PIL BILINEAR -> 32x32 uint8 -> the frozen
  DomainNet whitener -> 1,923-atom triangle activation -> 21-bin pyramid ->
  40,383-d codes. RAW C2 output; no power-normalisation (C4's transform was
  measured on DomainNet only).

Premises (gates; failure = VOID, not negative):

- g1 encoder pin: the rebuilt whitener + dictionary encode the first 256
  DomainNet train rows bit-identically to the sealed
  ``spm1923_fulltrain.npy`` memmap (max-abs delta == 0.0, registered
  tolerance 0.0).
- g2 split pin: the torchvision-reproduced Flowers-102 bounded split's
  labels and image ids equal the cached M19 npz arrays exactly.

Verdict (registered before running): transfer holds if the SPM arm's
best-penalty test accuracy >= the baseline arm's best-penalty test
accuracy; otherwise B closes as a scoped negative for THIS encoder (a
loss says nothing about the deep-patch deployment arm). Disclosures:
not a matched-cost fight (384 vs 40,383 width); the baseline reads
full-resolution images while the SPM arm reads 32x32; 510-row five-shot
fit.

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v24_m175_cell_b
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m104_experts import (
    RidgeAccumulator,
    _chunk_rows,
)
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m113_learned import (
    _build_whitener_and_candidates,
    _random_dictionary,
)
from experiments.tier4.eval_v16_m142_c2 import (
    SPM_LEVELS,
    _spm_encode_block_device,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m175_cell_b.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v24" / "m175_cell_b"

CLASSES = 102
OFFICIAL_SPLITS = {"train": "train", "dev": "val", "test": "test"}


# --------------------------------------------------------------------------
# flowers: split reproduction + 32x32 image loading
# --------------------------------------------------------------------------
def _reproduce_split(root: Path, split: str, per_class: int,
                     seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The M19 ``_select_split`` exactly: per-class seeded draws from the
    official split, sorted by image id. Returns (labels, image_ids)."""
    from torchvision.datasets import Flowers102

    dataset = Flowers102(root=root, split=OFFICIAL_SPLITS[split],
                         download=False)
    labels = np.asarray(dataset._labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in range(102):
        candidates = np.flatnonzero(labels == class_id)
        if len(candidates) < per_class:
            raise ValueError(
                f"Flowers class {class_id} has {len(candidates)} samples; "
                f"{per_class} required.")
        selected.extend(rng.choice(candidates, per_class,
                                   replace=False).tolist())
    pattern = re.compile(r"image_(\d+)\.jpg")
    ids = np.asarray(
        [int(pattern.fullmatch(Path(p).name).group(1))
         for p in np.asarray(dataset._image_files)[selected]],
        dtype=np.int64,
    )
    order = np.argsort(ids)
    return labels[selected][order], ids[order]


def _load_split_images(root: Path, jpg_dir: str, ids: np.ndarray,
                       size: int) -> np.ndarray:
    """Load the id-addressed jpg files, RGB -> BILINEAR -> ``size`` uint8."""
    folder = root / jpg_dir
    images = np.empty((len(ids), size, size, 3), dtype=np.uint8)
    for i, image_id in enumerate(ids):
        path = folder / f"image_{int(image_id):05d}.jpg"
        if not path.exists():
            raise ValueError(f"missing flowers image {path.name}")
        with Image.open(path) as im:
            rgb = im.convert("RGB").resize((size, size),
                                           resample=Image.Resampling.BILINEAR)
        images[i] = np.asarray(rgb, dtype=np.uint8)
    return images


def _load_flowers(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Flowers corpus + baseline features, split-pinned against the M19 npz.

    Returns one entry per split with images (32x32 uint8), labels, image
    ids, and the cached baseline features (float64)."""
    fl = config["flowers"]
    root = REPO_ROOT / fl["dataset_root"]
    out: dict[str, dict[str, Any]] = {}
    for split, (per_class, seed, feature_file) in fl["splits"].items():
        labels, ids = _reproduce_split(root, split, int(per_class), int(seed))
        cache = np.load(REPO_ROOT / fl["features_dir"] / feature_file)
        pinned_labels = np.asarray(cache["labels"], dtype=np.int64)
        pinned_ids = np.asarray(cache["indices"], dtype=np.int64)
        labels_ok = np.array_equal(labels, pinned_labels)
        ids_ok = np.array_equal(ids, pinned_ids)
        features = np.asarray(cache["features"], dtype=np.float64)
        if (not labels_ok or not ids_ok
                or features.shape != (len(ids), int(fl["feature_dimension"]))):
            raise SystemExit(
                f"M175 cell B VOID: split pin failed for '{split}' "
                f"(labels_ok={labels_ok}, ids_ok={ids_ok}, "
                f"features={features.shape})")
        images = _load_split_images(root, fl["jpg_dir"], ids,
                                    int(fl["image_size"]))
        out[split] = {
            "images": images,
            "labels": labels,
            "ids": ids,
            "features": features,
        }
    return out


# --------------------------------------------------------------------------
# encoding + fit + score (the sealed direct path)
# --------------------------------------------------------------------------
def _encode_spm(images: np.ndarray, table: torch.Tensor, whitener,
                grid: int, width: int) -> np.ndarray:
    """Encode uint8 images to the full 21-bin SPM codes (float32)."""
    out = np.empty((len(images), width), dtype=np.float32)
    step = _chunk_rows(table.shape[0], grid, len(images))
    offset = 0
    for start in range(0, len(images), step):
        stop = min(start + step, len(images))
        out[offset:offset + stop - start] = _spm_encode_block_device(
            images[start:stop], table, whitener, grid)
        offset += stop - start
    return out


def _fit_ladder(train_codes: np.ndarray, train_labels: np.ndarray,
                width: int, penalties: list[float]
                ) -> tuple[dict[str, np.ndarray], Any]:
    acc = RidgeAccumulator(width, CLASSES)
    block = 4096
    for start in range(0, len(train_codes), block):
        stop = min(start + block, len(train_codes))
        acc.add(np.asarray(train_codes[start:stop]),
                train_labels[start:stop])
    solved = acc.solve_many(penalties)
    return {str(p): w for p, w in solved.items()}, acc.standardiser()


def _score(test_codes: np.ndarray, test_labels: np.ndarray,
           weights: np.ndarray, standardiser) -> dict[str, Any]:
    hits = 0
    n = len(test_labels)
    block = 4096
    for start in range(0, n, block):
        stop = min(start + block, n)
        xs = standardiser(test_codes[start:stop]).astype(np.float64)
        scores = xs @ weights[:-1] + weights[-1]
        hits += int((np.argmax(scores, axis=1)
                     == test_labels[start:stop]).sum())
    return {"accuracy": hits / n, "rows": n}


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def run_m175_cell_b(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible

    torch.set_num_threads(int(config["numerics"]["torch_threads"]))
    configure_external_cache_environment()
    _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    rep = config["sparse"]
    size = int(config["corpus"]["image_size"])
    grid = (size - int(rep["patch"])) // int(rep["stride"]) + 1
    spm_atoms = int(config["sparse"]["spm_atoms"])
    spm_width = sum(level * level for level in SPM_LEVELS) * spm_atoms

    print("loading DomainNet corpus + rebuilding the sealed encoder",
          flush=True)
    corpus, _train_index, _test_index = _load_corpus(config)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)
    dict_spm = _random_dictionary(candidates, len(candidates),
                                  int(rep["dictionary_seed"]), spm_atoms)
    table = torch.from_numpy(
        np.ascontiguousarray(dict_spm)).to(torch.float32).to(device)

    # ---- g1: encoder pin against the sealed spm1923 memmap ----------------
    pin = config["encoder_pin"]
    check_rows = int(pin["check_rows"])
    check_codes = _encode_spm(corpus["train_images"][:check_rows], table,
                              whitener, grid, spm_width)
    sealed = np.load(
        data_cache_root() / pin["cache_relpath"] / pin["train_file"],
        mmap_mode="r")
    delta = float(np.abs(
        np.asarray(sealed[:check_rows], dtype=np.float64)
        - check_codes.astype(np.float64)).max())
    g1_ok = delta <= float(pin["tolerance"])
    print(f"g1 encoder pin: max-abs delta {delta:.3e} ok={g1_ok}",
          flush=True)
    if not g1_ok:
        evidence = {
            "milestone": "M175", "cell": "B",
            "admissible_as_evidence": False,
            "void": True,
            "void_reason": "g1 encoder pin failed (not bit-exact)",
            "g1_max_abs_delta": delta,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_canonical_json(output_dir / "evidence.json", evidence)
        build_artifact_index(output_dir)
        return evidence
    del check_codes, sealed, dict_spm
    torch.cuda.empty_cache()

    # ---- flowers: g2 split pin + 32x32 images + baseline features ----------
    print("loading flowers (split pin + 32x32 adapter)", flush=True)
    flowers = _load_flowers(config)
    for split in flowers:
        print(f"  {split}: {len(flowers[split]['images'])} rows")
    g2_pin = {
        split: {"rows": len(f["images"]),
                "labels_sha256": _array_digest(f["labels"]),
                "ids_sha256": _array_digest(f["ids"])}
        for split, f in flowers.items()
    }

    # ---- SPM arm: encode flowers with the frozen DomainNet encoder ---------
    print("encoding flowers with the frozen SPM encoder", flush=True)
    spm_codes = {
        split: _encode_spm(flowers[split]["images"], table, whitener, grid,
                           spm_width)
        for split in flowers
    }
    del table
    torch.cuda.empty_cache()

    # ---- both arms: the same exact ridge read ------------------------------
    penalties = [float(p) for p in config["cell"]["penalty_ladder"]]
    results: dict[str, Any] = {}
    for arm, (train_codes, width) in {
        "baseline_dinov2_cls": (
            flowers["train"]["features"], int(config["flowers"]
                                              ["feature_dimension"])),
        "spm_domainnet_encoder": (spm_codes["train"], spm_width),
    }.items():
        weights, standardiser = _fit_ladder(
            train_codes, flowers["train"]["labels"], width, penalties)
        arm_results: dict[str, Any] = {}
        for p in penalties:
            row: dict[str, Any] = {}
            for split in ("test", "dev"):
                codes = (flowers[split]["features"] if arm.startswith("baseline")
                         else spm_codes[split])
                row[split] = _score(codes, flowers[split]["labels"],
                                    weights[str(p)], standardiser)
            arm_results[str(p)] = row
        results[arm] = {
            "width": width,
            "accuracy_by_penalty": arm_results,
            "best_test_accuracy": max(
                arm_results[str(p)]["test"]["accuracy"] for p in penalties),
        }
        print(f"  {arm} best test {results[arm]['best_test_accuracy']:.4f}",
              flush=True)

    spm_best = results["spm_domainnet_encoder"]["best_test_accuracy"]
    base_best = results["baseline_dinov2_cls"]["best_test_accuracy"]
    transfer_holds = spm_best >= base_best

    evidence: dict[str, Any] = {
        "milestone": "M175",
        "cell": "B vision->vision DomainNet-32 -> Flowers102",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
        "verdict_registered_before_running": config["verdict"],
        "gates": {
            "g1_encoder_pin": {"max_abs_delta": delta, "ok": g1_ok,
                               "tolerance": float(pin["tolerance"])},
            "g2_split_pin": g2_pin,
        },
        "results": results,
        "verdict": {
            "spm_best_test_accuracy": spm_best,
            "baseline_best_test_accuracy": base_best,
            "transfer_holds": transfer_holds,
            "reading": config["verdict"]["consequence_pass"]
            if transfer_holds else config["verdict"]["consequence_fail"],
        },
        "disclosures": config["disclosures"],
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({
        "g1_delta": delta,
        "baseline_best": base_best,
        "spm_best": spm_best,
        "transfer_holds": transfer_holds,
    }, indent=1), flush=True)
    print(f"M175 cell B complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def _array_digest(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m175_cell_b(args.config, args.output)


if __name__ == "__main__":
    main()
