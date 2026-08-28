"""M175 cell B2 — HEAD transfer: the frozen DomainNet 345-way head read on
Flowers-102 (the user's registered expectation).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` section 12
(18 Aug 2026) before any number existed.

The user's expectation: transferring to Flowers-102 should NOT produce
flower-species predictions; it should produce the GENERIC DomainNet
classes the encoder knows — "flower" and its neighbours. Cell B measured
feature transfer (a NEW 102-way head fitted on flowers labels). This cell
measures HEAD transfer: the bit-exact DomainNet SPM-1923 encoder (g1 pin
vs the sealed memmap), the DomainNet head re-fitted from the sealed
full-train codes and labels (the M142 C2 exact ridge path), and the 306
flowers test rows read through that frozen 345-way head. Zero flowers
labels enter this cell anywhere.

Registered reading: the expectation is confirmed if the modal predicted
class is "flower" (or the flower-adjacent class mass dominates top-k);
everything measured is reported either way.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
from experiments.tier4.eval_v24_m175_cell_b import _load_flowers

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m175_cell_b2.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v24" / "m175_cell_b2"

CLASSES = 345


def _domainnet_class_names() -> list[str]:
    import glob

    import pyarrow.parquet as pq

    files = sorted(glob.glob(
        str(data_cache_root() / "domainnet" / "repository" / "data"
            / "*.parquet")))
    metadata = pq.ParquetFile(files[0]).metadata.metadata
    info = json.loads(metadata[b"huggingface"].decode("utf-8"))
    names = list(info["info"]["features"]["label"]["names"])
    if len(names) != CLASSES:
        raise SystemExit(f"M175 B2 VOID: {len(names)} class names != 345")
    return names


def run_m175_cell_b2(config_path: Path, output_dir: Path) -> dict[str, Any]:
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

    rep = config["sparse"]
    size = int(config["corpus"]["image_size"])
    grid = (size - int(rep["patch"])) // int(rep["stride"]) + 1
    spm_atoms = int(config["sparse"]["spm_atoms"])
    spm_width = sum(level * level for level in SPM_LEVELS) * spm_atoms

    # ---- encoder (same as cell B) + g1 pin ---------------------------------
    print("rebuilding the sealed encoder (g1 pin)", flush=True)
    corpus, _ti, _tei = _load_corpus(config)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)
    dict_spm = _random_dictionary(candidates, len(candidates),
                                  int(rep["dictionary_seed"]), spm_atoms)
    table = torch.from_numpy(
        np.ascontiguousarray(dict_spm)).to(torch.float32).to(device)
    pin = config["encoder_pin"]
    check_rows = int(pin["check_rows"])
    check = np.empty((check_rows, spm_width), dtype=np.float32)
    offset = 0
    step = _chunk_rows(spm_atoms, grid, check_rows)
    for start in range(0, check_rows, step):
        stop = min(start + step, check_rows)
        block = _spm_encode_block_device(corpus["train_images"][start:stop],
                                         table, whitener, grid)
        check[offset:offset + stop - start] = block
        offset += stop - start
    sealed = np.load(
        data_cache_root() / pin["cache_relpath"] / pin["train_file"],
        mmap_mode="r")
    delta = float(np.abs(np.asarray(sealed[:check_rows], dtype=np.float64)
                         - check.astype(np.float64)).max())
    g1_ok = delta <= float(pin["tolerance"])
    print(f"g1 encoder pin: {delta:.3e} ok={g1_ok}", flush=True)
    del check, sealed
    torch.cuda.empty_cache()

    # ---- DomainNet head from the sealed full-train codes --------------------
    print("fitting the DomainNet 345-way head (sealed full-train codes)",
          flush=True)
    train_mem = np.load(
        data_cache_root() / config["head"]["cache_relpath"]
        / config["head"]["train_file"], mmap_mode="r")
    full_labels = np.load(
        data_cache_root() / config["head"]["labels_relpath"]
        / config["head"]["labels_file"])["labels"]
    penalties = [float(p) for p in config["head"]["penalty_ladder"]]
    acc = RidgeAccumulator(spm_width, CLASSES)
    for start in range(0, len(train_mem), 4096):
        stop = min(start + 4096, len(train_mem))
        acc.add(np.asarray(train_mem[start:stop]), full_labels[start:stop])
    solved = acc.solve_many(penalties)
    std = acc.standardiser()
    print(f"  head fitted on {acc.rows} rows", flush=True)
    del train_mem, full_labels
    torch.cuda.empty_cache()

    # ---- read the flowers test rows through the frozen head ------------------
    print("reading flowers through the frozen DomainNet head", flush=True)
    flowers = _load_flowers(config)
    test_images = flowers["test"]["images"]
    test_codes = np.empty((len(test_images), spm_width), dtype=np.float32)
    offset = 0
    step = _chunk_rows(spm_atoms, grid, len(test_images))
    for start in range(0, len(test_images), step):
        stop = min(start + step, len(test_images))
        block = _spm_encode_block_device(test_images[start:stop], table,
                                         whitener, grid)
        test_codes[offset:offset + stop - start] = block
        offset += stop - start
    del table
    torch.cuda.empty_cache()

    names = _domainnet_class_names()
    flower_idx = names.index("flower")
    adjacent = {names.index(n) for n in config["head"]["flower_adjacent"]}
    histogram: dict[str, Any] = {}
    for p in penalties:
        weights = solved[p]
        xs = std(test_codes).astype(np.float64)
        scores = xs @ weights[:-1] + weights[-1]
        preds = np.argmax(scores, axis=1)
        counts = np.bincount(preds, minlength=CLASSES)
        order = np.argsort(-counts)
        top = [
            {"class": names[int(i)], "count": int(counts[i])}
            for i in order[:10]
        ]
        adjacent_mass = float(counts[list(adjacent)].sum() / len(preds))
        flower_mass = float(counts[flower_idx] / len(preds))
        histogram[str(p)] = {
            "modal_class": names[int(order[0])],
            "modal_count": int(counts[order[0]]),
            "flower_class_count": int(counts[flower_idx]),
            "flower_class_mass": flower_mass,
            "adjacent_mass": adjacent_mass,
            "top10": top,
        }
        print(f"  p={p}: modal {names[int(order[0])]} "
              f"({counts[order[0]]}/{len(preds)}), flower mass "
              f"{flower_mass:.3f}, adjacent mass {adjacent_mass:.3f}",
              flush=True)

    best = histogram["1.0"]
    expectation_confirmed = bool(
        best["modal_class"] == "flower"
        or best["adjacent_mass"] >= float(config["head"]["adjacent_mass_min"])
    )

    evidence: dict[str, Any] = {
        "milestone": "M175",
        "cell": "B2 head transfer (frozen DomainNet head on Flowers-102)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "gates": {"g1_encoder_pin": {"max_abs_delta": delta,
                                     "ok": g1_ok}},
        "head": {
            "fit_rows": int(acc.rows),
            "width": spm_width,
            "classes": CLASSES,
            "flowers_rows_scored": len(test_codes),
            "flowers_labels_used": 0,
        },
        "predicted_class_histogram": histogram,
        "verdict": {
            "expectation_confirmed": expectation_confirmed,
            "reading": config["verdict"]["consequence_pass"]
            if expectation_confirmed
            else config["verdict"]["consequence_fail"],
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"modal": best["modal_class"],
                      "flower_mass": best["flower_class_mass"],
                      "adjacent_mass": best["adjacent_mass"],
                      "expectation_confirmed": expectation_confirmed},
                     indent=1), flush=True)
    print(f"M175 cell B2 complete -> {output_dir / 'evidence.json'}",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m175_cell_b2(args.config, args.output)


if __name__ == "__main__":
    main()
