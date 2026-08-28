"""M175 cell B R3 — flowers-native SPM at 32x32: does the construction
transfer when the whitener and dictionary are fitted ON flowers?

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` section 12
(18 Aug 2026) before building. Same sparse parameters as cell B (patch
6, stride 1, contrast 10, zca 0.1, seeds 11, pool 8192, 1923 atoms,
21 bins); the difference is ONLY where the whitener and the candidate
pool come from (flowers vs DomainNet).

g1 machinery pin (before the flowers-native arm runs): rebuild cell
B's DomainNet encoder and pin it bit-exact against the sealed
`spm1923_fulltrain.npy[:256]` — both arms share the proven SPM path.

Registered reading: R3 best >= 0.8 -> the DomainNet-native
whitener/dictionary is the blocker; R3 <= 0.4 -> the SPM construction
itself is weak at 32x32 fine-grained; in-between -> both contribute.
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
                  / "m175_cell_b_r3.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v24" / "m175_cell_b_r3"

CLASSES = 102
PENALTIES = [0.1, 1.0, 10.0]


def _encode_spm(images: np.ndarray, table: torch.Tensor, whitener,
                grid: int, width: int) -> np.ndarray:
    out = np.empty((len(images), width), dtype=np.float32)
    step = _chunk_rows(table.shape[0], grid, len(images))
    offset = 0
    for start in range(0, len(images), step):
        stop = min(start + step, len(images))
        block = _spm_encode_block_device(images[start:stop], table,
                                         whitener, grid)
        out[offset:offset + stop - start] = block
        offset += stop - start
    return out


def _fit_and_score(train: np.ndarray, train_y: np.ndarray,
                   test: np.ndarray, test_y: np.ndarray,
                   width: int) -> dict[str, float]:
    acc = RidgeAccumulator(width, CLASSES)
    acc.add(train, train_y)
    solved = acc.solve_many(PENALTIES)
    std = acc.standardiser()
    out: dict[str, float] = {}
    for p in PENALTIES:
        weights = solved[p]
        hits = 0
        n = len(test_y)
        for start in range(0, n, 4096):
            stop = min(start + 4096, n)
            xs = std(test[start:stop]).astype(np.float64)
            scores = xs @ weights[:-1] + weights[-1]
            hits += int((np.argmax(scores, axis=1)
                         == test_y[start:stop]).sum())
        out[str(p)] = hits / n
    return out


def run_m175_cell_b_r3(config_path: Path, output_dir: Path) -> dict[str, Any]:
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
    atoms = int(rep["spm_atoms"])
    width = sum(level * level for level in SPM_LEVELS) * atoms

    # ---- g1: the DomainNet encoder rebuilt + pinned bit-exact --------------
    print("g1: rebuilding the sealed DomainNet encoder", flush=True)
    corpus, _ti, _tei = _load_corpus(config)
    whitener_dn, candidates_dn = _build_whitener_and_candidates(
        config, corpus)
    dict_dn = _random_dictionary(candidates_dn, len(candidates_dn),
                                 int(rep["dictionary_seed"]), atoms)
    table_dn = torch.from_numpy(
        np.ascontiguousarray(dict_dn)).to(torch.float32).to(device)
    check_rows = int(config["encoder_pin"]["check_rows"])
    check = _encode_spm(corpus["train_images"][:check_rows], table_dn,
                        whitener_dn, grid, width)
    sealed = np.load(
        data_cache_root() / config["encoder_pin"]["cache_relpath"]
        / config["encoder_pin"]["train_file"], mmap_mode="r")
    delta = float(np.abs(np.asarray(sealed[:check_rows], dtype=np.float64)
                         - check.astype(np.float64)).max())
    g1_ok = delta <= float(config["encoder_pin"]["tolerance"])
    print(f"g1 encoder pin: {delta:.3e} ok={g1_ok}", flush=True)
    del check, sealed, dict_dn, candidates_dn, corpus
    torch.cuda.empty_cache()

    # ---- flowers-native whitener + dictionary -------------------------------
    print("fitting the flowers-native whitener + dictionary", flush=True)
    flowers = _load_flowers(config)
    flowers_corpus = {"train_images": flowers["train"]["images"]}
    whitener_fl, candidates_fl = _build_whitener_and_candidates(
        config, flowers_corpus)
    dict_fl = _random_dictionary(candidates_fl, len(candidates_fl),
                                 int(rep["dictionary_seed"]), atoms)
    table_fl = torch.from_numpy(
        np.ascontiguousarray(dict_fl)).to(torch.float32).to(device)

    train_codes = _encode_spm(flowers["train"]["images"], table_fl,
                              whitener_fl, grid, width)
    test_codes = _encode_spm(flowers["test"]["images"], table_fl,
                             whitener_fl, grid, width)
    del table_fl
    torch.cuda.empty_cache()
    r3 = _fit_and_score(train_codes, flowers["train"]["labels"],
                        test_codes, flowers["test"]["labels"], width)
    best = max(r3.values())
    print(f"R3 flowers-native SPM: {r3} best {best:.4f}", flush=True)

    reading = (config["reading"]["high"] if best >= 0.8
               else config["reading"]["low"] if best <= 0.4
               else config["reading"]["between"])

    evidence: dict[str, Any] = {
        "milestone": "M175",
        "cell": "B R3 flowers-native SPM at 32x32",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "gates": {"g1_encoder_pin": {"max_abs_delta": delta, "ok": g1_ok}},
        "results": {
            "flowers_native_spm_accuracy_by_penalty": r3,
            "best_test_accuracy": best,
            "reference_cell_b_spm_arm": 0.16666666666666666,
            "reference_32x32_cls": 0.826797385620915,
        },
        "reading": {
            "conclusion": reading,
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"r3": r3, "best": best, "g1_delta": delta,
                      "reading": reading}, indent=1), flush=True)
    print(f"M175 cell B R3 complete -> {output_dir / 'evidence.json'}",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m175_cell_b_r3(args.config, args.output)


if __name__ == "__main__":
    main()
