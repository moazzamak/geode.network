"""M176c candidate 1 — deep-patch SPM on frozen DINOv2-small.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` §12 (18 Aug
2026). The first better-code arm (deployment-phase, budget-capped):
DINOv2-small encodes the DomainNet schedule at 224 px (streaming decode
from the raw parquets — the 32 px decoded cache cannot serve deep
patches); the 16x16 patch-token grid is SPM-pooled at levels 1x1+2x2+4x4
(21 bins); the dictionary is a seeded prefix of whitened deep-patch
candidates (the M117/M126 construction pattern, registered fresh for
this encoder); the readout is the sealed closed-form intercept ridge.

Ledger (registered): the full per-image MACs INCLUDE the DINOv2 backbone
(shared by all arms — excluding it would make the readout comparison
meaningless). The gate: beat the sealed dense ladder per-MAC (r70 0.3118
/ r98 0.4476) or serve a task axis no frozen arm serves. The sealed
DomainNet anchors (0.2605 / 0.2274 / 0.2786) are recorded for reference,
not as this arm's gate.

Premise gate (void on failure): the corpus digest gate inside
`_load_corpus` (the same rows), and the pixel-identity check.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.common.data_cache import (
    configure_external_cache_environment,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m104_experts import (
    RidgeAccumulator,
    _score,
)
from experiments.tier4.eval_v15_m107_dense import _row_group_blobs
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m142_factorial import power_norm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m176c_c1.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v24" / "m176c_c1")

CLASSES = 345
SPM_LEVELS = (1, 2, 4)


# ---------------------------------------------------------------------------
# streaming 224 px decode of the subsample row indices
# ---------------------------------------------------------------------------
def _decode_rows(split: str, index: np.ndarray, size: int
                 ) -> np.ndarray:
    """Decode subsample row indices from the raw parquets at `size` px."""
    from PIL import Image

    wanted = set(int(i) for i in index)
    order = np.argsort(index)
    out = np.empty((len(index), size, size, 3), dtype=np.uint8)
    pos = {int(idx): k for k, idx in enumerate(np.asarray(index)[order])}
    for start, rows, (handle, group) in _row_group_blobs(split):
        if not wanted:
            break
        group_wanted = [i for i in wanted if start <= i < start + rows]
        if not group_wanted:
            continue
        blobs = handle.read_row_group(group, columns=["image"]).column(
            "image").to_pylist()
        for i in sorted(group_wanted):
            record = blobs[i - start]
            picture = Image.open(io.BytesIO(record["bytes"])).convert("RGB")
            out[pos[i]] = np.asarray(
                picture.resize((size, size), Image.BILINEAR),
                dtype=np.uint8)
            wanted.discard(i)
    return out


# ---------------------------------------------------------------------------
# DINOv2 patch tokens + SPM pooling
# ---------------------------------------------------------------------------
def _load_backbone(config: dict[str, Any]):
    from transformers import AutoImageProcessor, Dinov2Model

    name = config["backbone"]["name"]
    processor = AutoImageProcessor.from_pretrained(name)
    model = Dinov2Model.from_pretrained(name)
    model.eval()
    return processor, model


def _patch_tokens(model, processor, device: torch.device,
                  images: np.ndarray) -> torch.Tensor:
    """DINOv2 patch tokens (B, 16, 16, 384) for 224 px images."""
    with torch.no_grad():
        inputs = processor(images=list(images), return_tensors="pt")
        pixels = inputs["pixel_values"].to(device)
        out = model(pixels)
        tokens = out.last_hidden_state[:, 1:, :]      # drop CLS
        b = tokens.shape[0]
        return tokens.reshape(b, 16, 16, -1)


def _spm_pool_tokens(tokens: torch.Tensor, atoms_table: torch.Tensor
                     ) -> np.ndarray:
    """SPM-pool the patch tokens against the atoms table (21 bins)."""
    b, g, _, d = tokens.shape
    x = tokens.reshape(b, g * g, d).float()
    table = atoms_table.float().to(x.device)
    with torch.no_grad():
        dist = torch.cdist(x, table)                  # (b, 196, atoms)
        mean = dist.mean(dim=1, keepdim=True)
        act = torch.clamp(mean - dist, min=0.0)       # triangle activation
        act = act.reshape(b, g, g, table.shape[0])
        blocks = []
        for level in SPM_LEVELS:
            edges = [round(g * i / level) for i in range(level + 1)]
            for iy in range(level):
                for ix in range(level):
                    blocks.append(
                        act[:, edges[iy]:edges[iy + 1],
                            edges[ix]:edges[ix + 1]].sum(dim=(1, 2)))
        pooled = torch.cat(blocks, dim=1)
    return pooled.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# deep-patch whitener + dictionary (M117/M126 pattern, fresh registration)
# ---------------------------------------------------------------------------
def _fit_deep_whitener(sample_tokens: np.ndarray, rep: dict[str, Any]
                       ) -> np.ndarray:
    rng = np.random.default_rng(int(rep["zca_fit_seed"]))
    patches = sample_tokens.reshape(-1, sample_tokens.shape[-1])
    take = min(int(rep["zca_fit_patches"]), len(patches))
    pool = patches[rng.choice(len(patches), take, replace=False)]
    mean = pool.mean(axis=0)
    cov = np.cov(pool.T) + float(rep["zca_epsilon"]) * np.eye(pool.shape[1])
    u, s, _ = np.linalg.svd(cov)
    whiten = u @ np.diag(1.0 / np.sqrt(s)) @ u.T
    return mean.astype(np.float32), whiten.astype(np.float32)


def _deep_dictionary(sample_tokens: np.ndarray, mean: np.ndarray,
                     whiten: np.ndarray, rep: dict[str, Any],
                     atoms: int) -> np.ndarray:
    rng = np.random.default_rng(int(rep["dictionary_seed"]))
    patches = sample_tokens.reshape(-1, sample_tokens.shape[-1])
    pool_size = int(rep["candidate_pool_size"])
    candidates = ((patches[rng.choice(len(patches), pool_size,
                                      replace=False)] - mean)
                  @ whiten).astype(np.float32)
    order = np.random.default_rng([int(rep["dictionary_seed"]), 100]
                                  ).permutation(pool_size)
    return candidates[order[:atoms]]


def run_m176c_c1(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible

    configure_external_cache_environment()
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # ---- the corpus schedule (digest gate) + pixel identity ----------------
    corpus, train_index, test_index = _load_corpus(config)
    # `_load_corpus` decoded the 32 px cache; the deep cells decode at 224
    # px straight from the parquets using the SAME indices.
    print("streaming 224px decode + backbone encode", flush=True)
    processor, model = _load_backbone(config)
    model = model.to(device)

    # ---- dictionary fit on a token sample ----------------------------------
    rep = config["sparse"]
    sample_rows = int(rep["dict_sample_images"])
    sample_images = _decode_rows("train", train_index[:sample_rows], 224)
    sample_tokens = []
    step = int(config["numerics"]["backbone_batch"])
    for start in range(0, len(sample_images), step):
        stop = min(start + step, len(sample_images))
        sample_tokens.append(
            _patch_tokens(model, processor, device,
                          sample_images[start:stop])
            .cpu().numpy())
    sample_tokens = np.concatenate(sample_tokens, axis=0)
    mean, whiten = _fit_deep_whitener(sample_tokens, rep)
    del sample_images

    atoms_ladder = ([int(a) for a in config.get("_smoke_atoms", [256])]
                    if smoke
                    else [int(a) for a in config["cell"]["atoms_ladder"]])

    n_train_rows = int(config["_smoke_train_rows"]) if smoke else len(
        corpus["train_labels"])
    n_test_rows = int(config["_smoke_test_rows"]) if smoke else len(
        corpus["test_labels"])
    test_idx = test_index[:n_test_rows]
    train_idx = train_index[:n_train_rows]

    results = {}
    for atoms in atoms_ladder:
        t0 = time.time()
        dictionary = _deep_dictionary(sample_tokens, mean, whiten, rep,
                                      atoms)
        table = torch.from_numpy(np.ascontiguousarray(dictionary)).to(
            device)
        width = 21 * atoms
        acc = RidgeAccumulator(width, CLASSES)
        block = int(config["numerics"]["block"])
        for start in range(0, n_train_rows, block):
            stop = min(start + block, n_train_rows)
            imgs = _decode_rows("train", train_idx[start:stop], 224)
            toks = _patch_tokens(model, processor, device, imgs)
            codes = _spm_pool_tokens(toks, table)
            acc.add(codes, corpus["train_labels"][start:stop])
            if start % (block * 8) == 0:
                print(f"  atoms {atoms}: train {stop}/{n_train_rows}",
                      flush=True)
        std = acc.standardiser()
        weights = {str(p): w for p, w in
                   acc.solve_many([0.1, 1.0, 10.0]).items()}
        # score on the test stream
        hits = {p: 0 for p in weights}
        n_seen = 0
        for start in range(0, n_test_rows, block):
            stop = min(start + block, n_test_rows)
            imgs = _decode_rows("test", test_idx[start:stop], 224)
            toks = _patch_tokens(model, processor, device, imgs)
            codes = _spm_pool_tokens(toks, table)
            for p, w in weights.items():
                xs = std(codes)
                hits[p] += int(_score(w, xs,
                                      corpus["test_labels"][start:stop]
                                      ).sum())
            n_seen += stop - start
        del acc
        torch.cuda.empty_cache()
        results[str(atoms)] = {
            "accuracy_by_penalty": {p: hits[p] / n_seen
                                    for p in hits},
            "width": width,
            "readout_macs": width * CLASSES,
            "encode_seconds": round(time.time() - t0, 1),
        }

    evidence: dict[str, Any] = {
        "milestone": "M176c-c1",
        "cell": "deep-patch SPM on frozen DINOv2-small",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
        "interpretation_registered_before_running":
            config["interpretation_registered_before_running"],
        "backbone": config["backbone"]["name"],
        "spm_levels": list(SPM_LEVELS),
        "results": results,
        "dense_ladder_reference": {"r70": 0.3118, "r98": 0.4476},
        "sealed_anchors_for_reference": {
            "raw_full": 0.2604927536231884,
            "p05_138k": 0.2273623188405797,
            "p05_full": 0.27855072463768116},
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps(results, indent=1), flush=True)
    print(f"M176c-c1 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m176c_c1(args.config, args.output)


if __name__ == "__main__":
    main()
