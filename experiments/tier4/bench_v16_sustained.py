"""Sustained GPU utilisation check for the two workloads v16 actually runs.

The one-shot benches in ``bench_v16_torch.py`` average over five iterations and
include synchronisation overhead, so they cannot show whether a *sustained* loop
keeps the GPU fed. A user watching Task Manager during a real run reported ~1%
GPU utilisation, which is what this script exists to explain or refute.

It measures, on the 9070 XT, in one long loop each:

* **(1) trunk training** — the real DINOv2-small model (the weights M109 will
  fine-tune), forward+backward at batch 32, 224 px, many steps, with an AdamW
  step so nothing is optimised away. This is M109's core loop.
* **(2) sparse encode** — the real M107/M108 sparse encoder (whitened 32x32
  patches -> ``torch.cdist`` against a 3072-atom dictionary -> 2x2 pool), with
  each phase timed separately so we can see where the wall clock actually goes:
  CPU whitening (numpy) vs GPU distance/pool vs CPU/GPU transfer.

Engineering measurement of the instrument. Not evidence, no corpus, no accuracy.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/results/v16"

from experiments.common.data_cache import data_cache_root  # noqa: E402
from experiments.tier4.eval_v15_m103_atoms import (  # noqa: E402
    Whitener,
    _contrast_normalise,
    _extract_patches,
    _fit_zca,
    _pool,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _report(torch) -> dict:
    index = int(os.environ.get("GEODE_GPU_INDEX", "0"))
    p = torch.cuda.get_device_properties(index)
    return {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device": p.name,
        "gcnArchName": getattr(p, "gcnArchName", None),
        "total_memory_gb": round(p.total_memory / 1024 ** 3, 2),
    }


def _trunk_training(torch, device, steps: int) -> dict:
    from transformers import Dinov2Model

    weights = data_cache_root() / "torch" / "dinov2-small"
    if not weights.exists():
        raise SystemExit(f"no torch weights at {weights}")
    model = Dinov2Model.from_pretrained(str(weights), dtype=torch.float32)
    model.train().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = 32
    rng = torch.Generator(device="cpu").manual_seed(20260805)
    data = torch.randint(0, 256, (batch, 224, 224, 3), generator=rng,
                         dtype=torch.uint8).to(device)
    mean = torch.from_numpy(IMAGENET_MEAN).to(device).view(1, 3, 1, 1)
    std = torch.from_numpy(IMAGENET_STD).to(device).view(1, 3, 1, 1)
    pixels = (data.float().permute(0, 3, 1, 2) / 255.0 - mean) / std

    def step():
        optimiser.zero_grad(set_to_none=True)
        out = model(pixel_values=pixels).last_hidden_state
        out.square().mean().backward()
        optimiser.step()

    step()
    torch.cuda.synchronize()
    started = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    seconds = time.time() - started
    return {
        "workload": "trunk training (DINOv2-small fwd+bwd+step)",
        "batch": batch,
        "steps": steps,
        "seconds": round(seconds, 2),
        "img_per_s": round(steps * batch / seconds, 2),
    }


def _sparse_encoder(torch, device) -> tuple:
    """Real M107 sparse encoder, one 3072-atom generalist, on synthetic data.

    Returns (encode_function, per_phase_timer). Phases: whiten (CPU numpy),
    to_gpu, cdist+activation+pool (GPU), to_numpy.
    """
    patch, stride, pool_grid = 6, 1, 2
    size, grid = 32, 27
    rng = np.random.default_rng(20260805)
    images = rng.integers(0, 256, (1024, size, size, 3), dtype=np.uint8)

    sample = images[:200]
    patches = _extract_patches(sample, patch, stride)
    take = min(200_000, len(patches))
    pool = _contrast_normalise(patches[rng.choice(len(patches), take,
                                                  replace=False)], 10.0)
    mean, whiten = _fit_zca(pool, 0.1)
    whitener = Whitener(patch, stride, 10.0, mean, whiten, grid)

    dict_rng = np.random.default_rng(11)
    dictionary = (pool[dict_rng.choice(len(pool), 3072, replace=False)]
                  - mean) @ whiten
    dictionary = np.ascontiguousarray(dictionary).astype(np.float32)
    table = torch.from_numpy(dictionary).to(device)
    dimension = patch * patch * 3
    atoms = 3072
    per_image = grid * grid
    chunk = max(1, int(1_200_000_000 / (4 * atoms * per_image)))

    timings = {"whiten_cpu": 0.0, "to_gpu": 0.0, "gpu_kernels": 0.0,
               "to_numpy": 0.0}

    def encode_loop():
        for start in range(0, len(images), chunk):
            block = images[start:start + chunk]
            t0 = time.time()
            white = np.ascontiguousarray(whitener(block))
            timings["whiten_cpu"] += time.time() - t0
            t0 = time.time()
            white_t = torch.from_numpy(white).to(device)
            torch.cuda.synchronize()
            timings["to_gpu"] += time.time() - t0
            t0 = time.time()
            with torch.no_grad():
                distances = torch.cdist(white_t, table)
                activation = torch.clamp(
                    distances.mean(dim=1, keepdim=True) - distances, min=0.0
                )
                pooled = _pool(activation, len(block), grid, pool_grid)
            torch.cuda.synchronize()
            timings["gpu_kernels"] += time.time() - t0
            t0 = time.time()
            pooled = pooled.to(torch.float32).cpu().numpy()
            timings["to_numpy"] += time.time() - t0
        return len(images)

    return encode_loop, timings


def main() -> int:
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
    if not torch.cuda.is_available():
        raise SystemExit("no GPU backend; HIP_VISIBLE_DEVICES=1 is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    report = {"_note": "engineering measurement of the instrument, NOT"
                       " evidence; no corpus, no accuracy, no operand",
              "instrument": _report(torch)}

    print("=== 1) sustained trunk training: DINOv2-small fwd+bwd, batch 32, "
          "224 px ===", flush=True)
    result = _trunk_training(torch, device, steps=60)
    report["trunk_training"] = result
    print(f"    {result['img_per_s']:.2f} img/s sustained over "
          f"{result['seconds']}s", flush=True)

    print("=== 2) sustained sparse encode: 3072-atom generalist, 1024 images, "
          "real chunking ===", flush=True)
    encode_loop, timings = _sparse_encoder(torch, device)
    encode_loop()
    torch.cuda.synchronize()
    started = time.time()
    encode_loop()
    torch.cuda.synchronize()
    seconds = time.time() - started
    total = sum(timings.values())
    report["sparse_encode"] = {
        "images": 1024,
        "seconds": round(seconds, 2),
        "img_per_s": round(1024 / seconds, 2),
        "phase_fraction": {k: round(v / total, 3) for k, v in timings.items()},
        "phase_seconds": {k: round(v, 2) for k, v in timings.items()},
    }
    print(f"    {1024 / seconds:.2f} img/s sustained", flush=True)
    for k, v in timings.items():
        print(f"      {k:<12} {v / total * 100:5.1f}%  ({v:.2f}s)", flush=True)

    out = OUT / "torch_benchmark_sustained.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
