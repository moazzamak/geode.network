"""Does a torch DINOv2 on the GPU reproduce the ONNX features M107 measured?

M107's dense arms are onnxruntime sessions on the CPU. Trunk training needs
gradients, which onnxruntime will not give, so v16's trunk arm has to be torch.
That is only sound if the torch model *starts* from M107's exact frozen
operating point -- otherwise the frozen and trained arms differ in two things at
once and neither can be read.

This compares the two implementations on the same input, through the same
feature definition M107 used (CLS concatenated with the mean patch token), and
reports the largest disagreement. Run the two halves in their own interpreters
(``onnx`` in .venv, ``torch`` in .venv-rocm) and then ``compare``.

Engineering measurement of the instrument. Not evidence, no corpus, no accuracy.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common.data_cache import data_cache_root

OUT = ROOT / "logs/results/v16"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def fixed_input(batch: int, resolution: int) -> np.ndarray:
    """A deterministic uint8 block, normalised exactly as M107 normalises."""
    rng = np.random.default_rng(20260805)
    images = rng.integers(0, 256, (batch, resolution, resolution, 3),
                          dtype=np.uint8)
    block = images.astype(np.float32) / 255.0
    block = (block - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(block.transpose(0, 3, 1, 2))


def feature(tokens: np.ndarray) -> np.ndarray:
    """M107's feature definition: CLS then the mean of the patch tokens."""
    return np.concatenate(
        [tokens[:, 0, :], tokens[:, 1:, :].mean(axis=1)], axis=1
    ).astype(np.float32)


def onnx_side(name: str, provider: str, batch: int, resolution: int) -> None:
    import onnxruntime as ort

    found = sorted((data_cache_root() / "huggingface" / "hub").glob(
        f"models--onnx-community--dinov2-{name}-ONNX/snapshots/*/onnx/model.onnx"))
    if not found:
        raise SystemExit(f"no ONNX export for dinov2-{name}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 16
    session = ort.InferenceSession(str(found[0]), options, providers=[provider])
    tokens = session.run(None, {"pixel_values": fixed_input(batch, resolution)})[0]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = provider.replace("ExecutionProvider", "").lower()
    np.save(OUT / f"parity_{name}_onnx_{tag}.npy", feature(tokens))
    print(f"onnx/{tag} {name}: tokens {tokens.shape} -> feature saved")


def torch_side(name: str, batch: int, resolution: int, device: str) -> None:
    import torch
    from transformers import Dinov2Model

    weights = data_cache_root() / "torch" / f"dinov2-{name}"
    if not weights.exists():
        raise SystemExit(f"no torch weights at {weights}")
    model = Dinov2Model.from_pretrained(str(weights), dtype=torch.float32)
    model.eval().to(device)
    data = torch.from_numpy(fixed_input(batch, resolution)).to(device)
    with torch.no_grad():
        tokens = model(pixel_values=data).last_hidden_state
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"parity_{name}_torch_{device.split(':')[0]}.npy",
            feature(tokens.float().cpu().numpy()))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"torch/{device} {name}: tokens {tuple(tokens.shape)},"
          f" {trainable:,} trainable parameters -> feature saved")


def compare(names: list[str]) -> int:
    report = {"_note": "engineering measurement of the instrument, NOT"
                       " evidence; no corpus, no accuracy, no operand",
              "pairs": []}
    worst = 0.0
    for name in names:
        available = sorted(OUT.glob(f"parity_{name}_*.npy"))
        if len(available) < 2:
            print(f"dinov2-{name}: need at least two sides, have"
                  f" {[p.name for p in available]}")
            continue
        reference_path = OUT / f"parity_{name}_onnx_cpu.npy"
        if not reference_path.exists():
            print(f"dinov2-{name}: no onnx/cpu reference")
            continue
        reference = np.load(reference_path)
        scale = float(np.abs(reference.astype(np.float64)).max())
        print(f"\ndinov2-{name}  reference onnx/cpu, feature {reference.shape},"
              f" max |value| {scale:.4f}")
        for path in available:
            if path == reference_path:
                continue
            other = np.load(path)
            gap = float(np.abs(reference.astype(np.float64)
                               - other.astype(np.float64)).max())
            cosine = float(np.mean(np.sum(reference * other, axis=1) / (
                np.linalg.norm(reference, axis=1)
                * np.linalg.norm(other, axis=1))))
            label = path.stem.replace(f"parity_{name}_", "")
            print(f"   vs {label:<12} max abs {gap:.3e}   relative"
                  f" {gap / scale:.3e}   mean cosine {cosine:.8f}")
            worst = max(worst, gap / scale)
            report["pairs"].append({
                "model": name, "against": label,
                "max_abs_difference": gap,
                "max_relative_difference": gap / scale,
                "mean_cosine_similarity": cosine})
    report["worst_relative_difference"] = worst
    (OUT / "parity.json").write_text(json.dumps(report, indent=2),
                                     encoding="utf-8")
    print(f"\nworst relative difference across all pairs: {worst:.3e}")
    print(f"written to {OUT / 'parity.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("side", choices=["onnx", "torch", "compare"])
    parser.add_argument("--models", default="small,base,large")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=224)
    arguments = parser.parse_args()
    names = [n for n in arguments.models.split(",") if n]

    if arguments.side == "compare":
        return compare(names)
    for name in names:
        if arguments.side == "onnx":
            onnx_side(name, arguments.provider, arguments.batch,
                      arguments.resolution)
        else:
            torch_side(name, arguments.batch, arguments.resolution,
                       arguments.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
