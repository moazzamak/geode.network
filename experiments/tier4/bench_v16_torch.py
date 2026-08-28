"""Measure the ROCm venv against CPU on the two operations v16 needs.

The prior note in this program says ROCm was "38x slower" and must never be
used. That was measured on some workload at some size; this re-measures it on
the two that decide v16: a forward+backward pass through a ViT-sized transformer
block (trunk training) and the tall-skinny matmul the sparse encoder is built
from (dictionary correlation).

Engineering measurement of the instrument. Not evidence, no corpus, no accuracy.
"""
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def device_report(torch) -> list[dict]:
    out = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        out.append({
            "index": index,
            "name": properties.name,
            "total_memory_gb": round(properties.total_memory / 1024 ** 3, 2),
            "multi_processor_count": getattr(
                properties, "multi_processor_count", None),
        })
    return out


def timed(function, iterations: int, synchronise) -> float:
    function()
    synchronise()
    started = time.time()
    for _ in range(iterations):
        function()
    synchronise()
    return (time.time() - started) / iterations


def transformer_step(torch, device, width=384, depth=12, tokens=257,
                     batch=32, train=True):
    """One forward (+backward) through a ViT-S-shaped stack."""
    block = torch.nn.TransformerEncoderLayer(
        d_model=width, nhead=6, dim_feedforward=4 * width,
        batch_first=True, dropout=0.0,
    )
    model = torch.nn.TransformerEncoder(block, num_layers=depth).to(device)
    data = torch.randn(batch, tokens, width, device=device)
    if not train:
        model.eval()

        def step():
            with torch.no_grad():
                model(data)
        return step, model
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step():
        optimiser.zero_grad(set_to_none=True)
        model(data).square().mean().backward()
        optimiser.step()
    return step, model


def sparse_step(torch, device, atoms=3072, dimension=108, patches=729,
                rows=64):
    """The sparse encoder's inner product: (rows*patches, dim) @ (dim, atoms)."""
    left = torch.randn(rows * patches, dimension, device=device)
    right = torch.randn(dimension, atoms, device=device)

    def step():
        result = left @ right
        torch.nn.functional.relu(result, inplace=True)
        return result
    return step


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    import torch

    report = {"_note": "engineering measurement of the instrument, NOT"
                       " evidence; no corpus, no accuracy, no operand",
              "torch": torch.__version__,
              "hip": getattr(torch.version, "hip", None),
              "target": which}

    if which == "gpu":
        if not torch.cuda.is_available():
            raise SystemExit("no GPU backend in this interpreter")
        report["devices"] = device_report(torch)
        index = int(os.environ.get("GEODE_GPU_INDEX", "0"))
        device = torch.device(f"cuda:{index}")
        torch.cuda.set_device(index)
        report["chosen_device"] = report["devices"][index]

        def synchronise():
            torch.cuda.synchronize()
    else:
        device = torch.device("cpu")
        torch.set_num_threads(16)

        def synchronise():
            return None

    print(f"torch {torch.__version__}  hip={getattr(torch.version, 'hip', None)}")
    if which == "gpu":
        print(f"device {report['chosen_device']}")
    print()

    for label, train, iterations in [("ViT-S forward only", False, 5),
                                     ("ViT-S forward+backward", True, 5)]:
        step, _ = transformer_step(torch, device, train=train)
        seconds = timed(step, iterations, synchronise)
        rate = 32 / seconds
        report[label] = {"seconds_per_batch32": round(seconds, 4),
                         "img_per_s": round(rate, 2)}
        print(f"{label:<26} {seconds * 1000:9.1f} ms/batch32   {rate:8.2f} img/s")

    step = sparse_step(torch, device)
    seconds = timed(step, 5, synchronise)
    report["sparse encode matmul"] = {
        "seconds_per_64_images": round(seconds, 4),
        "img_per_s": round(64 / seconds, 2)}
    print(f"{'sparse encode matmul':<26} {seconds * 1000:9.1f} ms/64 images"
          f"   {64 / seconds:8.2f} img/s")

    out = ROOT / f"logs/results/v16/torch_benchmark_{which}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
