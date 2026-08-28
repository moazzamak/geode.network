"""Measure DINOv2 ONNX inference on DirectML against the CPU provider M107 used.

Not an experiment and not evidence: this is an engineering measurement of the
instrument, taken to decide whether v16 can afford trunk training. It touches no
corpus and produces no accuracy. Bit-exactness between providers is checked and
reported, because a provider that is fast and wrong is worse than a slow one.
"""
import json
import pathlib
import sys
import time

import numpy as np
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common.data_cache import data_cache_root

CACHE = data_cache_root() / "huggingface" / "hub"


def model_path(name: str) -> pathlib.Path:
    found = sorted(CACHE.glob(
        f"models--onnx-community--dinov2-{name}-ONNX/snapshots/*/onnx/model.onnx"
    ))
    if not found:
        raise SystemExit(f"no ONNX export for dinov2-{name} under {CACHE}")
    return found[0]


def session(name: str, provider: str, threads: int = 16) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path(name)), options,
                                providers=[provider])


def bench(name: str, provider: str, resolution: int, batch: int,
          batches: int) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(0)
    block = rng.random((batch, 3, resolution, resolution),
                       dtype=np.float32)
    sess = session(name, provider)
    out = sess.run(None, {"pixel_values": block})[0]      # warm up / compile
    started = time.time()
    for _ in range(batches):
        out = sess.run(None, {"pixel_values": block})[0]
    elapsed = time.time() - started
    return (batch * batches) / elapsed, out


def main() -> int:
    print(f"onnxruntime {ort.__version__}")
    print(f"providers   {ort.get_available_providers()}\n")
    have_dml = "DmlExecutionProvider" in ort.get_available_providers()
    if not have_dml:
        raise SystemExit("DirectML provider not present in this interpreter")

    plan = [("small", 224, 32, 4), ("base", 224, 32, 3), ("large", 224, 16, 3)]
    rows = []
    for name, resolution, batch, batches in plan:
        print(f"dinov2-{name} at {resolution}px, batch {batch}")
        cpu_rate, cpu_out = bench(name, "CPUExecutionProvider", resolution,
                                  batch, batches)
        print(f"   CPU       {cpu_rate:8.2f} img/s")
        dml_rate, dml_out = bench(name, "DmlExecutionProvider", resolution,
                                  batch, batches)
        print(f"   DirectML  {dml_rate:8.2f} img/s   speedup"
              f" {dml_rate / cpu_rate:6.2f}x")
        gap = float(np.abs(cpu_out.astype(np.float64)
                           - dml_out.astype(np.float64)).max())
        scale = float(np.abs(cpu_out.astype(np.float64)).max())
        print(f"   max |CPU-DML| {gap:.3e}   relative {gap / scale:.3e}\n")
        rows.append({"model": name, "resolution": resolution, "batch": batch,
                     "cpu_img_per_s": round(cpu_rate, 3),
                     "dml_img_per_s": round(dml_rate, 3),
                     "speedup": round(dml_rate / cpu_rate, 3),
                     "max_abs_difference": gap,
                     "max_relative_difference": gap / scale})

    m107 = 138000 + 34500
    print("what this would have meant for M107's ten dense arms"
          " (172,500 images each):")
    total_cpu = total_dml = 0.0
    for row in rows:
        cpu_h = m107 / row["cpu_img_per_s"] / 3600
        dml_h = m107 / row["dml_img_per_s"] / 3600
        total_cpu += cpu_h
        total_dml += dml_h
        print(f"   dinov2-{row['model']:<6} CPU {cpu_h:6.2f} h ->"
              f" DirectML {dml_h:5.2f} h")
    print(f"   one arm of each: {total_cpu:.2f} h -> {total_dml:.2f} h")

    out = ROOT / "logs/results/v16/provider_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"_note": "engineering measurement of the instrument, NOT evidence;"
                  " no corpus, no accuracy, no operand",
         "onnxruntime": ort.__version__,
         "providers": ort.get_available_providers(),
         "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
