"""Benchmark the frozen DINOv2-small feature extractor across execution providers.

Feature extraction is the throughput bottleneck for enlarging the corpus, so
this measures where the time actually goes before any bulk run is launched.

Two things are measured separately, because they scale differently:

* **Preprocessing** -- PIL decode, resize, crop, normalize. Pure CPU, and
  trivially parallel across processes.
* **Inference** -- the INT8 ONNX graph, which is what a GPU provider could
  accelerate.

The provider comparison also reports numerical agreement. The backbone is an
INT8-quantized graph, and quantized operators are the ones most likely to be
implemented differently (or silently decomposed) by a non-CPU provider. A
provider that is fast but disagrees with the sealed CPU reference cannot be
used for a corpus that must stay comparable with the existing v12/v13 record.

Run with the frozen replay interpreter::

    .\\.venv\\Scripts\\python.exe tools\\benchmark_feature_extraction.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.tier4.prepare_v5_frozen_features import (  # noqa: E402
    preprocess_image_dinov2,
)

ONNX_PATH = REPOSITORY_ROOT / "data/v5/backbones/dinov2-small/onnx/model_int8.onnx"
PREPROCESSOR_PATH = REPOSITORY_ROOT / "data/v5/backbones/dinov2-small/preprocessor_config.json"

#: Enough images to amortize session warm-up without making the probe slow.
PROBE_IMAGE_COUNT = 64


def _preprocessor_config() -> dict:
    return json.loads(PREPROCESSOR_PATH.read_text(encoding="utf-8"))


def _probe_images(count: int) -> np.ndarray:
    """Deterministic synthetic images at native DomainNet-like resolution."""
    generator = np.random.default_rng(20260729)
    return generator.integers(0, 256, size=(count, 512, 512, 3), dtype=np.uint8)


def _time_preprocessing(images: np.ndarray, config: dict) -> dict[str, float]:
    start = time.perf_counter()
    processed = np.stack([preprocess_image_dinov2(image, config) for image in images], axis=0)
    elapsed = time.perf_counter() - start
    return {
        "seconds": round(elapsed, 3),
        "images_per_second": round(len(images) / elapsed, 1),
        "_processed": processed,
    }


def _time_provider(
    provider: str, batch: np.ndarray, batch_size: int
) -> dict[str, object] | None:
    import onnxruntime as ort

    try:
        session = ort.InferenceSession(str(ONNX_PATH), providers=[provider])
    except Exception as error:  # noqa: BLE001 - capability probe
        return {"available": False, "error": f"{type(error).__name__}: {error}"}

    if provider not in session.get_providers():
        return {"available": False, "error": f"provider fell back to {session.get_providers()}"}

    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    def run_all() -> np.ndarray:
        collected = []
        for start in range(0, len(batch), batch_size):
            chunk = batch[start : start + batch_size]
            outputs = dict(
                zip(output_names, session.run(output_names, {input_name: chunk}), strict=True)
            )
            collected.append(outputs["last_hidden_state"][:, 0, :])
        return np.concatenate(collected, axis=0)

    run_all()  # warm up: first call pays graph compilation and allocator setup
    start = time.perf_counter()
    features = run_all()
    elapsed = time.perf_counter() - start

    return {
        "available": True,
        "seconds": round(elapsed, 3),
        "images_per_second": round(len(batch) / elapsed, 1),
        "features": features,
    }


def main() -> None:
    config = _preprocessor_config()
    images = _probe_images(PROBE_IMAGE_COUNT)

    report: dict[str, object] = {"probe_image_count": PROBE_IMAGE_COUNT}

    preprocessing = _time_preprocessing(images, config)
    batch = preprocessing.pop("_processed")
    report["preprocessing"] = preprocessing

    providers: dict[str, object] = {}
    reference: np.ndarray | None = None
    for provider, batch_size in (("CPUExecutionProvider", 32), ("DmlExecutionProvider", 32)):
        result = _time_provider(provider, batch, batch_size)
        if result is None or not result.get("available"):
            providers[provider] = result
            continue
        features = result.pop("features")
        if reference is None:
            reference = features
            result["max_absolute_difference_vs_cpu"] = 0.0
            result["bit_identical_to_cpu"] = True
        else:
            difference = float(np.abs(features - reference).max())
            scale = float(np.abs(reference).max())
            result["max_absolute_difference_vs_cpu"] = difference
            result["relative_difference_vs_cpu"] = round(difference / scale, 6)
            result["bit_identical_to_cpu"] = difference == 0.0
        providers[provider] = result

    report["providers"] = providers

    cpu = providers.get("CPUExecutionProvider", {})
    if isinstance(cpu, dict) and cpu.get("available"):
        inference_rate = cpu["images_per_second"]
        preprocess_rate = preprocessing["images_per_second"]
        combined = 1.0 / (1.0 / inference_rate + 1.0 / preprocess_rate)
        report["single_process_cpu_pipeline"] = {
            "images_per_second": round(combined, 1),
            "hours_for_81920_images": round(81_920 / combined / 3600, 2),
        }

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
