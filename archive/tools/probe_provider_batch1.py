"""Compare execution providers at batch size 1, where quantization is per-image.

The earlier provider comparison ran at batch size 32, which conflates two
effects: a genuine kernel difference between providers, and the batch-composition
sensitivity introduced by DynamicQuantizeLinear. At batch size 1 the activation
scale is derived from a single image, so any residual disagreement is
attributable to the provider's quantized kernels alone.

This decides whether the GPU can be used for corpus extraction at all.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.tier4.prepare_v5_frozen_features import (  # noqa: E402
    preprocess_image_dinov2,
)

ONNX_PATH = REPOSITORY_ROOT / "data/v5/backbones/dinov2-small/onnx/model_int8.onnx"
PREPROCESSOR_PATH = REPOSITORY_ROOT / "data/v5/backbones/dinov2-small/preprocessor_config.json"
PROBE_COUNT = 48


def _run(session, batch: np.ndarray, batch_size: int) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    collected = []
    for offset in range(0, len(batch), batch_size):
        chunk = batch[offset : offset + batch_size]
        outputs = dict(
            zip(output_names, session.run(output_names, {input_name: chunk}), strict=True)
        )
        collected.append(outputs["last_hidden_state"][:, 0, :])
    return np.concatenate(collected, axis=0)


def main() -> None:
    import onnxruntime as ort

    config = json.loads(PREPROCESSOR_PATH.read_text(encoding="utf-8"))
    generator = np.random.default_rng(20260729)
    images = generator.integers(0, 256, size=(PROBE_COUNT, 384, 384, 3), dtype=np.uint8)
    batch = np.stack([preprocess_image_dinov2(image, config) for image in images], axis=0)

    report: dict[str, object] = {}
    reference: np.ndarray | None = None

    for provider in ("CPUExecutionProvider", "DmlExecutionProvider"):
        session = ort.InferenceSession(str(ONNX_PATH), providers=[provider])
        if provider not in session.get_providers():
            report[provider] = {"available": False}
            continue
        features = _run(session, batch, 1)
        _run(session, batch[:4], 1)
        start = time.perf_counter()
        _run(session, batch, 1)
        elapsed = time.perf_counter() - start

        entry: dict[str, object] = {
            "available": True,
            "images_per_second_batch_1": round(PROBE_COUNT / elapsed, 1),
            "feature_norm": round(float(np.linalg.norm(features[0])), 4),
        }
        if reference is None:
            reference = features
        else:
            difference = float(np.abs(features - reference).max())
            entry["max_absolute_difference_vs_cpu"] = difference
            entry["relative_difference_vs_cpu"] = round(
                difference / float(np.abs(reference).max()), 6
            )
            entry["usable_as_cpu_substitute"] = difference < 1e-3
        report[provider] = entry

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
