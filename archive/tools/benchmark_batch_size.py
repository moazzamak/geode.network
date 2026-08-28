"""Measure INT8 backbone throughput at batch size 1 against batch size 32.

Batch size 1 is the only setting under which the corpus is a well-defined
function of its images, because the graph's DynamicQuantizeLinear operators
otherwise derive activation scales from the whole batch tensor. This measures
what that guarantee costs, including whether restricting intra-op threads and
running several processes recovers the throughput.
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
PROBE_COUNT = 96


def main() -> None:
    import onnxruntime as ort

    config = json.loads(PREPROCESSOR_PATH.read_text(encoding="utf-8"))
    generator = np.random.default_rng(1)
    images = generator.integers(0, 256, size=(PROBE_COUNT, 384, 384, 3), dtype=np.uint8)
    batch = np.stack([preprocess_image_dinov2(image, config) for image in images], axis=0)

    results: dict[str, float] = {}
    for thread_count in (0, 1, 2, 4):
        options = ort.SessionOptions()
        if thread_count:
            options.intra_op_num_threads = thread_count
        session = ort.InferenceSession(
            str(ONNX_PATH), options, providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        label = "auto" if thread_count == 0 else str(thread_count)
        for batch_size in (1, 32):
            session.run(output_names, {input_name: batch[:batch_size]})
            start = time.perf_counter()
            for offset in range(0, PROBE_COUNT, batch_size):
                session.run(output_names, {input_name: batch[offset : offset + batch_size]})
            elapsed = time.perf_counter() - start
            results[f"threads_{label}_batch_{batch_size}"] = round(PROBE_COUNT / elapsed, 1)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
