"""Test whether the frozen INT8 backbone is a per-image function.

A feature extractor is normally assumed to compute each image independently, so
that a corpus is a function of its images alone. If the ONNX graph uses dynamic
quantization, activation scales are computed per tensor at run time, over the
whole batch -- which makes every image's features depend on which other images
happened to share its batch.

This probe answers three questions:

1. Does an image's feature vector change when its batch neighbours change?
2. Does it change with batch size?
3. Does the graph actually contain dynamic quantization operators?
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.tier4.prepare_v5_frozen_features import (  # noqa: E402
    preprocess_image_dinov2,
)

ONNX_PATH = REPOSITORY_ROOT / "data/v5/backbones/dinov2-small/onnx/model_int8.onnx"
PREPROCESSOR_PATH = REPOSITORY_ROOT / "data/v5/backbones/dinov2-small/preprocessor_config.json"


def _features(session, batch: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    outputs = dict(zip(output_names, session.run(output_names, {input_name: batch}), strict=True))
    return outputs["last_hidden_state"][:, 0, :]


def main() -> None:
    import onnx
    import onnxruntime as ort

    config = json.loads(PREPROCESSOR_PATH.read_text(encoding="utf-8"))
    generator = np.random.default_rng(20260729)
    images = generator.integers(0, 256, size=(32, 384, 384, 3), dtype=np.uint8)
    processed = np.stack([preprocess_image_dinov2(image, config) for image in images], axis=0)

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    alone = _features(session, processed[:1])[0]
    in_batch_of_8 = _features(session, processed[:8])[0]
    in_batch_of_32 = _features(session, processed[:32])[0]

    shuffled_order = np.concatenate([[0], generator.permutation(np.arange(1, 32))])
    in_shuffled_batch = _features(session, processed[shuffled_order])[0]

    report = {
        "image_0_alone_vs_batch_of_8": float(np.abs(alone - in_batch_of_8).max()),
        "image_0_alone_vs_batch_of_32": float(np.abs(alone - in_batch_of_32).max()),
        "image_0_batch_of_32_vs_same_batch_reordered": float(
            np.abs(in_batch_of_32 - in_shuffled_batch).max()
        ),
        "image_0_feature_norm": float(np.linalg.norm(alone)),
    }
    report["is_per_image_function"] = (
        report["image_0_alone_vs_batch_of_8"] == 0.0
        and report["image_0_alone_vs_batch_of_32"] == 0.0
        and report["image_0_batch_of_32_vs_same_batch_reordered"] == 0.0
    )

    model = onnx.load(str(ONNX_PATH))
    operator_counts: dict[str, int] = {}
    for node in model.graph.node:
        operator_counts[node.op_type] = operator_counts.get(node.op_type, 0) + 1
    report["dynamic_quantization_operators"] = {
        name: count
        for name, count in sorted(operator_counts.items())
        if "Dynamic" in name or "Quantize" in name
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
