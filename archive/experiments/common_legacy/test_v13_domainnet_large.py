"""Tests for the enlarged v13 DomainNet corpus preparation.

The central risk this module guards against is silent reintroduction of batched
extraction. Batching is a large speed win and looks harmless, but the frozen
INT8 backbone's DynamicQuantizeLinear operators make batched features depend on
batch membership, which would make the corpus a function of its chunking rather
than of its images.

Following design principle 9, every measurement operand here ships with a
positive control that fails if the operand is not measuring what it names.
"""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from experiments.tier4.prepare_v13_domainnet_large import (
    DEFAULT_CONFIG,
    _image_dimensions,
    _source_files,
    _verify,
    select_and_extract,
)
from experiments.tier4.prepare_v5_frozen_features import extract_features_batch

REPO_ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = REPO_ROOT / "data/v5/backbones/dinov2-small/onnx/model_int8.onnx"
PREPROCESSOR_PATH = REPO_ROOT / "data/v5/backbones/dinov2-small/preprocessor_config.json"


def _encoded_image(width: int, height: int, seed: int = 0) -> bytes:
    generator = np.random.default_rng(seed)
    array = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _corpus_shards_available() -> bool:
    if not DEFAULT_CONFIG.is_file():
        return False
    try:
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        download = json.loads(_verify(config["domainnet_download_record"]).read_text())
        _source_files(download)
    except (OSError, ValueError, FileNotFoundError):
        return False
    return True


class ImageDimensionTests(unittest.TestCase):
    def test_reads_dimensions_from_header(self) -> None:
        payload = _encoded_image(640, 480, seed=1)
        self.assertEqual(_image_dimensions(payload), (640, 480))

    def test_distinguishes_images_below_the_short_edge_filter(self) -> None:
        """Positive control: the filter operand must separate the two cases."""
        wide_but_short = _image_dimensions(_encoded_image(1024, 128, seed=2))
        large = _image_dimensions(_encoded_image(300, 300, seed=3))
        self.assertLess(min(wide_but_short), 256)
        self.assertGreaterEqual(min(large), 256)


class BackboneBatchDependenceTests(unittest.TestCase):
    """Assert the defect that forces batch size one is real and still present.

    If a future change quantizes the backbone differently, or replaces it with a
    graph that is genuinely per-image, this test fails and the batch-size-one
    constraint can be revisited deliberately rather than by accident.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not ONNX_PATH.is_file():
            raise unittest.SkipTest("frozen DINOv2 backbone is unavailable")
        cls.config = json.loads(PREPROCESSOR_PATH.read_text(encoding="utf-8"))
        generator = np.random.default_rng(20260729)
        cls.images = [
            generator.integers(0, 256, size=(288, 288, 3), dtype=np.uint8) for _ in range(4)
        ]

    def _extract(self, images: list[np.ndarray], batch_size: int) -> np.ndarray:
        return extract_features_batch(
            images,
            "dinov2-small",
            str(ONNX_PATH),
            self.config,
            "cls_token",
            batch_size=batch_size,
        )

    def test_batched_extraction_is_not_a_per_image_function(self) -> None:
        alone = self._extract(self.images[:1], 1)[0]
        together = self._extract(self.images, 4)[0]
        self.assertGreater(
            float(np.abs(alone - together).max()),
            1e-3,
            "batched extraction unexpectedly matched per-image extraction; "
            "the dynamic-quantization dependence may have been removed",
        )

    def test_per_image_extraction_is_independent_of_grouping(self) -> None:
        one_call = self._extract(self.images, 1)
        split = np.concatenate(
            [self._extract(self.images[:1], 1), self._extract(self.images[1:], 1)], axis=0
        )
        np.testing.assert_array_equal(one_call, split)


@unittest.skipUnless(_corpus_shards_available(), "DomainNet shards are unavailable")
class SelectAndExtractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        download = json.loads(_verify(cls.config["domainnet_download_record"]).read_text())
        cls.source_files = _source_files(download)
        cls.backbone = cls.config["backbone"]
        cls.preprocessing = json.loads(PREPROCESSOR_PATH.read_text(encoding="utf-8"))

    def _run(self, *, worker_count: int, chunk_size: int, samples_per_class: int = 4):
        return select_and_extract(
            self.source_files,
            classes=np.arange(3),
            samples_per_class=samples_per_class,
            minimum_short_edge=int(self.config["minimum_native_short_edge"]),
            backbone=self.backbone,
            preprocessing=self.preprocessing,
            onnx_path=ONNX_PATH,
            dispatch_chunk_size=chunk_size,
            worker_count=worker_count,
            progress=False,
        )

    def test_output_is_class_major(self) -> None:
        _, labels, manifest = self._run(worker_count=2, chunk_size=5)
        np.testing.assert_array_equal(labels, np.repeat(np.arange(3), 4))
        self.assertEqual([entry["class_label"] for entry in manifest], list(labels))

    def test_selection_respects_the_short_edge_filter(self) -> None:
        _, _, manifest = self._run(worker_count=2, chunk_size=5)
        minimum = int(self.config["minimum_native_short_edge"])
        for entry in manifest:
            self.assertGreaterEqual(
                min(entry["native_width"], entry["native_height"]), minimum
            )

    def test_result_is_invariant_to_sharding(self) -> None:
        """The positive control for the extraction operand itself."""
        sequential, labels_a, manifest_a = self._run(worker_count=1, chunk_size=64)
        sharded, labels_b, manifest_b = self._run(worker_count=4, chunk_size=3)
        np.testing.assert_array_equal(labels_a, labels_b)
        self.assertEqual(manifest_a, manifest_b)
        np.testing.assert_array_equal(sequential, sharded)


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def test_extraction_is_registered_at_batch_size_one(self) -> None:
        self.assertEqual(int(self.config["extraction_batch_size"]), 1)

    def test_geometry_budget_clears_the_registered_floor(self) -> None:
        geometry = int(self.config["samples_per_class"]) - 40
        self.assertGreaterEqual(geometry / 32.0, 10.0)

    def test_final_labels_remain_sealed(self) -> None:
        self.assertFalse(self.config["final_labels_opened"])


if __name__ == "__main__":
    unittest.main()
