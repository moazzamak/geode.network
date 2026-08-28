"""Build an enlarged native DomainNet feature corpus for the v13 program.

Motivation
----------
M78 established that no cell of the v12/v13 rank sweep reached the registered
floor of ten samples per fitted dimension, and that no per-class basis was
identified at any rank. The sealed M70 corpus supplies only 100 samples per
class, of which at most 60 remain for geometry once calibration and evaluation
are held out. That is a property of the corpus, not of the estimator, and no
change to the method can repair it.

This script produces the corpus that makes the floor reachable: 128 classes at
640 samples per class. With calibration and evaluation held out at 20 each,
600 samples remain for geometry, which clears ten samples per fitted dimension
at ranks up to 60.

Why extraction runs at batch size one
-------------------------------------
The frozen DINOv2-small INT8 graph contains 49 ``DynamicQuantizeLinear``
operators. These derive activation scales from the whole input tensor at run
time, so at any batch size above one an image's features depend on which other
images shared its batch. Measured directly, the same image extracted alone and
inside a batch of 32 differs by 1.21 in absolute terms against a feature norm
of roughly 50; reordering within a fixed batch changes nothing, confirming the
dependence is on batch membership rather than position.

Two consequences follow. First, the sealed M70 corpus is a function of how it
was chunked, not of its images alone, and cannot be reproduced by any corpus
with a different layout. Second, extracting at batch size one makes the corpus
a well-defined function of ``(image, backbone)``.

That second property is what makes this fast despite the smaller batch. Because
each image is independent, partitioning the work across processes cannot change
the result, so extraction shards freely over the CPU. The shard-invariance
control asserts exactly this, and would fail if the batching dependence were
still present.

Usage::

    .\\.venv\\Scripts\\python.exe -m experiments.tier4.prepare_v13_domainnet_large
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import pyarrow.parquet as parquet

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.tier4.prepare_v5_frozen_features import extract_features_batch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v13" / "domainnet_large.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v13" / "domainnet_large"

#: Per-worker session state, populated by :func:`_worker_initializer`.
_WORKER_STATE: dict[str, Any] = {}


def _resolve(path: str) -> Path:
    resolved = (REPO_ROOT / Path(path)).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError("v13 corpus paths must remain inside the repository")
    return resolved


def _verify(specification: dict[str, str]) -> Path:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"v13 corpus immutable artifact hash mismatch: {path}")
    return path


def _source_files(download: dict[str, Any]) -> list[Path]:
    repository = Path(download["dataset_root"]) / "repository"
    files = [repository / path for path in download["verified_files"] if "train-" in path]
    if not files or not all(path.is_file() for path in files):
        raise FileNotFoundError("DomainNet training shards are unavailable")
    return files


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


def _worker_initializer(
    onnx_path: str,
    backbone_text: str,
    preprocessing_text: str,
    intra_op_threads: int,
) -> None:
    """Build one ONNX session per worker process.

    Each worker is restricted to a single intra-op thread; parallelism comes
    from running many workers, which scales better than one wide session at
    batch size one.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    _WORKER_STATE["session"] = ort.InferenceSession(
        onnx_path, options, providers=["CPUExecutionProvider"]
    )
    _WORKER_STATE["onnx_path"] = onnx_path
    _WORKER_STATE["backbone"] = json.loads(backbone_text)
    _WORKER_STATE["preprocessing"] = json.loads(preprocessing_text)


def _worker_extract(task: tuple[int, bytes]) -> tuple[int, np.ndarray]:
    """Decode and extract a single image, independently of every other image."""
    position, image_bytes = task
    backbone = _WORKER_STATE["backbone"]
    with Image.open(BytesIO(image_bytes)) as image:
        array = np.asarray(image.convert("RGB"))
    features = extract_features_batch(
        [array],
        backbone["id"],
        _WORKER_STATE["onnx_path"],
        _WORKER_STATE["preprocessing"],
        backbone["token_pooling_policy"],
        batch_size=1,
        session=_WORKER_STATE["session"],
    )
    return position, features[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Selection and extraction
# ---------------------------------------------------------------------------


def _image_dimensions(image_bytes: bytes) -> tuple[int, int]:
    """Read an image's dimensions from its header without decoding pixels."""
    with Image.open(BytesIO(image_bytes)) as image:
        return int(image.width), int(image.height)


def select_and_extract(
    source_files: list[Path],
    *,
    classes: np.ndarray,
    samples_per_class: int,
    minimum_short_edge: int,
    backbone: dict[str, Any],
    preprocessing: dict[str, Any],
    onnx_path: Path,
    dispatch_chunk_size: int,
    worker_count: int,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Select images by deterministic prefix scan and extract them in parallel.

    The reader is sequential and therefore fully deterministic: it walks the
    hash-locked shards in order and accepts the first ``samples_per_class``
    images of each class that clear the short-edge filter. Only the extraction
    is parallel, and because extraction is per-image the worker count cannot
    influence the result.
    """
    target = {int(label): samples_per_class for label in classes}
    accepted_counts: dict[int, int] = {label: 0 for label in target}
    ordered_metadata: list[dict[str, Any]] = []
    feature_slots: list[np.ndarray | None] = []

    pending: list[tuple[int, bytes]] = []
    extracted_total = 0
    started = time.perf_counter()

    executor = ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_worker_initializer,
        initargs=(
            str(onnx_path),
            json.dumps(backbone),
            json.dumps(preprocessing),
            1,
        ),
    )

    def drain() -> None:
        nonlocal extracted_total
        if not pending:
            return
        for position, features in executor.map(_worker_extract, pending, chunksize=4):
            feature_slots[position] = features
        extracted_total += len(pending)
        pending.clear()
        if progress:
            elapsed = time.perf_counter() - started
            complete = sum(1 for label in target if accepted_counts[label] >= target[label])
            remaining = len(target) * samples_per_class - extracted_total
            print(
                f"  extracted {extracted_total:6d} | "
                f"{extracted_total / elapsed:6.1f} img/s | "
                f"classes filled {complete:3d}/{len(target)} | "
                f"eta {remaining / max(extracted_total / elapsed, 1e-9) / 60:5.1f} min",
                flush=True,
            )

    try:
        for source_path in source_files:
            source = parquet.ParquetFile(source_path)
            row_offset = 0
            for batch in source.iter_batches(
                batch_size=256, columns=["image", "label", "domain", "image_path"]
            ):
                for local_index, row in enumerate(batch.to_pylist()):
                    class_label = int(row["label"])
                    if (
                        class_label not in target
                        or accepted_counts[class_label] >= target[class_label]
                    ):
                        continue
                    image_bytes = row["image"]["bytes"]
                    width, height = _image_dimensions(image_bytes)
                    if min(width, height) < minimum_short_edge:
                        continue
                    image_path = row["image_path"]
                    ordered_metadata.append(
                        {
                            "source_file": source_path.name,
                            "source_row": int(row_offset + local_index),
                            "class_label": class_label,
                            "domain": int(row["domain"]),
                            "image_path": image_path.decode("utf-8")
                            if isinstance(image_path, bytes)
                            else str(image_path),
                            "native_width": width,
                            "native_height": height,
                        }
                    )
                    pending.append((len(feature_slots), image_bytes))
                    feature_slots.append(None)
                    accepted_counts[class_label] += 1
                    if len(pending) >= dispatch_chunk_size:
                        drain()
                row_offset += len(batch)
                if all(accepted_counts[label] >= target[label] for label in target):
                    break
            if all(accepted_counts[label] >= target[label] for label in target):
                break
        drain()
    finally:
        executor.shutdown()

    short = {label: count for label, count in accepted_counts.items() if count < target[label]}
    if short:
        raise ValueError(f"DomainNet classes lack sufficient samples: {short}")
    if any(slot is None for slot in feature_slots):
        raise RuntimeError("Extraction left unfilled feature slots")

    scan_features = np.stack([slot for slot in feature_slots if slot is not None], axis=0)
    scan_labels = np.asarray(
        [entry["class_label"] for entry in ordered_metadata], dtype=np.int64
    )

    # Re-order from scan order into class-major order, matching the M70 layout.
    order = np.concatenate(
        [np.flatnonzero(scan_labels == int(label)) for label in classes]
    )
    manifest = [ordered_metadata[int(index)] for index in order]
    return scan_features[order], scan_labels[order], manifest


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def shard_invariance_control(
    source_files: list[Path],
    *,
    config: dict[str, Any],
    backbone: dict[str, Any],
    preprocessing: dict[str, Any],
    onnx_path: Path,
) -> dict[str, Any]:
    """Extract a probe set under different shardings and require identity.

    This is the positive control for the extraction operand. It passes only if
    features are genuinely per-image; had extraction kept the batch-32 layout,
    changing the worker count or the dispatch chunk size would change the
    numbers.
    """
    specification = config["shard_invariance_control"]
    probe_images = int(specification["probe_images"])
    probe_classes = np.arange(4)
    per_class = probe_images // len(probe_classes)

    runs: list[np.ndarray] = []
    settings = [
        (int(worker_count), chunk)
        for worker_count, chunk in zip(
            specification["worker_counts"],
            (probe_images, max(8, probe_images // 7)),
            strict=False,
        )
    ]
    for worker_count, chunk_size in settings:
        features, _, _ = select_and_extract(
            source_files,
            classes=probe_classes,
            samples_per_class=per_class,
            minimum_short_edge=int(config["minimum_native_short_edge"]),
            backbone=backbone,
            preprocessing=preprocessing,
            onnx_path=onnx_path,
            dispatch_chunk_size=chunk_size,
            worker_count=worker_count,
            progress=False,
        )
        runs.append(features)

    difference = float(np.abs(runs[0] - runs[1]).max())
    tolerance = float(specification["maximum_absolute_difference"])
    return {
        "probe_images": int(runs[0].shape[0]),
        "settings": [
            {"worker_count": worker_count, "dispatch_chunk_size": chunk_size}
            for worker_count, chunk_size in settings
        ],
        "maximum_absolute_difference": difference,
        "tolerance": tolerance,
        "passed": difference <= tolerance,
    }


def m70_divergence(
    features: np.ndarray, labels: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    """Quantify how far the sealed M70 corpus sits from per-image extraction.

    Reported, never gated. M70 was extracted at batch size 32, so its features
    carry a batch-composition perturbation that this corpus deliberately does
    not reproduce.
    """
    specification = config["m70_divergence_report"]
    index_path = _verify(specification["m70_index"])
    reference_directory = index_path.parent
    reference_features = np.load(
        reference_directory / "arrays" / "features.npy", allow_pickle=False
    )
    reference_labels = np.load(
        reference_directory / "arrays" / "labels.npy", allow_pickle=False
    )
    prefix_count = int(specification["samples_per_class"])

    absolute: list[float] = []
    relative: list[float] = []
    candidate_classes = {int(value) for value in np.unique(labels)}
    for class_label in np.unique(reference_labels):
        if int(class_label) not in candidate_classes:
            continue
        left = reference_features[reference_labels == class_label][:prefix_count]
        right = features[labels == class_label][:prefix_count]
        if left.shape != right.shape:
            continue
        absolute.append(float(np.abs(left - right).max()))
        relative.append(
            float(np.linalg.norm(left - right, axis=1).mean() / np.linalg.norm(left, axis=1).mean())
        )

    return {
        "compared_classes": len(absolute),
        "prefix_samples_per_class": prefix_count,
        "maximum_absolute_difference": max(absolute) if absolute else None,
        "mean_relative_difference": round(float(np.mean(relative)), 6) if relative else None,
        "interpretation": (
            "Non-zero by construction. M70 was extracted at batch size 32, where "
            "DynamicQuantizeLinear makes each feature depend on its batch neighbours."
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_corpus(
    config_path: Path,
    output_dir: Path,
    *,
    workers: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    download = json.loads(_verify(config["domainnet_download_record"]).read_text())
    source_files = _source_files(download)

    backbone = config["backbone"]
    onnx_path = _resolve(backbone["onnx_path"])
    preprocessor_path = _resolve(backbone["preprocessor_path"])
    if sha256_file(onnx_path) != backbone["onnx_sha256"]:
        raise ValueError("DINOv2 weight hash mismatch")
    if sha256_file(preprocessor_path) != backbone["preprocessor_sha256"]:
        raise ValueError("DINOv2 preprocessing hash mismatch")
    preprocessing = json.loads(preprocessor_path.read_text(encoding="utf-8"))

    worker_count = workers or max(1, (os.cpu_count() or 2) - 2)

    if progress:
        print("Running shard-invariance control...", flush=True)
    control = shard_invariance_control(
        source_files,
        config=config,
        backbone=backbone,
        preprocessing=preprocessing,
        onnx_path=onnx_path,
    )
    if not control["passed"]:
        raise RuntimeError(
            "Shard-invariance control failed; extraction is not a per-image "
            f"function: {control}"
        )
    if progress:
        print(f"  control passed (difference {control['maximum_absolute_difference']})", flush=True)
        print(f"Extracting with {worker_count} workers...", flush=True)

    classes = np.arange(int(config["class_count"]))
    samples_per_class = int(config["samples_per_class"])

    started = time.perf_counter()
    features, labels, selection = select_and_extract(
        source_files,
        classes=classes,
        samples_per_class=samples_per_class,
        minimum_short_edge=int(config["minimum_native_short_edge"]),
        backbone=backbone,
        preprocessing=preprocessing,
        onnx_path=onnx_path,
        dispatch_chunk_size=int(config["dispatch_chunk_size"]),
        worker_count=worker_count,
        progress=progress,
    )
    extraction_seconds = time.perf_counter() - started

    expected_shape = (len(classes) * samples_per_class, int(backbone["output_dimension"]))
    if features.shape != expected_shape:
        raise RuntimeError(f"Extraction returned {features.shape}, expected {expected_shape}")

    features = features.astype(np.float32)
    divergence = m70_divergence(features, labels, config)

    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    features_path = arrays_dir / "features.npy"
    labels_path = arrays_dir / "labels.npy"
    np.save(features_path, features)
    np.save(labels_path, labels)

    domain_counts: dict[str, int] = {}
    for entry in selection:
        key = str(entry["domain"])
        domain_counts[key] = domain_counts.get(key, 0) + 1

    geometry_per_class = samples_per_class - 40
    evidence = {
        "schema_version": 2,
        "artifact": config["artifact"],
        "configuration_hash": payload_hash(config),
        "class_count": len(classes),
        "samples_per_class": samples_per_class,
        "total_samples": int(features.shape[0]),
        "output_dimension": int(features.shape[1]),
        "domain_counts": domain_counts,
        "extraction_batch_size": int(config["extraction_batch_size"]),
        "worker_count": worker_count,
        "extraction_seconds": round(extraction_seconds, 1),
        "images_per_second": round(features.shape[0] / extraction_seconds, 1),
        "shard_invariance_control": control,
        "m70_divergence_report": divergence,
        "geometry_budget": {
            "calibration_per_class": 20,
            "evaluation_per_class": 20,
            "geometry_per_class": geometry_per_class,
            "samples_per_fitted_dimension_at_rank_32": round(geometry_per_class / 32, 2),
            "samples_per_fitted_dimension_at_rank_60": round(geometry_per_class / 60, 2),
            "registered_floor": 10.0,
        },
        "final_labels_opened": False,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)

    manifest = dict(evidence)
    manifest["features_sha256"] = sha256_file(features_path)
    manifest["labels_sha256"] = sha256_file(labels_path)
    manifest["selection"] = selection
    write_canonical_json(output_dir / "selection_manifest.json", manifest)

    build_artifact_index(output_dir)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    evidence = build_corpus(
        arguments.config,
        arguments.output,
        workers=arguments.workers,
        progress=not arguments.quiet,
    )
    evidence.pop("domain_counts", None)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
