"""Download, convert, and verify the pinned ModelNet40 point-cloud source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.common.data_cache import configure_external_cache_environment
from src.runtime.modelnet_manifest import ModelNet40Manifest, ModelNetFile


REPOSITORY = "jxie/modelnet40-2048"
REVISION = "5ae9a4a64ed57c5340517862ab0b0bf5d8a99831"
SOURCES = {
    "train": ModelNetFile(
        "repository/data/train-00000-of-00001-4fece5076596e98a.parquet",
        "3fb005b7862b134deeef42ba65f193e6ec2255d787dd60143410824e1406d027",
        237_093_619,
    ),
    "test": ModelNetFile(
        "repository/data/test-00000-of-00001-baa2ae7c6a5df7e0.parquet",
        "ae364f886bf1032df15b81f9829c0c20031c5e5e259585b721e2969ed04d6a1d",
        59_313_912,
    ),
}
SPLIT_SAMPLES = {"train": 9_840, "test": 2_468}
ARTIFACT_PATH = "modelnet40_2048.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source(root: Path, item: ModelNetFile) -> None:
    path = root / item.path
    if path.stat().st_size != item.size:
        raise ValueError(f"ModelNet40 file size mismatch: {item.path}")
    if _sha256(path) != item.sha256:
        raise ValueError(f"ModelNet40 file hash mismatch: {item.path}")


def _convert(root: Path, output: Path) -> None:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportError("pyarrow is required to prepare ModelNet40") from error
    total = sum(SPLIT_SAMPLES.values())
    pointclouds = np.empty((total, 2048, 3), dtype=np.float32)
    labels = np.empty(total, dtype=np.int32)
    splits = np.empty(total, dtype=np.uint8)
    offset = 0
    for split_id, split_name in enumerate(("train", "test")):
        path = root / SOURCES[split_name].path
        split_start = offset
        source = parquet.ParquetFile(path)
        for batch in source.iter_batches(batch_size=32, columns=["inputs", "label"]):
            batch_points = np.asarray(batch.column(0).to_pylist(), dtype=np.float32)
            batch_labels = np.asarray(batch.column(1), dtype=np.int32)
            if batch_points.ndim != 3 or batch_points.shape[1:] != (2048, 3):
                raise ValueError("ModelNet40 source points must have shape (N, 2048, 3)")
            end = offset + len(batch_points)
            pointclouds[offset:end] = batch_points
            labels[offset:end] = batch_labels
            splits[offset:end] = split_id
            offset = end
        if offset - split_start != SPLIT_SAMPLES[split_name]:
            raise ValueError(f"ModelNet40 {split_name} row count mismatch")
    if offset != total or not np.isfinite(pointclouds).all():
        raise ValueError("ModelNet40 conversion produced invalid points")
    if not np.array_equal(np.unique(labels), np.arange(40, dtype=np.int32)):
        raise ValueError("ModelNet40 conversion requires labels 0 through 39")
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary, pointclouds=pointclouds, labels=labels, splits=splits,
    )
    temporary.replace(output)


def prepare_modelnet40(cache_root: Path | None = None) -> dict:
    configured_root = configure_external_cache_environment()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError("huggingface_hub is required for ModelNet40") from error
    root = (cache_root.resolve() if cache_root is not None else configured_root) / "modelnet40"
    repository_root = root / "repository"
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPOSITORY,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=[
            item.path.removeprefix("repository/") for item in SOURCES.values()
        ],
        local_dir=repository_root,
    )
    for item in SOURCES.values():
        _verify_source(root, item)
    artifact_path = root / ARTIFACT_PATH
    _convert(root, artifact_path)
    artifact = ModelNetFile(
        ARTIFACT_PATH, _sha256(artifact_path), artifact_path.stat().st_size,
    )
    manifest = ModelNet40Manifest(
        source_repository=REPOSITORY,
        source_revision=REVISION,
        source_files=tuple(SOURCES.values()),
        artifact=artifact,
        split_samples=tuple(SPLIT_SAMPLES.items()),
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"cache_root": str(configured_root), **manifest.verify(root)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = prepare_modelnet40(arguments.cache_root)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8", newline="\n")
    print(serialized, end="")


if __name__ == "__main__":
    main()