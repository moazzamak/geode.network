"""M85: the transfer corpus and the resolution control it cannot be read without.

Two sealed artifacts, both extracted through the same frozen DINOv2-small INT8
graph as the v13 corpus, at batch size one, on the CPU provider:

* **cifar100** — the registered transfer corpus, 20 coarse superclasses.
* **degraded** — the v13 corpus's own held-out evaluation rows, the identical
  images, put through the identical graph at CIFAR-100's 32x32 resolution.

The second exists because of N85.2. Degrading corpus images to 32x32 costs 41 %
of nearest-class-mean accuracy and displaces a row further than the distance to
its own class mean, so a CIFAR-100 number read alone would be a corpus effect
and a resolution effect superimposed with no way to attribute either.

Before either artifact is written the builder re-extracts a sample of the
corpus's own evaluation rows *natively* and requires exact agreement with the
sealed corpus (N85.3). If this instrument is not the instrument that built the
corpus, nothing it produces is comparable to v13 and the run stops.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
from PIL import Image
import pyarrow.parquet as parquet

from experiments.common.v5_artifacts import (
    build_artifact_index,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v13_boundary import (
    domain_matched_partition,
    domain_stratified_halves,
)
from experiments.tier4.prepare_v5_frozen_features import extract_features_batch
from experiments.tier4.eval_v13_m80_sparse_dictionary import (
    _load_corpus,
    _verify_corpus,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v13" / "m85_transfer.json"

_WORKER_STATE: dict[str, Any] = {}


def _resolve(path: str) -> Path:
    return (REPO_ROOT / Path(path)).resolve()


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


def _worker_initializer(
    onnx_path: str,
    backbone_text: str,
    preprocessing_text: str,
    intra_op_threads: int,
) -> None:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    _WORKER_STATE["session"] = ort.InferenceSession(
        onnx_path, options, providers=["CPUExecutionProvider"]
    )
    _WORKER_STATE["onnx_path"] = onnx_path
    _WORKER_STATE["backbone"] = json.loads(backbone_text)
    _WORKER_STATE["preprocessing"] = json.loads(preprocessing_text)


def degrade(image: Image.Image, size: int) -> Image.Image:
    """Convert to RGB first, then resample, so every source degrades alike.

    Order matters. Resampling a palette image before conversion would go
    through nearest-neighbour on indices and produce a different picture from
    the one the confound was measured on.
    """
    return image.convert("RGB").resize((size, size), resample=Image.Resampling.BILINEAR)


def _worker_extract(task: tuple[int, Any, int]) -> tuple[int, np.ndarray]:
    """Extract one image. ``size`` of zero means leave the resolution alone."""
    position, payload, size = task
    if isinstance(payload, bytes):
        with Image.open(BytesIO(payload)) as image:
            prepared = degrade(image, size) if size else image.convert("RGB")
            array = np.asarray(prepared)
    else:
        array = np.asarray(payload)
    backbone = _WORKER_STATE["backbone"]
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
# Extraction driver
# ---------------------------------------------------------------------------


def _extract(
    tasks: list[tuple[int, Any, int]],
    *,
    config: dict[str, Any],
    backbone: dict[str, Any],
    preprocessing: dict[str, Any],
    onnx_path: Path,
    worker_count: int,
    chunk_size: int,
    dimension: int,
    label: str,
) -> np.ndarray:
    features = np.zeros((len(tasks), dimension), dtype=np.float32)
    started = time.time()
    done = 0
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_worker_initializer,
        initargs=(
            str(onnx_path),
            json.dumps(backbone),
            json.dumps(preprocessing),
            1,
        ),
    ) as pool:
        for position, vector in pool.map(_worker_extract, tasks, chunksize=8):
            features[position] = vector
            done += 1
            if label and done % chunk_size == 0:
                rate = done / max(time.time() - started, 1e-9)
                print(f"    {label} {done}/{len(tasks)} at {rate:.1f} img/s")
    return features


def _read_rows(
    parquet_root: Path, records: list[dict[str, Any]]
) -> list[bytes]:
    """Pull the exact source rows the corpus manifest names, one pass per shard.

    Random access into a 5 GB parquet file is not affordable, so each shard is
    scanned once in row order and the wanted rows are picked out as they go by.
    """
    wanted: dict[str, dict[int, int]] = defaultdict(dict)
    for position, record in enumerate(records):
        wanted[record["source_file"]][int(record["source_row"])] = position

    payloads: list[bytes | None] = [None] * len(records)
    for shard_name, rows in wanted.items():
        shard = parquet_root / shard_name
        offset = 0
        reader = parquet.ParquetFile(shard)
        for batch in reader.iter_batches(batch_size=256, columns=["image"]):
            column = batch.column("image").to_pylist()
            for index, value in enumerate(column):
                position = rows.get(offset + index)
                if position is not None:
                    payloads[position] = value["bytes"]
            offset += len(column)
            if all(payloads[position] is not None for position in rows.values()):
                break
    missing = [index for index, value in enumerate(payloads) if value is None]
    if missing:
        raise ValueError(f"{len(missing)} manifest rows were not found in the shards")
    return [value for value in payloads if value is not None]


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def _reproduction_control(
    payloads: list[bytes],
    sealed: np.ndarray,
    *,
    config: dict[str, Any],
    backbone: dict[str, Any],
    preprocessing: dict[str, Any],
    onnx_path: Path,
) -> dict[str, Any]:
    """N85.3, gating. Re-extract corpus rows natively and demand exact equality."""
    tasks = [(index, payload, 0) for index, payload in enumerate(payloads)]
    fresh = _extract(
        tasks,
        config=config,
        backbone=backbone,
        preprocessing=preprocessing,
        onnx_path=onnx_path,
        worker_count=int(config["worker_count"]),
        chunk_size=int(config["dispatch_chunk_size"]),
        dimension=int(backbone["output_dimension"]),
        label="",
    )
    difference = float(np.abs(fresh - sealed).max())
    tolerance = float(config["reproduction_control"]["maximum_absolute_difference"])
    return {
        "sample_rows": len(payloads),
        "maximum_absolute_difference": difference,
        "tolerance": tolerance,
        "passes": bool(difference <= tolerance),
        "rationale": config["reproduction_control"]["description"],
    }


def _shard_invariance_control(
    payloads: list[bytes],
    *,
    size: int,
    config: dict[str, Any],
    backbone: dict[str, Any],
    preprocessing: dict[str, Any],
    onnx_path: Path,
) -> dict[str, Any]:
    specification = config["shard_invariance_control"]
    runs = []
    for worker_count in specification["worker_counts"]:
        tasks = [(index, payload, size) for index, payload in enumerate(payloads)]
        runs.append(
            _extract(
                tasks,
                config=config,
                backbone=backbone,
                preprocessing=preprocessing,
                onnx_path=onnx_path,
                worker_count=int(worker_count),
                chunk_size=int(config["dispatch_chunk_size"]),
                dimension=int(backbone["output_dimension"]),
                label="",
            )
        )
    difference = float(np.abs(runs[0] - runs[1]).max())
    tolerance = float(specification["maximum_absolute_difference"])
    return {
        "probe_images": len(payloads),
        "worker_counts": [int(value) for value in specification["worker_counts"]],
        "maximum_absolute_difference": difference,
        "tolerance": tolerance,
        "passes": bool(difference <= tolerance),
    }


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    started = time.time()

    backbone = config["backbone"]
    onnx_path = _resolve(backbone["onnx_path"])
    preprocessor_path = _resolve(backbone["preprocessor_path"])
    for path, expected in (
        (onnx_path, backbone["onnx_sha256"]),
        (preprocessor_path, backbone["preprocessor_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen backbone artifact hash mismatch: {path}")
    preprocessing = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    dimension = int(backbone["output_dimension"])

    # ---- the corpus's own evaluation rows -------------------------------
    corpus_index = _verify_corpus(config["corpus"])
    corpus_features, corpus_labels = _load_corpus(corpus_index)
    manifest = json.loads(
        (corpus_index.parent / "selection_manifest.json").read_text(encoding="utf-8")
    )
    records = manifest["selection"]
    corpus_domains = np.asarray(
        [int(record["domain"]) for record in records], dtype=np.int64
    )
    partition = config["partition"]
    _, evaluation_rows, partition_report = domain_matched_partition(
        corpus_labels,
        corpus_domains,
        quota=tuple(int(value) for value in partition["evaluation_domain_quota"]),
        fit_per_class=int(partition["fit_per_class"]),
        domain_count=int(config["corpus"]["domain_count"]),
    )
    calibration_rows, report_rows = domain_stratified_halves(
        corpus_labels, corpus_domains, evaluation_rows
    )
    print(
        f"corpus evaluation rows {len(evaluation_rows)} "
        f"({len(calibration_rows)} calibration, {len(report_rows)} report)"
    )

    parquet_root = Path(config["parquet_root"])
    print("reading source rows from the shards ...")
    payloads = _read_rows(parquet_root, [records[int(row)] for row in evaluation_rows])

    # ---- N85.3, before anything is written ------------------------------
    generator = np.random.default_rng(int(config["reproduction_control"]["sample_seed"]))
    sample = generator.choice(
        len(evaluation_rows),
        size=int(config["reproduction_control"]["sample_rows"]),
        replace=False,
    )
    print("N85.3 reproduction control ...")
    reproduction = _reproduction_control(
        [payloads[int(index)] for index in sample],
        corpus_features[evaluation_rows[sample]],
        config=config,
        backbone=backbone,
        preprocessing=preprocessing,
        onnx_path=onnx_path,
    )
    print(
        f"  max |diff| {reproduction['maximum_absolute_difference']:.3e} "
        f"-> {'reproduced' if reproduction['passes'] else 'MISMATCH'}"
    )
    if not reproduction["passes"]:
        raise ValueError(
            "N85.3 failed: this builder does not reproduce the sealed corpus, "
            "so nothing it extracts is comparable to v13"
        )

    size = int(config["degradation"]["size"])
    invariance = _shard_invariance_control(
        payloads[: int(config["shard_invariance_control"]["probe_images"])],
        size=size,
        config=config,
        backbone=backbone,
        preprocessing=preprocessing,
        onnx_path=onnx_path,
    )
    print(f"  shard invariance max |diff| {invariance['maximum_absolute_difference']:.3e}")

    # ---- the degraded artifact -----------------------------------------
    print(f"extracting {len(payloads)} evaluation rows at {size}x{size} ...")
    degraded_features = _extract(
        [(index, payload, size) for index, payload in enumerate(payloads)],
        config=config,
        backbone=backbone,
        preprocessing=preprocessing,
        onnx_path=onnx_path,
        worker_count=int(config["worker_count"]),
        chunk_size=int(config["dispatch_chunk_size"]),
        dimension=dimension,
        label="degraded",
    )

    # ---- the CIFAR-100 artifact ----------------------------------------
    cifar = config["cifar100"]
    source = np.load(_resolve(cifar["source_path"]))
    images = source[cifar["images_key"]]
    coarse = source[cifar["labels_key"]].astype(np.int64)
    per_class = int(cifar["samples_per_class"])
    rng = np.random.default_rng(int(cifar["sample_seed"]))
    chosen: list[int] = []
    for label in range(int(cifar["class_count"])):
        available = np.flatnonzero(coarse == label)
        if len(available) < per_class:
            raise ValueError(
                f"CIFAR-100 superclass {label} holds {len(available)} images, "
                f"below the registered {per_class}"
            )
        chosen.extend(rng.choice(available, size=per_class, replace=False).tolist())
    chosen_array = np.asarray(sorted(chosen), dtype=np.int64)
    print(f"extracting {len(chosen_array)} CIFAR-100 images ...")
    cifar_features = _extract(
        [
            (position, images[int(row)], 0)
            for position, row in enumerate(chosen_array)
        ],
        config=config,
        backbone=backbone,
        preprocessing=preprocessing,
        onnx_path=onnx_path,
        worker_count=int(config["worker_count"]),
        chunk_size=int(config["dispatch_chunk_size"]),
        dimension=dimension,
        label="cifar100",
    )

    # ---- write -----------------------------------------------------------
    output_dir = _resolve(config["output_dir"])
    arrays = output_dir / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    np.save(arrays / "degraded_features.npy", degraded_features)
    np.save(arrays / "degraded_labels.npy", corpus_labels[evaluation_rows])
    np.save(arrays / "degraded_domains.npy", corpus_domains[evaluation_rows])
    np.save(arrays / "degraded_corpus_rows.npy", evaluation_rows)
    np.save(arrays / "cifar100_features.npy", cifar_features)
    np.save(arrays / "cifar100_labels.npy", coarse[chosen_array])
    np.save(arrays / "cifar100_source_rows.npy", chosen_array)

    evidence: dict[str, Any] = {
        "milestone": "M85",
        "component": "transfer_artifacts",
        "generated_at": datetime.now(UTC).isoformat(),
        "purpose": config["purpose"],
        "registration_notes": config["registration_notes"],
        "corpus": config["corpus"],
        "partition": partition_report,
        "degradation": config["degradation"],
        "cifar100": config["cifar100"],
        "backbone": backbone,
        "minimum_native_short_edge_waiver": config[
            "minimum_native_short_edge_waiver"
        ],
        "degraded_rows": int(degraded_features.shape[0]),
        "cifar100_rows": int(cifar_features.shape[0]),
        "reproduction_control": reproduction,
        "shard_invariance_control": invariance,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "runtime_seconds": None,
    }
    evidence["feature_hash"] = hashlib.sha256(
        degraded_features.tobytes() + cifar_features.tobytes()
    ).hexdigest()
    evidence["runtime_seconds"] = round(time.time() - started, 2)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)

    print(f"\ndegraded rows   {evidence['degraded_rows']}")
    print(f"cifar100 rows   {evidence['cifar100_rows']}")
    print(f"feature_hash    {evidence['feature_hash']}")
    print(f"runtime         {evidence['runtime_seconds'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
