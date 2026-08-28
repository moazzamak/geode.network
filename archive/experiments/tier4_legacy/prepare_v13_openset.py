"""Seal the v13 out-of-set feature artifact for M83 and M84.

What this builds
----------------
DINOv2-small features for the 217 DomainNet classes that the v13 corpus does
not contain, drawn so that the sample's **domain** distribution matches the
corpus's own. Registered by N83.2: the corpus is 60.76% quickdraw, and an
out-of-set sample drawn by availability would be roughly 29% quickdraw and 29%
real. A boundary rejecting that sample could be detecting rendering style
rather than novelty, which is not a hypothetical concern — M82 showed a channel
over this corpus takes a style name for 82% of atoms when style is the easier
signal.

Why the corpus is skewed at all
-------------------------------
The shards are ordered by domain, and the corpus builder filled each class by a
prefix scan. The scan therefore drained clipart, infograph and painting before
reaching quickdraw, and only reached ``real`` and ``sketch`` for classes that
ran short earlier. The corpus's 60.76% quickdraw is a selection-order artifact
rather than a property of DomainNet, whose known-class rows are 31% quickdraw
and 31% real. This builder allocates an explicit per-cell quota instead, so the
match to the corpus is deliberate rather than incidental.

What this deliberately does **not** build
-----------------------------------------
N83.3's known-class control — novel images of *known* classes, which a novelty
detector must accept. That control needs no extraction: the corpus's own 64-row
per-class held-out evaluation partition is never fitted by Phase A, so it is
already novel to the geometry and already sealed.

Usage::

    .\\.venv\\Scripts\\python.exe -m experiments.tier4.prepare_v13_openset
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import json
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
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v13" / "m83_openset.json"
DOMAIN_COUNT = 6

_WORKER_STATE: dict[str, Any] = {}


def _resolve(path: str) -> Path:
    resolved = (REPO_ROOT / Path(path)).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError("v13 artifact paths must remain inside the repository")
    return resolved


def _verify(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"immutable artifact hash mismatch: {path}")


def largest_remainder_quota(counts: list[int], total: int) -> list[int]:
    """Allocate ``total`` samples across cells in proportion to ``counts``.

    Deterministic and tie-broken by cell order, so the quota is a function of
    the sealed domain counts alone. Kept separate from the builder so the
    configuration's stated quota can be checked against it before any image is
    read.
    """
    grand = sum(counts)
    if grand <= 0:
        raise ValueError("domain counts must be positive")
    exact = [count * total / grand for count in counts]
    quota = [int(value) for value in exact]
    remaining = total - sum(quota)
    order = sorted(
        range(len(counts)), key=lambda index: (-(exact[index] - quota[index]), index)
    )
    for index in order[:remaining]:
        quota[index] += 1
    return quota


# ---------------------------------------------------------------------------
# Worker side — identical in behaviour to the corpus builder's worker
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


def _worker_extract(task: tuple[int, bytes]) -> tuple[int, np.ndarray]:
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


def _image_dimensions(image_bytes: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(image_bytes)) as image:
        return int(image.width), int(image.height)


# ---------------------------------------------------------------------------
# Selection and extraction
# ---------------------------------------------------------------------------


def select_and_extract(
    source_files: list[Path],
    *,
    labels: list[int],
    quota: list[int],
    minimum_short_edge: int,
    backbone: dict[str, Any],
    preprocessing: dict[str, Any],
    onnx_path: Path,
    dispatch_chunk_size: int,
    worker_count: int,
    progress: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    """Fill one ``(class, domain)`` quota cell at a time by prefix scan.

    The reader is sequential and accepts the first images in shard order that
    clear the short-edge filter, so selection is a pure function of the shards.
    Only extraction is parallel, and because extraction runs at batch size one
    the worker count cannot influence the values.
    """
    wanted = {
        (label, domain): quota[domain]
        for label in labels
        for domain in range(DOMAIN_COUNT)
        if quota[domain] > 0
    }
    accepted: dict[tuple[int, int], int] = dict.fromkeys(wanted, 0)
    target_total = sum(wanted.values())
    unfilled = len(wanted)

    metadata: list[dict[str, Any]] = []
    slots: list[np.ndarray | None] = []
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
            slots[position] = features
        extracted_total += len(pending)
        pending.clear()
        if progress:
            elapsed = time.perf_counter() - started
            rate = extracted_total / max(elapsed, 1e-9)
            filled = sum(1 for cell in wanted if accepted[cell] >= wanted[cell])
            print(
                f"  extracted {extracted_total:6d}/{target_total} | "
                f"{rate:6.1f} img/s | cells {filled:4d}/{len(wanted)} | "
                f"eta {(target_total - extracted_total) / max(rate, 1e-9) / 60:5.1f} min",
                flush=True,
            )

    try:
        for source_path in source_files:
            if unfilled == 0:
                break
            source = parquet.ParquetFile(source_path)
            row_offset = 0
            for batch in source.iter_batches(
                batch_size=256, columns=["image", "label", "domain", "image_path"]
            ):
                for local_index, row in enumerate(batch.to_pylist()):
                    cell = (int(row["label"]), int(row["domain"]))
                    if cell not in wanted or accepted[cell] >= wanted[cell]:
                        continue
                    image_bytes = row["image"]["bytes"]
                    width, height = _image_dimensions(image_bytes)
                    if min(width, height) < minimum_short_edge:
                        continue
                    image_path = row["image_path"]
                    metadata.append(
                        {
                            "source_file": source_path.name,
                            "source_row": int(row_offset + local_index),
                            "class_label": cell[0],
                            "domain": cell[1],
                            "image_path": image_path.decode("utf-8")
                            if isinstance(image_path, bytes)
                            else str(image_path),
                            "native_width": width,
                            "native_height": height,
                        }
                    )
                    pending.append((len(slots), image_bytes))
                    slots.append(None)
                    accepted[cell] += 1
                    if accepted[cell] >= wanted[cell]:
                        unfilled -= 1
                    if len(pending) >= dispatch_chunk_size:
                        drain()
                row_offset += len(batch)
                if unfilled == 0:
                    break
        drain()
    finally:
        executor.shutdown()

    if any(slot is None for slot in slots):
        raise RuntimeError("extraction left unfilled feature slots")

    shortfall = {
        f"{label}:{domain}": wanted[(label, domain)] - accepted[(label, domain)]
        for (label, domain) in wanted
        if accepted[(label, domain)] < wanted[(label, domain)]
    }

    scan_features = np.stack([slot for slot in slots if slot is not None], axis=0)
    scan_labels = np.asarray(
        [entry["class_label"] for entry in metadata], dtype=np.int64
    )
    order = np.concatenate(
        [np.flatnonzero(scan_labels == label) for label in labels]
    )
    ordered = [metadata[int(index)] for index in order]
    return scan_features[order], ordered, shortfall


def shard_invariance_control(
    source_files: list[Path],
    *,
    config: dict[str, Any],
    backbone: dict[str, Any],
    preprocessing: dict[str, Any],
    onnx_path: Path,
) -> dict[str, Any]:
    """Extract a probe set twice under different worker counts and compare.

    Passes only if features are per-image. Had the INT8 graph's batch coupling
    survived, changing the worker count would change the numbers and the
    artifact would be a function of its own scheduling.
    """
    specification = config["shard_invariance_control"]
    probe_images = int(specification["probe_images"])
    probe_labels = list(range(128, 132))
    per_class = probe_images // len(probe_labels)
    quota = largest_remainder_quota(config["corpus"]["domain_counts"], per_class)

    runs: list[np.ndarray] = []
    for worker_count, chunk in zip(
        specification["worker_counts"],
        (probe_images, max(8, probe_images // 7)),
        strict=True,
    ):
        features, _, _ = select_and_extract(
            source_files,
            labels=probe_labels,
            quota=quota,
            minimum_short_edge=int(config["minimum_native_short_edge"]),
            backbone=backbone,
            preprocessing=preprocessing,
            onnx_path=onnx_path,
            dispatch_chunk_size=int(chunk),
            worker_count=int(worker_count),
            progress=False,
        )
        runs.append(features)

    difference = float(np.abs(runs[0] - runs[1]).max())
    tolerance = float(specification["maximum_absolute_difference"])
    return {
        "probe_images": int(runs[0].shape[0]),
        "worker_counts": [int(value) for value in specification["worker_counts"]],
        "maximum_absolute_difference": difference,
        "tolerance": tolerance,
        "passes": bool(difference <= tolerance),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=14)
    arguments = parser.parse_args()

    config_path = Path(arguments.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = _resolve(config["output_dir"])
    started = time.perf_counter()

    corpus_index = _resolve(config["corpus"]["path"])
    _verify(corpus_index, config["corpus"]["sha256"])

    backbone = config["backbone"]
    onnx_path = _resolve(backbone["onnx_path"])
    _verify(onnx_path, backbone["onnx_sha256"])
    preprocessor_path = _resolve(backbone["preprocessor_path"])
    _verify(preprocessor_path, backbone["preprocessor_sha256"])
    preprocessing = json.loads(preprocessor_path.read_text(encoding="utf-8"))

    record_path = _resolve(config["domainnet_download_record"]["path"])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    repository = Path(record["dataset_root"]) / "repository"
    source_files = [
        repository / path for path in record["verified_files"] if "train-" in path
    ]
    if not source_files or not all(path.is_file() for path in source_files):
        raise FileNotFoundError("DomainNet training shards are unavailable")

    # The quota is stated in the sealed configuration and recomputed here. A
    # disagreement means the stratification target drifted from the corpus it
    # claims to match, which must stop the build rather than be reported.
    per_class = int(config["samples_per_unseen_class"])
    domain_counts = [int(value) for value in config["corpus"]["domain_counts"]]
    quota = largest_remainder_quota(domain_counts, per_class)
    declared = [int(value) for value in config["domain_quota_per_unseen_class"]]
    if quota != declared:
        raise ValueError(
            f"declared domain quota {declared} is not the largest-remainder "
            f"allocation {quota} of the corpus distribution"
        )

    unseen = config["unseen_classes"]
    labels = list(range(int(unseen["first_label"]), int(unseen["last_label"]) + 1))
    if len(labels) != int(unseen["count"]):
        raise ValueError("unseen class range disagrees with the declared count")

    print(f"{config['artifact']}: {len(labels)} classes x {per_class} samples")
    print(f"  domain quota per class {quota} (corpus fractions matched)")
    print(f"  target {len(labels) * per_class} images, {arguments.workers} workers")

    features, manifest, shortfall = select_and_extract(
        source_files,
        labels=labels,
        quota=quota,
        minimum_short_edge=int(config["minimum_native_short_edge"]),
        backbone=backbone,
        preprocessing=preprocessing,
        onnx_path=onnx_path,
        dispatch_chunk_size=int(config["dispatch_chunk_size"]),
        worker_count=int(arguments.workers),
    )
    extraction_seconds = time.perf_counter() - started
    class_labels = np.asarray(
        [entry["class_label"] for entry in manifest], dtype=np.int64
    )
    domains = np.asarray([entry["domain"] for entry in manifest], dtype=np.int64)

    underfilled = sorted(
        {int(cell.split(":")[0]) for cell in shortfall}
    )
    complete = np.asarray(
        [label not in set(underfilled) for label in class_labels], dtype=bool
    )

    print(f"  extracted {len(manifest)} rows in {extraction_seconds / 60:.1f} min")
    if underfilled:
        print(
            f"  {len(underfilled)} classes underfilled and excluded from the "
            f"stratified set: {underfilled[:12]}"
        )

    # Disjointness against the corpus, keyed on the parquet row identity.
    corpus_manifest = json.loads(
        (corpus_index.parent / "selection_manifest.json").read_text(encoding="utf-8")
    )
    corpus_rows = {
        (row["source_file"], int(row["source_row"]))
        for row in corpus_manifest["selection"]
    }
    shared = sum(
        1
        for entry in manifest
        if (entry["source_file"], int(entry["source_row"])) in corpus_rows
    )
    disjointness = {
        "shared_rows": int(shared),
        "maximum_shared_rows": int(
            config["corpus_disjointness_control"]["maximum_shared_rows"]
        ),
        "corpus_rows": len(corpus_rows),
        "passes": bool(
            shared <= int(config["corpus_disjointness_control"]["maximum_shared_rows"])
        ),
    }
    print(f"  disjointness control: {shared} shared rows, passes {disjointness['passes']}")

    print("  shard-invariance control...", flush=True)
    invariance = shard_invariance_control(
        source_files,
        config=config,
        backbone=backbone,
        preprocessing=preprocessing,
        onnx_path=onnx_path,
    )
    print(
        f"  shard invariance: max |diff| {invariance['maximum_absolute_difference']:.3e}, "
        f"passes {invariance['passes']}"
    )

    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    np.save(arrays_dir / "features.npy", features.astype(np.float32))
    np.save(arrays_dir / "labels.npy", class_labels)
    np.save(arrays_dir / "domains.npy", domains)
    np.save(arrays_dir / "stratified.npy", complete)

    achieved = [int(np.sum(domains == domain)) for domain in range(DOMAIN_COUNT)]
    corpus_total = sum(domain_counts)
    evidence = {
        "schema_version": 1,
        "artifact": config["artifact"],
        "milestone": "M83",
        "program": "v13",
        "configuration_hash": sha256_file(config_path),
        "corpus_index_sha256": config["corpus"]["sha256"],
        "class_count": len(labels),
        "samples_per_class": per_class,
        "row_count": int(features.shape[0]),
        "dimension": int(features.shape[1]),
        "domain_quota_per_class": quota,
        "domain_counts": achieved,
        "domain_fractions": [
            value / max(int(features.shape[0]), 1) for value in achieved
        ],
        "corpus_domain_fractions": [
            value / corpus_total for value in domain_counts
        ],
        "maximum_domain_fraction_deviation": max(
            abs(
                achieved[index] / max(int(features.shape[0]), 1)
                - domain_counts[index] / corpus_total
            )
            for index in range(DOMAIN_COUNT)
        ),
        "underfilled_classes": underfilled,
        "underfilled_cells": shortfall,
        "stratified_row_count": int(np.sum(complete)),
        "corpus_disjointness_control": disjointness,
        "shard_invariance_control": invariance,
        "extraction_seconds": round(extraction_seconds, 1),
        "images_per_second": round(len(manifest) / max(extraction_seconds, 1e-9), 1),
        "worker_count": int(arguments.workers),
        "final_labels_opened": False,
    }
    evidence["feature_hash"] = payload_hash(
        {
            "features": sha256_file(arrays_dir / "features.npy"),
            "labels": sha256_file(arrays_dir / "labels.npy"),
            "domains": sha256_file(arrays_dir / "domains.npy"),
        }
    )
    write_canonical_json(output_dir / "selection_manifest.json", {"selection": manifest})
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)

    print(f"\nfeature_hash {evidence['feature_hash']}")
    print(
        f"domain fractions {[round(value, 4) for value in evidence['domain_fractions']]}"
    )
    print(
        f"max deviation from corpus {evidence['maximum_domain_fraction_deviation']:.4f}"
    )
    print(f"wrote {output_dir} in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
