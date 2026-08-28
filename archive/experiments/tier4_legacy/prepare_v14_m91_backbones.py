"""V14-M91: re-extract v13's exact rows under three DINOv2 sizes.

Plan v14 section 7.1. The capacity question is only answerable if the backbone
is the only factor that changes, so this builder holds everything else fixed and
then *checks* that it did.

Three things here are checks rather than claims.

**The rows.** v13's selection scan is a deterministic prefix walk over hash-locked
shards and never consults the backbone. Re-extraction therefore re-derives the
selection, and the manifest produced here must equal the sealed v13 manifest
entry for entry (N91.2). A mismatch means the reader changed, not the backbone,
and nothing below it is comparable.

**The reference arm.** dinov2-small is re-extracted by this code path rather than
copied from the sealed corpus, so all three arms are produced identically. Its
arrays must hash to v13's sealed digests (N91.8). If they do not, this code path
is not v13's and no arm can be read against a sealed number.

**The quantisation.** A larger model can be quantised worse than a smaller one, so
a capacity null could be a quantisation artefact. The INT8-versus-fp32 divergence
is measured per backbone on a fixed probe set and reported, never gated (N91.6).

Extraction is CPU-only. That is not a preference: DirectML disagrees with the CPU
provider by 21.7 percent relative on dinov2-base at batch size one, measured
before this file was written (N91.3).
"""

from __future__ import annotations

import argparse
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as parquet
from PIL import Image

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.tier4.prepare_v13_domainnet_large import (
    select_and_extract as select_known,
)
from experiments.tier4.prepare_v13_domainnet_large import (
    shard_invariance_control,
)
from experiments.tier4.prepare_v13_openset import (
    select_and_extract as select_openset,
)
from experiments.tier4.prepare_v5_frozen_features import extract_features_batch

DEFAULT_CONFIG = Path("experiments/configs/v14/m91_backbones.json")
DOMAIN_COUNT = 6


def _resolve(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    return resolved


def _source_files(record_path: Path) -> list[Path]:
    download = json.loads(record_path.read_text(encoding="utf-8"))
    repository = Path(download["dataset_root"]) / "repository"
    files = [
        repository / path for path in download["verified_files"] if "train-" in path
    ]
    if not files or not all(path.is_file() for path in files):
        raise FileNotFoundError("DomainNet training shards are unavailable")
    return files


# ---------------------------------------------------------------------------
# N91.2 -- identical rows, checked
# ---------------------------------------------------------------------------


def manifest_identity(
    produced: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    *,
    fields: list[str],
    maximum_differing: int,
) -> dict[str, Any]:
    """Compare a re-derived selection against v13's sealed one, entry for entry.

    Selection depends on the shards, the class list and the short-edge filter and
    never on the backbone, so this is expected to pass exactly. It is run because
    the alternative is to assume it.
    """
    differing: list[dict[str, Any]] = []
    for position, (left, right) in enumerate(zip(produced, reference, strict=False)):
        mismatched = [field for field in fields if left.get(field) != right.get(field)]
        if mismatched:
            differing.append(
                {
                    "row": position,
                    "fields": mismatched,
                    "produced": {field: left.get(field) for field in mismatched},
                    "reference": {field: right.get(field) for field in mismatched},
                }
            )
        if len(differing) >= 8:
            break
    return {
        "produced_rows": len(produced),
        "reference_rows": len(reference),
        "row_counts_agree": len(produced) == len(reference),
        "compared_fields": list(fields),
        "differing_entries": len(differing),
        "first_differences": differing,
        "maximum_differing_entries": maximum_differing,
        "passes": len(produced) == len(reference) and len(differing) <= maximum_differing,
    }


# ---------------------------------------------------------------------------
# N91.6 -- quantisation divergence, reported never gated
# ---------------------------------------------------------------------------


def probe_images(
    source_files: list[Path], *, count: int, minimum_short_edge: int
) -> list[np.ndarray]:
    """A fixed probe set: the first accepted images in shard order.

    Backbone-independent by construction, so every backbone's divergence is
    measured on the same pixels.
    """
    images: list[np.ndarray] = []
    for source_path in source_files:
        source = parquet.ParquetFile(source_path)
        for batch in source.iter_batches(batch_size=128, columns=["image"]):
            for row in batch.to_pylist():
                image_bytes = row["image"]["bytes"]
                with Image.open(BytesIO(image_bytes)) as image:
                    if min(image.width, image.height) < minimum_short_edge:
                        continue
                    images.append(np.asarray(image.convert("RGB")))
                if len(images) >= count:
                    return images
    return images


def quantisation_divergence(
    images: list[np.ndarray],
    *,
    backbone: dict[str, Any],
    preprocessing: dict[str, Any],
    int8_path: Path,
    float_path: Path,
) -> dict[str, Any]:
    """Relative L2 divergence between the INT8 graph and its fp32 original."""
    quantised = extract_features_batch(
        images,
        backbone["id"],
        str(int8_path),
        preprocessing,
        backbone["token_pooling_policy"],
        batch_size=1,
    )
    exact = extract_features_batch(
        images,
        backbone["id"],
        str(float_path),
        preprocessing,
        backbone["token_pooling_policy"],
        batch_size=1,
    )
    relative = np.linalg.norm(quantised - exact, axis=1) / np.linalg.norm(
        exact, axis=1
    )
    return {
        "probe_images": len(images),
        "float_graph_sha256": sha256_file(float_path),
        "mean_relative_divergence": float(relative.mean()),
        "maximum_relative_divergence": float(relative.max()),
        "gated": False,
        "interpretation": (
            "N91.6. Reported so that a capacity verdict of "
            "capacity_not_demonstrated has a candidate cause on the record. It "
            "cannot rescue a failing arm."
        ),
    }


def _float_graph(backbone: dict[str, Any]) -> Path:
    """Fetch the producer's fp32 graph, caching it beside the INT8 one."""
    local = Path(backbone["onnx_path"]).with_name("model.onnx")
    if not local.is_file():
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            backbone["float_repository"], backbone["float_file"]
        )
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(Path(downloaded).read_bytes())
    return local


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _verify_backbone(backbone: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    onnx_path = _resolve(backbone["onnx_path"])
    if sha256_file(onnx_path) != backbone["onnx_sha256"]:
        raise ValueError(f"{backbone['id']} weight hash mismatch")
    preprocessor_path = _resolve(backbone["preprocessor_path"])
    if sha256_file(preprocessor_path) != backbone["preprocessor_sha256"]:
        raise ValueError(f"{backbone['id']} preprocessing hash mismatch")
    preprocessing = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    return onnx_path, preprocessing


def extract_known(
    source_files: list[Path],
    *,
    config: dict[str, Any],
    backbone: dict[str, Any],
    preprocessing: dict[str, Any],
    onnx_path: Path,
    output_dir: Path,
    progress: bool,
) -> dict[str, Any]:
    known = config["known"]
    extraction = config["extraction"]
    classes = np.arange(int(known["class_count"]))
    samples_per_class = int(known["samples_per_class"])

    started = time.perf_counter()
    features, labels, selection = select_known(
        source_files,
        classes=classes,
        samples_per_class=samples_per_class,
        minimum_short_edge=int(known["minimum_native_short_edge"]),
        backbone=backbone,
        preprocessing=preprocessing,
        onnx_path=onnx_path,
        dispatch_chunk_size=int(extraction["dispatch_chunk_size"]),
        worker_count=int(extraction["worker_count"]),
        progress=progress,
    )
    seconds = time.perf_counter() - started

    expected = (len(classes) * samples_per_class, int(backbone["output_dimension"]))
    if features.shape != expected:
        raise RuntimeError(f"extraction returned {features.shape}, expected {expected}")

    domains = np.asarray([entry["domain"] for entry in selection], dtype=np.int64)
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    np.save(arrays_dir / "features.npy", features.astype(np.float32))
    np.save(arrays_dir / "labels.npy", labels)
    np.save(arrays_dir / "domains.npy", domains)

    reference = json.loads(
        _resolve(known["reference_manifest"]).read_text(encoding="utf-8")
    )
    identity = manifest_identity(
        selection,
        reference["selection"],
        fields=list(config["manifest_identity"]["fields"]),
        maximum_differing=int(config["manifest_identity"]["maximum_differing_entries"]),
    )

    features_digest = sha256_file(arrays_dir / "features.npy")
    labels_digest = sha256_file(arrays_dir / "labels.npy")
    evidence = {
        "backbone": backbone["id"],
        "split": "known",
        "row_count": int(features.shape[0]),
        "dimension": int(features.shape[1]),
        "extraction_seconds": round(seconds, 1),
        "images_per_second": round(features.shape[0] / max(seconds, 1e-9), 1),
        "features_sha256": features_digest,
        "labels_sha256": labels_digest,
        "manifest_identity": identity,
        "reproduces_v13_features": features_digest
        == known["reference_features_sha256"],
        "reproduces_v13_labels": labels_digest == known["reference_labels_sha256"],
    }
    write_canonical_json(
        output_dir / "selection_manifest.json", {"selection": selection}
    )
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    return evidence


def extract_openset(
    source_files: list[Path],
    *,
    config: dict[str, Any],
    backbone: dict[str, Any],
    preprocessing: dict[str, Any],
    onnx_path: Path,
    output_dir: Path,
    progress: bool,
) -> dict[str, Any]:
    openset = config["openset"]
    extraction = config["extraction"]
    labels = list(range(int(openset["first_label"]), int(openset["last_label"]) + 1))

    started = time.perf_counter()
    features, selection, shortfall = select_openset(
        source_files,
        labels=labels,
        quota=list(openset["domain_quota_per_unseen_class"]),
        minimum_short_edge=int(openset["minimum_native_short_edge"]),
        backbone=backbone,
        preprocessing=preprocessing,
        onnx_path=onnx_path,
        dispatch_chunk_size=int(extraction["dispatch_chunk_size"]),
        worker_count=int(extraction["worker_count"]),
        progress=progress,
    )
    seconds = time.perf_counter() - started

    class_labels = np.asarray(
        [entry["class_label"] for entry in selection], dtype=np.int64
    )
    domains = np.asarray([entry["domain"] for entry in selection], dtype=np.int64)
    underfilled = sorted({int(cell.split(":")[0]) for cell in shortfall})
    complete = np.asarray(
        [label not in set(underfilled) for label in class_labels], dtype=bool
    )

    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    np.save(arrays_dir / "features.npy", features.astype(np.float32))
    np.save(arrays_dir / "labels.npy", class_labels)
    np.save(arrays_dir / "domains.npy", domains)
    np.save(arrays_dir / "stratified.npy", complete)

    reference = json.loads(
        _resolve(openset["reference_manifest"]).read_text(encoding="utf-8")
    )
    identity = manifest_identity(
        selection,
        reference["selection"],
        fields=list(config["manifest_identity"]["fields"]),
        maximum_differing=int(config["manifest_identity"]["maximum_differing_entries"]),
    )

    feature_hash = payload_hash(
        {
            "features": sha256_file(arrays_dir / "features.npy"),
            "labels": sha256_file(arrays_dir / "labels.npy"),
            "domains": sha256_file(arrays_dir / "domains.npy"),
        }
    )
    evidence = {
        "backbone": backbone["id"],
        "split": "openset",
        "row_count": int(features.shape[0]),
        "dimension": int(features.shape[1]),
        "stratified_row_count": int(np.sum(complete)),
        "underfilled_classes": underfilled,
        "extraction_seconds": round(seconds, 1),
        "images_per_second": round(features.shape[0] / max(seconds, 1e-9), 1),
        "feature_hash": feature_hash,
        "manifest_identity": identity,
        "reproduces_v13_feature_hash": feature_hash
        == openset["reference_feature_hash"],
    }
    write_canonical_json(
        output_dir / "selection_manifest.json", {"selection": selection}
    )
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    return evidence


def build_backbone(
    backbone: dict[str, Any],
    *,
    config: dict[str, Any],
    source_files: list[Path],
    output_root: Path,
    progress: bool,
    skip_divergence: bool,
) -> dict[str, Any]:
    onnx_path, preprocessing = _verify_backbone(backbone)
    directory = output_root / backbone["id"]

    print(f"\n{backbone['id']} ({backbone['output_dimension']}-d)", flush=True)
    print("  shard-invariance control...", flush=True)
    invariance = shard_invariance_control(
        source_files,
        config={
            "shard_invariance_control": {
                "probe_images": 96,
                "worker_counts": [1, 4],
                "maximum_absolute_difference": 0.0,
            },
            "minimum_native_short_edge": config["known"]["minimum_native_short_edge"],
        },
        backbone=backbone,
        preprocessing=preprocessing,
        onnx_path=onnx_path,
    )
    print(
        f"  shard invariance max |diff| "
        f"{invariance['maximum_absolute_difference']:.3e}, "
        f"passes {invariance['passed']}",
        flush=True,
    )
    if not invariance["passed"]:
        raise RuntimeError(
            f"{backbone['id']} extraction is not per-image; the artifact would be "
            "a function of its own scheduling"
        )

    divergence: dict[str, Any] | None = None
    if not skip_divergence:
        print("  quantisation divergence control (N91.6)...", flush=True)
        specification = config["quantisation_divergence_control"]
        images = probe_images(
            source_files,
            count=int(specification["probe_images"]),
            minimum_short_edge=int(config["known"]["minimum_native_short_edge"]),
        )
        divergence = quantisation_divergence(
            images,
            backbone=backbone,
            preprocessing=preprocessing,
            int8_path=onnx_path,
            float_path=_float_graph(backbone),
        )
        print(
            f"  INT8 vs fp32 relative divergence mean "
            f"{divergence['mean_relative_divergence']:.4f}, "
            f"max {divergence['maximum_relative_divergence']:.4f}",
            flush=True,
        )

    print("  known corpus...", flush=True)
    known = extract_known(
        source_files,
        config=config,
        backbone=backbone,
        preprocessing=preprocessing,
        onnx_path=onnx_path,
        output_dir=directory / "known",
        progress=progress,
    )
    print("  open set...", flush=True)
    openset = extract_openset(
        source_files,
        config=config,
        backbone=backbone,
        preprocessing=preprocessing,
        onnx_path=onnx_path,
        output_dir=directory / "openset",
        progress=progress,
    )

    return {
        "backbone": backbone["id"],
        "role": backbone["role"],
        "output_dimension": int(backbone["output_dimension"]),
        "onnx_sha256": backbone["onnx_sha256"],
        "preprocessor_sha256": backbone["preprocessor_sha256"],
        "shard_invariance_control": invariance,
        "quantisation_divergence_control": divergence,
        "known": known,
        "openset": openset,
        "rows_identical": bool(
            known["manifest_identity"]["passes"]
            and openset["manifest_identity"]["passes"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--backbone",
        action="append",
        default=None,
        help="restrict to one backbone id; repeatable",
    )
    parser.add_argument("--skip-divergence", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args(argv)

    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    output_root = Path(config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    source_files = _source_files(_resolve(config["domainnet_download_record"]["path"]))

    selected = [
        backbone
        for backbone in config["backbones"]
        if arguments.backbone is None or backbone["id"] in arguments.backbone
    ]
    if not selected:
        raise SystemExit("no backbone selected")

    reports = [
        build_backbone(
            backbone,
            config=config,
            source_files=source_files,
            output_root=output_root,
            progress=not arguments.quiet,
            skip_divergence=arguments.skip_divergence,
        )
        for backbone in selected
    ]

    for report in reports:
        summary_path = output_root / f"{report['backbone']}.json"
        write_canonical_json(summary_path, report)
        known = report["known"]
        openset = report["openset"]
        print(
            f"\n{report['backbone']:<14} dim {report['output_dimension']:>5}"
            f"  known {known['row_count']}"
            f"  openset {openset['row_count']}"
            f"  rows-identical {report['rows_identical']}"
            f"  v13-features {known['reproduces_v13_features']}"
            f"  v13-openset {openset['reproduces_v13_feature_hash']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
