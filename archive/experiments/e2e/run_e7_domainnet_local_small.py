"""Run a bounded real-DomainNet episode across a logical Ray cluster."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.data_cache import data_cache_root
from src.runtime.domainnet_manifest import DomainNetManifest
from src.runtime.ray_executor import RayExecutor


DOMAIN_NAMES = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _validate_config(config: dict[str, Any]) -> None:
    class_ids = [int(value) for value in config["class_ids"]]
    source_domains = [int(value) for value in config["source_domain_ids"]]
    validation_domain = int(config["validation_domain_id"])
    final_domain = int(config["final_domain_id"])
    image_size = int(config["image_size"])
    if not class_ids or len(class_ids) != len(set(class_ids)):
        raise ValueError("E7 local-small class IDs must be non-empty and unique")
    if min(class_ids) < 0 or max(class_ids) >= 345:
        raise ValueError("E7 local-small class IDs must be within DomainNet")
    domains = [*source_domains, validation_domain, final_domain]
    if len(domains) != len(set(domains)) or any(
        value < 0 or value >= len(DOMAIN_NAMES) for value in domains
    ):
        raise ValueError("E7 local-small domains must be valid and disjoint")
    if int(config["samples_per_class_domain"]) < 1:
        raise ValueError("E7 local-small sample budget must be positive")
    if image_size < 4 or image_size % 4:
        raise ValueError("E7 local-small image size must be divisible by four")


def _image_feature(image_bytes: bytes, image_size: int) -> list[float]:
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as image:
        pixels = np.asarray(
            image.convert("RGB").resize((image_size, image_size)),
            dtype=np.float32,
        ) / 255.0
    means = pixels.mean(axis=(0, 1))
    deviations = pixels.std(axis=(0, 1))
    pooled = pixels.reshape(4, image_size // 4, 4, image_size // 4, 3).mean(
        axis=(1, 3),
    )
    return np.concatenate([means, deviations, pooled.reshape(-1)]).tolist()


def _extract_row_group(item: dict[str, Any]) -> dict[str, Any]:
    import importlib
    import pyarrow.parquet as parquet

    ray = importlib.import_module("ray")

    cache_root = Path(os.environ["GEODE_CACHE_DIR"])
    source = parquet.ParquetFile(cache_root / item["path"])
    table = source.read_row_group(
        int(item["row_group"]), columns=["image", "label", "domain", "image_path"],
    )
    allowed = {
        (int(domain), int(label))
        for domain, label in item["allowed_pairs"]
    }
    records = []
    for row_offset, row in enumerate(table.to_pylist()):
        key = (int(row["domain"]), int(row["label"]))
        if key not in allowed:
            continue
        records.append({
            "domain": key[0],
            "label": key[1],
            "image_path": str(row["image_path"]),
            "row_offset": row_offset,
            "feature": _image_feature(row["image"]["bytes"], int(item["image_size"])),
        })
    return {
        "path": item["path"],
        "row_group": int(item["row_group"]),
        "node_id": ray.get_runtime_context().get_node_id(),
        "records": records,
    }


def _locate_row_groups(
    repository_root: Path,
    manifest: DomainNetManifest,
    required: dict[tuple[int, int], int],
    *,
    split: str,
    image_size: int,
) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    remaining = dict(required)
    work = []
    for source_file in manifest.files:
        if f"/{split}-" not in f"/{source_file.path}":
            continue
        source = parquet.ParquetFile(repository_root / source_file.path)
        for row_group in range(source.metadata.num_row_groups):
            if not remaining:
                return work
            metadata = source.read_row_group(row_group, columns=["label", "domain"])
            pairs = [
                (int(domain), int(label))
                for label, domain in zip(
                    metadata.column("label").to_pylist(),
                    metadata.column("domain").to_pylist(),
                    strict=True,
                )
            ]
            available = set(pairs) & set(remaining)
            if not available:
                continue
            work.append({
                "path": source_file.path,
                "row_group": row_group,
                "allowed_pairs": sorted(available),
                "image_size": image_size,
            })
            counts: dict[tuple[int, int], int] = defaultdict(int)
            for pair in pairs:
                if pair in available:
                    counts[pair] += 1
            for pair, count in counts.items():
                remaining[pair] -= count
                if remaining[pair] <= 0:
                    del remaining[pair]
    if remaining:
        raise ValueError(f"DomainNet rows unavailable for pairs: {sorted(remaining)}")
    return work


def _bounded_records(
    results: list[dict[str, Any]],
    required: dict[tuple[int, int], int],
) -> list[dict[str, Any]]:
    records = sorted(
        (record for result in results for record in result["records"]),
        key=lambda record: (
            record["domain"], record["label"], record["image_path"],
        ),
    )
    selected = []
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for record in records:
        pair = (record["domain"], record["label"])
        if pair in required and counts[pair] < required[pair]:
            selected.append(record)
            counts[pair] += 1
    if counts != required:
        raise ValueError("extracted DomainNet records do not satisfy the frozen budget")
    return selected


def _nearest_centroid_metrics(
    source_records: list[dict[str, Any]],
    evaluation_records: list[dict[str, Any]],
    class_ids: list[int],
) -> dict[str, Any]:
    source_features = np.asarray([record["feature"] for record in source_records])
    source_labels = np.asarray([record["label"] for record in source_records])
    evaluation_features = np.asarray([
        record["feature"] for record in evaluation_records
    ])
    evaluation_labels = np.asarray([record["label"] for record in evaluation_records])
    centroids = np.stack([
        source_features[source_labels == class_id].mean(axis=0)
        for class_id in class_ids
    ])
    distances = np.sum(
        (evaluation_features[:, None, :] - centroids[None, :, :]) ** 2,
        axis=2,
    )
    predictions = np.asarray(class_ids)[np.argmin(distances, axis=1)]
    recalls = [
        float(np.mean(predictions[evaluation_labels == class_id] == class_id))
        for class_id in class_ids
    ]
    return {
        "accuracy": float(np.mean(predictions == evaluation_labels)),
        "balanced_accuracy": float(np.mean(recalls)),
        "prediction_hash": hashlib.sha256(predictions.tobytes()).hexdigest(),
        "centroid_hash": hashlib.sha256(centroids.tobytes()).hexdigest(),
    }


def run_local_small(config_path: Path, *, address: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported E7 local-small configuration schema")
    _validate_config(config)
    cache_root = data_cache_root()
    manifest = DomainNetManifest.load(cache_root / config["manifest_path"])
    repository_root = cache_root / config["data_root"]
    manifest_report = manifest.verify(repository_root)
    class_ids = [int(value) for value in config["class_ids"]]
    source_domains = [int(value) for value in config["source_domain_ids"]]
    validation_domain = int(config["validation_domain_id"])
    final_domain = int(config["final_domain_id"])
    budget = int(config["samples_per_class_domain"])
    train_required = {
        (domain, class_id): budget
        for domain in [*source_domains, validation_domain]
        for class_id in class_ids
    }
    test_required = {
        (final_domain, class_id): budget for class_id in class_ids
    }
    work = [
        *_locate_row_groups(
            repository_root, manifest, train_required,
            split="train", image_size=int(config["image_size"]),
        ),
        *_locate_row_groups(
            repository_root, manifest, test_required,
            split="test", image_size=int(config["image_size"]),
        ),
    ]
    for item in work:
        item["path"] = (Path(config["data_root"]) / item["path"]).as_posix()

    executor = RayExecutor(address=address)
    try:
        resources = executor.resource_report().to_dict()
        if resources["nodes"] < int(config["ray"]["minimum_nodes"]):
            raise ValueError("E7 local-small requires multiple logical Ray nodes")
        first = executor.map_on_nodes(
            _extract_row_group, work,
            node_ids=resources["node_ids"],
            max_retries=int(config["ray"]["max_retries"]),
        )
        second = executor.map_on_nodes(
            _extract_row_group, work,
            node_ids=resources["node_ids"],
            max_retries=int(config["ray"]["max_retries"]),
        )
    finally:
        executor.shutdown()

    required = {**train_required, **test_required}
    first_records = _bounded_records(first, required)
    second_records = _bounded_records(second, required)
    first_hash = _canonical_hash(first_records)
    second_hash = _canonical_hash(second_records)
    source = [record for record in first_records if record["domain"] in source_domains]
    validation = [
        record for record in first_records if record["domain"] == validation_domain
    ]
    final = [record for record in first_records if record["domain"] == final_domain]
    validation_metrics = _nearest_centroid_metrics(source, validation, class_ids)
    final_metrics = _nearest_centroid_metrics(source, final, class_ids)
    executing_nodes = sorted({result["node_id"] for result in first})
    checks = {
        "manifest_verified": manifest_report["class_count"] == 345,
        "multiple_logical_nodes": resources["nodes"] >= 2,
        "tasks_executed_on_multiple_nodes": len(executing_nodes) >= 2,
        "exact_feature_replay": first_hash == second_hash,
        "source_validation_final_domains_disjoint": not (
            set(source_domains) & {validation_domain, final_domain}
            or validation_domain == final_domain
        ),
        "final_domain_observational": True,
    }
    return {
        "schema_version": 1,
        "milestone": "E7-local-small",
        "qualification_status": "passed" if all(checks.values()) else "failed",
        "gate_passed": all(checks.values()),
        "config_hash": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "domainnet": manifest_report,
        "protocol": {
            "class_ids": class_ids,
            "source_domains": [DOMAIN_NAMES[value] for value in source_domains],
            "validation_domain": DOMAIN_NAMES[validation_domain],
            "final_domain": DOMAIN_NAMES[final_domain],
            "samples_per_class_domain": budget,
            "source_samples": len(source),
            "validation_samples": len(validation),
            "final_samples": len(final),
            "final_used_for_selection": False,
        },
        "ray": {
            "resources": resources,
            "work_items": len(work),
            "executing_node_ids": executing_nodes,
        },
        "replay": {
            "first_feature_hash": first_hash,
            "second_feature_hash": second_hash,
            "exact": first_hash == second_hash,
        },
        "validation_metrics": validation_metrics,
        "final_observational_metrics": final_metrics,
        "checks": checks,
        "claim_boundary": (
            "bounded nearest-centroid systems episode; not full DomainNet training "
            "or predictive-performance evidence"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("experiments/configs/e7_domainnet_local_small.json"),
    )
    parser.add_argument("--address", default="ray://127.0.0.1:10001")
    parser.add_argument(
        "--output", type=Path,
        default=Path("logs/results/e7_domainnet_local_small.json"),
    )
    arguments = parser.parse_args()
    result = run_local_small(arguments.config, address=arguments.address)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["gate_passed"]:
        raise RuntimeError("E7 DomainNet local-small qualification failed")


if __name__ == "__main__":
    main()