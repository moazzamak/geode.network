from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pyarrow.parquet as parquet

from experiments.common.v11_directional_envelope import (
    DirectionalTube,
    class_score_matrix,
    normalize_directions,
    spherical_log_map,
    split_conformal_quantile,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.tier4.eval_v11_m65_directional_envelope_screen import (
    _geometry_states,
    _quantile_tube,
)
from experiments.tier4.eval_v12_m70_d1_sample_sensitivity import (
    _probe_acceptance,
)
from experiments.tier4.prepare_v5_frozen_features import extract_features_batch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v12" / "m70_native_domainnet.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v12" / "m70_native_domainnet"


def _resolve(path: str) -> Path:
    resolved = (REPO_ROOT / Path(path)).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError("M70 native repository paths must remain inside the repository")
    return resolved


def _verify(specification: dict[str, str]) -> Path:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"M70 native immutable artifact hash mismatch: {path}")
    return path


def _source_files(download: dict[str, Any]) -> list[Path]:
    repository = Path(download["dataset_root"]) / "repository"
    files = [repository / path for path in download["verified_files"] if "train-" in path]
    if not files or not all(path.is_file() for path in files):
        raise FileNotFoundError("M70 native DomainNet training shards are unavailable")
    return files


def _select_native_images(
    source_files: list[Path],
    *,
    classes: np.ndarray,
    samples_per_class: int,
    minimum_short_edge: int,
) -> tuple[list[np.ndarray], np.ndarray, list[dict[str, Any]]]:
    selected: dict[int, list[tuple[np.ndarray, dict[str, Any]]]] = {
        int(class_label): [] for class_label in classes
    }
    for source_path in source_files:
        source = parquet.ParquetFile(source_path)
        row_offset = 0
        for batch in source.iter_batches(
            batch_size=256, columns=["image", "label", "domain", "image_path"]
        ):
            for local_index, row in enumerate(batch.to_pylist()):
                class_label = int(row["label"])
                if (
                    class_label not in selected
                    or len(selected[class_label]) >= samples_per_class
                ):
                    continue
                with Image.open(BytesIO(row["image"]["bytes"])) as image:
                    rgb = image.convert("RGB")
                    if min(rgb.size) < minimum_short_edge:
                        continue
                    array = np.asarray(rgb)
                    metadata = {
                        "source_file": source_path.name,
                        "source_row": int(row_offset + local_index),
                        "class_label": class_label,
                        "domain": int(row["domain"]),
                        "image_path": row["image_path"].decode("utf-8")
                        if isinstance(row["image_path"], bytes)
                        else str(row["image_path"]),
                        "native_width": int(rgb.width),
                        "native_height": int(rgb.height),
                    }
                selected[class_label].append((array, metadata))
            row_offset += len(batch)
            if all(len(values) >= samples_per_class for values in selected.values()):
                break
        if all(len(values) >= samples_per_class for values in selected.values()):
            break
    short = {
        class_label: len(values)
        for class_label, values in selected.items()
        if len(values) < samples_per_class
    }
    if short:
        raise ValueError(f"M70 native classes lack sufficient samples: {short}")
    images = []
    labels = []
    manifest = []
    for class_label in classes:
        for image, metadata in selected[int(class_label)]:
            images.append(image)
            labels.append(int(class_label))
            manifest.append(metadata)
    return images, np.asarray(labels, dtype=np.int64), manifest


def _load_or_extract(
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    features_path = output_dir / "arrays" / "features.npy"
    labels_path = output_dir / "arrays" / "labels.npy"
    manifest_path = output_dir / "selection_manifest.json"
    if features_path.is_file() and labels_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            sha256_file(features_path) == manifest["features_sha256"]
            and sha256_file(labels_path) == manifest["labels_sha256"]
            and manifest["configuration_hash"] == payload_hash(config)
        ):
            return (
                np.load(features_path, allow_pickle=False).astype(np.float64),
                np.load(labels_path, allow_pickle=False),
                manifest,
            )
        raise ValueError("M70 native cache exists but fails its immutable manifest")

    download = json.loads(_verify(config["domainnet_download_record"]).read_text())
    source_files = _source_files(download)
    classes = np.arange(max(int(value) for value in config["class_counts"]))
    images, labels, selection = _select_native_images(
        source_files,
        classes=classes,
        samples_per_class=int(config["samples_per_class"]),
        minimum_short_edge=int(config["minimum_native_short_edge"]),
    )
    backbone = config["backbone"]
    onnx_path = _resolve(backbone["onnx_path"])
    preprocessing_path = _resolve(backbone["preprocessor_path"])
    if sha256_file(onnx_path) != backbone["onnx_sha256"]:
        raise ValueError("M70 native DINOv2 weight hash mismatch")
    if sha256_file(preprocessing_path) != backbone["preprocessor_sha256"]:
        raise ValueError("M70 native preprocessing hash mismatch")
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    features = extract_features_batch(
        images,
        backbone["id"],
        str(onnx_path),
        preprocessing,
        backbone["token_pooling_policy"],
        batch_size=int(config["extraction_batch_size"]),
    )
    if features.shape != (len(labels), int(backbone["output_dimension"])):
        raise RuntimeError("M70 native feature extraction returned the wrong shape")
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    np.save(features_path, features.astype(np.float32), allow_pickle=False)
    np.save(labels_path, labels, allow_pickle=False)
    manifest = {
        "schema_version": 1,
        "configuration_hash": payload_hash(config),
        "dataset": "DomainNet",
        "source_repository": download["source_repository"],
        "source_revision": download["source_revision"],
        "source_files": [
            {
                "path": str(path),
                "sha256": next(
                    item["sha256"]
                    for item in json.loads(
                        Path(download["manifest_path"]).read_text(encoding="utf-8")
                    )["files"]
                    if Path(item["path"]).name == path.name
                ),
            }
            for path in source_files
        ],
        "class_labels": classes.tolist(),
        "samples_per_class": int(config["samples_per_class"]),
        "minimum_native_short_edge": int(config["minimum_native_short_edge"]),
        "selection": selection,
        "features_sha256": sha256_file(features_path),
        "labels_sha256": sha256_file(labels_path),
        "preprocessing": {
            "resize_shortest_edge": preprocessing["size"]["shortest_edge"],
            "center_crop": preprocessing["crop_size"],
            "interpolation": "PIL bicubic",
            "upsampling_excluded_by_selection": True,
        },
        "final_labels_opened": False,
    }
    write_canonical_json(manifest_path, manifest)
    return features, labels, manifest


def _partition_classes(
    features: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    config: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    geometry = []
    extent = []
    conformal = []
    for class_label in classes:
        rows = np.flatnonzero(labels == class_label)
        expected = int(config["samples_per_class"])
        if len(rows) != expected:
            raise ValueError("M70 native class sample count changed")
        geometry_count = int(config["geometry_per_class"])
        extent_count = int(config["extent_per_class"])
        conformal_count = int(config["conformal_per_class"])
        if geometry_count + extent_count + conformal_count != expected:
            raise ValueError("M70 native partition counts do not exhaust each class")
        geometry.extend(rows[:geometry_count])
        extent.extend(rows[geometry_count : geometry_count + extent_count])
        conformal.extend(rows[-conformal_count:])
    return {
        name: (features[indices], labels[indices])
        for name, indices in {
            "geometry_fit": np.asarray(geometry, dtype=np.int64),
            "extent_fit": np.asarray(extent, dtype=np.int64),
            "conformal_calibration": np.asarray(conformal, dtype=np.int64),
        }.items()
    }


def _evaluate_class_count(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    classes = np.arange(class_count, dtype=np.int64)
    partitions = _partition_classes(features, labels, classes, config)
    geometry_x, geometry_y = partitions["geometry_fit"]
    extent_x, extent_y = partitions["extent_fit"]
    conformal_x, conformal_y = partitions["conformal_calibration"]
    states, _ = _geometry_states(
        geometry_x,
        geometry_y,
        known_classes=classes,
        patch_count=1,
        seed=int(config["seed"]),
        maximum_rank=int(config["rank"]),
    )
    tubes: list[DirectionalTube] = []
    thresholds = []
    threshold_ratios = {}
    penetration = {}
    for state in states:
        class_label = int(state["class_label"])
        tube = _quantile_tube(
            state,
            normalize_directions(extent_x[extent_y == class_label]),
            rank=int(config["rank"]),
            penalty_weight=float(config["penalty_weight"]),
        )
        tubes.append(tube)
        own_scores = tube.score(conformal_x[conformal_y == class_label])
        threshold = split_conformal_quantile(
            own_scores, miscoverage=float(config["miscoverage"])
        )
        thresholds.append(threshold)
        threshold_ratios[str(class_label)] = float(
            threshold / np.median(own_scores)
        )
        own_extent_coordinates = np.abs(
            spherical_log_map(
                tube.center,
                normalize_directions(extent_x[extent_y == class_label]),
            )
            @ tube.basis
        )
        floor = np.quantile(
            own_extent_coordinates, 0.90, axis=0, method="higher"
        )
        other = normalize_directions(conformal_x[conformal_y != class_label])
        other_coordinates = np.abs(
            spherical_log_map(tube.center, other) @ tube.basis
        )
        penetration[str(class_label)] = float(
            np.mean(np.all(other_coordinates <= floor[None, :], axis=1))
        )
    threshold_array = np.asarray(thresholds, dtype=np.float64)
    probes = _probe_acceptance(
        tubes,
        threshold_array,
        replicates_per_axis_sign=int(config["probe_replicates_per_axis_sign"]),
        seed=int(config["seed"]) + class_count,
    )
    scores, score_classes = class_score_matrix(tubes, conformal_x)
    if not np.array_equal(score_classes, classes):
        raise RuntimeError("M70 native class order changed")
    predicted = classes[np.argmin(scores / threshold_array[None, :], axis=1)]
    return {
        "class_count": class_count,
        "threshold_ratio_by_class": threshold_ratios,
        "median_threshold_ratio": float(
            np.median(list(threshold_ratios.values()))
        ),
        "probe_acceptance": probes,
        "other_class_penetration_below_q90_floor_by_class": penetration,
        "mean_other_class_penetration_below_q90_floor": float(
            np.mean(list(penetration.values()))
        ),
        "conformal_classification_accuracy": float(
            np.mean(predicted == conformal_y)
        ),
        "partition_counts": {
            name: int(len(values[0])) for name, values in partitions.items()
        },
        "geometry_replay_hash": payload_hash(
            [tube.to_dict() for tube in tubes]
        ),
    }


def run_diagnostic(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _verify(config["v11_parent_index"])
    features, labels, selection_manifest = _load_or_extract(config, output_dir)
    class_count_results = [
        _evaluate_class_count(
            features,
            labels,
            class_count=int(class_count),
            config=config,
        )
        for class_count in config["class_counts"]
    ]
    system_4x = [
        result["probe_acceptance"]["4"]["system_acceptance"]
        for result in class_count_results
    ]
    penetration = [
        result["mean_other_class_penetration_below_q90_floor"]
        for result in class_count_results
    ]
    evidence = {
        "schema_version": 1,
        "milestone": "M70-D2-D3",
        "configuration_hash": sha256_file(config_path),
        "selection_manifest_sha256": sha256_file(
            output_dir / "selection_manifest.json"
        ),
        "dataset": "DomainNet",
        "native_resolution": True,
        "upsampling_excluded": all(
            min(item["native_width"], item["native_height"])
            >= int(config["minimum_native_short_edge"])
            for item in selection_manifest["selection"]
        ),
        "class_count_results": class_count_results,
        "gate": {
            "all_class_counts_reported": [
                result["class_count"] for result in class_count_results
            ]
            == config["class_counts"],
            "native_resolution_verified": True,
            "probe_count_increased_at_least_eight_x": all(
                result["probe_acceptance"]["4"]["per_patch_count"]
                >= 2 * int(config["rank"]) * 8
                for result in class_count_results
            ),
            "four_x_system_acceptance_non_decreasing": all(
                left <= right for left, right in zip(system_4x, system_4x[1:])
            ),
            "penetration_non_decreasing": all(
                left <= right
                for left, right in zip(penetration, penetration[1:])
            ),
            "final_labels_opened": False,
        },
        "final_labels_opened": False,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = run_diagnostic(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "class_count_results": [
                    {
                        "class_count": item["class_count"],
                        "median_threshold_ratio": item[
                            "median_threshold_ratio"
                        ],
                        "four_x_system_acceptance": item[
                            "probe_acceptance"
                        ]["4"]["system_acceptance"],
                        "eight_x_system_acceptance": item[
                            "probe_acceptance"
                        ]["8"]["system_acceptance"],
                        "mean_other_class_penetration": item[
                            "mean_other_class_penetration_below_q90_floor"
                        ],
                    }
                    for item in result["class_count_results"]
                ],
                "gate": result["gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
