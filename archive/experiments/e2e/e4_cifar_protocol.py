"""Frozen data and partition protocol for E4 CIFAR qualification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.experiment_manifest import array_fingerprint
from src.runtime import EpisodePartitionAudit, build_stratified_episode_partitions


@dataclass(frozen=True)
class E4Data:
    id_features: np.ndarray
    id_labels: np.ndarray
    id_source_indices: np.ndarray
    near_features: np.ndarray
    near_labels: np.ndarray
    near_source_indices: np.ndarray
    far_features: np.ndarray
    far_source_indices: np.ndarray
    fingerprints: dict[str, str]


def load_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported E4 configuration schema")
    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("E4 requires exactly five unique confirmatory seeds")
    if payload.get("deployment_seed") not in seeds:
        raise ValueError("deployment_seed must be one of the confirmatory seeds")
    margin = payload.get("non_inferiority_margin")
    if not isinstance(margin, (int, float)) or not 0.0 < margin < 1.0:
        raise ValueError("non_inferiority_margin must be frozen in (0, 1)")
    return payload


def _reconstructed_indices(population: int, count: int, seed: int) -> np.ndarray:
    if count < 1 or count > population:
        raise ValueError("feature sample count is outside the raw dataset")
    return np.random.default_rng(seed).permutation(population)[:count]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_e4_data(config: dict[str, Any]) -> E4Data:
    id_config = config["id"]
    id_raw_path = Path(id_config["raw_path"])
    id_feature_path = Path(id_config["feature_path"])
    with np.load(id_raw_path, allow_pickle=False) as raw:
        raw_id_labels = raw["coarse_labels"].astype(np.int32)
    with np.load(id_feature_path, allow_pickle=False) as cached:
        id_features = cached["features"].astype(np.float64)
        id_labels = cached["labels"].astype(np.int32)
    id_source_indices = _reconstructed_indices(
        len(raw_id_labels), id_config["feature_sample_count"],
        id_config["feature_sample_seed"],
    )
    if len(id_features) != len(id_source_indices) or not np.array_equal(
        id_labels, raw_id_labels[id_source_indices],
    ):
        raise ValueError("CIFAR-100 feature cache does not match reconstructed raw rows")

    near_config = config["near_ood"]
    near_raw_path = Path(near_config["raw_path"])
    near_feature_path = Path(near_config["feature_path"])
    with np.load(near_raw_path, allow_pickle=False) as raw:
        raw_near_labels = raw["labels"].astype(np.int32)
    with np.load(near_feature_path, allow_pickle=False) as cached:
        near_features = cached[near_config["feature_key"]].astype(np.float64)
    near_source_indices = _reconstructed_indices(
        len(raw_near_labels), near_config["feature_sample_count"],
        near_config["feature_sample_seed"],
    )
    if len(near_features) != len(near_source_indices):
        raise ValueError("CIFAR-10 feature cache row count does not match its protocol")
    near_labels = raw_near_labels[near_source_indices]

    far_config = config["far_ood"]
    far_feature_path = Path(far_config["feature_path"])
    with np.load(far_feature_path, allow_pickle=False) as cached:
        far_features = cached["features"].astype(np.float64)
        far_source_indices = cached["source_indices"].astype(np.int64)
        if str(cached["split"].item()) != "test":
            raise ValueError("SVHN cache must come from the official test split")
    if len(far_features) != len(far_source_indices):
        raise ValueError("SVHN feature cache has misaligned source indices")
    if len(np.unique(far_source_indices)) != len(far_source_indices):
        raise ValueError("SVHN source indices must be unique")

    return E4Data(
        id_features=id_features,
        id_labels=id_labels,
        id_source_indices=id_source_indices,
        near_features=near_features,
        near_labels=near_labels,
        near_source_indices=near_source_indices,
        far_features=far_features,
        far_source_indices=far_source_indices,
        fingerprints={
            "id_raw_file": _file_hash(id_raw_path),
            "id_feature_file": _file_hash(id_feature_path),
            "id_source_indices": array_fingerprint(id_source_indices),
            "near_raw_file": _file_hash(near_raw_path),
            "near_feature_file": _file_hash(near_feature_path),
            "near_source_indices": array_fingerprint(near_source_indices),
            "far_feature_file": _file_hash(far_feature_path),
            "far_source_indices": array_fingerprint(far_source_indices),
        },
    )


def build_id_partitions(
    data: E4Data, config: dict[str, Any], seed: int,
) -> tuple[dict[str, np.ndarray], EpisodePartitionAudit]:
    official_train_count = int(config["id"]["official_train_count"])
    development = np.flatnonzero(data.id_source_indices < official_train_count)
    final_test = np.flatnonzero(data.id_source_indices >= official_train_count)
    fractions = config["partition_fractions"]
    return build_stratified_episode_partitions(
        data.id_labels,
        development_indices=development,
        final_test_indices=final_test,
        seed=seed,
        readout_fraction=float(fractions["readout_calibration"]),
        risk_fraction=float(fractions["risk_control"]),
        validation_fraction=float(fractions["validation"]),
    )


def build_ood_partitions(
    data: E4Data, config: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    near_config = config["near_ood"]
    near_development = np.flatnonzero(
        data.near_source_indices < int(near_config["official_train_count"])
    )
    near_final = np.flatnonzero(
        data.near_source_indices >= int(near_config["official_train_count"])
    )
    near_order = np.random.default_rng(
        int(near_config["feature_sample_seed"])
    ).permutation(near_development)
    validation_count = int(round(
        len(near_order) * float(near_config["development_validation_fraction"])
    ))
    near = {
        "validation": np.sort(near_order[:validation_count]),
        "risk_control": np.sort(near_order[validation_count:]),
        "final_test": np.sort(near_final),
    }

    far_config = config["far_ood"]
    far_order = np.random.default_rng(int(far_config["split_seed"])).permutation(
        len(data.far_features)
    )
    far_validation_count = int(far_config["validation_count"])
    far_risk_count = int(far_config["risk_control_count"])
    far = {
        "validation": np.sort(far_order[:far_validation_count]),
        "risk_control": np.sort(
            far_order[far_validation_count:far_validation_count + far_risk_count]
        ),
        "final_test": np.sort(far_order[far_validation_count + far_risk_count:]),
    }
    for family, partitions in (("near", near), ("far", far)):
        values = list(partitions.values())
        if any(len(value) == 0 for value in values):
            raise ValueError(f"{family} OOD partitions must be non-empty")
        if any(
            np.intersect1d(values[left], values[right]).size
            for left in range(len(values))
            for right in range(left + 1, len(values))
        ):
            raise ValueError(f"{family} OOD partitions overlap")
    audit = {
        family: {
            "counts": {name: len(indices) for name, indices in partitions.items()},
            "hashes": {
                name: array_fingerprint(indices) for name, indices in partitions.items()
            },
            "pairwise_disjoint": True,
        }
        for family, partitions in (("near", near), ("far", far))
    }
    return {"near": near, "far": far}, audit