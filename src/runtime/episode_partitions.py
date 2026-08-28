"""Deterministic index partitions and leakage audits for dataset episodes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import numpy as np


PARTITION_NAMES = (
    "geometry",
    "readout_calibration",
    "risk_control",
    "validation",
    "final_test",
)


def _array_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class EpisodePartitionAudit:
    seed: int
    dataset_size: int
    partition_counts: tuple[tuple[str, int], ...]
    partition_hashes: tuple[tuple[str, str], ...]
    class_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    complete_coverage: bool
    pairwise_disjoint: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "dataset_size": self.dataset_size,
            "partition_counts": dict(self.partition_counts),
            "partition_hashes": dict(self.partition_hashes),
            "class_counts": {
                name: dict(counts) for name, counts in self.class_counts
            },
            "complete_coverage": self.complete_coverage,
            "pairwise_disjoint": self.pairwise_disjoint,
        }


def validate_episode_partitions(
    partitions: Mapping[str, np.ndarray],
    *,
    dataset_size: int,
    expected_indices: np.ndarray | None = None,
) -> None:
    if set(partitions) != set(PARTITION_NAMES):
        raise ValueError(f"partitions must be exactly {PARTITION_NAMES}")
    normalized: dict[str, np.ndarray] = {}
    for name in PARTITION_NAMES:
        indices = np.asarray(partitions[name])
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"partition {name!r} must contain one-dimensional integers")
        indices = indices.astype(np.int64, copy=False)
        if len(indices) == 0:
            raise ValueError(f"partition {name!r} must not be empty")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"partition {name!r} contains duplicate indices")
        if indices.min() < 0 or indices.max() >= dataset_size:
            raise ValueError(f"partition {name!r} contains out-of-range indices")
        normalized[name] = indices

    names = list(PARTITION_NAMES)
    for position, name in enumerate(names):
        for other_name in names[position + 1:]:
            if np.intersect1d(normalized[name], normalized[other_name]).size:
                raise ValueError(f"partitions {name!r} and {other_name!r} overlap")

    if expected_indices is not None:
        expected = np.sort(np.asarray(expected_indices, dtype=np.int64))
        actual = np.sort(np.concatenate(list(normalized.values())))
        if not np.array_equal(actual, expected):
            raise ValueError("partitions do not exactly cover the expected indices")


def build_stratified_episode_partitions(
    labels: np.ndarray,
    *,
    development_indices: np.ndarray,
    final_test_indices: np.ndarray,
    seed: int,
    readout_fraction: float = 0.15,
    risk_fraction: float = 0.10,
    validation_fraction: float = 0.15,
) -> tuple[dict[str, np.ndarray], EpisodePartitionAudit]:
    """Split development data while preserving a fixed official final test."""
    labels = np.asarray(labels)
    development = np.asarray(development_indices, dtype=np.int64)
    final_test = np.asarray(final_test_indices, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    fractions = (readout_fraction, risk_fraction, validation_fraction)
    if any(not 0.0 < value < 1.0 for value in fractions) or sum(fractions) >= 1.0:
        raise ValueError("development partition fractions must be positive and sum below one")
    if np.intersect1d(development, final_test).size:
        raise ValueError("development and final-test indices overlap")

    rng = np.random.default_rng(seed)
    grouped: dict[str, list[np.ndarray]] = {
        name: [] for name in PARTITION_NAMES if name != "final_test"
    }
    for class_id in np.unique(labels[development]):
        class_indices = development[labels[development] == class_id].copy()
        rng.shuffle(class_indices)
        readout_count = max(1, int(round(readout_fraction * len(class_indices))))
        risk_count = max(1, int(round(risk_fraction * len(class_indices))))
        validation_count = max(1, int(round(validation_fraction * len(class_indices))))
        held_out_count = readout_count + risk_count + validation_count
        if len(class_indices) - held_out_count < 2:
            raise ValueError(f"class {class_id!r} has insufficient geometry samples")
        grouped["readout_calibration"].append(class_indices[:readout_count])
        grouped["risk_control"].append(
            class_indices[readout_count:readout_count + risk_count]
        )
        grouped["validation"].append(
            class_indices[readout_count + risk_count:held_out_count]
        )
        grouped["geometry"].append(class_indices[held_out_count:])

    partitions = {
        name: np.sort(np.concatenate(grouped[name])).astype(np.int64)
        for name in grouped
    }
    partitions["final_test"] = np.sort(final_test).astype(np.int64)
    expected = np.concatenate((development, final_test))
    validate_episode_partitions(
        partitions, dataset_size=len(labels), expected_indices=expected,
    )
    partition_counts = tuple(
        (name, len(partitions[name])) for name in PARTITION_NAMES
    )
    partition_hashes = tuple(
        (name, _array_hash(partitions[name])) for name in PARTITION_NAMES
    )
    class_counts = tuple(
        (
            name,
            tuple(
                (str(class_id), int(np.sum(labels[partitions[name]] == class_id)))
                for class_id in np.unique(labels[partitions[name]])
            ),
        )
        for name in PARTITION_NAMES
    )
    return partitions, EpisodePartitionAudit(
        seed=seed,
        dataset_size=len(labels),
        partition_counts=partition_counts,
        partition_hashes=partition_hashes,
        class_counts=class_counts,
        complete_coverage=True,
        pairwise_disjoint=True,
    )