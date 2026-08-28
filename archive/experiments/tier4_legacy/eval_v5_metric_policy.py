"""Evaluate a frozen M18 metric policy on an explicit feature split."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from experiments.tier1.eval_v5_metric_support_sweep import (
    _evaluate_candidate,
    _support_bin,
)


def evaluate_frozen_metric_policy(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    policy: Mapping[str, Any],
    *,
    intrinsic_rank: int,
    eigenvalue_floor: float = 1e-6,
) -> dict[str, Any]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    if features.ndim != 2 or labels.shape != (len(features),):
        raise ValueError("features and labels must be aligned.")
    if train_indices.ndim != 1 or test_indices.ndim != 1:
        raise ValueError("train_indices and test_indices must be one-dimensional.")
    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("train_indices and test_indices must be non-empty.")
    for name, indices in (
        ("train_indices", train_indices),
        ("test_indices", test_indices),
    ):
        if np.any(indices < 0) or np.any(indices >= len(features)):
            raise ValueError(f"{name} must be within the feature array bounds.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{name} must not contain duplicate indices.")
    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError("train and test indices must be disjoint.")
    classes = np.unique(labels[train_indices])
    if set(classes.tolist()) != set(np.unique(labels[test_indices]).tolist()):
        raise ValueError("train and test splits must contain the same classes.")
    class_points = {
        class_id.item() if isinstance(class_id, np.generic) else class_id: features[
            train_indices[labels[train_indices] == class_id]
        ]
        for class_id in classes
    }
    minimum_support = min(len(points) for points in class_points.values())
    support_record = {
        "samples_per_class": minimum_support,
        "dimension": features.shape[1],
        "intrinsic_rank": intrinsic_rank,
    }
    bin_name = _support_bin(support_record, policy["support_bins"])
    candidate = policy["support_bin_selections"].get(bin_name)
    warnings = []
    if candidate is None:
        candidate = policy["global_candidate"]
        warnings.append("unseen_support_bin_used_global_candidate")
    result = _evaluate_candidate(
        class_points,
        features[test_indices],
        labels[test_indices],
        candidate,
        eigenvalue_floor=eigenvalue_floor,
    )
    return {
        "candidate": candidate,
        "support_bin": bin_name,
        "minimum_class_support": minimum_support,
        "warnings": warnings + result.pop("warnings"),
        **result,
    }
