"""Deterministic equal-budget selectors and M47 paired statistics."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def core_indices(features: np.ndarray, budget: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) < budget:
        raise ValueError("core selection requires a candidate matrix of at least budget rows")
    distance = np.linalg.norm(values - np.mean(values, axis=0), axis=1)
    return np.argsort(distance, kind="stable")[:budget]


def kcenter_indices(features: np.ndarray, budget: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) < budget:
        raise ValueError("k-center selection requires at least budget candidate rows")
    center = np.mean(values, axis=0)
    selected = [int(np.argmax(np.linalg.norm(values - center, axis=1)))]
    minimum_distance = np.linalg.norm(values - values[selected[0]], axis=1)
    while len(selected) < budget:
        minimum_distance[np.asarray(selected)] = -np.inf
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(values - values[next_index], axis=1),
        )
    return np.asarray(selected, dtype=np.int64)


def random_stratified_indices(
    features: np.ndarray, budget: int, seed: int
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) < budget:
        raise ValueError("stratified selection requires at least budget candidate rows")
    modes = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(values)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for mode in (0, 1):
        members = np.flatnonzero(modes == mode)
        take = min(len(members), budget // 2)
        selected.extend(rng.choice(members, size=take, replace=False).tolist())
    if len(selected) < budget:
        remaining = np.setdiff1d(np.arange(len(values)), np.asarray(selected))
        selected.extend(
            rng.choice(remaining, size=budget - len(selected), replace=False).tolist()
        )
    return np.asarray(selected[:budget], dtype=np.int64)


def boundary_inclusive_indices(
    features: np.ndarray,
    novelty_margin: np.ndarray,
    budget: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    margin = np.asarray(novelty_margin, dtype=np.float64)
    if values.ndim != 2 or margin.shape != (len(values),) or len(values) < budget:
        raise ValueError("boundary selection inputs are misaligned")
    selected: list[int] = []
    selected.extend(core_indices(values, max(1, budget // 5)).tolist())
    boundary_order = np.argsort(margin, kind="stable")
    quantile_positions = np.linspace(
        0, len(boundary_order) - 1, num=max(2, budget // 2), dtype=int
    )
    selected.extend(boundary_order[quantile_positions].tolist())
    modes = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(values)
    center = np.mean(values, axis=0)
    distance = np.linalg.norm(values - center, axis=1)
    for mode in (0, 1):
        members = np.flatnonzero(modes == mode)
        mode_order = members[np.argsort(distance[members], kind="stable")]
        selected.extend(mode_order[:3].tolist())
        selected.extend(mode_order[-3:].tolist())
    deduplicated = list(dict.fromkeys(selected))
    if len(deduplicated) < budget:
        for index in kcenter_indices(values, budget):
            if int(index) not in deduplicated:
                deduplicated.append(int(index))
            if len(deduplicated) == budget:
                break
    return np.asarray(deduplicated[:budget], dtype=np.int64)


def paired_bootstrap_interval(
    first: np.ndarray,
    second: np.ndarray,
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> dict[str, float | int]:
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if (
        first_values.ndim != 1
        or first_values.shape != second_values.shape
        or not len(first_values)
    ):
        raise ValueError("paired bootstrap inputs must be equal non-empty vectors")
    if not 0.0 < confidence < 1.0 or n_resamples <= 0:
        raise ValueError("invalid bootstrap policy")
    difference = first_values - second_values
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(difference), size=(n_resamples, len(difference)))
    means = np.mean(difference[samples], axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "difference": float(np.mean(difference)),
        "lower": float(np.quantile(means, tail)),
        "upper": float(np.quantile(means, 1.0 - tail)),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "seed": seed,
    }
