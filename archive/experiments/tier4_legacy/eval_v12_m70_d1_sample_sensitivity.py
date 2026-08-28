from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v11_directional_envelope import (
    DirectionalTube,
    class_score_matrix,
    normalize_directions,
    spherical_exp_map,
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
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.tier4.eval_v9_m51_surface_diagnostics import _partition_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "configs"
    / "v12"
    / "m70_d1_sample_sensitivity.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "logs" / "results" / "v12" / "m70_d1_sample_sensitivity"
)


def _resolve(path: str) -> Path:
    resolved = (REPO_ROOT / Path(path)).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError("M70-D1 paths must remain inside the repository")
    return resolved


def _verify(specification: dict[str, str]) -> Path:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"M70-D1 immutable artifact hash mismatch: {path}")
    return path


def _nested_indices(
    count: int, *, seed: int
) -> np.ndarray:
    return np.random.default_rng(seed).permutation(count)


def _bootstrap_thresholds(
    normalized_scores: np.ndarray,
    *,
    sample_sizes: list[int],
    miscoverage: float,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(normalized_scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("M70-D1 normalized scores must be a finite vector")
    rng = np.random.default_rng(seed)
    order = _nested_indices(len(values), seed=seed)
    result = {}
    for sample_size in sample_sizes:
        extrapolated = sample_size > len(values)
        if extrapolated:
            point_sample = rng.choice(values, size=sample_size, replace=True)
        else:
            point_sample = values[order[:sample_size]]
        point = split_conformal_quantile(
            point_sample, miscoverage=miscoverage
        )
        estimates = np.empty(resamples, dtype=np.float64)
        for index in range(resamples):
            sample = rng.choice(values, size=sample_size, replace=True)
            estimates[index] = split_conformal_quantile(
                sample, miscoverage=miscoverage
            )
        result[str(sample_size)] = {
            "threshold_ratio": float(point),
            "bootstrap_lower": float(np.quantile(estimates, 0.025)),
            "bootstrap_upper": float(np.quantile(estimates, 0.975)),
            "empirical_pool_count": int(len(values)),
            "extrapolated_with_replacement": bool(extrapolated),
        }
    return result


def _expanded_tangent_probes(
    tubes: list[DirectionalTube],
    *,
    multiplier: float,
    replicates_per_axis_sign: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    points = []
    owners = []
    for owner, tube in enumerate(tubes):
        count = 2 * tube.rank * replicates_per_axis_sign
        for _ in range(count):
            direction = rng.normal(size=tube.rank)
            direction /= np.max(
                np.abs(direction) / tube.tangent_extents
            )
            vector = tube.basis @ (multiplier * direction)
            points.append(spherical_exp_map(tube.center, vector[None, :])[0])
            owners.append(owner)
    return np.vstack(points), np.asarray(owners, dtype=np.int64)


def _probe_acceptance(
    tubes: list[DirectionalTube],
    thresholds: np.ndarray,
    *,
    replicates_per_axis_sign: int,
    seed: int,
) -> dict[str, Any]:
    classes = np.asarray([tube.class_label for tube in tubes], dtype=np.int64)
    result = {}
    for multiplier in (4.0, 8.0):
        points, owners = _expanded_tangent_probes(
            tubes,
            multiplier=multiplier,
            replicates_per_axis_sign=replicates_per_axis_sign,
            seed=seed + int(multiplier * 100),
        )
        scores, observed_classes = class_score_matrix(tubes, points)
        if not np.array_equal(classes, observed_classes):
            raise RuntimeError("M70-D1 class order changed during probe scoring")
        source = np.asarray(
            [
                scores[row, owner] <= thresholds[owner]
                for row, owner in enumerate(owners)
            ]
        )
        system = np.any(scores <= thresholds[None, :], axis=1)
        result[str(int(multiplier))] = {
            "probe_count": int(len(points)),
            "per_patch_count": int(len(points) // len(tubes)),
            "source_acceptance": float(np.mean(source)),
            "system_acceptance": float(np.mean(system)),
        }
    return result


def _penetration_rate(
    tubes: list[DirectionalTube],
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
) -> dict[str, float]:
    rates = {}
    directions = normalize_directions(calibration_x)
    for tube in tubes:
        other = directions[calibration_y != tube.class_label]
        logarithms = spherical_log_map(tube.center, other)
        coordinates = np.abs(logarithms @ tube.basis)
        rates[str(tube.class_label)] = float(
            np.mean(np.all(coordinates <= tube.tangent_extents[None, :], axis=1))
        )
    return rates


def _fit_seed(
    seed: int, source: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    loaded = _load_seed_data(source["seed_inputs"][str(seed)])
    train_x, train_y = loaded["datasets"]["train"]
    _, dev_y = loaded["datasets"]["dev"]
    known_classes = np.asarray(config["known_classes"], dtype=np.int64)
    unknown_classes = np.asarray(config["proxy_unknown_classes"], dtype=np.int64)
    partitions = _partition_seed(
        train_y,
        dev_y,
        seed=seed,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        geometry_fraction=float(config["geometry_fraction"]),
    )
    geometry_x = train_x[partitions["geometry_fit"]]
    geometry_y = train_y[partitions["geometry_fit"]]
    calibration_x = train_x[partitions["score_calibration"]]
    calibration_y = train_y[partitions["score_calibration"]]
    states, _ = _geometry_states(
        geometry_x,
        geometry_y,
        known_classes=known_classes,
        patch_count=1,
        seed=seed,
        maximum_rank=int(config["rank"]),
    )
    tubes = []
    normalized_scores: dict[int, np.ndarray] = {}
    raw_thresholds = []
    for state in states:
        class_label = int(state["class_label"])
        own_geometry = normalize_directions(
            geometry_x[geometry_y == class_label]
        )
        tube = _quantile_tube(
            state,
            own_geometry,
            rank=int(config["rank"]),
            penalty_weight=float(config["penalty_weight"]),
        )
        tubes.append(tube)
        own_calibration = calibration_x[calibration_y == class_label]
        scores = tube.score(own_calibration)
        median = float(np.median(scores))
        if median <= 0.0:
            raise ValueError("M70-D1 score median must be positive")
        normalized_scores[class_label] = scores / median
        raw_thresholds.append(
            split_conformal_quantile(
                scores, miscoverage=float(config["miscoverage"])
            )
        )
    thresholds = np.asarray(raw_thresholds, dtype=np.float64)
    record = {
        "seed": seed,
        "partition_hashes": {
            name: payload_hash(indices.tolist()) for name, indices in partitions.items()
        },
        "calibration_count_per_class": {
            str(class_label): int(np.sum(calibration_y == class_label))
            for class_label in known_classes
        },
        "raw_thresholds": thresholds.tolist(),
        "threshold_ratios": {
            str(class_label): float(
                split_conformal_quantile(
                    normalized_scores[int(class_label)],
                    miscoverage=float(config["miscoverage"]),
                )
            )
            for class_label in known_classes
        },
        "probe_acceptance": _probe_acceptance(
            tubes,
            thresholds,
            replicates_per_axis_sign=int(
                config["probe_replicates_per_axis_sign"]
            ),
            seed=seed,
        ),
        "other_class_penetration_at_geometry_q95_extent": _penetration_rate(
            tubes, calibration_x, calibration_y
        ),
        "geometry_replay_hash": payload_hash(
            [tube.to_dict() for tube in tubes]
        ),
    }
    return record, normalized_scores


def run_diagnostic(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _verify(config["source_config"])
    _verify(config["v11_parent_index"])
    source = json.loads(_resolve(config["source_config"]["path"]).read_text())
    seed_results = []
    pooled: dict[int, list[np.ndarray]] = {
        int(class_label): [] for class_label in config["known_classes"]
    }
    for seed in config["seeds"]:
        record, scores = _fit_seed(int(seed), source, config)
        seed_results.append(record)
        for class_label, values in scores.items():
            pooled[class_label].append(values)
    sensitivity = {
        str(class_label): _bootstrap_thresholds(
            np.concatenate(values),
            sample_sizes=[int(value) for value in config["sample_sizes"]],
            miscoverage=float(config["miscoverage"]),
            resamples=int(config["bootstrap_resamples"]),
            seed=int(config["bootstrap_seed"]) + class_label,
        )
        for class_label, values in pooled.items()
    }
    medians = {
        str(sample_size): float(
            np.median(
                [
                    sensitivity[str(class_label)][str(sample_size)][
                        "threshold_ratio"
                    ]
                    for class_label in config["known_classes"]
                ]
            )
        )
        for sample_size in config["sample_sizes"]
    }
    evidence = {
        "schema_version": 1,
        "milestone": "M70-D1",
        "configuration_hash": sha256_file(config_path),
        "ratio_definition": (
            "split-conformal own-class threshold divided by the own-class "
            "calibration-score median for the same frozen score field"
        ),
        "pooling_protocol": (
            "200 disjoint calibration scores per class and seed; 600 unique "
            "scores per class pooled after within-seed median normalization"
        ),
        "n800_protocol_caveat": (
            "n=800 exceeds the 600 unique disjoint frozen scores per class and "
            "is therefore an explicitly labeled empirical-bootstrap extrapolation"
        ),
        "seed_results": seed_results,
        "sample_size_sensitivity": sensitivity,
        "median_threshold_ratio_by_sample_size": medians,
        "gate": {
            "all_registered_sample_sizes_reported": set(
                int(value) for value in medians
            )
            == set(int(value) for value in config["sample_sizes"]),
            "probe_count_increased_at_least_eight_x": all(
                record["probe_acceptance"]["4"]["per_patch_count"]
                >= 2 * int(config["rank"]) * 8
                for record in seed_results
            ),
            "n800_marked_extrapolated": all(
                sensitivity[str(class_label)]["800"][
                    "extrapolated_with_replacement"
                ]
                for class_label in config["known_classes"]
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
                "median_threshold_ratio_by_sample_size": result[
                    "median_threshold_ratio_by_sample_size"
                ],
                "gate": result["gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
