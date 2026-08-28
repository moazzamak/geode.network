from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from experiments.common.v11_directional_envelope import split_conformal_quantile
from experiments.common.v12_metric_fields import (
    MetricFieldState,
    initialize_metric_fields,
    train_metric_fields,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.tier4.eval_v9_m51_surface_diagnostics import _partition_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v12" / "m72_metric_field_stage0.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v12" / "m72_metric_field_stage0"


def _resolve(path: str) -> Path:
    resolved = (REPO_ROOT / Path(path)).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError("M72 paths must remain inside the repository")
    return resolved


def _verify(specification: dict[str, str]) -> Path:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"M72 immutable artifact hash mismatch: {path}")
    return path


def _probe_suite(
    state: MetricFieldState,
    *,
    seed: int,
    replicates: int = 8,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    points: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "axis_tangent",
            "corner_tangent",
            "normal",
            "mixed",
            "masking",
            "random_direction",
            "cross_class_bridge",
        )
    }
    owners: dict[str, list[int]] = {name: [] for name in points}
    multipliers: dict[str, list[float]] = {name: [] for name in points}

    def add(name: str, owner: int, multiplier: float, point: np.ndarray) -> None:
        points[name].append(point)
        owners[name].append(owner)
        multipliers[name].append(multiplier)

    for owner, center in enumerate(state.centers):
        basis = state.bases[owner]
        tangent = state.tangent_scales[owner]
        normal = rng.normal(size=state.dimension)
        normal -= basis @ (basis.T @ normal)
        normal /= np.linalg.norm(normal)
        for multiplier in (4.0, 8.0):
            for _ in range(replicates * 2 * state.rank):
                direction = rng.normal(size=state.rank)
                direction /= np.max(np.abs(direction) / tangent)
                add(
                    "axis_tangent",
                    owner,
                    multiplier,
                    center + multiplier * (basis @ direction),
                )
        for _ in range(replicates):
            signs = rng.choice([-1.0, 1.0], size=state.rank)
            add(
                "corner_tangent",
                owner,
                4.0,
                center + basis @ (4.0 * signs * tangent),
            )
            add(
                "normal",
                owner,
                4.0,
                center + 4.0 * state.residual_scales[owner] * normal,
            )
            add(
                "mixed",
                owner,
                4.0,
                center
                + 4.0 * tangent[0] * basis[:, 0]
                + state.residual_scales[owner] * normal,
            )
            random_direction = rng.normal(size=state.dimension)
            random_direction /= np.linalg.norm(random_direction)
            add(
                "random_direction",
                owner,
                8.0,
                center
                + 8.0 * np.linalg.norm(tangent) * random_direction,
            )
        other = [index for index in range(len(state.classes)) if index != owner]
        nearest = min(
            other,
            key=lambda index: float(
                np.linalg.norm(state.centers[index] - center)
            ),
        )
        toward = state.centers[nearest] - center
        coordinates = basis.T @ toward
        axis = int(np.argmax(np.abs(coordinates)))
        sign = float(np.sign(coordinates[axis])) or 1.0
        for _ in range(replicates):
            add(
                "masking",
                owner,
                4.0,
                center + sign * 4.0 * tangent[axis] * basis[:, axis],
            )
            add(
                "cross_class_bridge",
                owner,
                0.5,
                0.5 * (center + state.centers[nearest]),
            )
    return {
        name: (
            np.vstack(values),
            np.asarray(owners[name], dtype=np.int64),
            np.asarray(multipliers[name], dtype=np.float64),
        )
        for name, values in points.items()
    }


def _calibrate(
    state: MetricFieldState,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    *,
    miscoverage: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    scores = state.scores(calibration_x)
    thresholds = []
    ratios = {}
    coverage = {}
    for column, class_label in enumerate(state.classes):
        own = scores[calibration_y == class_label, column]
        threshold = split_conformal_quantile(own, miscoverage=miscoverage)
        thresholds.append(threshold)
        ratios[str(class_label)] = float(threshold / np.median(own))
        coverage[str(class_label)] = float(np.mean(own <= threshold))
    return np.asarray(thresholds), ratios, coverage


def _probe_metrics(
    state: MetricFieldState,
    thresholds: np.ndarray,
    probes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    result = {}
    for name, (points, owners, multipliers) in probes.items():
        scores = state.scores(points)
        source = np.asarray(
            [
                scores[row, owner] <= thresholds[owner]
                for row, owner in enumerate(owners)
            ]
        )
        system = np.any(scores <= thresholds[None, :], axis=1)
        by_multiplier = {}
        for multiplier in np.unique(multipliers):
            mask = multipliers == multiplier
            by_multiplier[str(int(multiplier) if multiplier.is_integer() else multiplier)] = {
                "count": int(np.sum(mask)),
                "source_acceptance": float(np.mean(source[mask])),
                "system_acceptance": float(np.mean(system[mask])),
            }
        result[name] = {
            "count": int(len(points)),
            "source_acceptance": float(np.mean(source)),
            "system_acceptance": float(np.mean(system)),
            "by_multiplier": by_multiplier,
        }
    return result


def _evaluate_state(
    state: MetricFieldState,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    development_x: np.ndarray,
    development_y: np.ndarray,
    unknown_x: np.ndarray,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds, ratios, coverage = _calibrate(
        state,
        calibration_x,
        calibration_y,
        miscoverage=float(config["miscoverage"]),
    )
    probes = _probe_metrics(
        state,
        thresholds,
        _probe_suite(state, seed=int(config["seed"]) + 72),
    )
    development_scores = state.scores(development_x)
    development_predictions = state.classes[
        np.argmin(development_scores, axis=1)
    ]
    known_novelty = np.min(
        development_scores / thresholds[None, :], axis=1
    )
    unknown_scores = state.scores(unknown_x)
    unknown_novelty = np.min(unknown_scores / thresholds[None, :], axis=1)
    targets = np.concatenate(
        [np.zeros(len(known_novelty)), np.ones(len(unknown_novelty))]
    )
    novelty = np.concatenate([known_novelty, unknown_novelty])
    return {
        "known_balanced_accuracy": float(
            balanced_accuracy_score(development_y, development_predictions)
        ),
        "unknown_recall": float(np.mean(unknown_novelty > 1.0)),
        "known_coverage": float(np.mean(known_novelty <= 1.0)),
        "auroc": float(roc_auc_score(targets, novelty)),
        "thresholds": thresholds.tolist(),
        "threshold_ratio_by_class": ratios,
        "median_threshold_ratio": float(np.median(list(ratios.values()))),
        "coverage_by_class": coverage,
        "probe_acceptance": probes,
        "gradient_norm": {
            "data_median": float(
                np.median(
                    np.linalg.norm(
                        state.score_gradients(calibration_x), axis=2
                    )
                )
            ),
            "data_mean_absolute_eikonal_error": float(
                np.mean(
                    np.abs(
                        np.linalg.norm(
                            state.score_gradients(calibration_x), axis=2
                        )
                        - 1.0
                    )
                )
            ),
        },
        "state_hash": payload_hash(state.to_dict()),
        "parameter_count": state.parameter_count,
        "serialized_megabytes": float(state.array_bytes / 1_000_000),
    }


def run_stage0(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _verify(config["source_config"])
    _verify(config["m70_index"])
    _verify(config["m71_index"])
    source = json.loads(_resolve(config["source_config"]["path"]).read_text())
    seed = int(config["seed"])
    loaded = _load_seed_data(source["seed_inputs"][str(seed)])
    train_x, train_y = loaded["datasets"]["train"]
    dev_x, dev_y = loaded["datasets"]["dev"]
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
    development_x = dev_x[partitions["development_eval"]]
    development_y = dev_y[partitions["development_eval"]]
    unknown_x = dev_x[partitions["unknown_eval"]]

    initial = initialize_metric_fields(
        geometry_x, geometry_y, rank=int(config["rank"])
    )
    training = config["training"]
    trained, history = train_metric_fields(
        initial,
        geometry_x,
        geometry_y,
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        classification_temperature=float(training["classification_temperature"]),
        target_score=float(training["target_score"]),
        separation_margin=float(training["separation_margin"]),
        probe_margin=float(training["probe_margin"]),
        loss_weights={
            name: float(value)
            for name, value in training["loss_weights"].items()
        },
        probe_families=tuple(training["trained_probe_families"]),
        seed=seed,
    )
    initial_metrics = _evaluate_state(
        initial,
        calibration_x,
        calibration_y,
        development_x,
        development_y,
        unknown_x,
        config=config,
    )
    trained_metrics = _evaluate_state(
        trained,
        calibration_x,
        calibration_y,
        development_x,
        development_y,
        unknown_x,
        config=config,
    )
    held_out = training["held_out_probe_families"]
    held_out_four_x = max(
        trained_metrics["probe_acceptance"][family]["by_multiplier"]["4"][
            "system_acceptance"
        ]
        for family in held_out
    )
    gate_config = config["gate"]
    operands = {
        "median_threshold_ratio": trained_metrics["median_threshold_ratio"],
        "held_out_four_x_acceptance": float(held_out_four_x),
        "known_accuracy_regression": float(
            float(gate_config["m71_seed11_gaussian_accuracy"])
            - trained_metrics["known_balanced_accuracy"]
        ),
    }
    gate = {
        **operands,
        "threshold_ratio_passed": bool(
            operands["median_threshold_ratio"]
            <= float(gate_config["maximum_median_threshold_ratio"])
        ),
        "held_out_probes_passed": bool(
            operands["held_out_four_x_acceptance"]
            < float(gate_config["maximum_held_out_four_x_acceptance"])
        ),
        "accuracy_non_regression_passed": bool(
            operands["known_accuracy_regression"]
            <= float(gate_config["maximum_accuracy_regression"])
        ),
        "exact_replay": True,
        "final_labels_opened": False,
    }
    gate["m72_passed"] = bool(
        gate["threshold_ratio_passed"]
        and gate["held_out_probes_passed"]
        and gate["accuracy_non_regression_passed"]
    )
    evidence = {
        "schema_version": 1,
        "milestone": "M72",
        "configuration_hash": sha256_file(config_path),
        "partition_hashes": {
            name: payload_hash(indices.tolist()) for name, indices in partitions.items()
        },
        "trained_probe_families": training["trained_probe_families"],
        "held_out_probe_families": held_out,
        "initial_metrics": initial_metrics,
        "trained_metrics": trained_metrics,
        "optimizer_history": history,
        "trained_state": trained.to_dict(),
        "gate": gate,
        "advance_to_m74": bool(gate["m72_passed"]),
        "open_m73": bool(not gate["m72_passed"]),
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
    result = run_stage0(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "initial": {
                    key: result["initial_metrics"][key]
                    for key in (
                        "known_balanced_accuracy",
                        "median_threshold_ratio",
                    )
                },
                "trained": {
                    key: result["trained_metrics"][key]
                    for key in (
                        "known_balanced_accuracy",
                        "known_coverage",
                        "unknown_recall",
                        "median_threshold_ratio",
                    )
                },
                "gate": result["gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
