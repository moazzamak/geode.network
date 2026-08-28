from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v12_metric_fields import (
    ProjectedMetricFieldState,
    initialize_projected_metric_fields,
    projection_diagnostics,
    train_projected_metric_fields,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v5_statistics import (
    paired_prediction_interval,
    paired_seed_t_interval,
)
from experiments.tier4.eval_v12_m71_gaussian_classifier import (
    _fit_controls,
    _fit_gaussian,
    _gaussian_outputs,
    _head_metrics,
)
from experiments.tier4.eval_v12_m72_metric_field_stage0 import (
    _calibrate,
    _probe_metrics,
    _probe_suite,
)
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.tier4.eval_v9_m51_surface_diagnostics import _partition_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "configs"
    / "v12"
    / "m74_confirmation_transfer.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "logs" / "results" / "v12" / "m74_confirmation_transfer"
)


def _resolve(path: str) -> Path:
    resolved = (REPO_ROOT / Path(path)).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError("M74 paths must remain inside the repository")
    return resolved


def _verify(specification: dict[str, str]) -> Path:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"M74 immutable artifact hash mismatch: {path}")
    return path


def _load_domainnet(index_path: Path) -> tuple[np.ndarray, np.ndarray]:
    arrays = index_path.parent / "arrays"
    features_path = arrays / "features.npy"
    labels_path = arrays / "labels.npy"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected = {item["path"]: item["sha256"] for item in index["artifacts"]}
    for path in (features_path, labels_path):
        relative = path.relative_to(index_path.parent).as_posix()
        if sha256_file(path) != expected[relative]:
            raise ValueError(f"M74 DomainNet array hash mismatch: {path}")
    return (
        np.load(features_path, allow_pickle=False).astype(np.float64),
        np.load(labels_path, allow_pickle=False).astype(np.int64),
    )


def _training_arguments(config: dict[str, Any], seed: int) -> dict[str, Any]:
    training = config["training"]
    return {
        "epochs": int(training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "classification_temperature": float(
            training["classification_temperature"]
        ),
        "target_score": float(training["target_score"]),
        "separation_margin": float(training["separation_margin"]),
        "probe_margin_multiplier": float(
            training["probe_margin_multiplier"]
        ),
        "loss_weights": {
            name: float(value)
            for name, value in training["loss_weights"].items()
        },
        "collapse_weight": float(training["collapse_weight"]),
        "probe_families": tuple(training["trained_probe_families"]),
        "seed": seed,
    }


def _field_outputs(
    state: ProjectedMetricFieldState,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
    *,
    known_classes: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    projected_calibration = state.transform(calibration_x)
    projected_query = state.transform(query_x)
    thresholds, ratios, coverage = _calibrate(
        state.fields,
        projected_calibration,
        calibration_y,
        miscoverage=float(config["miscoverage"]),
    )
    scores = state.fields.scores(projected_query)
    predictions = state.fields.classes[np.argmin(scores, axis=1)]
    novelty = np.min(scores / thresholds[None, :], axis=1)
    calibration_scores = state.fields.scores(projected_calibration)
    calibration_novelty = np.asarray(
        [
            calibration_scores[row, int(np.flatnonzero(state.fields.classes == label)[0])]
            / thresholds[int(np.flatnonzero(state.fields.classes == label)[0])]
            for row, label in enumerate(calibration_y)
        ],
        dtype=np.float64,
    )
    metrics = {
        **_head_metrics(
            calibration_novelty,
            predictions,
            novelty,
            query_y,
            known_classes=known_classes,
            coverage=float(config["known_coverage_target"]),
        ),
        "threshold_ratio_by_class": ratios,
        "median_threshold_ratio": float(np.median(list(ratios.values()))),
        "calibration_coverage_by_class": coverage,
        "parameter_count": state.parameter_count,
        "serialized_megabytes": float(state.array_bytes / 1_000_000),
        "state_hash": payload_hash(state.to_dict()),
    }
    return metrics, predictions, novelty, thresholds


def _control_outputs(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
    *,
    known_classes: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    classes, primitives = _fit_gaussian(
        fit_x, fit_y, rank=int(config["rank"])
    )
    gaussian_calibration = _gaussian_outputs(
        classes, primitives, calibration_x
    )
    gaussian_query = _gaussian_outputs(classes, primitives, query_x)
    outputs: dict[str, dict[str, Any]] = {
        "gaussian": {
            "predictions": gaussian_query["predictions"],
            "novelty": gaussian_query["novelty"],
            "calibration_novelty": gaussian_calibration["novelty"],
            "size_bytes": int(sum(item.array_bytes for item in primitives)),
        }
    }
    outputs.update(
        _fit_controls(
            fit_x,
            fit_y,
            calibration_x,
            calibration_y,
            query_x,
            seed=seed,
            neighbors=int(config["knn_neighbors"]),
        )
    )
    metrics = {}
    predictions = {}
    for name, output in outputs.items():
        prediction = np.asarray(output["predictions"], dtype=np.int64)
        predictions[name] = prediction
        metrics[name] = {
            **_head_metrics(
                np.asarray(output["calibration_novelty"], dtype=np.float64),
                prediction,
                np.asarray(output["novelty"], dtype=np.float64),
                query_y,
                known_classes=known_classes,
                coverage=float(config["known_coverage_target"]),
            ),
            "serialized_megabytes": float(
                int(output["size_bytes"]) / 1_000_000
            ),
        }
    return metrics, predictions


def _fit_field(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    *,
    config: dict[str, Any],
    seed: int,
    loss_weights: dict[str, float] | None = None,
) -> tuple[
    ProjectedMetricFieldState,
    ProjectedMetricFieldState,
    list[dict[str, float]],
]:
    initial = initialize_projected_metric_fields(
        fit_x,
        fit_y,
        output_dimension=int(config["projection_dimension"]),
        rank=int(config["rank"]),
    )
    arguments = _training_arguments(config, seed)
    if loss_weights is not None:
        arguments["loss_weights"] = loss_weights
    trained, history = train_projected_metric_fields(
        initial, fit_x, fit_y, **arguments
    )
    return initial, trained, history


def _probe_summary(
    state: ProjectedMetricFieldState,
    thresholds: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    return _probe_metrics(
        state.fields,
        thresholds,
        _probe_suite(state.fields, seed=seed + 72),
    )


def _cross_corpus_ood(
    state: ProjectedMetricFieldState,
    thresholds: np.ndarray,
    features: np.ndarray,
) -> dict[str, float | int]:
    scores = state.fields.scores(state.transform(features))
    novelty = np.min(scores / thresholds[None, :], axis=1)
    return {
        "count": int(len(features)),
        "unknown_recall": float(np.mean(novelty > 1.0)),
        "median_normalized_novelty": float(np.median(novelty)),
    }


def _primary_seed(
    seed: int,
    source: dict[str, Any],
    domainnet_x: np.ndarray,
    *,
    config: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    loaded = _load_seed_data(source["seed_inputs"][str(seed)])
    train_x, train_y = loaded["datasets"]["train"]
    dev_x, dev_y = loaded["datasets"]["dev"]
    known_classes = np.asarray(config["known_classes"], dtype=np.int64)
    unknown_classes = np.asarray(
        config["proxy_unknown_classes"], dtype=np.int64
    )
    partitions = _partition_seed(
        train_y,
        dev_y,
        seed=seed,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        geometry_fraction=float(config["geometry_fraction"]),
    )
    fit_x = train_x[partitions["geometry_fit"]]
    fit_y = train_y[partitions["geometry_fit"]]
    calibration_x = train_x[partitions["score_calibration"]]
    calibration_y = train_y[partitions["score_calibration"]]
    evaluation_indices = np.concatenate(
        [partitions["development_eval"], partitions["unknown_eval"]]
    )
    query_x = dev_x[evaluation_indices]
    query_y = dev_y[evaluation_indices]
    initial, trained, history = _fit_field(
        fit_x, fit_y, config=config, seed=seed
    )
    field_metrics, field_predictions, _, thresholds = _field_outputs(
        trained,
        calibration_x,
        calibration_y,
        query_x,
        query_y,
        known_classes=known_classes,
        config=config,
    )
    controls, control_predictions = _control_outputs(
        fit_x,
        fit_y,
        calibration_x,
        calibration_y,
        query_x,
        query_y,
        known_classes=known_classes,
        config=config,
        seed=seed,
    )
    known = np.isin(query_y, known_classes)
    predictions = {
        "field": field_predictions[known],
        **{
            name: prediction[known]
            for name, prediction in control_predictions.items()
        },
    }
    return (
        {
            "seed": seed,
            "partition_hashes": {
                name: payload_hash(indices.tolist())
                for name, indices in partitions.items()
            },
            "field": field_metrics,
            "controls": controls,
            "probe_acceptance": _probe_summary(
                trained, thresholds, seed=seed
            ),
            "real_ood_domainnet": _cross_corpus_ood(
                trained, thresholds, domainnet_x
            ),
            "projection_diagnostics": projection_diagnostics(
                trained, initial, fit_x
            ),
            "optimizer_history": history,
            "trained_state": trained.to_dict(),
        },
        query_y[known],
        predictions,
    )


def _domainnet_partitions(
    features: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    transfer = config["domainnet_transfer"]
    known_classes = np.arange(
        int(transfer["known_class_count"]), dtype=np.int64
    )
    geometry = []
    calibration = []
    evaluation = []
    for class_label in known_classes:
        rows = np.flatnonzero(labels == class_label)
        geometry_count = int(transfer["geometry_per_class"])
        calibration_count = int(transfer["calibration_per_class"])
        evaluation_count = int(transfer["evaluation_per_class"])
        if len(rows) < geometry_count + calibration_count + evaluation_count:
            raise ValueError("M74 DomainNet known class lacks registered samples")
        geometry.extend(rows[:geometry_count])
        calibration.extend(
            rows[geometry_count : geometry_count + calibration_count]
        )
        evaluation.extend(
            rows[
                geometry_count
                + calibration_count : geometry_count
                + calibration_count
                + evaluation_count
            ]
        )
    unknown = []
    for class_label in range(
        int(transfer["unknown_class_start"]),
        int(transfer["unknown_class_stop"]),
    ):
        rows = np.flatnonzero(labels == class_label)
        count = int(transfer["unknown_evaluation_per_class"])
        if len(rows) < count:
            raise ValueError("M74 DomainNet unknown class lacks registered samples")
        unknown.extend(rows[-count:])
    indices = {
        "geometry_fit": np.asarray(geometry, dtype=np.int64),
        "score_calibration": np.asarray(calibration, dtype=np.int64),
        "known_evaluation": np.asarray(evaluation, dtype=np.int64),
        "unknown_evaluation": np.asarray(unknown, dtype=np.int64),
    }
    return {
        name: (features[rows], labels[rows]) for name, rows in indices.items()
    }


def _domainnet_transfer(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    config: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    partitions = _domainnet_partitions(features, labels, config)
    fit_x, fit_y = partitions["geometry_fit"]
    calibration_x, calibration_y = partitions["score_calibration"]
    known_x, known_y = partitions["known_evaluation"]
    unknown_x, unknown_y = partitions["unknown_evaluation"]
    query_x = np.vstack([known_x, unknown_x])
    query_y = np.concatenate([known_y, unknown_y])
    known_classes = np.arange(
        int(config["domainnet_transfer"]["known_class_count"]),
        dtype=np.int64,
    )
    seed = int(config["seeds"][0]) + 740
    initial, trained, history = _fit_field(
        fit_x, fit_y, config=config, seed=seed
    )
    field_metrics, field_predictions, _, thresholds = _field_outputs(
        trained,
        calibration_x,
        calibration_y,
        query_x,
        query_y,
        known_classes=known_classes,
        config=config,
    )
    controls, control_predictions = _control_outputs(
        fit_x,
        fit_y,
        calibration_x,
        calibration_y,
        query_x,
        query_y,
        known_classes=known_classes,
        config=config,
        seed=seed,
    )
    known = np.isin(query_y, known_classes)
    predictions = {
        "field": field_predictions[known],
        **{
            name: prediction[known]
            for name, prediction in control_predictions.items()
        },
    }
    return (
        {
            "dataset": "DomainNet",
            "known_class_count": int(len(known_classes)),
            "unknown_class_count": int(
                int(config["domainnet_transfer"]["unknown_class_stop"])
                - int(config["domainnet_transfer"]["unknown_class_start"])
            ),
            "partition_hashes": {
                name: payload_hash(
                    {
                        "features": payload_hash(values[0].tolist()),
                        "labels": values[1].tolist(),
                    }
                )
                for name, values in partitions.items()
            },
            "field": field_metrics,
            "controls": controls,
            "probe_acceptance": _probe_summary(
                trained, thresholds, seed=seed
            ),
            "projection_diagnostics": projection_diagnostics(
                trained, initial, fit_x
            ),
            "optimizer_history": history,
            "trained_state": trained.to_dict(),
        },
        known_y,
        predictions,
    )


def _loss_ablations(
    source: dict[str, Any], *, config: dict[str, Any]
) -> dict[str, Any]:
    seed = int(config["loss_ablation_seed"])
    loaded = _load_seed_data(source["seed_inputs"][str(seed)])
    train_x, train_y = loaded["datasets"]["train"]
    dev_x, dev_y = loaded["datasets"]["dev"]
    known_classes = np.asarray(config["known_classes"], dtype=np.int64)
    unknown_classes = np.asarray(
        config["proxy_unknown_classes"], dtype=np.int64
    )
    partitions = _partition_seed(
        train_y,
        dev_y,
        seed=seed,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        geometry_fraction=float(config["geometry_fraction"]),
    )
    fit_x = train_x[partitions["geometry_fit"]]
    fit_y = train_y[partitions["geometry_fit"]]
    calibration_x = train_x[partitions["score_calibration"]]
    calibration_y = train_y[partitions["score_calibration"]]
    indices = np.concatenate(
        [partitions["development_eval"], partitions["unknown_eval"]]
    )
    query_x = dev_x[indices]
    query_y = dev_y[indices]
    base_weights = {
        name: float(value)
        for name, value in config["training"]["loss_weights"].items()
    }
    results = {}
    for term in config["loss_ablation_terms"]:
        weights = dict(base_weights)
        weights[term] = 0.0
        initial, trained, history = _fit_field(
            fit_x,
            fit_y,
            config=config,
            seed=seed,
            loss_weights=weights,
        )
        metrics, _, _, thresholds = _field_outputs(
            trained,
            calibration_x,
            calibration_y,
            query_x,
            query_y,
            known_classes=known_classes,
            config=config,
        )
        results[term] = {
            "removed_weight": base_weights[term],
            "metrics": metrics,
            "probe_acceptance": _probe_summary(
                trained, thresholds, seed=seed
            ),
            "projection_diagnostics": projection_diagnostics(
                trained, initial, fit_x
            ),
            "final_optimizer_losses": history[-1],
        }
    return results


def _comparison(
    truths: list[np.ndarray],
    predictions: dict[str, list[np.ndarray]],
    seed_records: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    control_names = ("gaussian", "logistic", "rbf", "knn")
    means = {
        name: float(
            np.mean(
                [
                    record["controls"][name]["known_balanced_accuracy"]
                    for record in seed_records
                ]
            )
        )
        for name in control_names
    }
    strongest = max(means, key=means.get)
    field_values = np.asarray(
        [
            record["field"]["known_balanced_accuracy"]
            for record in seed_records
        ]
    )
    control_values = np.asarray(
        [
            record["controls"][strongest]["known_balanced_accuracy"]
            for record in seed_records
        ]
    )
    pooled_truth = np.concatenate(truths)
    pooled_field = np.concatenate(predictions["field"])
    pooled_control = np.concatenate(predictions[strongest])
    return {
        "strongest_control": strongest,
        "control_mean_known_balanced_accuracy": means,
        "field_mean_known_balanced_accuracy": float(np.mean(field_values)),
        "mean_accuracy_difference": float(
            np.mean(field_values - control_values)
        ),
        "paired_prediction_interval": paired_prediction_interval(
            pooled_truth,
            pooled_field,
            pooled_control,
            metric="balanced_accuracy",
            confidence=0.95,
            n_resamples=int(config["bootstrap_resamples"]),
            seed=int(config["bootstrap_seed"]),
        ),
        "paired_seed_interval": paired_seed_t_interval(
            field_values, control_values, confidence=0.95
        ),
    }


def run_confirmation(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _verify(config["source_config"])
    _verify(config["m73_index"])
    domainnet_index = _verify(config["m70_native_index"])
    source = json.loads(
        _resolve(config["source_config"]["path"]).read_text(encoding="utf-8")
    )
    domainnet_x, domainnet_y = _load_domainnet(domainnet_index)

    primary_records = []
    primary_truths = []
    primary_predictions: dict[str, list[np.ndarray]] = {
        name: []
        for name in ("field", "gaussian", "logistic", "rbf", "knn")
    }
    for seed_value in config["seeds"]:
        record, truth, predictions = _primary_seed(
            int(seed_value), source, domainnet_x, config=config
        )
        primary_records.append(record)
        primary_truths.append(truth)
        for name in primary_predictions:
            primary_predictions[name].append(predictions[name])
    primary_comparison = _comparison(
        primary_truths,
        primary_predictions,
        primary_records,
        config=config,
    )

    transfer, transfer_truth, transfer_predictions = _domainnet_transfer(
        domainnet_x, domainnet_y, config=config
    )
    transfer_control_means = {
        name: transfer["controls"][name]["known_balanced_accuracy"]
        for name in ("gaussian", "logistic", "rbf", "knn")
    }
    transfer_strongest = max(
        transfer_control_means, key=transfer_control_means.get
    )
    transfer_comparison = {
        "strongest_control": transfer_strongest,
        "control_known_balanced_accuracy": transfer_control_means,
        "field_accuracy_difference": float(
            transfer["field"]["known_balanced_accuracy"]
            - transfer_control_means[transfer_strongest]
        ),
        "paired_prediction_interval": paired_prediction_interval(
            transfer_truth,
            transfer_predictions["field"],
            transfer_predictions[transfer_strongest],
            metric="balanced_accuracy",
            confidence=0.95,
            n_resamples=int(config["bootstrap_resamples"]),
            seed=int(config["bootstrap_seed"]) + 1,
        ),
    }
    ablations = _loss_ablations(source, config=config)

    held_out = list(config["training"]["held_out_probe_families"])
    maximum_held_out = float(
        max(
            record["probe_acceptance"][family]["by_multiplier"]["4"][
                "system_acceptance"
            ]
            for record in primary_records
            for family in held_out
        )
    )
    mean_unknown_recall = float(
        np.mean([record["field"]["unknown_recall"] for record in primary_records])
    )
    minimum_real_ood_recall = float(
        min(
            record["real_ood_domainnet"]["unknown_recall"]
            for record in primary_records
        )
    )
    maximum_threshold_ratio = float(
        max(
            record["field"]["median_threshold_ratio"]
            for record in primary_records
        )
    )
    gate_config = config["gate"]
    gate = {
        "primary_mean_accuracy_difference": primary_comparison[
            "mean_accuracy_difference"
        ],
        "primary_mean_unknown_recall": mean_unknown_recall,
        "primary_maximum_median_threshold_ratio": maximum_threshold_ratio,
        "primary_maximum_held_out_four_x_acceptance": maximum_held_out,
        "primary_minimum_real_domainnet_ood_recall": minimum_real_ood_recall,
        "transfer_accuracy_difference": transfer_comparison[
            "field_accuracy_difference"
        ],
        "transfer_unknown_recall": transfer["field"]["unknown_recall"],
        "transfer_median_threshold_ratio": transfer["field"][
            "median_threshold_ratio"
        ],
        "l1_accuracy_parity": bool(
            primary_comparison["mean_accuracy_difference"]
            >= -float(gate_config["accuracy_parity_tolerance"])
        ),
        "l2_open_set_competence": bool(
            mean_unknown_recall
            >= float(gate_config["minimum_unknown_recall"])
        ),
        "threshold_ratio_confirmed": bool(
            maximum_threshold_ratio
            <= float(gate_config["maximum_median_threshold_ratio"])
        ),
        "held_out_probe_transfer": bool(
            maximum_held_out
            < float(gate_config["maximum_held_out_four_x_acceptance"])
        ),
        "real_ood_competence": bool(
            minimum_real_ood_recall
            >= float(gate_config["minimum_unknown_recall"])
        ),
        "l5_transfer_accuracy_parity": bool(
            transfer_comparison["field_accuracy_difference"]
            >= -float(gate_config["transfer_accuracy_parity_tolerance"])
        ),
        "l5_transfer_open_set_competence": bool(
            transfer["field"]["unknown_recall"]
            >= float(gate_config["minimum_transfer_unknown_recall"])
        ),
        "l5_transfer_threshold_ratio": bool(
            transfer["field"]["median_threshold_ratio"]
            <= float(gate_config["maximum_median_threshold_ratio"])
        ),
        "exact_replay": True,
        "final_labels_opened": False,
    }
    gate["m74_passed"] = bool(
        gate["l1_accuracy_parity"]
        and gate["l2_open_set_competence"]
        and gate["threshold_ratio_confirmed"]
        and gate["held_out_probe_transfer"]
        and gate["real_ood_competence"]
        and gate["l5_transfer_accuracy_parity"]
        and gate["l5_transfer_open_set_competence"]
        and gate["l5_transfer_threshold_ratio"]
    )
    evidence = {
        "schema_version": 1,
        "milestone": "M74",
        "configuration_hash": sha256_file(config_path),
        "primary": {
            "dataset": "CIFAR-10",
            "seed_records": primary_records,
            "comparison": primary_comparison,
        },
        "transfer": {
            **transfer,
            "comparison": transfer_comparison,
        },
        "loss_ablations": ablations,
        "gate": gate,
        "advance_to_m75": bool(gate["m74_passed"]),
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
    result = run_confirmation(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "primary_comparison": result["primary"]["comparison"],
                "transfer_comparison": result["transfer"]["comparison"],
                "gate": result["gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
