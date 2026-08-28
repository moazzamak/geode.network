from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from experiments.common.v10_manifold_support import (
    DimensionlessTube,
    SafetyPenaltySelectionError,
    fit_dimensionless_tube,
    generate_axis_tangent_probes,
    generate_safety_probes,
    probe_acceptance,
    select_smallest_safety_penalty,
    system_scores,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v7_acceptance import AcceptanceOutput
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.tier4.eval_v9_m51_surface_diagnostics import _partition_seed
from src.subspace_primitive import SubspacePrimitive


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v10" / "m58_affine_screen.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v10" / "m58_affine_screen"


def _resolve(path: str) -> Path:
    return REPO_ROOT / Path(path)


def _verify(specification: dict[str, str]) -> Path:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"M58 immutable artifact hash mismatch: {path}")
    return path


def _tube_output(
    tubes: Sequence[DimensionlessTube],
    query: np.ndarray,
    *,
    bounded: bool,
) -> AcceptanceOutput:
    if bounded:
        scores = np.column_stack([tube.score(query) for tube in tubes])
    else:
        scores = np.column_stack([tube.score_terms(query)[0] for tube in tubes])
    classes = np.asarray([tube.class_label for tube in tubes], dtype=np.int64)
    return AcceptanceOutput(
        predictions=classes[np.argmin(scores, axis=1)],
        novelty=np.min(scores, axis=1),
        state_hash=payload_hash(
            {
                "bounded": bounded,
                "tubes": [tube.to_dict() for tube in tubes],
            }
        ),
        size_bytes=sum(tube.parameter_count for tube in tubes) * 8,
    )


def _metrics(
    output: AcceptanceOutput,
    *,
    calibration_count: int,
    evaluation_labels: np.ndarray,
    unknown_classes: np.ndarray,
    threshold: float | None = None,
    known_coverage_target: float = 0.92,
) -> dict[str, Any]:
    calibration_novelty = output.novelty[:calibration_count]
    evaluation_novelty = output.novelty[calibration_count:]
    predictions = output.predictions[calibration_count:]
    unknown = np.isin(evaluation_labels, unknown_classes)
    known = ~unknown
    frozen_threshold = (
        float(
            np.quantile(
                calibration_novelty, known_coverage_target, method="higher"
            )
        )
        if threshold is None
        else float(threshold)
    )
    accepted = evaluation_novelty <= frozen_threshold
    accepted_known = accepted & known
    accepted_accuracy = (
        float(
            balanced_accuracy_score(
                evaluation_labels[accepted_known], predictions[accepted_known]
            )
        )
        if len(np.unique(evaluation_labels[accepted_known])) > 1
        else 0.0
    )
    return {
        "threshold": frozen_threshold,
        "known_balanced_accuracy": float(
            balanced_accuracy_score(evaluation_labels[known], predictions[known])
        ),
        "known_coverage": float(np.mean(accepted[known])),
        "unknown_recall": float(np.mean(~accepted[unknown])),
        "accepted_known_balanced_accuracy": accepted_accuracy,
        "auroc": float(roc_auc_score(unknown.astype(np.int64), evaluation_novelty)),
        "prediction_class_count": int(len(np.unique(predictions[known]))),
        "collapsed_to_one_class": len(np.unique(predictions[known])) == 1,
        "state_hash": output.state_hash,
        "serialized_megabytes": float(output.size_bytes / (1024 * 1024)),
    }


def _fit_cell(
    geometry_x: np.ndarray,
    geometry_y: np.ndarray,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    query: np.ndarray,
    evaluation_y: np.ndarray,
    known_classes: np.ndarray,
    unknown_classes: np.ndarray,
    *,
    rank: int,
    extent_quantile: float,
    outer_scale_policy: str,
    config: dict[str, Any],
    parent_parameter_count: int,
) -> dict[str, Any]:
    initial = [
        fit_dimensionless_tube(
            geometry_x[geometry_y == class_label],
            calibration_x[calibration_y == class_label],
            rank=rank,
            extent_quantile=extent_quantile,
            outer_scale_policy=outer_scale_policy,
            penalty_weight=float(config["penalty_grid"][0]),
            class_label=int(class_label),
        )
        for class_label in known_classes
    ]
    selected = select_smallest_safety_penalty(
        initial,
        calibration_x,
        penalty_grid=tuple(float(value) for value in config["penalty_grid"]),
        known_coverage_target=float(config["known_coverage_target"]),
    )
    tubes = selected["tubes"]
    threshold = float(selected["threshold"])
    bounded_output = _tube_output(tubes, query, bounded=True)
    replay_output = _tube_output(
        [
            fit_dimensionless_tube(
                geometry_x[geometry_y == class_label],
                calibration_x[calibration_y == class_label],
                rank=rank,
                extent_quantile=extent_quantile,
                outer_scale_policy=outer_scale_policy,
                penalty_weight=float(selected["selected_penalty"]),
                class_label=int(class_label),
            )
            for class_label in known_classes
        ],
        query,
        bounded=True,
    )
    unbounded_output = _tube_output(tubes, query, bounded=False)
    bounded = _metrics(
        bounded_output,
        calibration_count=len(calibration_x),
        evaluation_labels=evaluation_y,
        unknown_classes=unknown_classes,
        threshold=threshold,
    )
    unbounded = _metrics(
        unbounded_output,
        calibration_count=len(calibration_x),
        evaluation_labels=evaluation_y,
        unknown_classes=unknown_classes,
    )
    probes = generate_safety_probes(tubes, seed=int(config["seed"]))
    acceptance = probe_acceptance(tubes, probes, threshold=threshold)
    tangent_acceptance = {}
    for multiplier in (0.5, 1.0, 2.0, 4.0, 8.0):
        tangent, _ = generate_axis_tangent_probes(tubes, multiplier=multiplier)
        tangent_acceptance[str(multiplier).rstrip("0").rstrip(".")] = float(
            np.mean(system_scores(tubes, tangent) <= threshold)
        )
    parameter_count = sum(tube.parameter_count for tube in tubes)
    fit_work_units = int(
        sum(
            np.sum(geometry_y == class_label) * geometry_x.shape[1] * rank
            for class_label in known_classes
        )
    )
    parent_fit_work_units = int(parent_parameter_count * len(geometry_x))
    return {
        "rank": rank,
        "extent_quantile": extent_quantile,
        "outer_scale_policy": outer_scale_policy,
        "selected_penalty": selected["selected_penalty"],
        "selection_attempts": selected["attempts"],
        "calibration_lineage_hash": selected["lineage_hash"],
        "bounded_tube": bounded,
        "unbounded_residual": unbounded,
        "probe_acceptance": acceptance,
        "tangent_acceptance_by_multiplier": tangent_acceptance,
        "parameter_count": parameter_count,
        "parent_parameter_count": parent_parameter_count,
        "parameter_ratio": float(parameter_count / parent_parameter_count),
        "fit_work_units": fit_work_units,
        "parent_fit_work_units": parent_fit_work_units,
        "fit_work_ratio": float(fit_work_units / parent_fit_work_units),
        "tube_state_hash": payload_hash([tube.to_dict() for tube in tubes]),
        "exact_replay": bool(
            bounded_output.state_hash == replay_output.state_hash
            and np.array_equal(bounded_output.predictions, replay_output.predictions)
            and np.array_equal(bounded_output.novelty, replay_output.novelty)
        ),
    }


def run_evaluation(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["final_labels_opened"]:
        raise PermissionError("M58 final labels must remain sealed")
    verified_locks = {
        name: {
            "path": specification["path"],
            "sha256": sha256_file(_verify(specification)),
        }
        for name, specification in sorted(config["parent_locks"].items())
    }
    v9_config = json.loads(
        _resolve(config["parent_locks"]["v9_m53_config"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    v9_evidence = json.loads(
        _resolve(config["parent_locks"]["v9_m53_evidence"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    m51_config_path = _verify(v9_config["m51_config"])
    parent_path = _verify(v9_config["parent_student"])
    m51_config = json.loads(m51_config_path.read_text(encoding="utf-8"))
    source_path = _verify(m51_config["source_config"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    loaded = _load_seed_data(source["seed_inputs"][str(seed)])
    train_features, train_labels = loaded["datasets"]["train"]
    dev_features, dev_labels = loaded["datasets"]["dev"]
    known_classes = np.asarray(config["known_classes"], dtype=np.int64)
    unknown_classes = np.asarray(config["proxy_unknown_classes"], dtype=np.int64)
    partitions = _partition_seed(
        train_labels,
        dev_labels,
        seed=seed,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        geometry_fraction=float(m51_config["geometry_fraction"]),
    )
    observed_partition_hashes = {
        name: payload_hash(values.tolist()) for name, values in partitions.items()
    }
    if observed_partition_hashes != v9_evidence["partition_hashes"]:
        raise ValueError("M58 partition lineage differs from frozen v9 M53")
    geometry_x = train_features[partitions["geometry_fit"]]
    geometry_y = train_labels[partitions["geometry_fit"]]
    calibration_x = train_features[partitions["score_calibration"]]
    calibration_y = train_labels[partitions["score_calibration"]]
    evaluation_indices = np.concatenate(
        [partitions["development_eval"], partitions["unknown_eval"]]
    )
    evaluation_x = dev_features[evaluation_indices]
    evaluation_y = dev_labels[evaluation_indices]
    query = np.vstack([calibration_x, evaluation_x])
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_parameter_count = sum(
        SubspacePrimitive.from_dict(item["payload"]).parameter_count
        for item in parent["selected_candidates"]
    )

    cells = []
    for rank in config["ranks"]:
        for extent_quantile in config["extent_quantiles"]:
            for outer_scale_policy in config["outer_scale_policies"]:
                try:
                    cell = _fit_cell(
                        geometry_x,
                        geometry_y,
                        calibration_x,
                        calibration_y,
                        query,
                        evaluation_y,
                        known_classes,
                        unknown_classes,
                        rank=int(rank),
                        extent_quantile=float(extent_quantile),
                        outer_scale_policy=str(outer_scale_policy),
                        config=config,
                        parent_parameter_count=parent_parameter_count,
                    )
                    cell["calibration_feasible"] = True
                except SafetyPenaltySelectionError as error:
                    cell = {
                        "rank": int(rank),
                        "extent_quantile": float(extent_quantile),
                        "outer_scale_policy": str(outer_scale_policy),
                        "calibration_feasible": False,
                        "selection_attempts": list(error.attempts),
                        "stop_reason": str(error),
                        "screen_passed": False,
                    }
                cells.append(cell)

    baseline = v9_evidence["controls"]["signed_volume_a2"]
    gate_config = config["gate"]
    for cell in cells:
        if not cell["calibration_feasible"]:
            continue
        metrics = cell["bounded_tube"]
        probes = cell["probe_acceptance"]["system"]
        tangent = cell["tangent_acceptance_by_multiplier"]
        operands = {
            "balanced_accuracy_gain": float(
                metrics["known_balanced_accuracy"]
                - baseline["known_balanced_accuracy"]
            ),
            "unknown_recall_loss": float(
                baseline["unknown_recall"] - metrics["unknown_recall"]
            ),
            "accepted_known_accuracy_loss": float(
                baseline["accepted_known_balanced_accuracy"]
                - metrics["accepted_known_balanced_accuracy"]
            ),
            "eight_x_tangent_acceptance": tangent["8"],
            "four_x_tangent_acceptance": tangent["4"],
            "bridge_acceptance": probes["bridge"],
            "random_direction_acceptance": probes["random_direction"],
            "mixed_acceptance": probes["mixed"],
            "prediction_class_count": metrics["prediction_class_count"],
            "parameter_ratio": cell["parameter_ratio"],
            "fit_work_ratio": cell["fit_work_ratio"],
            "exact_replay": cell["exact_replay"],
        }
        cell["gate_operands"] = operands
        cell["screen_passed"] = bool(
            operands["balanced_accuracy_gain"]
            >= gate_config["minimum_balanced_accuracy_gain"]
            and operands["unknown_recall_loss"]
            <= gate_config["maximum_unknown_recall_loss"]
            and operands["accepted_known_accuracy_loss"]
            <= gate_config["maximum_accepted_known_accuracy_loss"]
            and operands["eight_x_tangent_acceptance"]
            <= gate_config["maximum_8x_tangent_acceptance"]
            and operands["four_x_tangent_acceptance"]
            <= gate_config["maximum_4x_tangent_acceptance"]
            and operands["bridge_acceptance"]
            <= gate_config["maximum_other_probe_acceptance"]
            and operands["random_direction_acceptance"]
            <= gate_config["maximum_other_probe_acceptance"]
            and operands["mixed_acceptance"]
            <= gate_config["maximum_other_probe_acceptance"]
            and operands["prediction_class_count"]
            >= gate_config["minimum_prediction_class_count"]
            and operands["parameter_ratio"] <= gate_config["maximum_parameter_ratio"]
            and operands["fit_work_ratio"] <= gate_config["maximum_fit_work_ratio"]
            and operands["exact_replay"]
        )
    eligible = [index for index, cell in enumerate(cells) if cell["screen_passed"]]
    retained_index = (
        min(
            eligible,
            key=lambda index: (
                cells[index]["parameter_count"],
                cells[index]["rank"],
                cells[index]["selected_penalty"],
                index,
            ),
        )
        if eligible
        else None
    )
    evidence = {
        "schema_version": 1,
        "milestone": "M58",
        "configuration_hash": sha256_file(config_path),
        "verified_parent_locks": verified_locks,
        "partition_hashes": observed_partition_hashes,
        "controls": v9_evidence["controls"],
        "resource_accounting": {
            "fit_work_unit_definition": "geometry_rows_x_ambient_dimension_x_rank",
            "a2_fit_work_budget_definition": "a2_parameter_count_x_geometry_rows",
        },
        "cells": cells,
        "eligible_cell_indices": eligible,
        "retained_cell_index": retained_index,
        "retained_cell": cells[retained_index] if retained_index is not None else None,
        "advance_to_m59": retained_index is not None,
        "open_m60_curvature_diagnostic": False,
        "final_labels_opened": False,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    write_canonical_json(
        output_dir / "verification.json",
        {
            "schema_version": 1,
            "milestone": "M58",
            "evidence_sha256": sha256_file(output_dir / "evidence.json"),
            "cell_count": len(cells),
            "eligible_cell_indices": eligible,
            "retained_cell_index": retained_index,
            "advance_to_m59": evidence["advance_to_m59"],
            "open_m60_curvature_diagnostic": evidence[
                "open_m60_curvature_diagnostic"
            ],
        },
    )
    build_artifact_index(output_dir)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    evidence = run_evaluation(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "cell_count": len(evidence["cells"]),
                "eligible_cell_indices": evidence["eligible_cell_indices"],
                "retained_cell_index": evidence["retained_cell_index"],
                "advance_to_m59": evidence["advance_to_m59"],
                "open_m60_curvature_diagnostic": evidence[
                    "open_m60_curvature_diagnostic"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
