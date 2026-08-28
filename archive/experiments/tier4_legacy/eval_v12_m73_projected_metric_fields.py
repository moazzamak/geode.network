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
from experiments.tier4.eval_v12_m72_metric_field_stage0 import _evaluate_state
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.tier4.eval_v9_m51_surface_diagnostics import _partition_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "configs"
    / "v12"
    / "m73_projected_metric_fields.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "logs" / "results" / "v12" / "m73_projected_metric_fields"
)


def _resolve(path: str) -> Path:
    resolved = (REPO_ROOT / Path(path)).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError("M73 paths must remain inside the repository")
    return resolved


def _verify(specification: dict[str, str]) -> Path:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"M73 immutable artifact hash mismatch: {path}")
    return path


def _evaluate_projected(
    state: ProjectedMetricFieldState,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    development_x: np.ndarray,
    development_y: np.ndarray,
    unknown_x: np.ndarray,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    metrics = _evaluate_state(
        state.fields,
        state.transform(calibration_x),
        calibration_y,
        state.transform(development_x),
        development_y,
        state.transform(unknown_x),
        config=config,
    )
    metrics["state_hash"] = payload_hash(state.to_dict())
    metrics["parameter_count"] = state.parameter_count
    metrics["serialized_megabytes"] = float(state.array_bytes / 1_000_000)
    return metrics


def _held_out_four_x(
    metrics: dict[str, Any], held_out_families: list[str]
) -> float:
    return float(
        max(
            metrics["probe_acceptance"][family]["by_multiplier"]["4"][
                "system_acceptance"
            ]
            for family in held_out_families
        )
    )


def run_stage1(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _verify(config["source_config"])
    m72_index_path = _verify(config["m72_index"])
    m72_evidence = json.loads(
        (m72_index_path.parent / "evidence.json").read_text(encoding="utf-8")
    )
    source = json.loads(
        _resolve(config["source_config"]["path"]).read_text(encoding="utf-8")
    )
    seed = int(config["seed"])
    loaded = _load_seed_data(source["seed_inputs"][str(seed)])
    train_x, train_y = loaded["datasets"]["train"]
    dev_x, dev_y = loaded["datasets"]["dev"]
    partitions = _partition_seed(
        train_y,
        dev_y,
        seed=seed,
        known_classes=np.asarray(config["known_classes"], dtype=np.int64),
        unknown_classes=np.asarray(
            config["proxy_unknown_classes"], dtype=np.int64
        ),
        geometry_fraction=float(config["geometry_fraction"]),
    )
    geometry_x = train_x[partitions["geometry_fit"]]
    geometry_y = train_y[partitions["geometry_fit"]]
    calibration_x = train_x[partitions["score_calibration"]]
    calibration_y = train_y[partitions["score_calibration"]]
    development_x = dev_x[partitions["development_eval"]]
    development_y = dev_y[partitions["development_eval"]]
    unknown_x = dev_x[partitions["unknown_eval"]]

    initial = initialize_projected_metric_fields(
        geometry_x,
        geometry_y,
        output_dimension=int(config["projection_dimension"]),
        rank=int(config["rank"]),
    )
    training = config["training"]
    common_arguments = {
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
        "probe_families": tuple(training["trained_probe_families"]),
        "seed": seed,
    }
    arms = {}
    for name, collapse_weight in (
        ("constrained", float(training["collapse_weight"])),
        ("zero_constraint_ablation", float(training["ablation_collapse_weight"])),
    ):
        state, history = train_projected_metric_fields(
            initial,
            geometry_x,
            geometry_y,
            collapse_weight=collapse_weight,
            **common_arguments,
        )
        arms[name] = {
            "collapse_weight": collapse_weight,
            "metrics": _evaluate_projected(
                state,
                calibration_x,
                calibration_y,
                development_x,
                development_y,
                unknown_x,
                config=config,
            ),
            "projection_diagnostics": projection_diagnostics(
                state, initial, geometry_x
            ),
            "optimizer_history": history,
            "trained_state": state.to_dict(),
        }

    constrained = arms["constrained"]
    ablation = arms["zero_constraint_ablation"]
    constrained_metrics = constrained["metrics"]
    m72_metrics = m72_evidence["trained_metrics"]
    held_out = list(training["held_out_probe_families"])
    constrained_held_out = _held_out_four_x(constrained_metrics, held_out)
    m72_held_out = _held_out_four_x(m72_metrics, held_out)
    accuracy_improvement = float(
        constrained_metrics["known_balanced_accuracy"]
        - m72_metrics["known_balanced_accuracy"]
    )
    held_out_reduction = float(m72_held_out - constrained_held_out)
    gate_config = config["gate"]
    accuracy_material = bool(
        accuracy_improvement
        >= float(gate_config["minimum_accuracy_improvement_over_m72"])
    )
    held_out_material = bool(
        held_out_reduction
        >= float(
            gate_config["minimum_held_out_four_x_reduction_over_m72"]
        )
    )
    constrained_diagnostics = constrained["projection_diagnostics"]
    ablation_diagnostics = ablation["projection_diagnostics"]
    distance_load_bearing = bool(
        constrained_diagnostics["mean_relative_distance_drift"]
        <= float(
            gate_config[
                "maximum_relative_distance_drift_fraction_vs_ablation"
            ]
        )
        * ablation_diagnostics["mean_relative_distance_drift"]
    )
    orthogonality_load_bearing = bool(
        constrained_diagnostics["row_orthogonality_error"]
        <= float(
            gate_config[
                "maximum_row_orthogonality_error_fraction_vs_ablation"
            ]
        )
        * ablation_diagnostics["row_orthogonality_error"]
    )
    full_rank = bool(
        constrained_diagnostics["effective_rank"]
        == int(config["projection_dimension"])
    )
    gate = {
        "median_threshold_ratio": constrained_metrics[
            "median_threshold_ratio"
        ],
        "m72_known_balanced_accuracy": m72_metrics[
            "known_balanced_accuracy"
        ],
        "known_accuracy_improvement_over_m72": accuracy_improvement,
        "m72_held_out_four_x_acceptance": m72_held_out,
        "held_out_four_x_acceptance": constrained_held_out,
        "held_out_four_x_reduction_over_m72": held_out_reduction,
        "threshold_ratio_passed": bool(
            constrained_metrics["median_threshold_ratio"]
            <= float(gate_config["maximum_median_threshold_ratio"])
        ),
        "accuracy_material_improvement": accuracy_material,
        "held_out_material_improvement": held_out_material,
        "material_improvement_passed": bool(
            accuracy_material or held_out_material
        ),
        "distance_constraint_load_bearing": distance_load_bearing,
        "orthogonality_constraint_load_bearing": orthogonality_load_bearing,
        "full_projection_rank": full_rank,
        "collapse_prevention_load_bearing": bool(
            distance_load_bearing and orthogonality_load_bearing and full_rank
        ),
        "exact_replay": True,
        "final_labels_opened": False,
    }
    gate["m73_passed"] = bool(
        gate["threshold_ratio_passed"]
        and gate["material_improvement_passed"]
        and gate["collapse_prevention_load_bearing"]
    )
    evidence = {
        "schema_version": 1,
        "milestone": "M73",
        "configuration_hash": sha256_file(config_path),
        "partition_hashes": {
            name: payload_hash(indices.tolist())
            for name, indices in partitions.items()
        },
        "projection_initialization": {
            "method": "centered_pca_with_deterministic_signs",
            "state_hash": payload_hash(initial.to_dict()),
            "projection_diagnostics": projection_diagnostics(
                initial, initial, geometry_x
            ),
        },
        "trained_probe_families": training["trained_probe_families"],
        "held_out_probe_families": held_out,
        "arms": arms,
        "gate": gate,
        "advance_to_m74": bool(gate["m73_passed"]),
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
    result = run_stage1(arguments.config, arguments.output)
    print(
        json.dumps(
            {
                "constrained": result["arms"]["constrained"]["metrics"],
                "zero_constraint_ablation": result["arms"][
                    "zero_constraint_ablation"
                ]["metrics"],
                "gate": result["gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
