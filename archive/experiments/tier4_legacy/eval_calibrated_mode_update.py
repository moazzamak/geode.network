from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from experiments.tier4.eval_calibrated_graph_migration import (
    _build_model,
    _fit_calibrator,
)
from src.model_editor import ModelEditor
from src.model_fingerprint import InputSpec
from src.model_network import ModelNetwork
from src.replay_constrained_fitter import fit_replay_constrained_expert


def run_calibrated_mode_update(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    centers = np.asarray([
        [-3.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
    ])

    def sample(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = np.vstack([
            rng.normal(center, 0.4, size=(count, centers.shape[1]))
            for center in centers
        ])
        mode_ids = np.repeat(np.arange(3, dtype=np.int64), count)
        class_ids = np.where(mode_ids == 2, 1, 0)
        return features, class_ids, mode_ids

    geometry_X, geometry_y, geometry_modes = sample(30)
    calibration_X, calibration_y, _ = sample(24)
    test_X, test_y, test_modes = sample(40)
    live_geometry = geometry_modes != 1
    live_calibration = np.concatenate([
        np.arange(24), np.arange(48, 72),
    ])

    live_source = _build_model(
        "source",
        InputSpec("passthrough", dim=centers.shape[1]),
        geometry_X[live_geometry],
        geometry_y[live_geometry],
        (0, 1),
    )
    live_source.calibrator = _fit_calibrator(
        live_source,
        calibration_X[live_calibration],
        calibration_y[live_calibration],
        seed,
    )
    live_scores = live_source.sdf_scores(geometry_X[live_geometry])
    live_downstream = _build_model(
        "downstream",
        InputSpec("sdf_scores", ("source",), dim=2),
        live_scores,
        geometry_y[live_geometry],
        (0, 1),
    )
    live_calibration_scores = live_source.sdf_scores(
        calibration_X[live_calibration],
    )
    live_downstream.calibrator = _fit_calibrator(
        live_downstream,
        live_calibration_scores,
        calibration_y[live_calibration],
        seed,
    )
    live_network = ModelNetwork()
    live_network.add_node("source", live_source)
    live_network.add_node("downstream", live_downstream, upstream=["source"])
    live_source_signature = live_source.fingerprint.signature
    live_predictions_before = live_network.run(test_X)

    candidate_network = copy.deepcopy(live_network)
    candidate_source = candidate_network._nodes["source"].model
    constrained_fit = fit_replay_constrained_expert(
        geometry_X[geometry_modes == 1],
        geometry_X[geometry_y == 1],
        exclusion_margin=0.1,
    )
    candidate_source.class_fusion_modes[0] = "hard_min"
    editor = ModelEditor(candidate_source.class_models)
    transaction = editor.apply_transaction(
        lambda: candidate_source.class_models[0].append(
            copy.deepcopy(constrained_fit.expert),
        ),
        lambda _models: (
            constrained_fit.exclusion_violations == 0
            and constrained_fit.positive_coverage >= 0.5
        ),
        operation_name="calibrated_existing_mode_update",
        class_id=0,
    )
    candidate_source.calibrator = _fit_calibrator(
        candidate_source, calibration_X, calibration_y, seed,
    )
    candidate_geometry_scores = candidate_source.sdf_scores(geometry_X)
    candidate_downstream = _build_model(
        "downstream",
        InputSpec("sdf_scores", ("source",), dim=2),
        candidate_geometry_scores,
        geometry_y,
        (0, 1),
    )
    candidate_calibration_scores = candidate_source.sdf_scores(calibration_X)
    candidate_downstream.calibrator = _fit_calibrator(
        candidate_downstream,
        candidate_calibration_scores,
        calibration_y,
        seed,
    )
    candidate_network._nodes["downstream"].model = candidate_downstream
    validation_issues = candidate_network.validate()
    candidate_predictions = candidate_network.run(test_X)
    old_mode_mask = test_modes == 0
    new_mode_mask = test_modes == 1
    live_predictions_after = live_network.run(test_X)

    return {
        "protocol": {
            "seed": seed,
            "geometry_count": len(geometry_X),
            "calibration_count": len(calibration_X),
            "test_count": len(test_X),
            "test_used_for_fitting_or_calibration": False,
            "mutation_published": False,
        },
        "update": {
            "transaction_accepted": transaction["accepted"],
            "class_fusion_mode": candidate_source.class_fusion_modes[0],
            "radius_scale": constrained_fit.radius_scale,
            "fit_positive_coverage": constrained_fit.positive_coverage,
            "exclusion_violations": constrained_fit.exclusion_violations,
            "source_calibrator_width": int(
                candidate_source.calibrator.n_features_in_
            ),
            "downstream_input_width": int(
                candidate_downstream.fingerprint.input_spec.dim
            ),
            "downstream_calibrator_width": int(
                candidate_downstream.calibrator.n_features_in_
            ),
            "validation_issues": validation_issues,
            "candidate_graph_executed": True,
            "live_graph_unchanged": (
                live_source.fingerprint.signature == live_source_signature
                and not live_source.class_fusion_modes
                and all(
                    np.array_equal(live_predictions_before[name], predictions)
                    for name, predictions in live_predictions_after.items()
                )
            ),
        },
        "observational_test": {
            "source_accuracy": float(np.mean(
                candidate_predictions["source"] == test_y,
            )),
            "downstream_accuracy": float(np.mean(
                candidate_predictions["downstream"] == test_y,
            )),
            "old_mode_source_accuracy": float(np.mean(
                candidate_predictions["source"][old_mode_mask] == 0,
            )),
            "new_mode_source_accuracy": float(np.mean(
                candidate_predictions["source"][new_mode_mask] == 0,
            )),
        },
    }


def run_multiseed(seeds: tuple[int, ...]) -> dict:
    runs = [run_calibrated_mode_update(seed) for seed in seeds]
    return {
        "protocol": {"seeds": list(seeds), "mutation_published": False},
        "summary": {
            "accepted_transactions": sum(
                run["update"]["transaction_accepted"] for run in runs
            ),
            "valid_graphs": sum(
                not run["update"]["validation_issues"] for run in runs
            ),
            "unchanged_live_graphs": sum(
                run["update"]["live_graph_unchanged"] for run in runs
            ),
            "source_accuracy_mean": float(np.mean([
                run["observational_test"]["source_accuracy"] for run in runs
            ])),
            "downstream_accuracy_mean": float(np.mean([
                run["observational_test"]["downstream_accuracy"] for run in runs
            ])),
            "old_mode_accuracy_minimum": float(np.min([
                run["observational_test"]["old_mode_source_accuracy"] for run in runs
            ])),
            "new_mode_accuracy_minimum": float(np.min([
                run["observational_test"]["new_mode_source_accuracy"] for run in runs
            ])),
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrated mode update dry run")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact_path = Path(config["artifact_path"])
    result = run_multiseed(tuple(config["seeds"]))
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()