from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.model_fingerprint import InputSpec, ModelFingerprint, OutputSpec
from src.model_migration import dry_run_add_class_migration
from src.model_network import FittedModel, ModelNetwork
from src.sdf_engine import EllipsoidExpert, Expert


def _fit_expert(points: np.ndarray) -> Expert:
    center = points.mean(axis=0)
    radii = np.maximum(np.max(np.abs(points - center), axis=0) * 1.2, 0.25)
    expert = Expert(alpha=2.0)
    expert.add_ellipsoid(EllipsoidExpert(center=center, radii=radii))
    return expert


def _build_model(
    task_name: str,
    input_spec: InputSpec,
    features: np.ndarray,
    labels: np.ndarray,
    classes: tuple[int, ...],
) -> FittedModel:
    class_models = {
        class_id: [_fit_expert(features[labels == class_id])]
        for class_id in classes
    }
    model = FittedModel(
        ModelFingerprint(
            task_name=task_name,
            input_spec=input_spec,
            output_spec=OutputSpec("sdf_scores", classes),
        ),
        class_models,
        {class_id: 1.0 for class_id in classes},
    )
    return model


def _fit_calibrator(
    model: FittedModel,
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> LogisticRegression:
    calibrator = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=seed,
    )
    calibrator.fit(model.sdf_scores(features), labels)
    return calibrator


def run_calibrated_graph_migration(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    centers = np.asarray([
        [-4.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0],
    ])

    def sample(count: int) -> tuple[np.ndarray, np.ndarray]:
        features = np.vstack([
            rng.normal(center, 0.45, size=(count, centers.shape[1]))
            for center in centers
        ])
        labels = np.repeat(np.arange(3, dtype=np.int64), count)
        return features, labels

    geometry_X, geometry_y = sample(30)
    calibration_X, calibration_y = sample(24)
    test_X, test_y = sample(40)
    known_geometry = geometry_y < 2
    known_calibration = calibration_y < 2

    live_source = _build_model(
        "source",
        InputSpec("passthrough", dim=centers.shape[1]),
        geometry_X[known_geometry],
        geometry_y[known_geometry],
        (0, 1),
    )
    live_source.calibrator = _fit_calibrator(
        live_source,
        calibration_X[known_calibration],
        calibration_y[known_calibration],
        seed,
    )
    live_geometry_scores = live_source.sdf_scores(geometry_X[known_geometry])
    downstream_geometry_y = (geometry_y[known_geometry] == 1).astype(np.int64)
    live_downstream = _build_model(
        "downstream",
        InputSpec("sdf_scores", ("source",), dim=2),
        live_geometry_scores,
        downstream_geometry_y,
        (0, 1),
    )
    live_calibration_scores = live_source.sdf_scores(
        calibration_X[known_calibration],
    )
    live_downstream.calibrator = _fit_calibrator(
        live_downstream,
        live_calibration_scores,
        (calibration_y[known_calibration] == 1).astype(np.int64),
        seed,
    )
    live_network = ModelNetwork()
    live_network.add_node("source", live_source)
    live_network.add_node("downstream", live_downstream, upstream=["source"])
    live_source_signature = live_source.fingerprint.signature
    live_downstream_signature = live_downstream.fingerprint.signature

    preview_source = copy.deepcopy(live_source)
    preview_source.class_models[2] = [_fit_expert(geometry_X[geometry_y == 2])]
    preview_source.score_scales[2] = 1.0
    preview_source.fingerprint = ModelFingerprint(
        task_name="source",
        input_spec=live_source.fingerprint.input_spec,
        output_spec=OutputSpec("sdf_scores", (0, 1, 2)),
    )
    replacement_calibrator = _fit_calibrator(
        preview_source, calibration_X, calibration_y, seed,
    )
    preview_source.calibrator = replacement_calibrator
    migrated_geometry_scores = preview_source.sdf_scores(geometry_X)
    downstream_all_y = (geometry_y != 0).astype(np.int64)
    replacement_downstream = _build_model(
        "downstream",
        InputSpec("sdf_scores", ("source",), dim=3),
        migrated_geometry_scores,
        downstream_all_y,
        (0, 1),
    )
    migrated_calibration_scores = preview_source.sdf_scores(calibration_X)
    replacement_downstream.calibrator = _fit_calibrator(
        replacement_downstream,
        migrated_calibration_scores,
        (calibration_y != 0).astype(np.int64),
        seed,
    )

    dry_run = dry_run_add_class_migration(
        live_network,
        source_node="source",
        new_class_id=2,
        new_class_models=preview_source.class_models[2],
        score_scale=1.0,
        replacement_calibrator=replacement_calibrator,
        downstream_replacements={"downstream": replacement_downstream},
    )
    candidate_labels = dry_run.candidate_network.run(test_X)
    source_accuracy = float(np.mean(candidate_labels["source"] == test_y))
    downstream_truth = (test_y != 0).astype(np.int64)
    downstream_accuracy = float(np.mean(
        candidate_labels["downstream"] == downstream_truth,
    ))
    candidate_source = dry_run.candidate_network._nodes["source"].model
    candidate_downstream = dry_run.candidate_network._nodes["downstream"].model
    return {
        "protocol": {
            "seed": seed,
            "geometry_count": len(geometry_X),
            "calibration_count": len(calibration_X),
            "test_count": len(test_X),
            "test_used_for_reconstruction": False,
            "migration_published": False,
        },
        "migration": {
            "valid": dry_run.valid,
            "validation_issues": list(dry_run.validation_issues),
            "old_signature": dry_run.old_signature,
            "new_signature": dry_run.new_signature,
            "source_calibrator_width": int(
                candidate_source.calibrator.n_features_in_
            ),
            "downstream_input_width": int(
                candidate_downstream.fingerprint.input_spec.dim
            ),
            "downstream_calibrator_width": int(
                candidate_downstream.calibrator.n_features_in_
            ),
            "candidate_graph_executed": True,
            "live_source_unchanged": (
                live_source.fingerprint.signature == live_source_signature
                and live_source.class_ids == [0, 1]
            ),
            "live_downstream_unchanged": (
                live_downstream.fingerprint.signature == live_downstream_signature
                and live_downstream.fingerprint.input_spec.dim == 2
            ),
        },
        "observational_test": {
            "source_accuracy": source_accuracy,
            "downstream_accuracy": downstream_accuracy,
        },
    }


def run_multiseed(seeds: tuple[int, ...]) -> dict:
    runs = [run_calibrated_graph_migration(seed) for seed in seeds]
    return {
        "protocol": {"seeds": list(seeds), "migration_published": False},
        "summary": {
            "valid_migrations": sum(run["migration"]["valid"] for run in runs),
            "executed_candidate_graphs": sum(
                run["migration"]["candidate_graph_executed"] for run in runs
            ),
            "source_accuracy_mean": float(np.mean([
                run["observational_test"]["source_accuracy"] for run in runs
            ])),
            "downstream_accuracy_mean": float(np.mean([
                run["observational_test"]["downstream_accuracy"] for run in runs
            ])),
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrated graph migration dry run")
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