"""Resumable Tier 4 smoke episode over immutable local stages."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.experiment_manifest import array_fingerprint, experiment_id
from experiments.common.moe_eval import fit_experts
from experiments.tier4.eval_complex_classification import (
    _apply_transform,
    _build_transform,
    compute_raw_scores,
    compute_score_scales,
    load_cifar_npz,
)
from src.runtime import (
    DatasetEpisodeContract,
    GeometryCapacityContract,
    LifecycleState,
    LocalArtifactStore,
    LocalExecutor,
    MetricEvent,
    MetricLedger,
    ModelSelectionContract,
    PretrainingLane,
    PretrainingProvenance,
    ReproducibilityContract,
    ReproducibilityLevel,
    RunContract,
    StageSpec,
    evaluate_geometry_feasibility,
)
from src.sdf_engine import EllipsoidExpert, Expert


FeatureLoader = Callable[[], tuple[np.ndarray, np.ndarray]]
ClassFitter = Callable[[np.ndarray, np.ndarray, int], list[Expert]]
PartitionBuilder = Callable[[np.ndarray, int], dict[str, np.ndarray]]


class InjectedStageFailure(RuntimeError):
    """Raised after a requested stage commit to exercise resume behavior."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(_canonical_json(value) + "\n", encoding="utf-8", newline="\n")


def _load_array(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        return np.load(stream, allow_pickle=False)


def _save_array(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)


def _hash_mapping(values: dict[str, str]) -> str:
    return hashlib.sha256(_canonical_json(values).encode("utf-8")).hexdigest()


def _partition_indices(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Build disjoint, stratified episode partitions with geometry as remainder."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    rng = np.random.default_rng(seed)
    partitions: dict[str, list[np.ndarray]] = {
        "geometry": [],
        "readout_calibration": [],
        "risk_control": [],
        "validation": [],
        "final_test": [],
    }
    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        if len(class_indices) < 10:
            raise ValueError(
                f"class {class_id!r} needs at least 10 samples for the smoke episode"
            )
        rng.shuffle(class_indices)
        final_count = max(1, int(round(0.20 * len(class_indices))))
        validation_count = max(1, int(round(0.15 * len(class_indices))))
        readout_count = max(1, int(round(0.15 * len(class_indices))))
        risk_count = max(1, int(round(0.10 * len(class_indices))))
        boundary_1 = final_count
        boundary_2 = boundary_1 + validation_count
        boundary_3 = boundary_2 + readout_count
        boundary_4 = boundary_3 + risk_count
        if len(class_indices) - boundary_4 < 2:
            raise ValueError(f"class {class_id!r} has insufficient geometry samples")
        partitions["final_test"].append(class_indices[:boundary_1])
        partitions["validation"].append(class_indices[boundary_1:boundary_2])
        partitions["readout_calibration"].append(class_indices[boundary_2:boundary_3])
        partitions["risk_control"].append(class_indices[boundary_3:boundary_4])
        partitions["geometry"].append(class_indices[boundary_4:])
    return {
        name: np.sort(np.concatenate(parts)).astype(np.int64)
        for name, parts in partitions.items()
    }


def _serialize_experts(experts: list[Expert]) -> dict[str, Any]:
    return {
        "experts": [
            {
                "alpha": expert.alpha,
                "ellipsoids": [
                    {
                        "center": ellipsoid.center.tolist(),
                        "radii": ellipsoid.radii.tolist(),
                        "orientation": ellipsoid.orientation.tolist(),
                        "polarity": ellipsoid.polarity,
                    }
                    for ellipsoid in expert.ellipsoids
                ],
            }
            for expert in experts
        ],
    }


def _deserialize_experts(payload: dict[str, Any]) -> list[Expert]:
    experts: list[Expert] = []
    for expert_record in payload["experts"]:
        expert = Expert(alpha=float(expert_record["alpha"]))
        for ellipsoid_record in expert_record["ellipsoids"]:
            expert.add_ellipsoid(EllipsoidExpert(
                center=np.asarray(ellipsoid_record["center"], dtype=np.float64),
                radii=np.asarray(ellipsoid_record["radii"], dtype=np.float64),
                orientation=np.asarray(
                    ellipsoid_record["orientation"], dtype=np.float64,
                ),
                polarity=int(ellipsoid_record["polarity"]),
            ))
        experts.append(expert)
    return experts


def _default_class_fitter(
    *,
    alpha: float,
    max_iterations: int | None,
    nudge_iterations: int,
) -> ClassFitter:
    def fit(points: np.ndarray, exclusions: np.ndarray, seed: int) -> list[Expert]:
        return fit_experts(
            points=points,
            exclude_points=exclusions,
            consensus_threshold=0.12,
            capture_threshold=0.08,
            alpha=alpha,
            max_iterations=max_iterations,
            nudge_iterations=nudge_iterations,
            nudge_learning_rate=0.02,
            use_gpu=False,
            seed=seed,
        )

    return fit


def _fail_if_requested(fail_after: str | None, stage_name: str) -> None:
    if fail_after == stage_name:
        raise InjectedStageFailure(f"injected failure after {stage_name}")


def _tier4_smoke_config(
    *,
    dataset_fingerprint: str,
    seed: int,
    pca_components: int,
    alpha: float,
    max_iterations: int | None,
    nudge_iterations: int,
    max_condition_number: float,
    max_parameter_sample_ratio: float,
    pretraining_lane: PretrainingLane,
    pretraining_source: str,
    partition_strategy: str = "default_stratified",
) -> dict[str, Any]:
    config = {
        "dataset_fingerprint": dataset_fingerprint,
        "seed": seed,
        "pca_components": pca_components,
        "alpha": alpha,
        "max_iterations": max_iterations,
        "nudge_iterations": nudge_iterations,
        "max_condition_number": max_condition_number,
        "max_parameter_sample_ratio": max_parameter_sample_ratio,
        "pretraining_lane": pretraining_lane.value,
        "pretraining_source": pretraining_source,
    }
    if partition_strategy != "default_stratified":
        config["partition_strategy"] = partition_strategy
    return config


def run_resumable_tier4_smoke(
    *,
    runtime_root: str | Path,
    attempt_id: str,
    dataset_fingerprint: str,
    feature_loader: FeatureLoader,
    seed: int = 42,
    pca_components: int = 32,
    alpha: float = 2.0,
    max_iterations: int | None = 10,
    nudge_iterations: int = 0,
    max_condition_number: float = 1e8,
    max_parameter_sample_ratio: float = 0.6,
    pretraining_lane: PretrainingLane = PretrainingLane.CONTROLLED,
    pretraining_source: str = "episode-data-only",
    fail_after: str | None = None,
    fail_during: str | None = None,
    class_fitter: ClassFitter | None = None,
    partition_builder: PartitionBuilder | None = None,
    partition_strategy: str = "default_stratified",
) -> dict[str, Any]:
    """Run or resume a validation-only Tier 4 smoke episode."""
    if (
        fail_during not in (None, "features")
        and not (
            fail_during.startswith("class-")
            and fail_during.removeprefix("class-").isdigit()
        )
    ):
        raise ValueError("fail_during must be None, 'features', or 'class-<id>'")
    config = _tier4_smoke_config(
        dataset_fingerprint=dataset_fingerprint,
        seed=seed,
        pca_components=pca_components,
        alpha=alpha,
        max_iterations=max_iterations,
        nudge_iterations=nudge_iterations,
        max_condition_number=max_condition_number,
        max_parameter_sample_ratio=max_parameter_sample_ratio,
        pretraining_lane=pretraining_lane,
        pretraining_source=pretraining_source,
        partition_strategy=partition_strategy,
    )
    run_id = experiment_id(config)
    store = LocalArtifactStore(runtime_root)
    executor = LocalExecutor(store)
    fitter = class_fitter or _default_class_fitter(
        alpha=alpha,
        max_iterations=max_iterations,
        nudge_iterations=nudge_iterations,
    )

    feature_inputs = {
        "dataset": dataset_fingerprint,
        "feature_config": _hash_mapping({
            "pretraining_lane": pretraining_lane.value,
            "pretraining_source": pretraining_source,
        }),
    }

    def write_features(path: Path) -> None:
        features, labels = feature_loader()
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)
        if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
            raise ValueError("feature loader must return aligned (n, d) features and labels")
        if not np.all(np.isfinite(features)):
            raise ValueError("features must contain only finite values")
        _save_array(path / "features.npy", features)
        if fail_during == "features":
            raise InjectedStageFailure("injected failure during features")
        _save_array(path / "labels.npy", labels)

    feature_manifest = executor.execute_stage(
        run_id,
        attempt_id,
        StageSpec(
            "features",
            LifecycleState.FEATURES_READY,
            write_features,
            feature_inputs,
        ),
    ).manifest
    _fail_if_requested(fail_after, "features")
    feature_path = store.stage_path(run_id, attempt_id, "features")
    features = _load_array(feature_path / "features.npy")
    labels = _load_array(feature_path / "labels.npy").astype(np.int32)

    transform_inputs = {
        "feature_stage": _hash_mapping(dict(feature_manifest.output_hashes)),
        "transform_config": hashlib.sha256(
            _canonical_json({"seed": seed, "pca_components": pca_components}).encode("utf-8")
        ).hexdigest(),
    }

    def write_transform(path: Path) -> None:
        partitions = (
            _partition_indices(labels, seed)
            if partition_builder is None
            else partition_builder(labels, seed)
        )
        from src.runtime import validate_episode_partitions

        validate_episode_partitions(
            partitions,
            dataset_size=len(labels),
            expected_indices=np.arange(len(labels), dtype=np.int64),
        )
        geometry_indices = partitions["geometry"]
        pca, lda, scaler = _build_transform(
            features[geometry_indices],
            labels[geometry_indices],
            pca_components,
            seed,
        )
        transformed = {
            name: _apply_transform(features[indices], pca, lda, scaler)
            for name, indices in partitions.items()
            if name != "final_test"
        }
        geometry_contract = GeometryCapacityContract(
            allowed_families=("full",),
            max_condition_number=max_condition_number,
            max_parameter_sample_ratio=max_parameter_sample_ratio,
            min_effective_rank=1.0,
        )
        feasibility = evaluate_geometry_feasibility(
            transformed["geometry"], labels[geometry_indices], geometry_contract,
        )
        feasibility.require_supportable()

        split_hashes = {
            name: array_fingerprint(indices) for name, indices in partitions.items()
        }
        run_contract = RunContract(
            run_id=run_id,
            config_hash=hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest(),
            episode=DatasetEpisodeContract(
                geometry_split=split_hashes["geometry"],
                readout_calibration_split=split_hashes["readout_calibration"],
                risk_control_split=split_hashes["risk_control"],
                validation_split=split_hashes["validation"],
                final_test_split=split_hashes["final_test"],
            ),
            pretraining=PretrainingProvenance(
                lane=pretraining_lane,
                source_datasets=(pretraining_source,),
                objective="fixed feature extraction",
                checkpoint_hash=feature_inputs["feature_config"],
                license="recorded by dataset/backbone provider",
                access_date="2026-07-25",
                overlap_risk=(
                    "none declared" if pretraining_lane is PretrainingLane.CONTROLLED
                    else "requires external provenance audit"
                ),
            ),
            geometry=geometry_contract,
            model_selection=ModelSelectionContract(
                validation_domains=("tier4-validation",),
                final_domains=("tier4-final-test",),
                selection_rule="smoke run; no hyperparameter selection",
                primary_metric="raw_validation_accuracy",
            ),
            reproducibility=ReproducibilityContract(
                level=ReproducibilityLevel.REPLAY_IDENTITY,
                environment_fingerprint="local-fixed-environment",
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
            ),
        )
        for name, indices in partitions.items():
            _save_array(path / f"{name}_indices.npy", indices)
            if name == "final_test":
                continue
            _save_array(path / f"{name}_features.npy", transformed[name])
            _save_array(path / f"{name}_labels.npy", labels[indices])
        _write_json(path / "geometry_feasibility.json", feasibility.to_dict())
        _write_json(path / "run_contract.json", run_contract.to_dict())

    transform_manifest = executor.execute_stage(
        run_id,
        attempt_id,
        StageSpec(
            "transform",
            LifecycleState.FEATURES_READY,
            write_transform,
            transform_inputs,
        ),
    ).manifest
    _fail_if_requested(fail_after, "transform")
    transform_path = store.stage_path(run_id, attempt_id, "transform")
    geometry_features = _load_array(transform_path / "geometry_features.npy")
    geometry_labels = _load_array(transform_path / "geometry_labels.npy").astype(np.int32)
    validation_features = _load_array(transform_path / "validation_features.npy")
    validation_labels = _load_array(transform_path / "validation_labels.npy").astype(np.int32)
    class_ids = np.unique(geometry_labels)

    class_manifests = {}
    transform_hash = _hash_mapping(dict(transform_manifest.output_hashes))
    for class_position, class_id_value in enumerate(class_ids):
        class_id = int(class_id_value)
        stage_name = f"class-{class_id}"

        def write_class(
            path: Path,
            *,
            current_class_id: int = class_id,
            current_class_seed: int = seed + class_position,
        ) -> None:
            experts = fitter(
                geometry_features[geometry_labels == current_class_id],
                geometry_features[geometry_labels != current_class_id],
                current_class_seed,
            )
            if fail_during == stage_name:
                raise InjectedStageFailure(f"injected failure during {stage_name}")
            _write_json(path / "class_model.json", {
                "class_id": current_class_id,
                **_serialize_experts(experts),
            })

        class_manifests[class_id] = executor.execute_stage(
            run_id,
            attempt_id,
            StageSpec(
                stage_name,
                LifecycleState.GEOMETRY_READY,
                write_class,
                {
                    "transform_stage": transform_hash,
                    "class_id": hashlib.sha256(
                        str(class_id).encode("ascii")
                    ).hexdigest(),
                },
            ),
        ).manifest
        _fail_if_requested(fail_after, stage_name)

    assembly_inputs = {
        f"class-{class_id}": _hash_mapping(dict(manifest.output_hashes))
        for class_id, manifest in sorted(class_manifests.items())
    }
    assembly_inputs["transform_stage"] = transform_hash

    def write_assembly(path: Path) -> None:
        models: dict[int, list[Expert]] = {}
        for class_id in class_ids:
            class_path = store.stage_path(run_id, attempt_id, f"class-{int(class_id)}")
            payload = json.loads((class_path / "class_model.json").read_text(encoding="utf-8"))
            models[int(class_id)] = _deserialize_experts(payload)
        scales = compute_score_scales(models, geometry_features, alpha=alpha)
        scores = compute_raw_scores(
            models, validation_features, alpha=alpha, score_scales=scales,
        )
        ordered_classes = np.asarray(sorted(models), dtype=np.int32)
        predictions = ordered_classes[np.argmin(scores, axis=1)]
        accuracy = float(np.mean(predictions == validation_labels))
        _save_array(path / "validation_scores.npy", scores)
        _save_array(path / "validation_predictions.npy", predictions)
        _write_json(path / "summary.json", {
            "run_id": run_id,
            "class_ids": ordered_classes.tolist(),
            "validation_accuracy": accuracy,
            "validation_sample_count": len(validation_labels),
            "final_test_observed": False,
        })

    assembly_manifest = executor.execute_stage(
        run_id,
        attempt_id,
        StageSpec(
            "assembly",
            LifecycleState.EVALUATED,
            write_assembly,
            assembly_inputs,
        ),
    ).manifest
    _fail_if_requested(fail_after, "assembly")
    assembly_path = store.stage_path(run_id, attempt_id, "assembly")
    summary = json.loads((assembly_path / "summary.json").read_text(encoding="utf-8"))

    metric_path = store.stage_path(run_id, attempt_id, "metrics-placeholder").parent / "metrics.jsonl"
    metric = MetricEvent(
        event_id=f"{run_id}-raw-validation-accuracy",
        run_id=run_id,
        attempt_id=attempt_id,
        stage_name="assembly",
        split="validation",
        metric_name="raw_accuracy",
        value=summary["validation_accuracy"],
        sample_count=summary["validation_sample_count"],
        created_at=datetime.now(timezone.utc).isoformat(),
        namespace="selection",
    )
    ledger = MetricLedger(metric_path)
    existing = {event.event_id: event for event in ledger.read_events()}
    if metric.event_id in existing:
        metric = existing[metric.event_id]
    ledger.append(metric)

    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "summary": summary,
        "assembly_output_hashes": dict(assembly_manifest.output_hashes),
        "metric_count": len(ledger.read_events()),
        "runtime_status": executor.status(
            run_id,
            attempt_id,
            ("features", "transform", *(f"class-{int(value)}" for value in class_ids), "assembly"),
        ).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable Tier 4 GEODE smoke episode")
    parser.add_argument("--runtime-root", default="runs/e2e")
    parser.add_argument("--attempt-id", default="attempt-1")
    parser.add_argument("--dataset-path", default="data/tier4/cifar10_features.npz")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--feature-extractor", choices=("cnn", "hog"), default="hog")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--fail-after")
    parser.add_argument("--fail-during", metavar="STAGE")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--status", action="store_true")
    action.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    dataset_fingerprint = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    pretraining_lane = (
        PretrainingLane.EXTERNAL
        if args.feature_extractor == "cnn"
        else PretrainingLane.CONTROLLED
    )
    pretraining_source = (
        "ImageNet-1K MobileNetV2"
        if args.feature_extractor == "cnn"
        else "episode images with deterministic HOG"
    )
    if args.status:
        config = _tier4_smoke_config(
            dataset_fingerprint=dataset_fingerprint,
            seed=args.seed,
            pca_components=args.pca_components,
            alpha=2.0,
            max_iterations=args.max_iterations,
            nudge_iterations=0,
            max_condition_number=1e8,
            max_parameter_sample_ratio=0.6,
            pretraining_lane=pretraining_lane,
            pretraining_source=pretraining_source,
        )
        run_id = experiment_id(config)
        status = LocalExecutor(LocalArtifactStore(args.runtime_root)).status(
            run_id,
            args.attempt_id,
            ("features", "transform", "assembly"),
        )
        print(_canonical_json(status.to_dict()))
        return
    result = run_resumable_tier4_smoke(
        runtime_root=args.runtime_root,
        attempt_id=args.attempt_id,
        dataset_fingerprint=dataset_fingerprint,
        feature_loader=lambda: load_cifar_npz(
            str(dataset_path),
            max_samples=args.max_samples,
            pca_components=args.pca_components,
            seed=args.seed,
            feature_extractor=args.feature_extractor,
        ),
        seed=args.seed,
        pca_components=args.pca_components,
        max_iterations=args.max_iterations,
        pretraining_lane=pretraining_lane,
        pretraining_source=pretraining_source,
        fail_after=args.fail_after,
        fail_during=args.fail_during,
    )
    print(_canonical_json(result))


if __name__ == "__main__":
    main()