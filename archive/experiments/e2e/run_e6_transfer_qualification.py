"""Qualify a parent-linked CIFAR-100 to CIFAR-10 transfer episode."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform

import numpy as np

from experiments.common.classification_baselines import fit_classification_baselines
from experiments.common.classification_metrics import classification_metrics
from experiments.common.experiment_manifest import array_fingerprint, canonical_json
from experiments.common.score_readouts import fit_score_readout
from experiments.e2e.e4_cifar_protocol import (
    build_id_partitions,
    load_config as load_e4_config,
    load_e4_data,
)
from experiments.e2e.e5_bundle_loader import load_e4_candidate
from experiments.e2e.run_e4_cifar_qualification import (
    _readout_bytes,
    _transform_bytes,
)
from experiments.e2e.run_tier4_smoke import _serialize_experts
from experiments.tier4.eval_complex_classification import (
    _apply_transform,
    _build_transform,
    compute_raw_scores,
    compute_score_scales,
    fit_class_models,
)
from src.model_fingerprint import InputSpec, ModelFingerprint, OutputSpec
from src.open_set import SupportProfile
from src.runtime import (
    BundleNode,
    BundleProvenance,
    LocalModelBundleStore,
    build_stratified_episode_partitions,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported E6 configuration schema")
    if config["source_forgetting_budget"] != 0.0:
        raise ValueError("E6 immutable-branch forgetting budget must be zero")
    if config["pretraining_lanes"]["external"]["status"] != "not_run":
        raise ValueError("an external lane requires a separately frozen checkpoint")
    return config


def _target_partitions(data, config):
    official_train_count = 50_000
    development = np.flatnonzero(data.near_source_indices < official_train_count)
    final_test = np.flatnonzero(data.near_source_indices >= official_train_count)
    fractions = config["partition_fractions"]
    return build_stratified_episode_partitions(
        data.near_labels,
        development_indices=development,
        final_test_indices=final_test,
        seed=int(config["seed"]),
        readout_fraction=float(fractions["readout_calibration"]),
        risk_fraction=float(fractions["risk_control"]),
        validation_fraction=float(fractions["validation"]),
    )


def _fit_target_geode(data, partitions, config):
    seed = int(config["seed"])
    model_config = config["model"]
    pca, lda, scaler = _build_transform(
        data.near_features[partitions["geometry"]],
        data.near_labels[partitions["geometry"]],
        int(model_config["pca_components"]),
        seed,
    )
    transformed = {
        name: _apply_transform(data.near_features[indices], pca, lda, scaler)
        for name, indices in partitions.items()
    }
    geometry_labels = data.near_labels[partitions["geometry"]]
    class_ids = np.unique(geometry_labels)
    models = fit_class_models(
        transformed["geometry"],
        geometry_labels,
        class_ids,
        consensus_threshold=float(model_config["consensus_threshold"]),
        capture_threshold=float(model_config["capture_threshold"]),
        alpha=float(model_config["alpha"]),
        max_iterations=int(model_config["max_iterations"]),
        nudge_iterations=int(model_config["nudge_iterations"]),
        nudge_learning_rate=0.02,
        seed=seed,
    )
    scales = compute_score_scales(
        models,
        transformed["geometry"],
        float(model_config["alpha"]),
        class_labels=geometry_labels,
    )
    readout_scores = compute_raw_scores(
        models,
        transformed["readout_calibration"],
        float(model_config["alpha"]),
        scales,
    )
    readout = fit_score_readout(
        "multinomial",
        readout_scores,
        data.near_labels[partitions["readout_calibration"]],
        class_ids,
        seed=seed,
    )
    final_scores = compute_raw_scores(
        models,
        transformed["final_test"],
        float(model_config["alpha"]),
        scales,
    )
    probabilities = readout.predict_proba(final_scores)
    transform_blob = _transform_bytes(pca, lda, scaler)
    model_blob = _json_bytes({
        "classes": class_ids.tolist(),
        "score_scales": {str(key): value for key, value in sorted(scales.items())},
        "class_models": {
            str(class_id): _serialize_experts(models[int(class_id)])
            for class_id in class_ids
        },
    })
    return {
        "classes": class_ids,
        "models": models,
        "scales": scales,
        "probabilities": probabilities,
        "predictions": class_ids[np.argmax(probabilities, axis=1)],
        "transform_blob": transform_blob,
        "model_blob": model_blob,
        "readout_blob": _readout_bytes(readout),
        "transformed": transformed,
    }


def _package_child(config, partitions, audit, geode, summary, registry: Path) -> str:
    parent_id = str(config["parent_bundle_id"])
    class_ids = tuple(int(value) for value in geode["classes"])
    fingerprint = ModelFingerprint(
        task_name="cifar10_transfer_classification",
        input_spec=InputSpec("raw_cnn", dim=1280),
        output_spec=OutputSpec("probabilities", class_ids),
        alpha=float(config["model"]["alpha"]),
        pca_components=int(config["model"]["pca_components"]),
    )
    transform_fingerprint = _sha256(geode["transform_blob"])
    partition_hashes = audit.to_dict()["partition_hashes"]
    support = SupportProfile(
        model_signature=fingerprint.signature,
        feature_transform_fingerprint=transform_fingerprint,
        training_dataset_fingerprint=_sha256(_json_bytes({
            "target": config["target_dataset"],
            "geometry": partition_hashes["geometry"],
        })),
        calibration_dataset_fingerprint=_sha256(_json_bytes({
            "readout": partition_hashes["readout_calibration"],
            "risk": partition_hashes["risk_control"],
        })),
        class_ids=class_ids,
        score_scales=tuple(float(geode["scales"][value]) for value in class_ids),
        novelty_score="not_qualified",
        global_threshold=0.0,
        version="e6-v1",
        fit_seed=int(config["seed"]),
        created_at="2026-07-26T00:00:00Z",
    )
    evidence_blob = _json_bytes(summary)
    config_blob = _json_bytes(config)
    node = BundleNode(
        name="cifar10_transfer",
        artifact_path="model_state.json",
        fingerprint=fingerprint,
        class_order=class_ids,
        feature_transform_fingerprint=transform_fingerprint,
        support_profile=support,
    )
    provenance = BundleProvenance(
        routing_mode="exhaustive",
        semantic_router_cache_version="disabled-e6",
        training_manifest_hash=_sha256(config_blob),
        evaluation_manifest_hash=_sha256(evidence_blob),
        metric_summary_hash=_sha256(evidence_blob),
        software_compatibility="python>=3.11,numpy>=2,scikit-learn>=1.6",
        environment_fingerprint=_sha256(
            f"{platform.platform()}|{platform.python_version()}".encode("utf-8")
        ),
        created_at="2026-07-26T00:00:00Z",
        created_by="E6 transfer qualification",
    )
    manifest = LocalModelBundleStore(registry).publish(
        {
            "model_state.json": geode["model_blob"],
            "transform.npz": geode["transform_blob"],
            "readout.npz": geode["readout_blob"],
            "evaluation_summary.json": evidence_blob,
            "frozen_config.json": config_blob,
        },
        [node],
        provenance=provenance,
        parent_bundle_id=parent_id,
    )
    return manifest.bundle_id


def run_qualification(config_path: Path, e4_config_path: Path) -> dict:
    config = _load_config(config_path)
    e4_config = load_e4_config(e4_config_path)
    data = load_e4_data(e4_config)
    registry = Path(config["parent_registry"])
    parent = load_e4_candidate(registry)
    if parent.manifest.bundle_id != config["parent_bundle_id"]:
        raise ValueError("E6 parent does not match the frozen bundle")

    source_partitions, _ = build_id_partitions(data, e4_config, int(config["seed"]))
    source_features = data.id_features[source_partitions["final_test"]]
    source_labels = data.id_labels[source_partitions["final_test"]]
    source_predictions_before = parent.predict(source_features)
    source_hash_before = array_fingerprint(source_predictions_before)
    source_accuracy_before = float(np.mean(source_predictions_before == source_labels))

    target_partitions, target_audit = _target_partitions(data, config)
    target_labels = data.near_labels[target_partitions["final_test"]]
    geometry_features = data.near_features[target_partitions["geometry"]]
    geometry_labels = data.near_labels[target_partitions["geometry"]]
    final_features = data.near_features[target_partitions["final_test"]]
    classes = np.unique(geometry_labels)

    linear_probe = fit_classification_baselines(
        geometry_features,
        geometry_labels,
        {int(class_id): 1 for class_id in classes},
        seed=int(config["seed"]),
        include_names={"logistic_regression"},
    )["logistic_regression"]
    linear_probabilities = linear_probe.predict_proba(final_features)

    geode = _fit_target_geode(data, target_partitions, config)
    adapter_probe = fit_classification_baselines(
        geode["transformed"]["geometry"],
        geometry_labels,
        {int(class_id): 1 for class_id in classes},
        seed=int(config["seed"]),
        include_names={"logistic_regression"},
    )["logistic_regression"]
    adapter_probabilities = adapter_probe.predict_proba(
        geode["transformed"]["final_test"]
    )
    geode_replay = _fit_target_geode(data, target_partitions, config)

    source_predictions_after = parent.predict(source_features)
    source_hash_after = array_fingerprint(source_predictions_after)
    source_accuracy_after = float(np.mean(source_predictions_after == source_labels))
    variants = {
        "frozen_source": {
            "target_metrics": None,
            "target_role": "unavailable_without_target_head",
            "source_accuracy": source_accuracy_after,
        },
        "linear_probe": {
            "target_metrics": classification_metrics(
                target_labels, linear_probabilities, classes,
            ),
            "prediction_hash": array_fingerprint(
                classes[np.argmax(linear_probabilities, axis=1)]
            ),
        },
        "supervised_adapter": {
            "target_metrics": classification_metrics(
                target_labels, adapter_probabilities, classes,
            ),
            "prediction_hash": array_fingerprint(
                classes[np.argmax(adapter_probabilities, axis=1)]
            ),
        },
        "geode_head": {
            "target_metrics": classification_metrics(
                target_labels, geode["probabilities"], classes,
            ),
            "prediction_hash": array_fingerprint(geode["predictions"]),
            "model_hashes": {
                "transform": _sha256(geode["transform_blob"]),
                "geometry": _sha256(geode["model_blob"]),
                "readout": _sha256(geode["readout_blob"]),
            },
        },
    }
    replay_identity = {
        "prediction_hash": (
            array_fingerprint(geode_replay["predictions"])
            == variants["geode_head"]["prediction_hash"]
        ),
        "transform": _sha256(geode_replay["transform_blob"])
        == variants["geode_head"]["model_hashes"]["transform"],
        "geometry": _sha256(geode_replay["model_blob"])
        == variants["geode_head"]["model_hashes"]["geometry"],
        "readout": _sha256(geode_replay["readout_blob"])
        == variants["geode_head"]["model_hashes"]["readout"],
    }
    replay_identity["passed"] = all(replay_identity.values())
    forgetting = source_accuracy_before - source_accuracy_after
    target_improvement = (
        variants["geode_head"]["target_metrics"]["balanced_accuracy"] - 0.1
    )
    summary = {
        "schema_version": 1,
        "milestone": "E6",
        "config_hash": _sha256(config_path.read_bytes()),
        "parent_bundle_id": parent.manifest.bundle_id,
        "partition_audit": target_audit.to_dict(),
        "pretraining_lanes": config["pretraining_lanes"],
        "variants": variants,
        "source_replay": {
            "accuracy_before": source_accuracy_before,
            "accuracy_after": source_accuracy_after,
            "forgetting": forgetting,
            "budget": float(config["source_forgetting_budget"]),
            "prediction_hash_before": source_hash_before,
            "prediction_hash_after": source_hash_after,
            "passed": (
                source_hash_before == source_hash_after
                and forgetting <= float(config["source_forgetting_budget"])
            ),
        },
        "target_gate": {
            "chance_balanced_accuracy": 0.1,
            "geode_improvement": target_improvement,
            "required_improvement": float(config["target_accuracy_floor"]),
            "passed": target_improvement >= float(config["target_accuracy_floor"]),
        },
        "replay_identity": replay_identity,
        "final_test_used_for_selection": False,
    }
    summary["core_gates_passed"] = (
        summary["source_replay"]["passed"]
        and summary["target_gate"]["passed"]
        and replay_identity["passed"]
        and summary["pretraining_lanes"]["external"]["status"] == "not_run"
    )
    if summary["core_gates_passed"]:
        summary["child_bundle_id"] = _package_child(
            config, target_partitions, target_audit, geode, summary, registry,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("experiments/configs/e6_transfer_qualification.json"),
    )
    parser.add_argument(
        "--e4-config", type=Path,
        default=Path("experiments/configs/e4_cifar_qualification.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("logs/results/e6_transfer_qualification.json"),
    )
    arguments = parser.parse_args()
    result = run_qualification(arguments.config, arguments.e4_config)
    if not result["core_gates_passed"]:
        raise RuntimeError(f"E6 qualification failed: {result}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({
        "core_gates_passed": result["core_gates_passed"],
        "child_bundle_id": result.get("child_bundle_id"),
        "source_replay": result["source_replay"],
        "target_gate": result["target_gate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()