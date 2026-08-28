"""Run the frozen five-seed E4 CIFAR qualification and package a candidate."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
from scipy import stats

from experiments.common.classification_baselines import fit_classification_baselines
from experiments.common.classification_metrics import classification_metrics
from experiments.common.experiment_manifest import array_fingerprint, canonical_json
from experiments.common.ood_metrics import (
    ood_detection_metrics,
    ood_operating_point,
    select_ood_threshold_at_known_coverage,
)
from experiments.common.ood_scores import (
    fit_feature_ood_scorers,
    maximum_probability_score,
    minimum_sdf_score,
    sdf_energy_score,
)
from experiments.common.score_readouts import fit_score_readout
from experiments.e2e.e4_cifar_protocol import (
    E4Data,
    build_id_partitions,
    build_ood_partitions,
    load_config,
    load_e4_data,
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
    GeometryCapacityContract,
    LocalModelBundleStore,
    evaluate_geometry_feasibility,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _transform_bytes(pca, lda, scaler) -> bytes:
    arrays = {
        "pca_components": pca.components_,
        "pca_mean": pca.mean_,
        "pca_explained_variance": pca.explained_variance_,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        "pca_singular_values": pca.singular_values_,
        "lda_classes": lda.classes_,
        "lda_priors": lda.priors_,
        "lda_means": lda.means_,
        "lda_scalings": lda.scalings_,
        "lda_xbar": lda.xbar_,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "scaler_var": scaler.var_,
    }
    return _npz_bytes(**arrays)


def _readout_bytes(readout) -> bytes:
    classifier = readout.classifier
    return _npz_bytes(
        classes=readout.classes,
        classifier_classes=classifier.classes_,
        classifier_coef=classifier.coef_,
        classifier_intercept=classifier.intercept_,
        classifier_mean=readout.classifier_mean,
        classifier_scale=readout.classifier_scale,
    )


def _score_all(models, features, readout, density_scorers, alpha, scales):
    raw_scores = compute_raw_scores(models, features, alpha, scales)
    probabilities = readout.predict_proba(raw_scores)
    return probabilities, {
        "minimum_sdf": minimum_sdf_score(raw_scores),
        "maximum_probability": maximum_probability_score(probabilities),
        "sdf_energy": sdf_energy_score(raw_scores),
        **density_scorers.score(features),
    }


def _transformed_splits(data: E4Data, partitions, ood_partitions, pca, lda, scaler):
    id_features = {
        name: _apply_transform(data.id_features[indices], pca, lda, scaler)
        for name, indices in partitions.items()
    }
    ood_features = {
        "near": {
            name: _apply_transform(data.near_features[indices], pca, lda, scaler)
            for name, indices in ood_partitions["near"].items()
        },
        "far": {
            name: _apply_transform(data.far_features[indices], pca, lda, scaler)
            for name, indices in ood_partitions["far"].items()
        },
    }
    return id_features, ood_features


def run_seed(data: E4Data, config: dict[str, Any], seed: int) -> tuple[dict, dict]:
    started = time.perf_counter()
    partitions, partition_audit = build_id_partitions(data, config, seed)
    ood_partitions, ood_audit = build_ood_partitions(data, config)
    model_config = config["model"]
    pca, lda, scaler = _build_transform(
        data.id_features[partitions["geometry"]],
        data.id_labels[partitions["geometry"]],
        int(model_config["pca_components"]),
        seed,
    )
    id_features, ood_features = _transformed_splits(
        data, partitions, ood_partitions, pca, lda, scaler,
    )
    geometry_labels = data.id_labels[partitions["geometry"]]
    class_ids = np.unique(geometry_labels)
    geometry_contract = GeometryCapacityContract(
        allowed_families=("sphere", "axis_aligned", "shrinkage", "full"),
        max_condition_number=1e8,
        max_parameter_sample_ratio=0.75,
        min_effective_rank=2.0,
    )
    feasibility = evaluate_geometry_feasibility(
        id_features["geometry"], geometry_labels, geometry_contract,
    )
    feasibility.require_supportable()

    models = fit_class_models(
        id_features["geometry"],
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
        id_features["geometry"],
        float(model_config["alpha"]),
        class_labels=geometry_labels,
    )
    readout_scores = compute_raw_scores(
        models,
        id_features["readout_calibration"],
        float(model_config["alpha"]),
        scales,
    )
    readout = fit_score_readout(
        "multinomial",
        readout_scores,
        data.id_labels[partitions["readout_calibration"]],
        class_ids,
        seed=seed,
    )
    density_scorers = fit_feature_ood_scorers(
        id_features["geometry"], gmm_components=len(class_ids), seed=seed,
    )

    split_probabilities = {}
    split_scores = {}
    for name in ("risk_control", "validation", "final_test"):
        split_probabilities[name], split_scores[name] = _score_all(
            models, id_features[name], readout, density_scorers,
            float(model_config["alpha"]), scales,
        )
    ood_scores = {"near": {}, "far": {}}
    for family in ood_scores:
        for name in ("validation", "risk_control", "final_test"):
            _, ood_scores[family][name] = _score_all(
                models, ood_features[family][name], readout, density_scorers,
                float(model_config["alpha"]), scales,
            )

    validation_ood = {}
    for score_name in split_scores["validation"]:
        validation_ood[score_name] = {
            family: ood_detection_metrics(
                split_scores["validation"][score_name],
                ood_scores[family]["validation"][score_name],
            )
            for family in ("near", "far")
        }
    authorized = list(config["authorized_ood_scores"])
    selected_score = max(
        authorized,
        key=lambda name: np.mean([
            validation_ood[name][family]["auroc"] for family in ("near", "far")
        ]),
    )
    threshold = select_ood_threshold_at_known_coverage(
        split_scores["risk_control"][selected_score], minimum_known_coverage=0.9,
    )
    final_ood = {
        score_name: {
            family: {
                "detection": ood_detection_metrics(
                    split_scores["final_test"][score_name],
                    ood_scores[family]["final_test"][score_name],
                ),
                **(
                    {"operating_point": ood_operating_point(
                        split_scores["final_test"][score_name],
                        ood_scores[family]["final_test"][score_name],
                        threshold,
                    )} if score_name == selected_score else {}
                ),
            }
            for family in ("near", "far")
        }
        for score_name in split_scores["final_test"]
    }

    baselines = fit_classification_baselines(
        id_features["geometry"],
        geometry_labels,
        {class_id: max(1, len(models[int(class_id)])) for class_id in class_ids},
        seed=seed,
        include_names=set(config["baselines"]),
    )
    final_labels = data.id_labels[partitions["final_test"]]
    final_metrics = {
        "geode_multinomial": classification_metrics(
            final_labels, split_probabilities["final_test"], class_ids,
        )
    }
    final_predictions = {
        "geode_multinomial": class_ids[
            split_probabilities["final_test"].argmax(axis=1)
        ]
    }
    baseline_fit_seconds = {}
    for name, estimator in baselines.items():
        probabilities = estimator.predict_proba(id_features["final_test"])
        final_metrics[name] = classification_metrics(
            final_labels, probabilities, class_ids,
        )
        final_predictions[name] = class_ids[probabilities.argmax(axis=1)]
        baseline_fit_seconds[name] = estimator.fit_seconds_

    transform_blob = _transform_bytes(pca, lda, scaler)
    model_blob = _json_bytes({
        "classes": class_ids.tolist(),
        "score_scales": {str(key): value for key, value in sorted(scales.items())},
        "class_models": {
            str(class_id): _serialize_experts(models[int(class_id)])
            for class_id in class_ids
        },
    })
    readout_blob = _readout_bytes(readout)
    transform_fingerprint = _sha256(transform_blob)
    record = {
        "seed": seed,
        "partition_audit": partition_audit.to_dict(),
        "ood_partition_audit": ood_audit,
        "geometry_feasibility": feasibility.to_dict(),
        "validation_ood": validation_ood,
        "selected_ood_score": selected_score,
        "risk_control_threshold": threshold,
        "final_classification": final_metrics,
        "final_ood": final_ood,
        "prediction_hashes": {
            name: array_fingerprint(value) for name, value in final_predictions.items()
        },
        "model_hashes": {
            "transform": transform_fingerprint,
            "geometry": _sha256(model_blob),
            "readout": _sha256(readout_blob),
        },
        "model_counts": {
            "classes": len(class_ids),
            "experts": sum(len(experts) for experts in models.values()),
            "primitives": sum(
                len(expert.ellipsoids)
                for experts in models.values()
                for expert in experts
            ),
        },
        "baseline_fit_seconds": baseline_fit_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "final_test_used_for_selection": False,
    }
    state = {
        "record": record,
        "partitions": partitions,
        "class_ids": class_ids,
        "scales": scales,
        "selected_score": selected_score,
        "threshold": threshold,
        "transform_blob": transform_blob,
        "model_blob": model_blob,
        "readout_blob": readout_blob,
    }
    return record, state


def _non_inferiority(records: list[dict], config: dict[str, Any]) -> dict[str, Any]:
    endpoint = config["primary_endpoint"]
    differences = np.asarray([
        record["final_classification"]["geode_multinomial"][endpoint]
        - record["final_classification"]["logistic_regression"][endpoint]
        for record in records
    ])
    mean = float(np.mean(differences))
    standard_error = float(stats.sem(differences))
    lower = (
        mean if standard_error == 0.0
        else float(mean - stats.t.ppf(0.95, len(differences) - 1) * standard_error)
    )
    margin = float(config["non_inferiority_margin"])
    return {
        "endpoint": endpoint,
        "margin": margin,
        "paired_differences": differences.tolist(),
        "mean_difference": mean,
        "one_sided_95_percent_lower_bound": lower,
        "passed": lower >= -margin,
        "decision_rule": "one-sided paired t lower bound >= -margin",
        "pilot_artifact": config["margin_pilot_artifact"],
        "pilot_seed": config["margin_pilot_seed"],
    }


def _package_candidate(
    config: dict[str, Any], data: E4Data, state: dict, summary: dict, root: Path,
) -> str:
    class_ids = tuple(int(value) for value in state["class_ids"])
    fingerprint = ModelFingerprint(
        task_name="cifar100_coarse_classification",
        input_spec=InputSpec("raw_cnn", dim=1280),
        output_spec=OutputSpec("probabilities", class_ids),
        alpha=float(config["model"]["alpha"]),
        pca_components=int(config["model"]["pca_components"]),
    )
    partition_hashes = state["record"]["partition_audit"]["partition_hashes"]
    support = SupportProfile(
        model_signature=fingerprint.signature,
        feature_transform_fingerprint=state["record"]["model_hashes"]["transform"],
        training_dataset_fingerprint=_sha256(_json_bytes({
            "data": data.fingerprints, "geometry": partition_hashes["geometry"],
        })),
        calibration_dataset_fingerprint=_sha256(_json_bytes({
            "readout": partition_hashes["readout_calibration"],
            "risk": partition_hashes["risk_control"],
        })),
        class_ids=class_ids,
        score_scales=tuple(float(state["scales"][value]) for value in class_ids),
        novelty_score=state["selected_score"],
        global_threshold=float(state["threshold"]),
        version="e4-v1",
        fit_seed=int(config["deployment_seed"]),
        created_at="2026-07-26T00:00:00Z",
    )
    evidence_blob = _json_bytes(summary)
    config_blob = _json_bytes(config)
    components = {
        "model_state.json": state["model_blob"],
        "transform.npz": state["transform_blob"],
        "readout.npz": state["readout_blob"],
        "evaluation_summary.json": evidence_blob,
        "frozen_config.json": config_blob,
    }
    node = BundleNode(
        name="cifar100",
        artifact_path="model_state.json",
        fingerprint=fingerprint,
        class_order=class_ids,
        feature_transform_fingerprint=support.feature_transform_fingerprint,
        support_profile=support,
    )
    provenance = BundleProvenance(
        routing_mode="exhaustive",
        semantic_router_cache_version="disabled-e4",
        training_manifest_hash=_sha256(config_blob),
        evaluation_manifest_hash=_sha256(evidence_blob),
        metric_summary_hash=_sha256(evidence_blob),
        software_compatibility="python>=3.11,numpy>=2,scikit-learn>=1.6",
        environment_fingerprint=_sha256(
            f"{platform.platform()}|{platform.python_version()}".encode("utf-8")
        ),
        created_at="2026-07-26T00:00:00Z",
        created_by="E4 CIFAR qualification",
    )
    manifest = LocalModelBundleStore(root).publish(
        components, [node], provenance=provenance,
    )
    LocalModelBundleStore(root).activate(manifest.bundle_id)
    return manifest.bundle_id


def run_qualification(config_path: Path, bundle_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    data = load_e4_data(config)
    records = []
    deployment_state = None
    for seed in config["seeds"]:
        print(f"E4 seed {seed}: fitting and evaluating", flush=True)
        record, state = run_seed(data, config, int(seed))
        records.append(record)
        if seed == config["deployment_seed"]:
            deployment_state = state
    non_inferiority = _non_inferiority(records, config)
    replay_record, _ = run_seed(data, config, int(config["deployment_seed"]))
    deployment_record = next(
        record for record in records if record["seed"] == config["deployment_seed"]
    )
    replay_identity = {
        "model_hashes": replay_record["model_hashes"] == deployment_record["model_hashes"],
        "prediction_hashes": (
            replay_record["prediction_hashes"] == deployment_record["prediction_hashes"]
        ),
        "classification_metrics": (
            replay_record["final_classification"]
            == deployment_record["final_classification"]
        ),
        "ood_metrics": replay_record["final_ood"] == deployment_record["final_ood"],
        "selected_ood_score": (
            replay_record["selected_ood_score"]
            == deployment_record["selected_ood_score"]
        ),
        "risk_control_threshold": (
            replay_record["risk_control_threshold"]
            == deployment_record["risk_control_threshold"]
        ),
    }
    replay_identity["passed"] = all(replay_identity.values())
    reproducibility = {
        "partition_hashes_unique_by_seed": len({
            canonical_json(record["partition_audit"]["partition_hashes"])
            for record in records
        }) == len(records),
        "fixed_ood_partitions": len({
            canonical_json(record["ood_partition_audit"]) for record in records
        }) == 1,
        "all_final_tests_observational": all(
            not record["final_test_used_for_selection"] for record in records
        ),
        "data_fingerprints": data.fingerprints,
        "pretraining": config["pretraining"],
        "deployment_seed_replay_identity": replay_identity,
    }
    summary = {
        "schema_version": 1,
        "milestone": "E4",
        "config_hash": _sha256(config_path.read_bytes()),
        "seeds": config["seeds"],
        "records": records,
        "non_inferiority": non_inferiority,
        "reproducibility_audit": reproducibility,
        "generalized_ood_complete": all(
            set(record["final_ood"][record["selected_ood_score"]]) == {"near", "far"}
            for record in records
        ),
    }
    core_passed = (
        non_inferiority["passed"]
        and reproducibility["partition_hashes_unique_by_seed"]
        and reproducibility["fixed_ood_partitions"]
        and reproducibility["all_final_tests_observational"]
        and replay_identity["passed"]
        and summary["generalized_ood_complete"]
    )
    summary["core_gates_passed"] = core_passed
    if core_passed:
        summary["candidate_bundle_id"] = _package_candidate(
            config, data, deployment_state, summary, bundle_root,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("experiments/configs/e4_cifar_qualification.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("logs/results/e4_cifar_qualification.json"),
    )
    parser.add_argument(
        "--bundle-root", type=Path,
        default=Path("logs/results/e4_model_registry"),
    )
    arguments = parser.parse_args()
    summary = run_qualification(arguments.config, arguments.bundle_root)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({
        "core_gates_passed": summary["core_gates_passed"],
        "non_inferiority": summary["non_inferiority"],
        "candidate_bundle_id": summary.get("candidate_bundle_id"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()