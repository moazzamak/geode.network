"""Package real text and point-cloud GEODE models under shared contracts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import platform

import numpy as np

from experiments.common.data_cache import data_cache_root
from experiments.common.experiment_manifest import array_fingerprint, canonical_json
from experiments.common.moe_eval import (
    fit_experts,
    mean_abs_sdf_error,
    split_train_test_indices,
)
from experiments.e2e.e8_bundle_loader import load_e8_bundle
from experiments.e2e.run_e4_cifar_qualification import (
    _readout_bytes,
    _transform_bytes,
)
from experiments.e2e.run_tier4_smoke import _serialize_experts
from experiments.tier6.eval_temporal_text_prediction import (
    VOCAB_SIZE,
    apply_transform_pipeline,
    compute_score_scales,
    fit_adaptive_class_models,
    fit_score_calibrator,
    fit_transform_pipeline,
    geometry_calibration_split,
    linear_context_accuracy,
    predict_calibrated_labels,
    predict_char_labels,
    probability_perplexity,
    sample_context_pairs,
    sampled_ngram_accuracy,
    top_k_probability_accuracy,
)
from src.inference_engine import InferenceEngine
from src.model_fingerprint import InputSpec, ModelFingerprint, OutputSpec
from src.open_set import SupportProfile
from src.runtime import BundleNode, BundleProvenance, LocalModelBundleStore
from src.runtime.modelnet_manifest import ModelNet40Manifest
from src.sdf_engine import EllipsoidExpert, Expert


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _readout_blob(calibrator, classes: np.ndarray) -> bytes:
    scaler = calibrator.named_steps["standardscaler"]
    classifier = calibrator.named_steps["logisticregression"]
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        classes=classes,
        classifier_classes=classifier.classes_,
        classifier_coef=classifier.coef_,
        classifier_intercept=classifier.intercept_,
        classifier_mean=scaler.mean_,
        classifier_scale=scaler.scale_,
    )
    return buffer.getvalue()


def _pointcloud_model(config: dict) -> dict:
    path = data_cache_root() / config["pointcloud_path"]
    with np.load(path, allow_pickle=False) as state:
        clouds = state["pointclouds"]
        labels = state["labels"]
        splits = state["splits"]
    if clouds.ndim != 3 or clouds.shape[-1] != 3:
        raise ValueError("pointclouds must have shape (shapes, points, 3)")
    if clouds.shape[1] != 2048 or len(labels) != len(clouds) or len(splits) != len(clouds):
        raise ValueError("ModelNet40 artifact arrays are incompatible")
    if not np.array_equal(np.unique(labels), np.arange(40)):
        raise ValueError("ModelNet40 artifact must contain all 40 labels")
    split_counts = {
        "train": int(np.count_nonzero(splits == 0)),
        "test": int(np.count_nonzero(splits == 1)),
    }
    points = clouds[:int(config["pointcloud_shapes"])].reshape(-1, 3).astype(np.float64)
    train_indices, test_indices = split_train_test_indices(
        len(points), test_fraction=0.2, seed=int(config["seed"]),
    )
    train = points[train_indices]
    test = points[test_indices]
    experts = fit_experts(
        train,
        consensus_threshold=0.15,
        capture_threshold=0.08,
        alpha=1.0,
        max_iterations=int(config["pointcloud_max_iterations"]),
        nudge_iterations=10,
        nudge_learning_rate=0.02,
        seed=int(config["seed"]),
    )
    center = train.mean(axis=0)
    radius = float(np.median(np.linalg.norm(train - center, axis=1)))
    sphere = Expert(alpha=1.0)
    sphere.add_ellipsoid(EllipsoidExpert(center=center, radii=np.full(3, radius)))
    predictions = InferenceEngine(experts, alpha=1.0).get_fused_sdf(test)
    model_blob = _json_bytes({
        **_serialize_experts(experts),
        "alpha": 1.0,
    })
    return {
        "experts": experts,
        "test": test,
        "prediction_hash": array_fingerprint(predictions),
        "metrics": {
            "geode_test_mean_abs_sdf": float(np.mean(np.abs(predictions))),
            "single_sphere_test_mean_abs_sdf": mean_abs_sdf_error([sphere], test, 1.0),
            "shapes": int(config["pointcloud_shapes"]),
            "points": len(points),
            "experts": len(experts),
        },
        "model_blob": model_blob,
        "dataset_fingerprint": array_fingerprint(points),
        "split_counts": split_counts,
    }


def _text_model(config: dict) -> dict:
    with np.load(config["text_path"], allow_pickle=False) as state:
        train_ids = state["train_ids"][:int(config["text_train_prefix"])]
        test_ids = state["test_ids"][:int(config["text_test_prefix"])]
    window = int(config["text_window"])
    seed = int(config["seed"])
    train_contexts, train_labels = sample_context_pairs(
        train_ids,
        window=window,
        max_samples=int(config["text_train_pairs"]),
        seed=seed,
    )
    test_contexts, test_labels = sample_context_pairs(
        test_ids,
        window=window,
        max_samples=int(config["text_test_pairs"]),
        seed=seed + 1,
    )
    counts = Counter(int(value) for value in train_labels)
    retained = np.asarray([
        class_id for class_id, _ in counts.most_common(int(config["text_active_classes"]))
    ], dtype=np.int32)
    train_mask = np.isin(train_labels, retained)
    test_mask = np.isin(test_labels, retained)
    train_contexts = train_contexts[train_mask]
    train_labels = train_labels[train_mask]
    retained_test_contexts = test_contexts[test_mask]
    retained_test_labels = test_labels[test_mask]
    geometry_indices, calibration_indices = geometry_calibration_split(
        np.arange(len(train_contexts)), calibration_fraction=0.15, gap=window,
    )
    pca, lda, scaler = fit_transform_pipeline(
        train_contexts[geometry_indices],
        train_labels[geometry_indices],
        int(config["text_pca_components"]),
        seed,
    )
    transformed_train = apply_transform_pipeline(train_contexts, pca, lda, scaler)
    transformed_test = apply_transform_pipeline(
        retained_test_contexts, pca, lda, scaler,
    )
    geometry_labels = train_labels[geometry_indices]
    classes = np.unique(geometry_labels)
    models, complexity = fit_adaptive_class_models(
        transformed_train[geometry_indices],
        geometry_labels,
        classes,
        consensus_threshold=0.10,
        capture_threshold=0.08,
        alpha=float(config["alpha"]),
        max_iterations=int(config["text_max_iterations"]),
        nudge_iterations=0,
        nudge_learning_rate=0.02,
        seed=seed,
    )
    scales = compute_score_scales(
        models,
        transformed_train[geometry_indices],
        float(config["alpha"]),
        class_labels=geometry_labels,
    )
    calibration_mask = np.isin(train_labels[calibration_indices], classes)
    _, calibration_scores = predict_char_labels(
        models,
        transformed_train[calibration_indices][calibration_mask],
        float(config["alpha"]),
        scales,
    )
    calibrator = fit_score_calibrator(
        calibration_scores,
        train_labels[calibration_indices][calibration_mask],
    )
    final_mask = np.isin(retained_test_labels, calibrator.classes_)
    final_contexts = retained_test_contexts[final_mask]
    final_labels = retained_test_labels[final_mask]
    final_transformed = transformed_test[final_mask]
    _, final_scores = predict_char_labels(
        models, final_transformed, float(config["alpha"]), scales,
    )
    predictions, probabilities = predict_calibrated_labels(calibrator, final_scores)
    linear_accuracy = linear_context_accuracy(
        train_contexts[geometry_indices],
        geometry_labels,
        final_contexts,
        final_labels,
        seed=seed,
    )
    ngram_accuracy = sampled_ngram_accuracy(
        train_contexts[geometry_indices],
        geometry_labels,
        final_contexts,
        final_labels,
        window,
    )
    model_blob = _json_bytes({
        "alpha": float(config["alpha"]),
        "classes": classes.tolist(),
        "score_scales": {str(key): value for key, value in sorted(scales.items())},
        "class_models": {
            str(class_id): _serialize_experts(models[int(class_id)])
            for class_id in classes
        },
    })
    return {
        "classes": calibrator.classes_,
        "models": models,
        "scales": scales,
        "final_contexts": final_contexts,
        "predictions": predictions,
        "prediction_hash": array_fingerprint(predictions),
        "metrics": {
            "accuracy": float(np.mean(predictions == final_labels)),
            "top_5_accuracy": top_k_probability_accuracy(
                final_labels, probabilities, calibrator.classes_, 5,
            ),
            "perplexity": probability_perplexity(
                final_labels, probabilities, calibrator.classes_,
            ),
            "linear_context_accuracy": linear_accuracy,
            "matched_ngram_accuracy": ngram_accuracy,
            "retained_test_coverage": float(len(final_labels) / len(test_labels)),
            "active_classes": len(classes),
            "complexity": dict(Counter(complexity.values())),
        },
        "model_blob": model_blob,
        "transform_blob": _transform_bytes(pca, lda, scaler),
        "readout_blob": _readout_blob(calibrator, calibrator.classes_),
        "dataset_fingerprint": _sha256(
            array_fingerprint(train_ids).encode("ascii")
            + array_fingerprint(test_ids).encode("ascii")
        ),
    }


def _package(config: dict, point: dict, text: dict, summary: dict, root: Path) -> str:
    point_fingerprint = ModelFingerprint(
        task_name="modelnet40_surface_reconstruction",
        input_spec=InputSpec("passthrough", dim=3),
        output_spec=OutputSpec("sdf_scores", ("surface",)),
        alpha=1.0,
        pca_components=3,
    )
    text_classes = tuple(int(value) for value in text["classes"])
    text_fingerprint = ModelFingerprint(
        task_name="wikitext103_character_prediction",
        input_spec=InputSpec(
            "passthrough", dim=int(config["text_window"]) * VOCAB_SIZE,
        ),
        output_spec=OutputSpec("probabilities", text_classes),
        alpha=float(config["alpha"]),
        pca_components=int(config["text_pca_components"]),
    )
    point_transform = "identity-3d"
    text_transform = _sha256(text["transform_blob"])
    point_support = SupportProfile(
        point_fingerprint.signature,
        point_transform,
        point["dataset_fingerprint"],
        point["dataset_fingerprint"],
        ("surface",),
        (1.0,),
        "absolute_sdf",
        0.08,
        "e8-modelnet40-v1",
        int(config["seed"]),
        "2026-07-26T00:00:00Z",
    )
    text_support = SupportProfile(
        text_fingerprint.signature,
        text_transform,
        text["dataset_fingerprint"],
        text["dataset_fingerprint"],
        text_classes,
        tuple(float(text["scales"][value]) for value in text_classes),
        "maximum_probability",
        0.0,
        "e8-wikitext103-v1",
        int(config["seed"]),
        "2026-07-26T00:00:00Z",
    )
    evidence = _json_bytes(summary)
    frozen_config = _json_bytes(config)
    provenance = BundleProvenance(
        routing_mode="exhaustive",
        semantic_router_cache_version="disabled-e8",
        training_manifest_hash=_sha256(frozen_config),
        evaluation_manifest_hash=_sha256(evidence),
        metric_summary_hash=_sha256(evidence),
        software_compatibility="python>=3.11,numpy>=2,scikit-learn>=1.6",
        environment_fingerprint=_sha256(
            f"{platform.platform()}|{platform.python_version()}".encode("utf-8")
        ),
        created_at="2026-07-26T00:00:00Z",
        created_by="E8 cross-modal qualification",
    )
    store = LocalModelBundleStore(root)
    manifest = store.publish(
        {
            "point_model.json": point["model_blob"],
            "text_model.json": text["model_blob"],
            "text_transform.npz": text["transform_blob"],
            "text_readout.npz": text["readout_blob"],
            "evaluation_summary.json": evidence,
            "frozen_config.json": frozen_config,
        },
        [
            BundleNode(
                "modelnet40",
                "point_model.json",
                point_fingerprint,
                ("surface",),
                point_transform,
                support_profile=point_support,
            ),
            BundleNode(
                "wikitext103",
                "text_model.json",
                text_fingerprint,
                text_classes,
                text_transform,
                support_profile=text_support,
            ),
        ],
        provenance=provenance,
    )
    store.activate(manifest.bundle_id)
    return manifest.bundle_id


def run_qualification(config_path: Path, registry: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported E8 configuration schema")
    cache_root = data_cache_root()
    point_manifest = ModelNet40Manifest.load(cache_root / config["pointcloud_manifest"])
    point_manifest_report = point_manifest.verify(cache_root / "modelnet40")
    point = _pointcloud_model(config)
    if point["split_counts"] != point_manifest_report["split_samples"]:
        raise ValueError("ModelNet40 artifact split counts do not match its manifest")
    text = _text_model(config)
    modelnet40_available = (
        config["required_pointcloud_dataset"] == "ModelNet40"
        and point_manifest_report["class_count"] == 40
    )
    summary = {
        "schema_version": 1,
        "milestone": "E8",
        "qualification_status": "passed" if modelnet40_available else "blocked",
        "gate_passed": modelnet40_available,
        "blockers": ([] if modelnet40_available else ["modelnet40_data_missing"]),
        "pointcloud": {
            "required_dataset": config["required_pointcloud_dataset"],
            "actual_dataset": "ModelNet40",
            "proxy_only": not modelnet40_available,
            "manifest": point_manifest_report,
            **point["metrics"],
        },
        "text": text["metrics"],
        "shared_lifecycle_contract": "content-addressed-json-numpy-bundle-v1",
    }
    summary["bundle_id"] = _package(config, point, text, summary, registry)
    loaded = load_e8_bundle(registry)
    point_hash = array_fingerprint(loaded.point_scores(point["test"]))
    text_hash = array_fingerprint(loaded.text_predict(text["final_contexts"]))
    summary["bundle_replay"] = {
        "point_prediction_hash": point_hash == point["prediction_hash"],
        "text_prediction_hash": text_hash == text["prediction_hash"],
        "manifest_verified": loaded.manifest.bundle_id == summary["bundle_id"],
    }
    summary["bundle_replay"]["passed"] = all(summary["bundle_replay"].values())
    if not summary["bundle_replay"]["passed"]:
        raise RuntimeError("E8 bundle replay failed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("experiments/configs/e8_cross_modal_qualification.json"),
    )
    parser.add_argument(
        "--registry", type=Path,
        default=Path("logs/results/e8_model_registry"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("logs/results/e8_cross_modal_qualification.json"),
    )
    parser.add_argument("--allow-blocked", action="store_true")
    arguments = parser.parse_args()
    result = run_qualification(arguments.config, arguments.registry)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["gate_passed"] and not arguments.allow_blocked:
        raise RuntimeError(f"E8 qualification blocked: {result['blockers']}")


if __name__ == "__main__":
    main()