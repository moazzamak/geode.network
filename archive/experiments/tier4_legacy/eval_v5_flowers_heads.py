"""Evaluate bounded Flowers-102 identity features with frozen-space probes."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.experiment_manifest import canonical_json
from experiments.common.representation_metrics import (
    compute_representation_diagnostics,
)
from experiments.common.v5_artifacts import build_artifact_index
from experiments.common.v5_frozen_representations import (
    FeatureCacheMetadata,
    RepresentationManifest,
    compute_preprocessing_digest,
    require_cache_binding,
    verify_cache_file_integrity,
)
from experiments.tier4.eval_v5_frozen_space_heads import (
    compute_compactness,
    evaluate_head,
    fit_logistic_head,
    fit_prototype_head,
    fit_rbf_head,
    fit_weighted_knn_head,
)


def _blocked_head(name: str, reason: str) -> dict[str, Any]:
    return {
        "head": name,
        "status": "blocked",
        "reason": reason,
        "accuracy": None,
        "balanced_accuracy": None,
        "nll": None,
    }


def _minimum_class_support(labels: np.ndarray) -> int:
    _, counts = np.unique(labels, return_counts=True)
    return int(counts.min())


def run_evaluation(
    config_path: str | Path = "experiments/configs/v5/m19_flowers102_s1.json",
    feature_dir: str | Path = "data/v5/features/m19_flowers102_s1",
    output_dir: str | Path = "logs/results/v5/m19_flowers102_s1",
) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    backbone_config = json.loads(
        (PROJECT_ROOT / config["backbone_config"]).read_text(encoding="utf-8")
    )
    feature_dir = Path(feature_dir)
    output_dir = Path(output_dir)
    summary_path = feature_dir / "extraction_summary.json"
    extraction = json.loads(summary_path.read_text(encoding="utf-8"))
    if extraction["dataset"] != config["dataset"]:
        raise ValueError("Flowers extraction dataset does not match the active config.")

    results: dict[str, Any] = {
        "schema_version": 1,
        "milestone": "M19",
        "stage": config["stage"],
        "dataset": config["dataset"],
        "seed": config["seed"],
        "samples_per_class": config["samples_per_class"],
        "split_hashes": extraction["split_hashes"],
        "identity_interfaces_only": True,
        "representations": {},
        "blocked_arms": [],
        "claim_boundary": (
            "Bounded one-seed transfer feasibility only; five train examples "
            "per class do not support population-level ranking claims."
        ),
    }

    knn_config = backbone_config["weighted_knn"]
    best_head = {"accuracy": -1.0, "backbone": None, "head": None}
    for backbone in backbone_config["backbones"]:
        if backbone.get("status") == "blocked":
            continue
        backbone_id = backbone["id"]
        extracted = extraction["representations"].get(backbone_id)
        if extracted is None or extracted.get("status") != "extracted":
            results["blocked_arms"].append({
                "backbone": backbone_id,
                "reason": "No extracted representation in the Flowers cache.",
            })
            continue

        manifest = RepresentationManifest.from_dict(extracted["manifest"])
        expected_preprocessing = compute_preprocessing_digest(
            PROJECT_ROOT / backbone["preprocessor_config"]
        )
        if (
            manifest.backbone_id != backbone_id
            or manifest.upstream_weights_digest != backbone["weights_sha256"]
            or manifest.preprocessing_digest != expected_preprocessing
            or manifest.training_split_hash != extraction["split_hashes"]["train"]
            or manifest.output_dimension != backbone["output_dimension"]
            or manifest.interface_architecture != "identity"
        ):
            raise ValueError(
                f"Flowers representation provenance mismatch for {backbone_id}."
            )

        split_data = {}
        for split in ("train", "dev", "test"):
            metadata = FeatureCacheMetadata.from_dict(
                extracted["cache_metadata"][split]
            )
            require_cache_binding(metadata, manifest)
            cache_path = (
                feature_dir
                / backbone_id
                / f"features_{split}_{manifest.representation_hash[:16]}.npz"
            )
            verify_cache_file_integrity(cache_path, metadata.feature_file_hash)
            with np.load(cache_path) as cache:
                split_data[split] = {
                    "features": cache["features"].astype(np.float64),
                    "labels": cache["labels"].astype(np.int64),
                    "indices": cache["indices"].astype(np.int64),
                }

        X_train = split_data["train"]["features"]
        y_train = split_data["train"]["labels"]
        X_dev = split_data["dev"]["features"]
        y_dev = split_data["dev"]["labels"]
        X_test = split_data["test"]["features"]
        y_test = split_data["test"]["labels"]
        minimum_support = _minimum_class_support(y_train)
        dimension = X_train.shape[1]
        diagnostic_neighbors = min(10, minimum_support - 1)

        fitters = (
            (
                "weighted_knn",
                lambda: fit_weighted_knn_head(
                    X_train,
                    y_train,
                    n_neighbors=knn_config["n_neighbors"],
                    temperature=knn_config["temperature"],
                    query_batch_size=knn_config["query_batch_size"],
                ),
            ),
            ("linear_logistic", lambda: fit_logistic_head(
                X_train, y_train, config["seed"]
            )),
            ("rbf_svm", lambda: fit_rbf_head(
                X_train, y_train, config["seed"]
            )),
            ("prototype", lambda: fit_prototype_head(
                X_train, y_train, config["seed"]
            )),
        )
        heads = {}
        for head_name, fitter in fitters:
            print(f"Fitting {backbone_id}/{head_name}...")
            started = time.perf_counter()
            head = fitter()
            fit_seconds = time.perf_counter() - started
            heads[head_name] = {
                "dev": evaluate_head(head, X_dev, y_dev),
                "test": evaluate_head(head, X_test, y_test),
                "fit_wall_seconds": fit_seconds,
            }
            test_accuracy = heads[head_name]["test"]["accuracy"]
            if test_accuracy is not None and test_accuracy > best_head["accuracy"]:
                best_head = {
                    "accuracy": test_accuracy,
                    "backbone": backbone_id,
                    "head": head_name,
                }

        heads["gaussian_mixture"] = {
            "dev": _blocked_head(
                "gaussian_mixture",
                f"Full-covariance GMM is under-supported: {minimum_support} "
                f"training samples per class for {dimension} dimensions.",
            ),
            "test": _blocked_head(
                "gaussian_mixture",
                f"Full-covariance GMM is under-supported: {minimum_support} "
                f"training samples per class for {dimension} dimensions.",
            ),
            "fit_wall_seconds": None,
        }
        sphere_seed = dimension + 2
        geode_reason = (
            f"Native spherical GEODE requires d+2={sphere_seed} seed points "
            f"per class; Flowers S1 provides {minimum_support}."
        )
        heads["current_geode"] = {
            "dev": _blocked_head("current_geode", geode_reason),
            "test": _blocked_head("current_geode", geode_reason),
            "fit_wall_seconds": None,
            "component_efficiency": {
                "status": "blocked",
                "reason": geode_reason,
                "target_coverage": backbone_config["geode_config"][
                    "component_efficiency_target_coverage"
                ],
            },
        }

        results["representations"][backbone_id] = {
            "representation_hash": manifest.representation_hash,
            "checkpoint_license": manifest.checkpoint_license,
            "output_dimension": dimension,
            "minimum_training_class_support": minimum_support,
            "compactness": compute_compactness(X_train, y_train),
            "diagnostic_neighbors": diagnostic_neighbors,
            "representation_diagnostics": compute_representation_diagnostics(
                X_train,
                y_train,
                n_neighbors=diagnostic_neighbors,
            ),
            "heads": heads,
        }

    results["best_test_head"] = best_head
    results["advancement_gates"] = {
        "cache_integrity_verified": True,
        "all_identity_backbones_evaluated": (
            len(results["representations"]) == 3
        ),
        "native_geode_support_adequate": False,
        "notes": (
            "Use the official larger training split or another preregistered "
            "support increase before native-dimensional spherical GEODE."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "m19_flowers102_s1_evidence.json"
    evidence_path.write_text(
        canonical_json(results) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    build_artifact_index(output_dir)
    return results


if __name__ == "__main__":
    run_evaluation()
