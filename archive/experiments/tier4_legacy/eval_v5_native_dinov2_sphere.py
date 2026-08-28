"""Evaluate native 384D DINOv2 with direct spherical GEODE fitting."""

from __future__ import annotations

import argparse
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
    compute_geode_component_efficiency,
    evaluate_head,
    fit_geode_head,
    fit_logistic_head,
    fit_prototype_head,
    fit_rbf_head,
    fit_weighted_knn_head,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/configs/v5/m19_native_dinov2_sphere.json",
    )
    parser.add_argument(
        "--feature-dir",
        default="data/v5/features/m19_native_dinov2_sphere",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/results/v5/m19_native_dinov2_sphere",
    )
    return parser.parse_args()


def _load_bound_cache(
    feature_dir: Path,
    backbone_id: str,
    manifest: RepresentationManifest,
    metadata_payload: dict[str, Any],
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    metadata = FeatureCacheMetadata.from_dict(metadata_payload)
    require_cache_binding(metadata, manifest)
    path = (
        feature_dir
        / backbone_id
        / f"features_{split}_{manifest.representation_hash[:16]}.npz"
    )
    verify_cache_file_integrity(path, metadata.feature_file_hash)
    with np.load(path) as cache:
        return (
            cache["features"].astype(np.float64),
            cache["labels"].astype(np.int64),
        )


def run_evaluation(
    config_path: str | Path = "experiments/configs/v5/m19_native_dinov2_sphere.json",
    feature_dir: str | Path = "data/v5/features/m19_native_dinov2_sphere",
    output_dir: str | Path = "logs/results/v5/m19_native_dinov2_sphere",
) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    feature_dir = Path(feature_dir)
    output_dir = Path(output_dir)
    extraction = json.loads(
        (feature_dir / "extraction_summary.json").read_text(encoding="utf-8")
    )
    backbone = config["backbones"][0]
    backbone_id = backbone["id"]
    extracted = extraction["representations"][backbone_id]
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
        raise ValueError("Native DINOv2 representation provenance mismatch.")

    datasets = {
        split: _load_bound_cache(
            feature_dir,
            backbone_id,
            manifest,
            extracted["cache_metadata"][split],
            split,
        )
        for split in ("train", "dev", "test")
    }
    X_train, y_train = datasets["train"]
    X_dev, y_dev = datasets["dev"]
    X_test, y_test = datasets["test"]
    dimension = X_train.shape[1]
    class_counts = np.bincount(y_train)
    minimum_support = int(class_counts[class_counts > 0].min())
    required_support = dimension + 2
    if minimum_support < required_support:
        raise ValueError(
            f"Native sphere needs {required_support} points per class; "
            f"only {minimum_support} are available."
        )

    knn = config["weighted_knn"]
    geode = config["geode_config"]
    fitters = (
        (
            "weighted_knn",
            lambda: fit_weighted_knn_head(
                X_train,
                y_train,
                n_neighbors=knn["n_neighbors"],
                temperature=knn["temperature"],
                query_batch_size=knn["query_batch_size"],
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
        (
            "current_geode",
            lambda: fit_geode_head(
                X_train,
                y_train,
                config["seed"],
                max_iterations=geode["max_iterations"],
                consensus_threshold=geode["consensus_threshold"],
                dimension_limit=dimension,
            ),
        ),
    )

    heads = {}
    fitted_geode = None
    for name, fitter in fitters:
        print(f"Fitting native DINOv2/{name}...")
        started = time.perf_counter()
        head = fitter()
        fit_seconds = time.perf_counter() - started
        heads[name] = {
            "dev": evaluate_head(head, X_dev, y_dev),
            "test": evaluate_head(head, X_test, y_test),
            "fit_wall_seconds": fit_seconds,
        }
        if name == "current_geode":
            fitted_geode = head
            heads[name]["component_efficiency"] = (
                compute_geode_component_efficiency(
                    head,
                    X_train,
                    y_train,
                    target_coverage=geode[
                        "component_efficiency_target_coverage"
                    ],
                    capture_threshold=geode["capture_threshold"],
                )
            )

    assert fitted_geode is not None
    per_class_primitives = [
        sum(
            len(expert.ellipsoids)
            for expert in class_model["model"]
        )
        if class_model["model"] is not None
        else 0
        for class_model in fitted_geode["model"]
    ]
    results = {
        "schema_version": 1,
        "milestone": config["milestone"],
        "stage": config["stage"],
        "dataset": config["dataset"],
        "seed": config["seed"],
        "split_hashes": extraction["split_hashes"],
        "representation_hash": manifest.representation_hash,
        "backbone": backbone_id,
        "output_dimension": dimension,
        "primitive_family": "sphere",
        "minimum_seed_rule": "d_plus_2",
        "minimum_seed_size": required_support,
        "minimum_training_class_support": minimum_support,
        "test_used_for_selection": False,
        "representation_diagnostics": compute_representation_diagnostics(
            X_train, y_train
        ),
        "heads": heads,
        "geode_structure": {
            "per_class_primitives": per_class_primitives,
            "total_primitives": int(sum(per_class_primitives)),
        },
        "claim_boundary": (
            "One-seed bounded native-support study. Meeting the algebraic seed "
            "minimum does not establish adequate statistical support."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = config.get("artifact_id", "m19_native_dinov2_sphere")
    evidence = output_dir / f"{artifact_id}_evidence.json"
    evidence.write_text(
        canonical_json(results) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    build_artifact_index(output_dir)
    return results


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        config_path=args.config,
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
    )
