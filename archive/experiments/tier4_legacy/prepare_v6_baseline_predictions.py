"""Lock M27 baseline predictions and the primary RBF teacher checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v5_frozen_representations import RepresentationManifest
from experiments.tier4.eval_v5_frozen_space_heads import (
    fit_geode_head,
    fit_rbf_head,
    fit_weighted_knn_head,
    predict_geode,
)
from experiments.tier4.eval_v5_native_dinov2_sphere import _load_bound_cache


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "configs"
    / "v5"
    / "m19_native_dinov2_sphere_support_pilot.json"
)
DEFAULT_FEATURE_DIR = (
    REPO_ROOT
    / "data"
    / "v5"
    / "features"
    / "m19_native_dinov2_sphere_support_pilot"
)
DEFAULT_SOURCE_EVIDENCE = (
    REPO_ROOT
    / "logs"
    / "results"
    / "v5"
    / "m19_native_dinov2_sphere_support_pilot"
    / "m19_native_dinov2_sphere_support_pilot_evidence.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "logs" / "results" / "v6" / "m27_baseline_predictions"
)


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        np.mean(
            [
                np.mean(predictions[labels == class_id] == class_id)
                for class_id in np.unique(labels)
            ]
        )
    )


def _predict(head: dict[str, Any], features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if head["name"] == "current_geode":
        return predict_geode(head["model"], features, head["classes"])
    model = head["model"]
    return model.predict(features), model.predict_proba(features)


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def prepare_baseline_predictions(
    config_path: Path = DEFAULT_CONFIG,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    source_evidence_path: Path = DEFAULT_SOURCE_EVIDENCE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    extraction = json.loads(
        (feature_dir / "extraction_summary.json").read_text(encoding="utf-8")
    )
    evidence = json.loads(source_evidence_path.read_text(encoding="utf-8"))
    backbone_id = config["backbones"][0]["id"]
    extracted = extraction["representations"][backbone_id]
    manifest = RepresentationManifest.from_dict(extracted["manifest"])
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
    train_features, train_labels = datasets["train"]
    geode = config["geode_config"]
    knn = config["weighted_knn"]
    heads = {
        "weighted_knn": fit_weighted_knn_head(
            train_features,
            train_labels,
            n_neighbors=knn["n_neighbors"],
            temperature=knn["temperature"],
            query_batch_size=knn["query_batch_size"],
        ),
        "rbf_svm": fit_rbf_head(train_features, train_labels, config["seed"]),
        "current_geode": fit_geode_head(
            train_features,
            train_labels,
            config["seed"],
            max_iterations=geode["max_iterations"],
            consensus_threshold=geode["consensus_threshold"],
            dimension_limit=train_features.shape[1],
        ),
    }

    if output_dir.exists():
        shutil.rmtree(output_dir)
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True)
    labels_metadata = {}
    for source_split, output_split in (("dev", "development"), ("test", "test")):
        labels_path = arrays_dir / f"{output_split}_labels.npy"
        np.save(labels_path, datasets[source_split][1], allow_pickle=False)
        labels_metadata[output_split] = {
            "path": _relative(labels_path),
            "sha256": sha256_file(labels_path),
        }

    head_metadata = {}
    for head_name, head in heads.items():
        split_metadata = {}
        for source_split, output_split in (("dev", "development"), ("test", "test")):
            features, labels = datasets[source_split]
            predictions, probabilities = _predict(head, features)
            predictions_path = arrays_dir / f"{head_name}_{output_split}_predictions.npy"
            probabilities_path = (
                arrays_dir / f"{head_name}_{output_split}_probabilities.npy"
            )
            np.save(predictions_path, predictions, allow_pickle=False)
            np.save(probabilities_path, probabilities, allow_pickle=False)
            balanced_accuracy = _balanced_accuracy(labels, predictions)
            expected = evidence["heads"][head_name][source_split]["balanced_accuracy"]
            if not np.isclose(
                balanced_accuracy, expected, rtol=0.0, atol=1e-12
            ):
                raise RuntimeError(
                    f"{head_name}/{output_split} reproduced {balanced_accuracy}, "
                    f"expected frozen value {expected}."
                )
            split_metadata[output_split] = {
                "predictions_path": _relative(predictions_path),
                "predictions_sha256": sha256_file(predictions_path),
                "probabilities_path": _relative(probabilities_path),
                "probabilities_sha256": sha256_file(probabilities_path),
                "balanced_accuracy": balanced_accuracy,
            }
        head_metadata[head_name] = split_metadata

    teacher_train_predictions, teacher_train_probabilities = _predict(
        heads["rbf_svm"], train_features
    )
    teacher_train_labels_path = arrays_dir / "teacher_train_labels.npy"
    teacher_train_predictions_path = arrays_dir / "teacher_train_predictions.npy"
    teacher_train_probabilities_path = arrays_dir / "teacher_train_probabilities.npy"
    np.save(teacher_train_labels_path, train_labels, allow_pickle=False)
    np.save(
        teacher_train_predictions_path,
        teacher_train_predictions,
        allow_pickle=False,
    )
    np.save(
        teacher_train_probabilities_path,
        teacher_train_probabilities,
        allow_pickle=False,
    )
    teacher_path = output_dir / "rbf_teacher.json"
    write_canonical_json(
        teacher_path,
        {
            "schema_version": 1,
            "family": "rbf_svm",
            "seed": config["seed"],
            "hyperparameters": {
                "C": 1.0,
                "kernel": "rbf",
                "calibration": "sigmoid",
                "calibration_folds": 3,
            },
            "representation_hash": manifest.representation_hash,
            "training_split_hash": extraction["split_hashes"]["train"],
            "classes": heads["rbf_svm"]["classes"].tolist(),
            "labels_path": _relative(teacher_train_labels_path),
            "labels_sha256": sha256_file(teacher_train_labels_path),
            "predictions_path": _relative(teacher_train_predictions_path),
            "predictions_sha256": sha256_file(teacher_train_predictions_path),
            "probabilities_path": _relative(teacher_train_probabilities_path),
            "probabilities_sha256": sha256_file(teacher_train_probabilities_path),
        },
    )

    metadata = {
        "schema_version": 1,
        "milestone": "M27",
        "representation_hash": manifest.representation_hash,
        "split_hashes": {
            "train": extraction["split_hashes"]["train"],
            "development": extraction["split_hashes"]["dev"],
            "test": extraction["split_hashes"]["test"],
        },
        "feature_hashes": {
            "train": extracted["cache_metadata"]["train"]["feature_file_hash"],
            "development": extracted["cache_metadata"]["dev"]["feature_file_hash"],
            "test": extracted["cache_metadata"]["test"]["feature_file_hash"],
        },
        "source_evidence_hash": sha256_file(source_evidence_path),
        "labels": labels_metadata,
        "heads": head_metadata,
        "teacher_checkpoint": {
            "family": "rbf_svm",
            "path": _relative(teacher_path),
            "sha256": sha256_file(teacher_path),
        },
    }
    metadata_path = output_dir / "baseline_predictions.json"
    write_canonical_json(metadata_path, metadata)
    index = build_artifact_index(output_dir)
    return {
        "metadata_path": _relative(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "teacher_checkpoint_hash": metadata["teacher_checkpoint"]["sha256"],
        "artifact_count": len(index["artifacts"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument(
        "--source-evidence", type=Path, default=DEFAULT_SOURCE_EVIDENCE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_baseline_predictions(
                args.config, args.feature_dir, args.source_evidence, args.output
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
