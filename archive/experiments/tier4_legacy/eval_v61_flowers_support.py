"""Run separate A4-F5 and A4-F34 Flowers support-tier evaluations."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.classification_metrics import balanced_accuracy
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    serialized_size,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v5_frozen_representations import (
    FeatureCacheMetadata,
    RepresentationManifest,
    require_cache_binding,
    verify_cache_file_integrity,
)
from experiments.common.v61_flowers_support import (
    deserialize_primitive_head,
    fit_global_temperature,
    fit_rank3_primitives,
    predict_primitives,
    primitive_logits,
    serialize_primitive_head,
    support_tier_status,
)
from experiments.tier4.eval_v5_frozen_space_heads import (
    evaluate_head,
    fit_logistic_head,
    fit_prototype_head,
    fit_weighted_knn_head,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v6_1" / "a4_flowers_support.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v6_1" / "a4_flowers_support"

_F5_CONTRACT = {
    "samples_per_class": 5,
    "rank": 3,
    "minimum_support_rule": "r_plus_2",
    "minimum_support": 5,
    "components_per_class": 1,
    "heads": [
        "linear_logistic",
        "prototype",
        "weighted_knn",
        "rank3_affine_subspace",
        "rank3_tangent_cap",
    ],
    "evaluation_split": "dev",
    "objective": "fit_and_replay_feasibility",
}
_F34_CONTRACT = {
    "rank": 32,
    "minimum_support_rule": "r_plus_2",
    "minimum_support": 34,
    "allowed_fit_splits": ["train"],
    "forbid_partition_combination": True,
}


def _resolve(path: str) -> Path:
    return REPO_ROOT / Path(path)


def _load_locked_json(item: dict[str, Any], name: str) -> tuple[Path, dict[str, Any]]:
    path = _resolve(str(item["path"]))
    actual = sha256_file(path)
    if actual != str(item["sha256"]):
        raise ValueError(
            f"{name} hash mismatch: expected {item['sha256']}, got {actual}."
        )
    return path, json.loads(path.read_text(encoding="utf-8"))


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != 1
        or config.get("amendment") != "v6.1"
        or config.get("milestone") != "A4"
        or config.get("stage") != "S1-Flowers"
        or config.get("seed") != 11
        or config.get("representation")
        != {
            "backbone": "dinov2-small",
            "representation_hash": "b0c0bb74c51684d184a4be1638a684ad55fcde9d4836c1012b6f736ba5a9763b",
            "dimension": 384,
        }
        or config.get("a4_f5") != _F5_CONTRACT
        or config.get("a4_f34") != _F34_CONTRACT
        or config.get("test_labels_opened") is not False
    ):
        raise ValueError("Unsupported, partition-open, or test-open A4 configuration.")


def _load_split(
    feature_dir: Path,
    backbone: str,
    manifest: RepresentationManifest,
    metadata_payload: dict[str, Any],
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    metadata = FeatureCacheMetadata.from_dict(metadata_payload)
    require_cache_binding(metadata, manifest)
    path = (
        feature_dir
        / backbone
        / f"features_{split}_{manifest.representation_hash[:16]}.npz"
    )
    verify_cache_file_integrity(path, metadata.feature_file_hash)
    with np.load(path) as cache:
        return (
            cache["features"].astype(np.float64),
            cache["labels"].astype(np.int64),
        )


def _primitive_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    class_to_column = {int(value): index for index, value in enumerate(classes)}
    columns = np.asarray([class_to_column[int(label)] for label in labels])
    nll = -float(
        np.mean(
            np.log(
                np.maximum(
                    probabilities[np.arange(len(labels)), columns],
                    np.finfo(np.float64).tiny,
                )
            )
        )
    )
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": balanced_accuracy(labels, predictions),
        "nll": nll,
    }


def _fit_primitive_head(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    dev_features: np.ndarray,
    dev_labels: np.ndarray,
    *,
    tangent: bool,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    started = time.perf_counter()
    candidates = fit_rank3_primitives(
        train_features, train_labels, tangent=tangent
    )
    train_logits, classes = primitive_logits(
        candidates, train_features, tangent=tangent
    )
    temperature = fit_global_temperature(
        train_logits,
        train_labels,
        classes,
        minimum=float(config["temperature"]["minimum"]),
        maximum=float(config["temperature"]["maximum"]),
    )
    fit_seconds = time.perf_counter() - started
    student = serialize_primitive_head(
        candidates,
        tangent=tangent,
        temperature=temperature,
        representation_hash=config["representation"]["representation_hash"],
    )
    started = time.perf_counter()
    predictions, probabilities = predict_primitives(
        candidates,
        dev_features,
        tangent=tangent,
        temperature=temperature,
    )
    inference_seconds = time.perf_counter() - started
    parameter_count = int(sum(candidate.parameter_count for candidate in candidates))
    metrics = {
        "development": _primitive_metrics(
            dev_labels, predictions, probabilities, classes
        ),
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "component_count": len(candidates),
        "parameter_count": parameter_count,
        "serialized_bytes": serialized_size(student),
        "global_temperature": temperature,
    }
    return student, metrics, predictions, probabilities


def run_evaluation(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    parent_config_path, parent_config = _load_locked_json(
        config["parent_config"], "Flowers parent configuration"
    )
    extraction_path, extraction = _load_locked_json(
        config["extraction_summary"], "Flowers extraction summary"
    )
    parent_evidence_path, _ = _load_locked_json(
        config["parent_evidence"], "Flowers parent evidence"
    )
    a2_path, a2 = _load_locked_json(config["a2_evidence"], "A2 evidence")
    if a2["final_outcome"] != "Outcome D" or a2["test_labels_opened"]:
        raise ValueError("A4 cannot alter or open final Outcome D.")
    if (
        parent_config["samples_per_class"] != {"train": 5, "dev": 2, "test": 3}
        or parent_config["official_split_mapping"]
        != {"train": "train", "dev": "val", "test": "test"}
    ):
        raise ValueError("Flowers official split contract changed.")
    backbone = config["representation"]["backbone"]
    extracted = extraction["representations"][backbone]
    manifest = RepresentationManifest.from_dict(extracted["manifest"])
    if (
        manifest.representation_hash
        != config["representation"]["representation_hash"]
        or manifest.output_dimension != config["representation"]["dimension"]
        or manifest.training_split_hash != extraction["split_hashes"]["train"]
    ):
        raise ValueError("A4 representation lineage mismatch.")
    feature_dir = extraction_path.parent
    train_features, train_labels = _load_split(
        feature_dir,
        backbone,
        manifest,
        extracted["cache_metadata"]["train"],
        "train",
    )
    dev_features, dev_labels = _load_split(
        feature_dir,
        backbone,
        manifest,
        extracted["cache_metadata"]["dev"],
        "dev",
    )
    f5_status = support_tier_status(
        train_labels, rank=3, allowed_fit_splits=["train"]
    )
    f34_status = support_tier_status(
        train_labels,
        rank=32,
        allowed_fit_splits=config["a4_f34"]["allowed_fit_splits"],
    )
    if f5_status["status"] != "feasible":
        raise ValueError("A4-F5 rank-3 support is unexpectedly unavailable.")

    heads: dict[str, Any] = {}
    baseline_fitters = {
        "linear_logistic": lambda: fit_logistic_head(
            train_features, train_labels, config["seed"]
        ),
        "prototype": lambda: fit_prototype_head(
            train_features, train_labels, config["seed"]
        ),
        "weighted_knn": lambda: fit_weighted_knn_head(
            train_features,
            train_labels,
            n_neighbors=int(config["weighted_knn"]["n_neighbors"]),
            temperature=float(config["weighted_knn"]["temperature"]),
            query_batch_size=int(config["weighted_knn"]["query_batch_size"]),
        ),
    }
    for name, fitter in baseline_fitters.items():
        started = time.perf_counter()
        head = fitter()
        fit_seconds = time.perf_counter() - started
        heads[name] = {
            "development": evaluate_head(head, dev_features, dev_labels),
            "fit_seconds": fit_seconds,
        }

    students = {}
    arrays = {}
    for name, tangent in (
        ("rank3_affine_subspace", False),
        ("rank3_tangent_cap", True),
    ):
        student, metrics, predictions, probabilities = _fit_primitive_head(
            train_features,
            train_labels,
            dev_features,
            dev_labels,
            tangent=tangent,
            config=config,
        )
        replay_candidates, replay_tangent, replay_temperature = (
            deserialize_primitive_head(student)
        )
        replay_predictions, replay_probabilities = predict_primitives(
            replay_candidates,
            dev_features,
            tangent=replay_tangent,
            temperature=replay_temperature,
        )
        metrics["exact_replay"] = (
            np.array_equal(predictions, replay_predictions)
            and np.array_equal(probabilities, replay_probabilities)
        )
        heads[name] = metrics
        students[name] = student
        arrays[f"{name}_development_predictions"] = predictions
        arrays[f"{name}_development_probabilities"] = probabilities

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for name, student in students.items():
        write_canonical_json(output_dir / f"{name}_student.json", student)
    array_artifacts = {}
    for name, values in arrays.items():
        path = output_dir / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        array_artifacts[name] = {
            "path": path.name,
            "sha256": sha256_file(path),
        }
    evidence = {
        "schema_version": 1,
        "amendment": "v6.1",
        "milestone": "A4",
        "stage": "S1-Flowers",
        "configuration_hash": payload_hash(config),
        "parent_hashes": {
            "parent_config": sha256_file(parent_config_path),
            "extraction_summary": sha256_file(extraction_path),
            "parent_evidence": sha256_file(parent_evidence_path),
            "a2_evidence": sha256_file(a2_path),
        },
        "representation": config["representation"],
        "split_hashes": extraction["split_hashes"],
        "loaded_splits": ["train", "dev"],
        "fit_splits": ["train"],
        "evaluation_split": "dev",
        "a4_f5": {
            "status": f5_status,
            "objective": "fit_and_replay_feasibility",
            "heads": heads,
            "all_primitive_replays_exact": all(
                heads[name]["exact_replay"]
                for name in ("rank3_affine_subspace", "rank3_tangent_cap")
            ),
        },
        "a4_f34": {
            "status": f34_status,
            "partition_combination_used": False,
            "reason": (
                "Official training support is 5 per class, below r+2=34; "
                "development and test partitions remain excluded from fitting."
            ),
        },
        "array_artifacts": array_artifacts,
        "test_features_loaded": False,
        "test_labels_opened": False,
        "predictive_outcome_unchanged": "Outcome D",
        "claim_boundary": (
            "A4-F5 establishes rank-3 fit and replay feasibility only. "
            "It is not a competitiveness or predictive-rescue study."
        ),
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    index = build_artifact_index(output_dir)
    return {
        "a4_f5_feasible": f5_status["status"] == "feasible",
        "rank3_affine_balanced_accuracy": heads["rank3_affine_subspace"][
            "development"
        ]["balanced_accuracy"],
        "rank3_tangent_balanced_accuracy": heads["rank3_tangent_cap"][
            "development"
        ]["balanced_accuracy"],
        "a4_f34_status": f34_status["status"],
        "test_labels_opened": False,
        "artifact_count": len(index["artifacts"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run_evaluation(args.config, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
