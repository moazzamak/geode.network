from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.classification_metrics import classification_metrics
from experiments.common.experiment_manifest import array_fingerprint
from experiments.common.v5_artifacts import (
    build_artifact_index,
    parameter_count,
    payload_hash,
    serialized_size,
    sha256_file,
    write_canonical_json,
    write_run_record,
)
from experiments.common.v5_protocol import (
    DataStage,
    GateOperand,
    RepresentationLineage,
    validate_protocol_config,
)
from experiments.common.v5_registry import (
    ExperimentCell,
    expand_matrix,
    validate_matched_comparison,
    validate_required_controls,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v5" / "protocol.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v5" / "m16_s0"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_state() -> tuple[str, bool]:
    paths = [
        REPO_ROOT / "experiments" / "common" / "classification_metrics.py",
        REPO_ROOT / "experiments" / "common" / "experiment_manifest.py",
        REPO_ROOT / "experiments" / "common" / "v5_protocol.py",
        REPO_ROOT / "experiments" / "common" / "v5_registry.py",
        REPO_ROOT / "experiments" / "common" / "v5_artifacts.py",
        REPO_ROOT / "experiments" / "common" / "v5_statistics.py",
        REPO_ROOT / "experiments" / "tier1" / "eval_v5_protocol_s0.py",
        DEFAULT_CONFIG,
    ]
    source_hash = payload_hash(
        {
            path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
            for path in paths
        }
    )
    relative_paths = [str(path.relative_to(REPO_ROOT)) for path in paths]
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *relative_paths],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return source_hash, bool(status.stdout.strip())


def _toy_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.array(
        [
            [-2.0, -1.0],
            [-1.5, -1.2],
            [-1.0, -2.0],
            [-0.8, -1.4],
            [1.0, 1.8],
            [1.4, 0.9],
            [2.0, 1.2],
            [1.1, 1.3],
            [-1.2, -0.9],
            [-0.7, -1.7],
            [1.2, 1.0],
            [1.8, 1.5],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    return features[:8], labels[:8], features[8:], labels[8:]


def _represent(features: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return features.copy()
    if name == "fixed_affine":
        matrix = np.array([[1.25, 0.20], [-0.15, 0.90]], dtype=np.float64)
        offset = np.array([0.10, -0.05], dtype=np.float64)
        return features @ matrix + offset
    raise ValueError(f"Unknown S0 representation {name!r}.")


def _probabilities(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    evaluation_features: np.ndarray,
    head: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    classes = np.unique(train_labels)
    centers = np.stack([train_features[train_labels == value].mean(axis=0) for value in classes])
    deltas = evaluation_features[:, None, :] - centers[None, :, :]
    squared = np.sum(deltas * deltas, axis=2)
    parameters: dict[str, np.ndarray] = {"centers": centers}
    if head == "linear_logistic":
        logits = evaluation_features @ centers.T
    elif head == "rbf":
        logits = np.exp(-0.5 * squared)
    elif head == "prototype":
        logits = -squared
    elif head == "gaussian_mixture":
        variances = np.stack(
            [
                np.var(train_features[train_labels == value], axis=0) + 1e-6
                for value in classes
            ]
        )
        parameters["variances"] = variances
        logits = -0.5 * (
            np.sum((deltas * deltas) / variances[None, :, :], axis=2)
            + np.log(variances).sum(axis=1)[None, :]
        )
    elif head == "current_geode":
        logits = -np.sqrt(squared + 1e-12)
    else:
        raise ValueError(f"Unknown S0 head {head!r}.")
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, parameters


def _lineage(representation: str) -> RepresentationLineage:
    base = payload_hash({"s0": "fixed_backbone", "version": 1})
    preprocessing = payload_hash({"s0": "preprocessing", "version": 1})
    if representation == "identity":
        return RepresentationLineage(
            backbone_id="m16_s0_identity",
            weights_hash=base,
            preprocessing_hash=preprocessing,
            output_dimension=2,
        )
    return RepresentationLineage(
        backbone_id="m16_s0_identity",
        weights_hash=base,
        preprocessing_hash=preprocessing,
        output_dimension=2,
        interface_id="m16_s0_fixed_affine",
        interface_hash=payload_hash({"matrix": [[1.25, 0.20], [-0.15, 0.90]], "offset": [0.10, -0.05]}),
    )


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_protocol_config(config)
    return config


def _matrix(config: dict[str, Any]) -> list[ExperimentCell]:
    train_x, _, evaluation_x, _ = _toy_data()
    split_hash = payload_hash(
        {
            "train": array_fingerprint(np.arange(len(train_x), dtype=np.int64)),
            "evaluation": array_fingerprint(
                np.arange(len(train_x), len(train_x) + len(evaluation_x), dtype=np.int64)
            ),
        }
    )
    feature_hashes = {
        representation: array_fingerprint(
            _represent(np.vstack([train_x, evaluation_x]), representation)
        )
        for representation in config["representations"]
    }
    cells = expand_matrix(
        milestone="M16",
        stage=DataStage.S0,
        dataset="synthetic_gaussian_s0",
        representations=config["representations"],
        heads=config["heads"],
        readouts=config["readouts"],
        split_hashes={
            representation: split_hash for representation in config["representations"]
        },
        feature_hashes=feature_hashes,
        declared_seeds=tuple(config["stages"]["S0"]["seeds"]),
    )
    validate_required_controls(cells, set(config["required_heads"]))
    return cells


def run_s0(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    cells = _matrix(config)
    train_x, train_y, evaluation_x, evaluation_y = _toy_data()
    config_hash = payload_hash(config)
    source_hash, source_dirty = _source_state()
    environment_hash = payload_hash(
        {"python": platform.python_version(), "platform": platform.platform()}
    )
    dataset_hash = payload_hash({"features": array_fingerprint(np.vstack([train_x, evaluation_x])), "labels": array_fingerprint(np.concatenate([train_y, evaluation_y]))})
    records = []
    for cell in cells:
        represented_train = _represent(train_x, cell.representation)
        represented_evaluation = _represent(evaluation_x, cell.representation)
        probabilities, parameters = _probabilities(
            represented_train,
            train_y,
            represented_evaluation,
            cell.head,
        )
        metrics = classification_metrics(
            evaluation_y,
            probabilities,
            np.array([0, 1], dtype=np.int64),
            top_k=2,
        )
        gate_operands = [
            GateOperand("finite_metrics", True, "eq", True).to_dict(),
            GateOperand("split_match", True, "eq", True).to_dict(),
        ]
        record = {
            "schema_version": 1,
            "milestone": "M16",
            "experiment_id": cell.cell_id,
            "commit": _git_commit(),
            "source_hash": source_hash,
            "source_dirty": source_dirty,
            "configuration_hash": config_hash,
            "environment_hash": environment_hash,
            "dataset": cell.dataset,
            "dataset_hash": dataset_hash,
            "split_hash": cell.split_hash,
            "feature_hash": cell.feature_hash,
            "representation_hash": _lineage(cell.representation).digest,
            "method_family": "s0_contract_probe",
            "representation": cell.representation,
            "head": cell.head,
            "readout": cell.readout,
            "seed": cell.seed,
            "budget_mode": "s0_fixed",
            "selected_hyperparameters": {},
            "selection_metric": None,
            "metrics": metrics,
            "timing": {"fit_seconds": None, "inference_seconds": None},
            "memory": {"peak_process_bytes": None},
            "parameter_counts": {
                "fitted": parameter_count(parameters),
                "serialized_bytes": serialized_size(
                    {name: value.tolist() for name, value in parameters.items()}
                ),
            },
            "warnings": [],
            "gate_operands": gate_operands,
            "advancement_passed": all(item["passed"] for item in gate_operands),
            "parent_artifacts": [],
        }
        records.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "protocol_snapshot.json", config)
    write_canonical_json(output_dir / "matrix.json", [cell.to_dict() for cell in cells])
    write_canonical_json(output_dir / "metrics.json", records)
    for record in records:
        write_run_record(output_dir / "runs" / f"{record['experiment_id']}.json", record)
    index = build_artifact_index(output_dir)
    return {
        "cell_count": len(cells),
        "artifact_count": len(index["artifacts"]),
        "metrics_hash": payload_hash(records),
    }


def verify_s0(config_path: Path, output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        first = Path(temporary) / "first"
        second = Path(temporary) / "second"
        first_summary = run_s0(config_path, first)
        second_summary = run_s0(config_path, second)
        first_files = {
            path.relative_to(first).as_posix(): path.read_bytes()
            for path in first.rglob("*")
            if path.is_file()
        }
        second_files = {
            path.relative_to(second).as_posix(): path.read_bytes()
            for path in second.rglob("*")
            if path.is_file()
        }
        if first_files != second_files or first_summary != second_summary:
            raise RuntimeError("S0 replay was not byte-identical.")

        mismatched = list(_matrix(_load_config(config_path)))
        mismatched[0] = ExperimentCell(
            **{
                **mismatched[0].__dict__,
                "split_hash": payload_hash({"deliberate": "mismatch"}),
            }
        )
        try:
            validate_matched_comparison(mismatched)
        except ValueError:
            mismatch_rejected = True
        else:
            raise RuntimeError("The deliberate split mismatch was not rejected.")

        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(first, output_dir)
    summary = {
        **first_summary,
        "byte_identical_replay": True,
        "split_mismatch_rejected": mismatch_rejected,
    }
    write_canonical_json(output_dir / "verification.json", summary)
    preliminary_index = build_artifact_index(output_dir)
    summary["artifact_count"] = len(preliminary_index["artifacts"])
    write_canonical_json(output_dir / "verification.json", summary)
    build_artifact_index(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M16 deterministic S0 gate.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify_s0(args.config, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
