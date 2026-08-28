from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.experiment_manifest import canonical_json
from experiments.common.v5_protocol import GateOperand, require_sha256

# The payload hash is a PRODUCT primitive; re-exported here so every
# experiment script keeps its imports unchanged (experiments -> geode
# only — geode never imports experiments).
from geode.hashing import payload_hash


RUN_REQUIRED_FIELDS = {
    "schema_version",
    "milestone",
    "experiment_id",
    "commit",
    "source_hash",
    "source_dirty",
    "configuration_hash",
    "environment_hash",
    "dataset",
    "dataset_hash",
    "split_hash",
    "feature_hash",
    "representation_hash",
    "method_family",
    "representation",
    "head",
    "readout",
    "seed",
    "budget_mode",
    "selected_hyperparameters",
    "selection_metric",
    "metrics",
    "timing",
    "memory",
    "parameter_counts",
    "warnings",
    "gate_operands",
    "advancement_passed",
    "parent_artifacts",
}


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_run_record(record: Mapping[str, Any]) -> None:
    missing = RUN_REQUIRED_FIELDS - set(record)
    if missing:
        raise ValueError(f"Run record is missing fields {sorted(missing)}.")
    if record["schema_version"] != 1:
        raise ValueError("Unsupported v5 run-record schema.")
    for field in (
        "source_hash",
        "configuration_hash",
        "environment_hash",
        "dataset_hash",
        "split_hash",
        "feature_hash",
        "representation_hash",
    ):
        require_sha256(record[field], field)
    operands = record["gate_operands"]
    if not isinstance(operands, list) or not operands:
        raise ValueError("gate_operands must be a non-empty list.")
    operand_results = []
    for operand in operands:
        parsed = GateOperand(
            name=str(operand["name"]),
            value=operand["value"],
            operator=str(operand["operator"]),
            threshold=operand["threshold"],
        )
        serialized = parsed.to_dict()
        if operand.get("passed") != serialized["passed"]:
            raise ValueError(f"Gate operand {parsed.name!r} has an inconsistent result.")
        operand_results.append(serialized["passed"])
    if bool(record["advancement_passed"]) != all(operand_results):
        raise ValueError("advancement_passed must equal the conjunction of gate operands.")


def write_canonical_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(payload) + "\n", encoding="utf-8", newline="\n")


def write_run_record(path: str | Path, record: Mapping[str, Any]) -> None:
    validate_run_record(record)
    write_canonical_json(path, dict(record))


def build_artifact_index(
    root: str | Path,
    *,
    output_name: str = "artifact_index.json",
) -> dict[str, Any]:
    root_path = Path(root)
    entries = []
    for path in sorted(root_path.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name == output_name:
            continue
        entries.append(
            {
                "path": path.relative_to(root_path).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    index = {"schema_version": 1, "artifacts": entries}
    write_canonical_json(root_path / output_name, index)
    return index


def require_representation_match(
    *,
    active_hash: str,
    artifact_hash: str,
    artifact_kind: str,
) -> None:
    require_sha256(active_hash, "active_hash")
    require_sha256(artifact_hash, "artifact_hash")
    if active_hash != artifact_hash:
        raise ValueError(
            f"{artifact_kind} representation hash {artifact_hash} does not match "
            f"active representation {active_hash}."
        )


def validate_migration_report(report: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "source_representation_hash",
        "target_representation_hash",
        "component_correspondence",
        "edit_survival",
        "invalidated_artifacts",
        "rollback_bundle_hash",
    }
    if set(report) != required or report.get("schema_version") != 1:
        raise ValueError("Unsupported migration-report schema.")
    source = require_sha256(report["source_representation_hash"], "source_representation_hash")
    target = require_sha256(report["target_representation_hash"], "target_representation_hash")
    if source == target:
        raise ValueError("Migration source and target representations must differ.")
    require_sha256(report["rollback_bundle_hash"], "rollback_bundle_hash")
    for field in ("component_correspondence", "edit_survival", "invalidated_artifacts"):
        if not isinstance(report[field], list):
            raise ValueError(f"{field} must be a list.")


def parameter_count(values: Any) -> int:
    if isinstance(values, np.ndarray):
        return int(values.size)
    if isinstance(values, Mapping):
        return sum(parameter_count(value) for value in values.values())
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        return sum(parameter_count(value) for value in values)
    if isinstance(values, np.generic):
        return 1
    if isinstance(values, (int, float, bool)):
        return 1
    raise TypeError(f"Unsupported parameter container {type(values).__name__}.")


def serialized_size(payload: Any) -> int:
    return len((canonical_json(payload) + "\n").encode("utf-8"))
