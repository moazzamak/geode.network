"""Reproduce final v6.1 conclusions from immutable artifacts only."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    sha256_file,
    write_canonical_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v6_1" / "final_artifact_replay.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v6_1" / "final_artifact_replay"


def _resolve(path: str) -> Path:
    return REPO_ROOT / Path(path)


def _load_locked_json(item: dict[str, Any], name: str) -> tuple[Path, Any]:
    path = _resolve(str(item["path"]))
    actual = sha256_file(path)
    if actual != str(item["sha256"]):
        raise ValueError(
            f"{name} hash mismatch: expected {item['sha256']}, got {actual}."
        )
    return path, json.loads(path.read_text(encoding="utf-8"))


def _value_at(payload: Any, path: str) -> Any:
    value = payload
    for segment in path.split("."):
        if not isinstance(value, dict) or segment not in value:
            raise ValueError(f"Artifact operand path {path!r} is missing.")
        value = value[segment]
    return value


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != 1
        or config.get("amendment") != "v6.1"
        or config.get("milestone") != "FINAL"
        or len(config.get("artifact_indexes", [])) != 7
        or set(config.get("evidence", {}))
        != {"a1_tangent", "a1_weighted", "a1_budget", "a2", "a3", "a4"}
        or len(config.get("operands", [])) != 11
        or len(config.get("branches", [])) != 13
        or config.get("test_labels_opened") is not False
        or config.get("training_data_allowed") is not False
    ):
        raise ValueError("Unsupported or data-open v6.1 closure configuration.")


def _validate_artifact_index(path: Path, index: dict[str, Any]) -> int:
    if set(index) != {"schema_version", "artifacts"} or index["schema_version"] != 1:
        raise ValueError(f"Unsupported artifact index {path}.")
    root = path.parent
    for artifact in index["artifacts"]:
        artifact_path = root / artifact["path"]
        if not artifact_path.is_file():
            raise ValueError(f"Indexed artifact is missing: {artifact_path}.")
        if artifact_path.stat().st_size != int(artifact["bytes"]):
            raise ValueError(f"Indexed artifact size mismatch: {artifact_path}.")
        if sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError(f"Indexed artifact hash mismatch: {artifact_path}.")
    return len(index["artifacts"])


def run_replay(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    artifact_count = 0
    verified_indexes = []
    for position, item in enumerate(config["artifact_indexes"]):
        path, index = _load_locked_json(item, f"artifact index {position}")
        count = _validate_artifact_index(path, index)
        artifact_count += count
        verified_indexes.append(
            {"path": item["path"], "sha256": item["sha256"], "artifact_count": count}
        )
    evidence = {}
    evidence_locks = {}
    for name, item in config["evidence"].items():
        path, payload = _load_locked_json(item, f"{name} evidence")
        evidence[name] = payload
        evidence_locks[name] = {
            "path": item["path"],
            "sha256": sha256_file(path),
        }
    verified_operands = []
    for operand in config["operands"]:
        actual = _value_at(evidence[operand["evidence"]], operand["path"])
        if actual != operand["expected"]:
            raise ValueError(
                f"Operand {operand['evidence']}:{operand['path']} mismatch: "
                f"expected {operand['expected']!r}, got {actual!r}."
            )
        verified_operands.append({**operand, "actual": actual, "passed": True})

    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(
        output_dir / "branch_ledger.json",
        {"schema_version": 1, "branches": config["branches"]},
    )
    write_canonical_json(
        output_dir / "conclusion_summary.json",
        {
            "schema_version": 1,
            "predictive_outcome": evidence["a2"]["final_outcome"],
            "weighted_balanced_accuracy": evidence["a2"]["mean_metrics"][
                "weighted_balanced_accuracy"
            ],
            "same_space_gap": evidence["a2"]["same_space_gap"],
            "lifecycle_claim": evidence["a3"]["claim_status"],
            "flowers_rank3_affine_balanced_accuracy": evidence["a4"]["a4_f5"][
                "heads"
            ]["rank3_affine_subspace"]["development"]["balanced_accuracy"],
            "flowers_rank3_tangent_balanced_accuracy": evidence["a4"]["a4_f5"][
                "heads"
            ]["rank3_tangent_cap"]["development"]["balanced_accuracy"],
            "flowers_rank32_status": evidence["a4"]["a4_f34"]["status"]["status"],
            "test_labels_opened": False,
        },
    )
    write_canonical_json(output_dir / "evidence_locks.json", evidence_locks)
    write_canonical_json(output_dir / "verified_indexes.json", verified_indexes)
    write_canonical_json(output_dir / "verified_operands.json", verified_operands)
    index = build_artifact_index(output_dir)
    return {
        "predictive_outcome": evidence["a2"]["final_outcome"],
        "verified_index_count": len(verified_indexes),
        "verified_artifact_count": artifact_count,
        "verified_operand_count": len(verified_operands),
        "branch_count": len(config["branches"]),
        "training_data_loaded": False,
        "test_labels_opened": False,
        "artifact_count": len(index["artifacts"]),
    }


def verify_replay(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "first"
        second = root / "second"
        first_summary = run_replay(config_path, first)
        second_summary = run_replay(config_path, second)
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
        if first_summary != second_summary or first_files != second_files:
            raise RuntimeError("v6.1 artifact-only replay was not byte-identical.")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(first, output_dir)
    summary = {**first_summary, "byte_identical_replay": True}
    write_canonical_json(output_dir / "verification.json", summary)
    build_artifact_index(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify_replay(args.config, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
