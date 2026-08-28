from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    write_canonical_json,
)
from experiments.common.v61_protocol import (
    validate_indexed_parent_locks,
    validate_parent_file_locks,
    validate_representation_lineage,
    validate_v61_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v6_1" / "a0_parent_lock.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v6_1" / "a0_parent_lock"
M30_EVIDENCE = REPO_ROOT / "logs" / "results" / "v6" / "m30_directional_s2" / "evidence.json"


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_v61_config(config)
    return config


def run_a0(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    parent_files = validate_parent_file_locks(
        config["parent_file_locks"], REPO_ROOT
    )
    parent_artifacts = validate_indexed_parent_locks(
        config["indexed_parent_locks"], REPO_ROOT
    )
    m30_evidence = json.loads(M30_EVIDENCE.read_text(encoding="utf-8"))
    lineage = validate_representation_lineage(
        config["representation_lineage"], m30_evidence=m30_evidence
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "protocol_snapshot.json", config)
    write_canonical_json(output_dir / "parent_file_locks.json", parent_files)
    write_canonical_json(output_dir / "parent_artifact_locks.json", parent_artifacts)
    write_canonical_json(output_dir / "representation_lineage.json", lineage)
    write_canonical_json(
        output_dir / "amendment_schema.json",
        {
            "schema_version": config["schema_version"],
            "amendment": config["amendment"],
            "stage": config["stage"],
            "primitive": config["amended_primitive"],
            "readout": config["weighted_readout"],
            "budgets": config["budgets"],
            "test_labels_opened": config["test_labels_opened"],
        },
    )
    write_canonical_json(
        output_dir / "claim_snapshot.json",
        {
            "claims": config["claims"],
            "closed_branches": sorted(config["closed_branches"]),
        },
    )
    index = build_artifact_index(output_dir)
    return {
        "parent_file_count": len(parent_files),
        "parent_directory_count": len(parent_artifacts),
        "parent_artifact_count": sum(
            item["artifact_count"] for item in parent_artifacts
        ),
        "representation_lineage_count": len(lineage),
        "claim_count": len(config["claims"]),
        "closed_branch_count": len(config["closed_branches"]),
        "training_data_loaded": False,
        "test_labels_opened": False,
        "artifact_count": len(index["artifacts"]),
    }


def verify_a0(config_path: Path, output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "first"
        second = root / "second"
        first_summary = run_a0(config_path, first)
        second_summary = run_a0(config_path, second)
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
            raise RuntimeError("v6.1 A0 parent-lock replay was not byte-identical.")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(first, output_dir)

    summary = {**first_summary, "byte_identical_replay": True}
    write_canonical_json(output_dir / "verification.json", summary)
    preliminary_index = build_artifact_index(output_dir)
    summary["artifact_count"] = len(preliminary_index["artifacts"])
    write_canonical_json(output_dir / "verification.json", summary)
    build_artifact_index(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the immutable v6.1 A0 parent-artifact lock."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify_a0(args.config, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
