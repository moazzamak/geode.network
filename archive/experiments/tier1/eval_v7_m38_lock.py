from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import build_artifact_index, sha256_file, write_canonical_json
from experiments.common.v7_protocol import (
    schedule_locks,
    synthetic_contract_fixture,
    validate_parent_locks,
    validate_v7_m38_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v7" / "m38_protocol_lock.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v7" / "m38_protocol_lock"


def run_m38(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_v7_m38_config(config)
    parent_locks = validate_parent_locks(config["parent_file_locks"], REPO_ROOT)
    audit_path = REPO_ROOT / config["literature_audit"]["path"]
    if not audit_path.is_file():
        raise ValueError("M38 literature audit is missing.")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "protocol_snapshot.json", config)
    write_canonical_json(output_dir / "parent_locks.json", parent_locks)
    write_canonical_json(output_dir / "schedule_locks.json", schedule_locks(config["schedules"]))
    write_canonical_json(output_dir / "synthetic_contract_fixture.json", synthetic_contract_fixture())
    write_canonical_json(
        output_dir / "literature_lock.json",
        {
            **config["literature_audit"],
            "sha256": sha256_file(audit_path),
            "qualified_positioning": (
                "engineered composition of established open-world stages"
            ),
        },
    )
    index = build_artifact_index(output_dir)
    return {
        "parent_lock_count": len(parent_locks),
        "schedule_count": len(config["schedules"]),
        "stage_count": len(config["stages"]),
        "all_seven_stage_system_found": False,
        "outcome_e_triggered": False,
        "training_data_loaded": False,
        "final_labels_opened": False,
        "artifact_count": len(index["artifacts"]),
    }


def verify_m38(config_path: Path, output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "first"
        second = root / "second"
        first_summary = run_m38(config_path, first)
        second_summary = run_m38(config_path, second)
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
            raise RuntimeError("v7 M38 replay was not byte-identical.")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(first, output_dir)
    summary = {**first_summary, "byte_identical_replay": True}
    write_canonical_json(output_dir / "verification.json", summary)
    build_artifact_index(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the immutable v7 M38 protocol lock.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify_m38(args.config, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
