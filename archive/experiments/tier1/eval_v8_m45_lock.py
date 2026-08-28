from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v8_protocol import (
    synthetic_episode_replay,
    validate_m45_config,
    validate_parent_locks,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v8" / "m45_protocol_lock.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v8" / "m45_protocol_lock"


def _load_indexed_json(index_path: Path, artifact_name: str) -> dict[str, Any]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if "artifacts" in index:
        entries = {
            str(entry["path"]): str(entry["sha256"])
            for entry in index["artifacts"]
        }
        expected = entries.get(artifact_name)
    else:
        expected = index.get(f"{artifact_name.removesuffix('.json')}_sha256")
    if expected is None:
        raise ValueError(f"{artifact_name} is absent from {index_path}")
    artifact_path = index_path.parent / artifact_name
    if "artifacts" in index:
        actual = sha256_file(artifact_path)
    else:
        actual = payload_hash(json.loads(artifact_path.read_text(encoding="utf-8")))
    if actual != expected:
        raise ValueError(f"indexed parent artifact drift: {artifact_path}")
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def run_m45(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_m45_config(config)
    parent_locks = validate_parent_locks(config["parent_locks"], REPO_ROOT)
    lock_paths = {
        str(lock["id"]): REPO_ROOT / str(lock["path"])
        for lock in config["parent_locks"]
    }
    v61 = _load_indexed_json(lock_paths["v6_1_final_index"], "verification.json")
    v7 = _load_indexed_json(lock_paths["v7_final_index"], "evidence.json")
    if v61.get("predictive_outcome") != "Outcome D":
        raise ValueError("v6.1 parent does not reproduce Outcome D")
    if v7.get("outcome") != "C" or v7.get("conclusions", {}).get("outcome_c") is not True:
        raise ValueError("v7 parent does not reproduce Outcome C")
    parent_outcomes = {
        "v6_1": {
            "outcome": v61["predictive_outcome"],
            "training_data_loaded": v61["training_data_loaded"],
            "test_labels_opened": v61["test_labels_opened"],
        },
        "v7": {
            "outcome": v7["outcome"],
            "training_data_loaded": v7["training_data_loaded"],
            "final_labels_opened": v7["final_labels_opened"],
        },
    }
    replay = synthetic_episode_replay(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "protocol_snapshot.json", config)
    write_canonical_json(output_dir / "parent_locks.json", parent_locks)
    write_canonical_json(output_dir / "parent_outcomes.json", parent_outcomes)
    write_canonical_json(output_dir / "episode_replay_contracts.json", replay)
    index = build_artifact_index(output_dir)
    return {
        "parent_lock_count": len(parent_locks),
        "parent_outcomes": ["v6.1 Outcome D", "v7 Outcome C"],
        "episode_count": len(replay["episode_contracts"]),
        "interface_count": len(replay["interface_audits"]),
        "all_interfaces_complete": all(
            not audit["unsupported_diagnostics"]
            and set(audit["required_statistics"]).issubset(audit["supplied_statistics"])
            for audit in replay["interface_audits"]
        ),
        "registered_failure_cases": len(config["failure_policy"]),
        "training_data_loaded": replay["training_data_loaded"],
        "final_labels_opened": replay["final_labels_opened"],
        "artifact_count": len(index["artifacts"]),
    }


def verify_m45(config_path: Path, output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "first"
        second = root / "second"
        first_summary = run_m45(config_path, first)
        second_summary = run_m45(config_path, second)
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
            raise RuntimeError("v8 M45 replay was not byte-identical")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(first, output_dir)
    summary = {**first_summary, "byte_identical_replay": True}
    write_canonical_json(output_dir / "verification.json", summary)
    build_artifact_index(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the immutable v8 M45 protocol lock")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify_m45(args.config, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
