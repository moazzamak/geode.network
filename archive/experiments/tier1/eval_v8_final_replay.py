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
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v8" / "final_outcome_replay.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v8" / "final_outcome_replay"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_index(index_path: Path) -> int:
    index = _load_json(index_path)
    for artifact in index["artifacts"]:
        path = index_path.parent / artifact["path"]
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"v8 indexed artifact drifted: {path}")
    return len(index["artifacts"])


def run_final_replay(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    if (
        config.get("milestone") != "V8_FINAL"
        or config.get("outcome") != "D"
        or config.get("training_data_loaded") is not False
        or config.get("final_labels_opened") is not False
    ):
        raise ValueError("invalid or data-open v8 final replay config")
    verified_artifacts = 0
    verified_indexes = []
    evidence = {}
    for lock in config["index_locks"]:
        index_path = REPO_ROOT / lock["path"]
        actual = sha256_file(index_path)
        if actual != lock["sha256"]:
            raise ValueError(f"v8 milestone index drifted: {lock['milestone']}")
        verified_artifacts += _verify_index(index_path)
        verified_indexes.append(
            {
                "milestone": lock["milestone"],
                "path": lock["path"],
                "sha256": actual,
            }
        )
        evidence_path = index_path.parent / "evidence.json"
        if evidence_path.is_file():
            evidence[lock["milestone"]] = _load_json(evidence_path)
        else:
            evidence[lock["milestone"]] = _load_json(
                index_path.parent / "verification.json"
            )
    conclusions = {
        "m45_protocol_qualified": (
            evidence["M45"]["byte_identical_replay"]
            and evidence["M45"]["parent_outcomes"]
            == ["v6.1 Outcome D", "v7 Outcome C"]
        ),
        "m46_anchor_quantile_retained": (
            evidence["M46"]["retained_threshold_rule"] == "anchor_quantile"
            and evidence["M46"]["threshold_transfer_supported"]
        ),
        "m47_outcome_d": (
            evidence["M47"]["outcome"] == "Outcome D"
            and not evidence["M47"]["gate"]["advance_to_m48"]
        ),
        "m48_blocked": evidence["M47"]["m48_status"] == "blocked_by_m47",
        "m49_no_residual_retained": (
            evidence["M49"]["retained_residual"] is None
            and evidence["M49"]["final_main_program_outcome"] == "Outcome D"
        ),
        "m50_blocked": evidence["M47"]["m50_status"] == "blocked_by_m47",
    }
    if conclusions != config["required_conclusions"]:
        raise ValueError("v8 final conclusions do not match registered operands")
    replay = {
        "schema_version": 1,
        "milestone": "V8_FINAL",
        "outcome": "D",
        "verified_indexes": verified_indexes,
        "verified_index_count": len(verified_indexes),
        "verified_artifact_count": verified_artifacts,
        "conclusions": conclusions,
        "training_data_loaded": False,
        "final_labels_opened": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", replay)
    index = build_artifact_index(output_dir)
    return {
        "outcome": "D",
        "verified_index_count": len(verified_indexes),
        "verified_artifact_count": verified_artifacts,
        "verified_conclusion_count": len(conclusions),
        "training_data_loaded": False,
        "final_labels_opened": False,
        "artifact_count": len(index["artifacts"]),
    }


def verify_final_replay(config_path: Path, output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "first"
        second = root / "second"
        first_summary = run_final_replay(config_path, first)
        second_summary = run_final_replay(config_path, second)
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
            raise RuntimeError("v8 final replay was not byte-identical")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(first, output_dir)
    summary = {**first_summary, "byte_identical_replay": True}
    write_canonical_json(output_dir / "verification.json", summary)
    build_artifact_index(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify_final_replay(args.config, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
