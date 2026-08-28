from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    payload_hash,
    sha256_file,
    write_canonical_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v7" / "final_stagewise_replay.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v7" / "final_stagewise_replay"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_replay(config: dict[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema_version") != 1
        or config.get("milestone") != "V7_FINAL"
        or config.get("outcome") != "C"
        or config.get("m44_status") != "blocked_by_m43"
    ):
        raise ValueError("Unsupported v7 final replay configuration.")
    if config.get("training_data_loaded") is not False:
        raise PermissionError("Final replay cannot load training data.")
    if config.get("final_labels_opened") is not False:
        raise PermissionError("Final replay cannot open final labels.")
    verified = []
    evidences: dict[str, dict[str, Any]] = {}
    for lock in config["indexes"]:
        path = REPO_ROOT / lock["path"]
        if sha256_file(path) != lock["sha256"]:
            raise ValueError(f"{lock['milestone']} artifact index drifted.")
        index = _load_json(path)
        if lock["mode"] == "artifact_manifest":
            for artifact in index["artifacts"]:
                artifact_path = path.parent / artifact["path"]
                if (
                    sha256_file(artifact_path) != artifact["sha256"]
                    or artifact_path.stat().st_size != artifact["bytes"]
                ):
                    raise ValueError(
                        f"{lock['milestone']} artifact {artifact['path']} drifted."
                    )
            artifact_count = len(index["artifacts"])
        elif lock["mode"] == "evidence_index":
            evidence_path = path.parent / "evidence.json"
            evidence = _load_json(evidence_path)
            if payload_hash(evidence) != index["evidence_sha256"]:
                raise ValueError(f"{lock['milestone']} evidence drifted.")
            if evidence.get("final_labels_opened") is not False:
                raise PermissionError(
                    f"{lock['milestone']} evidence opened final labels."
                )
            evidences[lock["milestone"]] = evidence
            artifact_count = 2
        else:
            raise ValueError("Unsupported final replay index mode.")
        verified.append(
            {
                "milestone": lock["milestone"],
                "path": lock["path"],
                "sha256": lock["sha256"],
                "artifact_count": artifact_count,
            }
        )
    conclusions = {
        "m39_rejection_qualified": evidences["M39"]["gate"]["advance_to_m40"],
        "m40_discovery_qualified": evidences["M40"]["advance_to_m41"],
        "m41_new_class_qualified": bool(
            evidences["M41"]["retained_operations"]["create_new"]
        ),
        "m41_existing_expansion_closed": not bool(
            evidences["M41"]["retained_operations"]["update_existing"]
        ),
        "m42_authoritative_routing_closed": (
            evidences["M42"]["routing_mode_for_m43"]
            == "shadow_only_exhaustive"
        ),
        "m43_integrated_gate_failed": not evidences["M43"]["gate"][
            "advance_to_m44"
        ],
        "m44_blocked": evidences["M43"]["m44_status"] == "blocked_by_m43",
        "outcome_c": True,
    }
    if not all(conclusions.values()):
        raise ValueError("Frozen v7 evidence does not reproduce Outcome C.")
    return {
        "schema_version": 1,
        "milestone": "V7_FINAL",
        "outcome": "C",
        "verified_indexes": verified,
        "verified_index_count": len(verified),
        "verified_artifact_count": sum(
            item["artifact_count"] for item in verified
        ),
        "conclusions": conclusions,
        "training_data_loaded": False,
        "final_labels_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    replay = build_replay(_load_json(args.config))
    args.output.mkdir(parents=True, exist_ok=True)
    write_canonical_json(args.output / "evidence.json", replay)
    write_canonical_json(
        args.output / "artifact_index.json",
        {
            "schema_version": 1,
            "milestone": "V7_FINAL",
            "evidence_sha256": payload_hash(replay),
            "outcome": "C",
        },
    )
    print(json.dumps(replay, indent=2))


if __name__ == "__main__":
    main()
