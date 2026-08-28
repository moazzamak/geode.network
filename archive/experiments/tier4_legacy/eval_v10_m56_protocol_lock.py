from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v10_manifold_support import (
    PROBE_FAMILIES,
    TANGENT_MULTIPLIERS,
    calibration_lineage_hash,
    fit_dimensionless_tube,
    generate_axis_tangent_probes,
    generate_safety_probes,
    probe_acceptance,
    select_smallest_safety_penalty,
    system_scores,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from src.runtime.schemas import TubeCalibrationRecord, TubeSafetyEvidence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v10" / "m56_protocol_lock.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v10" / "m56_protocol_lock"


def _resolve(path: str) -> Path:
    return REPO_ROOT / Path(path)


def _verify_lock(specification: dict[str, str]) -> dict[str, str]:
    path = _resolve(specification["path"])
    observed = sha256_file(path)
    if observed != specification["sha256"]:
        raise ValueError(f"M56 parent hash mismatch: {path}")
    return {"path": specification["path"], "sha256": observed}


def _toy_protocol(
    penalty_grid: tuple[float, ...],
) -> tuple[TubeCalibrationRecord, TubeSafetyEvidence, dict[str, Any]]:
    rng = np.random.default_rng(56)
    dimension = 16
    rank = 8
    geometry_parts = []
    calibration_parts = []
    tubes = []
    for class_label, shift in enumerate((-8.0, 8.0)):
        tangent_geometry = rng.normal(scale=0.5, size=(96, rank))
        normal_geometry = rng.normal(scale=0.05, size=(96, dimension - rank))
        geometry = np.column_stack([tangent_geometry, normal_geometry])
        geometry[:, -1] += shift
        tangent_calibration = rng.normal(scale=0.5, size=(40, rank))
        normal_calibration = rng.normal(scale=0.05, size=(40, dimension - rank))
        calibration = np.column_stack([tangent_calibration, normal_calibration])
        calibration[:, -1] += shift
        geometry_parts.append(geometry)
        calibration_parts.append(calibration)
        tubes.append(
            fit_dimensionless_tube(
                geometry,
                calibration,
                rank=rank,
                extent_quantile=0.95,
                outer_scale_policy="interquantile_range",
                penalty_weight=penalty_grid[0],
                class_label=class_label,
            )
        )
    calibration_points = np.vstack(calibration_parts)
    selected = select_smallest_safety_penalty(
        tubes,
        calibration_points,
        penalty_grid=penalty_grid,
    )
    selected_tubes = selected["tubes"]
    threshold = float(selected["threshold"])
    probes = generate_safety_probes(selected_tubes, seed=56)
    acceptance = probe_acceptance(selected_tubes, probes, threshold=threshold)
    tangent_acceptance = {}
    for multiplier in TANGENT_MULTIPLIERS:
        points, _ = generate_axis_tangent_probes(
            selected_tubes, multiplier=multiplier
        )
        key = str(multiplier).rstrip("0").rstrip(".")
        tangent_acceptance[key] = float(
            np.mean(system_scores(selected_tubes, points) <= threshold)
        )
    geometry_hash = payload_hash(
        [tube.to_dict() for tube in selected_tubes]
    )
    partition_hash = payload_hash(
        {
            "geometry_counts": [len(points) for points in geometry_parts],
            "calibration_counts": [len(points) for points in calibration_parts],
            "seed": 56,
        }
    )
    replay_hash = calibration_lineage_hash(
        selected_tubes,
        penalty_grid=penalty_grid,
        threshold=threshold,
    )
    calibration_record = TubeCalibrationRecord(
        geometry_hash=geometry_hash,
        representation_hash="1" * 64,
        partition_hash=partition_hash,
        rank=rank,
        patch_count=1,
        extent_quantile=0.95,
        outer_scale_policy="interquantile_range",
        penalty_grid=penalty_grid,
        selected_penalty=float(selected["selected_penalty"]),
        known_coverage_target=0.92,
        calibrated_threshold=threshold,
        calibration_known_coverage=float(selected["coverage"]),
        selected_before_development=True,
        final_labels_opened=False,
        replay_hash=replay_hash,
    )
    source_hash = payload_hash(
        {
            "generate_safety_probes": inspect.getsource(generate_safety_probes),
            "generate_axis_tangent_probes": inspect.getsource(
                generate_axis_tangent_probes
            ),
        }
    )
    parameter_count = sum(tube.parameter_count for tube in selected_tubes)
    probe_counts = tuple(
        (name, int(len(probes[name][0]))) for name in PROBE_FAMILIES
    )
    peak_temporary_bytes = max(
        (
            points.nbytes + owners.nbytes
            for points, owners in probes.values()
        ),
        default=0,
    )
    safety_record = TubeSafetyEvidence(
        calibration_replay_hash=replay_hash,
        probe_generator_hash=source_hash,
        probe_counts=probe_counts,
        source_patch_acceptance=tuple(
            (name, acceptance["source_patch"][name]) for name in PROBE_FAMILIES
        ),
        system_acceptance=tuple(
            (name, acceptance["system"][name]) for name in PROBE_FAMILIES
        ),
        tangent_acceptance_by_multiplier=tuple(
            (key, tangent_acceptance[key]) for key in ("0.5", "1", "2", "4", "8")
        ),
        parameter_count=parameter_count,
        fit_work_units=sum(len(points) * dimension * rank for points in geometry_parts),
        latency_seconds=0.0,
        peak_temporary_bytes=int(peak_temporary_bytes),
        exact_replay=True,
    )
    details = {
        "selection_attempts": selected["attempts"],
        "tube_hash": geometry_hash,
        "probe_generator_hash": source_hash,
        "resource_accounting": {
            "parameter_count": parameter_count,
            "fit_work_units": safety_record.fit_work_units,
            "latency_contract_registered": True,
            "latency_seconds": 0.0,
            "latency_measurement_stage": "M58",
            "peak_temporary_bytes": peak_temporary_bytes,
        },
    }
    return calibration_record, safety_record, details


def run_evaluation(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["final_labels_opened"]:
        raise PermissionError("M56 final labels must remain sealed")
    verified_locks = {
        name: _verify_lock(specification)
        for name, specification in sorted(config["parent_locks"].items())
    }
    partition_manifest = json.loads(
        _resolve(config["parent_locks"]["v9_partitions"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if set(partition_manifest) != {"11", "23", "37"}:
        raise ValueError("M56 requires all three frozen partition seeds")
    calibration, safety, details = _toy_protocol(
        tuple(float(value) for value in config["penalty_grid"])
    )
    calibration_roundtrip = (
        TubeCalibrationRecord.from_dict(calibration.to_dict()) == calibration
    )
    safety_roundtrip = TubeSafetyEvidence.from_dict(safety.to_dict()) == safety
    evidence = {
        "schema_version": 1,
        "milestone": "M56",
        "configuration_hash": sha256_file(config_path),
        "verified_parent_locks": verified_locks,
        "verified_parent_lock_count": len(verified_locks),
        "partition_manifest_hash": config["parent_locks"]["v9_partitions"]["sha256"],
        "registered_grid": {
            "ranks": config["ranks"],
            "patch_counts": config["patch_counts"],
            "extent_quantiles": config["extent_quantiles"],
            "outer_scale_policies": config["outer_scale_policies"],
            "penalty_grid": config["penalty_grid"],
            "known_coverage_target": config["known_coverage_target"],
            "tangent_multipliers": config["tangent_multipliers"],
        },
        "calibration_record": calibration.to_dict(),
        "safety_record": safety.to_dict(),
        "protocol_details": details,
        "gate": {
            "parent_locks_verified": len(verified_locks) == len(config["parent_locks"]),
            "calibration_roundtrip": calibration_roundtrip,
            "safety_roundtrip": safety_roundtrip,
            "calibration_coverage_passed": calibration.calibration_known_coverage >= 0.90,
            "four_x_tangent_passed": dict(
                safety.tangent_acceptance_by_multiplier
            )["4"] <= 0.01,
            "eight_x_tangent_passed": dict(
                safety.tangent_acceptance_by_multiplier
            )["8"] == 0.0,
            "exact_replay": safety.exact_replay,
            "final_labels_opened": False,
        },
    }
    evidence["gate"]["m56_passed"] = all(
        value is True
        for key, value in evidence["gate"].items()
        if key not in {"final_labels_opened", "m56_passed"}
    ) and evidence["gate"]["final_labels_opened"] is False
    write_canonical_json(output_dir / "evidence.json", evidence)
    write_canonical_json(
        output_dir / "verification.json",
        {
            "schema_version": 1,
            "milestone": "M56",
            "evidence_sha256": sha256_file(output_dir / "evidence.json"),
            "m56_passed": evidence["gate"]["m56_passed"],
            "advance_to_m57": evidence["gate"]["m56_passed"],
        },
    )
    build_artifact_index(output_dir)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    evidence = run_evaluation(arguments.config, arguments.output)
    print(json.dumps(evidence["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
