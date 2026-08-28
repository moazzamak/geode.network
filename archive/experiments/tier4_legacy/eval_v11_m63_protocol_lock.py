from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v11_directional_envelope import (
    CONTRAST_MARGINS,
    PROBE_FAMILIES,
    DirectionalTube,
    calibrate_class_thresholds,
    class_score_matrix,
    composite_endpoint_records,
    contrast_acceptance,
    directional_replay_hash,
    fit_delegated_rbf_head,
    fit_directional_tube,
    generate_geodesic_probes,
    spherical_exp_map,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.tier4.eval_v9_m51_surface_diagnostics import _partition_seed
from src.runtime.schemas import (
    ConformalCalibrationRecord,
    ContrastAcceptanceRecord,
    DirectionalGeometryRecord,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v11" / "m63_protocol_lock.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v11" / "m63_protocol_lock"


def _resolve(path: str) -> Path:
    return REPO_ROOT / Path(path)


def _verify_lock(specification: dict[str, str]) -> dict[str, str]:
    path = _resolve(specification["path"])
    observed = sha256_file(path)
    if observed != specification["sha256"]:
        raise ValueError(f"M63 parent hash mismatch: {path}")
    return {"path": specification["path"], "sha256": observed}


def _class_fixture(
    rng: np.random.Generator,
    *,
    class_label: int,
    count: int,
    dimension: int,
    rank: int,
) -> np.ndarray:
    angle = 0.8
    if class_label == 0:
        center = np.eye(dimension)[0]
        first_tangent = np.eye(dimension)[1]
    else:
        center = np.cos(angle) * np.eye(dimension)[0] + np.sin(angle) * np.eye(
            dimension
        )[1]
        first_tangent = np.sin(angle) * np.eye(dimension)[0] - np.cos(
            angle
        ) * np.eye(dimension)[1]
    basis = np.column_stack(
        [first_tangent, *[np.eye(dimension)[axis] for axis in range(2, rank + 1)]]
    )
    normal_basis = np.column_stack(
        [np.eye(dimension)[axis] for axis in range(rank + 1, dimension)]
    )
    coefficients = rng.normal(scale=0.055, size=(count, rank))
    coefficients[:, 0] *= 1.4
    vectors = coefficients @ basis.T
    vectors += rng.normal(scale=0.0015, size=(count, dimension - rank - 1)) @ (
        normal_basis.T
    )
    return spherical_exp_map(center, vectors)


def _axis_negatives(tube: DirectionalTube) -> np.ndarray:
    vectors = []
    for axis in range(tube.rank):
        for sign in (-1.0, 1.0):
            vectors.append(
                sign
                * 4.0
                * tube.tangent_extents[axis]
                * tube.basis[:, axis]
            )
    return spherical_exp_map(tube.center, np.vstack(vectors))


def _probe_acceptance(
    tubes: list[DirectionalTube],
    thresholds: np.ndarray,
    classes: np.ndarray,
    probes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    margin: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]]]:
    class_columns = {int(value): column for column, value in enumerate(classes)}
    source: dict[str, float] = {}
    system: dict[str, float] = {}
    by_multiplier: dict[str, dict[str, float]] = {}
    for name in PROBE_FAMILIES:
        points, owners, multipliers = probes[name]
        if not len(points):
            source[name] = 0.0
            system[name] = 0.0
            by_multiplier[name] = {}
            continue
        source_accepted = np.asarray(
            [
                tubes[int(owner)].score(points[index : index + 1])[0]
                <= thresholds[class_columns[tubes[int(owner)].class_label]]
                for index, owner in enumerate(owners)
            ],
            dtype=bool,
        )
        scores, observed_classes = class_score_matrix(tubes, points)
        if not np.array_equal(observed_classes, classes):
            raise ValueError("probe class ordering changed")
        system_accepted = contrast_acceptance(
            scores, thresholds, classes, margin=margin
        )["accepted"]
        source[name] = float(np.mean(source_accepted))
        system[name] = float(np.mean(system_accepted))
        by_multiplier[name] = {
            str(value).rstrip("0").rstrip("."): float(
                np.mean(system_accepted[multipliers == value])
            )
            for value in sorted(set(multipliers.tolist()))
        }
    return source, system, by_multiplier


def _toy_protocol(
    config: dict[str, Any], *, delegated_head_hash: str
) -> dict[str, Any]:
    rng = np.random.default_rng(63)
    dimension = 16
    rank = 8
    geometry_parts = [
        _class_fixture(
            rng,
            class_label=label,
            count=96,
            dimension=dimension,
            rank=rank,
        )
        for label in (0, 1)
    ]
    calibration_parts = [
        _class_fixture(
            rng,
            class_label=label,
            count=49,
            dimension=dimension,
            rank=rank,
        )
        for label in (0, 1)
    ]
    quantile_tubes = [
        fit_directional_tube(
            geometry_parts[label],
            calibration_parts[label],
            rank=rank,
            extent_policy="quantile",
            extent_quantile=0.95,
            penalty_weight=16.0,
            class_label=label,
        )
        for label in (0, 1)
    ]
    tubes = [
        fit_directional_tube(
            geometry_parts[label],
            calibration_parts[label],
            rank=rank,
            extent_policy="negative_guided",
            extent_quantile=0.95,
            penalty_weight=16.0,
            class_label=label,
            negative_points=_axis_negatives(quantile_tubes[label]),
        )
        for label in (0, 1)
    ]
    calibration_points = np.vstack(calibration_parts)
    calibration_labels = np.repeat(np.asarray([0, 1]), 49)
    scores, classes = class_score_matrix(tubes, calibration_points)
    thresholds = calibrate_class_thresholds(
        scores,
        calibration_labels,
        classes,
        miscoverage=float(config["miscoverage"]),
    )
    probes = generate_geodesic_probes(tubes, seed=63)
    attempts = []
    selected_margin = None
    selected_acceptance = None
    for margin in tuple(float(value) for value in config["contrast_margin_grid"]):
        source, system, by_multiplier = _probe_acceptance(
            tubes, thresholds, classes, probes, margin=margin
        )
        axis = by_multiplier["axis_tangent"]
        masking = by_multiplier["masking"]
        feasible = bool(
            axis.get("4", 0.0) <= 0.01
            and axis.get("8", 0.0) == 0.0
            and masking.get("4", 0.0) <= 0.01
        )
        attempts.append(
            {
                "margin": margin,
                "source_patch_acceptance": source,
                "system_acceptance": system,
                "system_acceptance_by_multiplier": by_multiplier,
                "feasible": feasible,
            }
        )
        if feasible:
            selected_margin = margin
            selected_acceptance = (source, system, by_multiplier)
            break
    if selected_margin is None or selected_acceptance is None:
        raise ValueError("no registered contrast margin passes fixture safety")
    source, system, by_multiplier = selected_acceptance
    decision = contrast_acceptance(
        scores, thresholds, classes, margin=selected_margin
    )
    records = composite_endpoint_records(
        {name: values[:8] for name, values in decision.items()},
        calibration_labels[:8],
        calibration_labels[:8],
    )
    geometry_hash = payload_hash([tube.to_dict() for tube in tubes])
    partition_hash = payload_hash(
        {
            "geometry_counts": [len(values) for values in geometry_parts],
            "calibration_counts": [len(values) for values in calibration_parts],
            "seed": 63,
        }
    )
    replay_hash = directional_replay_hash(
        tubes,
        thresholds,
        miscoverage=float(config["miscoverage"]),
        contrast_margin=selected_margin,
    )
    parameter_count = sum(tube.parameter_count for tube in tubes)
    fit_work_units = sum(
        len(values) * dimension * rank for values in geometry_parts
    )
    geometry_record = DirectionalGeometryRecord(
        geometry_hash=geometry_hash,
        representation_hash=config["delegated_head_lineage"]["representation_hash"],
        partition_hash=partition_hash,
        rank=rank,
        patch_count=1,
        extent_policy="negative_guided",
        extent_quantile=0.95,
        parameter_count=parameter_count,
        fit_work_units=fit_work_units,
        replay_hash=replay_hash,
    )
    conformal_hash = payload_hash(
        {
            "geometry_replay_hash": replay_hash,
            "thresholds": thresholds.tolist(),
            "miscoverage": config["miscoverage"],
        }
    )
    conformal_record = ConformalCalibrationRecord(
        geometry_replay_hash=replay_hash,
        delegated_head_hash=delegated_head_hash,
        miscoverage=float(config["miscoverage"]),
        class_counts=tuple((str(label), 49) for label in classes),
        class_thresholds=tuple(
            (str(label), float(threshold))
            for label, threshold in zip(classes, thresholds, strict=True)
        ),
        selected_before_development=True,
        final_labels_opened=False,
        replay_hash=conformal_hash,
    )
    peak_temporary_bytes = max(
        (
            points.nbytes + owners.nbytes + multipliers.nbytes
            for points, owners, multipliers in probes.values()
        ),
        default=0,
    )
    contrast_record = ContrastAcceptanceRecord(
        calibration_replay_hash=conformal_hash,
        margin_grid=CONTRAST_MARGINS,
        selected_margin=selected_margin,
        probe_counts=tuple(
            (name, int(len(probes[name][0]))) for name in PROBE_FAMILIES
        ),
        source_patch_acceptance=tuple(
            (name, source[name]) for name in PROBE_FAMILIES
        ),
        system_acceptance=tuple(
            (name, system[name]) for name in PROBE_FAMILIES
        ),
        endpoint_count=len(records),
        latency_seconds=0.0,
        peak_temporary_bytes=int(peak_temporary_bytes),
        exact_replay=True,
    )
    near_tie = contrast_acceptance(
        np.asarray([[0.92, 0.97]]),
        np.ones(2),
        np.asarray([0, 1]),
        margin=0.1,
    )
    coverage_by_class = {
        str(label): float(
            np.mean(scores[calibration_labels == label, column] <= thresholds[column])
        )
        for column, label in enumerate(classes)
    }
    return {
        "geometry_record": geometry_record.to_dict(),
        "conformal_record": conformal_record.to_dict(),
        "contrast_record": contrast_record.to_dict(),
        "contrast_selection_attempts": attempts,
        "conformal_coverage_by_class": coverage_by_class,
        "negative_guided_extent_ratios": [
            (
                tubes[index].tangent_extents
                / quantile_tubes[index].tangent_extents
            ).tolist()
            for index in range(len(tubes))
        ],
        "composite_endpoint_records": records,
        "near_tie_fixture": {
            "v10_minimum_rule_accepted": True,
            "v11_contrast_rule_accepted": bool(near_tie["accepted"][0]),
        },
        "resource_accounting": {
            "parameter_count": parameter_count,
            "fit_work_units": fit_work_units,
            "latency_contract_registered": True,
            "latency_seconds": 0.0,
            "latency_measurement_stage": "M65",
            "peak_temporary_bytes": int(peak_temporary_bytes),
        },
        "probe_generator_hash": payload_hash(
            {
                "generate_geodesic_probes": inspect.getsource(
                    generate_geodesic_probes
                ),
                "spherical_exp_map": inspect.getsource(spherical_exp_map),
            }
        ),
    }


def _delegated_head_lock(
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    source = json.loads(
        _resolve(config["parent_locks"]["source_config"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    partition_config = json.loads(
        _resolve(config["parent_locks"]["v9_partition_config"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    seed = 11
    loaded = _load_seed_data(source["seed_inputs"][str(seed)])
    train_features, train_labels = loaded["datasets"]["train"]
    dev_features, dev_labels = loaded["datasets"]["dev"]
    known_classes = np.asarray(config["known_classes"], dtype=np.int64)
    unknown_classes = np.asarray(config["proxy_unknown_classes"], dtype=np.int64)
    partitions = _partition_seed(
        train_labels,
        dev_labels,
        seed=seed,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        geometry_fraction=float(partition_config["geometry_fraction"]),
    )
    geometry_x = train_features[partitions["geometry_fit"]]
    geometry_y = train_labels[partitions["geometry_fit"]]
    calibration_x = train_features[partitions["score_calibration"]]
    calibration_y = train_labels[partitions["score_calibration"]]
    query = np.vstack([calibration_x, dev_features])
    arguments = {
        "known_classes": tuple(int(value) for value in known_classes),
        "seed": seed,
        "c_value": 1.0,
        "gamma": "scale",
    }
    first = fit_delegated_rbf_head(
        geometry_x,
        geometry_y,
        calibration_x,
        calibration_y,
        query,
        **arguments,
    )
    second = fit_delegated_rbf_head(
        geometry_x,
        geometry_y,
        calibration_x,
        calibration_y,
        query,
        **arguments,
    )
    if not np.array_equal(
        first["predictions"], second["predictions"]
    ) or not np.array_equal(first["probabilities"], second["probabilities"]):
        raise RuntimeError("delegated-head predictions did not replay exactly")
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = arrays_dir / "delegated_head_predictions.npy"
    probabilities_path = arrays_dir / "delegated_head_probabilities.npy"
    np.save(predictions_path, first["predictions"], allow_pickle=False)
    np.save(probabilities_path, first["probabilities"], allow_pickle=False)
    known_dev = np.isin(dev_labels, known_classes)
    dev_predictions = first["predictions"][len(calibration_x) :]
    return {
        "family": "rbf_svm",
        "seed": seed,
        "fit_partition": "geometry_fit",
        "calibration_partition": "score_calibration",
        "query_layout": {
            "score_calibration_count": int(len(calibration_x)),
            "development_count": int(len(dev_features)),
        },
        "known_classes": known_classes.tolist(),
        "proxy_unknown_classes_excluded_from_fit": bool(
            not np.any(np.isin(geometry_y, unknown_classes))
            and not np.any(np.isin(calibration_y, unknown_classes))
        ),
        "representation_hash": config["delegated_head_lineage"][
            "representation_hash"
        ],
        "geometry_partition_hash": payload_hash(
            partitions["geometry_fit"].tolist()
        ),
        "calibration_partition_hash": payload_hash(
            partitions["score_calibration"].tolist()
        ),
        "predictions_path": str(
            predictions_path.relative_to(REPO_ROOT)
        ).replace("\\", "/"),
        "predictions_sha256": sha256_file(predictions_path),
        "probabilities_path": str(
            probabilities_path.relative_to(REPO_ROOT)
        ).replace("\\", "/"),
        "probabilities_sha256": sha256_file(probabilities_path),
        "development_known_balanced_accuracy": float(
            np.mean(dev_predictions[known_dev] == dev_labels[known_dev])
        ),
        "support_vector_count": first["support_vector_count"],
        "support_vectors_unchanged_by_calibration": first[
            "support_vectors_unchanged_by_calibration"
        ],
        "fit_class_count": first["fit_class_count"],
        "calibration_class_count": first["calibration_class_count"],
        "exact_replay": True,
    }


def run_evaluation(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["final_labels_opened"]:
        raise PermissionError("M63 final labels must remain sealed")
    verified_locks = {
        name: _verify_lock(specification)
        for name, specification in sorted(config["parent_locks"].items())
    }
    partitions = json.loads(
        _resolve(config["parent_locks"]["v9_partitions"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if set(partitions) != {"11", "23", "37"}:
        raise ValueError("M63 requires all three frozen partition seeds")
    for seed, manifest in partitions.items():
        split_hashes = tuple(manifest["hashes"].values())
        if len(split_hashes) != len(set(split_hashes)):
            raise ValueError(f"M63 partition split hashes overlap for seed {seed}")
    delegated_head = _delegated_head_lock(config, output_dir)
    first = _toy_protocol(
        config, delegated_head_hash=delegated_head["predictions_sha256"]
    )
    second = _toy_protocol(
        config, delegated_head_hash=delegated_head["predictions_sha256"]
    )
    exact_fixture_replay = payload_hash(first) == payload_hash(second)
    geometry_roundtrip = (
        DirectionalGeometryRecord.from_dict(first["geometry_record"]).to_dict()
        == first["geometry_record"]
    )
    conformal_roundtrip = (
        ConformalCalibrationRecord.from_dict(first["conformal_record"]).to_dict()
        == first["conformal_record"]
    )
    contrast_roundtrip = (
        ContrastAcceptanceRecord.from_dict(first["contrast_record"]).to_dict()
        == first["contrast_record"]
    )
    minimum_coverage = min(first["conformal_coverage_by_class"].values())
    evidence = {
        "schema_version": 1,
        "milestone": "M63",
        "configuration_hash": sha256_file(config_path),
        "verified_parent_locks": verified_locks,
        "verified_parent_lock_count": len(verified_locks),
        "partition_seed_count": len(partitions),
        "delegated_head": delegated_head,
        "registered_grid": {
            key: config[key]
            for key in (
                "ranks",
                "patch_counts",
                "extent_policies",
                "extent_quantiles",
                "miscoverage",
                "contrast_margin_grid",
                "tangent_multipliers",
                "masking_multipliers",
            )
        },
        "protocol_fixture": first,
        "gate": {
            "parent_locks_verified": len(verified_locks)
            == len(config["parent_locks"]),
            "partition_identities_verified": len(partitions) == 3,
            "delegated_head_lineage_verified": delegated_head[
                "proxy_unknown_classes_excluded_from_fit"
            ]
            and delegated_head["fit_partition"] == "geometry_fit"
            and delegated_head["calibration_partition"] == "score_calibration",
            "delegated_head_exact_replay": delegated_head["exact_replay"],
            "geometry_schema_roundtrip": geometry_roundtrip,
            "conformal_schema_roundtrip": conformal_roundtrip,
            "contrast_schema_roundtrip": contrast_roundtrip,
            "finite_sample_coverage_passed": minimum_coverage >= 0.92,
            "near_tie_contrast_rejected": not first["near_tie_fixture"][
                "v11_contrast_rule_accepted"
            ],
            "four_x_tangent_passed": first["contrast_selection_attempts"][-1][
                "system_acceptance_by_multiplier"
            ]["axis_tangent"]["4"]
            <= 0.01,
            "eight_x_tangent_passed": first["contrast_selection_attempts"][-1][
                "system_acceptance_by_multiplier"
            ]["axis_tangent"]["8"]
            == 0.0,
            "four_x_masking_passed": first["contrast_selection_attempts"][-1][
                "system_acceptance_by_multiplier"
            ]["masking"]["4"]
            <= 0.01,
            "composite_endpoints_separated": all(
                set(record)
                >= {
                    "envelope_accepted",
                    "envelope_class",
                    "head_class",
                    "composite_class",
                }
                for record in first["composite_endpoint_records"]
            ),
            "exact_fixture_replay": exact_fixture_replay,
            "final_labels_opened": False,
        },
    }
    evidence["gate"]["m63_passed"] = all(
        value is True
        for key, value in evidence["gate"].items()
        if key not in {"final_labels_opened", "m63_passed"}
    ) and evidence["gate"]["final_labels_opened"] is False
    write_canonical_json(output_dir / "evidence.json", evidence)
    write_canonical_json(
        output_dir / "verification.json",
        {
            "schema_version": 1,
            "milestone": "M63",
            "evidence_sha256": sha256_file(output_dir / "evidence.json"),
            "m63_passed": evidence["gate"]["m63_passed"],
            "advance_to_m64": evidence["gate"]["m63_passed"],
        },
    )
    write_canonical_json(
        output_dir / "replay_verification.json",
        {
            "schema_version": 1,
            "milestone": "M63",
            "first_fixture_sha256": payload_hash(first),
            "second_fixture_sha256": payload_hash(second),
            "byte_identical": exact_fixture_replay,
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
