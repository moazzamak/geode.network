from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v6_factorial import primitive_field_matrix
from experiments.common.v9_surface_support import (
    STRATA,
    assign_strata,
    class_minimum_fields,
    deterministic_equal_mass_bands,
    metric_field_matrix,
    permuted_labels,
    random_orientation,
    replay_digest,
    validate_disjoint_partitions,
)
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from src.runtime.schemas import SurfaceSupportDiagnostic
from src.subspace_primitive import SubspacePrimitive


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v9" / "m51_surface_diagnostics.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v9" / "m51_surface_diagnostics"


def _resolve(path: str) -> Path:
    return REPO_ROOT / Path(path)


def _verify_artifact(specification: dict[str, str]) -> Path:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"immutable artifact hash mismatch: {path}")
    return path


def _partition_seed(
    train_labels: np.ndarray,
    dev_labels: np.ndarray,
    *,
    seed: int,
    known_classes: np.ndarray,
    unknown_classes: np.ndarray,
    geometry_fraction: float,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    geometry: list[int] = []
    calibration: list[int] = []
    for class_label in known_classes:
        indices = np.flatnonzero(train_labels == class_label)
        shuffled = rng.permutation(indices)
        count = int(np.floor(len(shuffled) * geometry_fraction))
        geometry.extend(shuffled[:count].tolist())
        calibration.extend(shuffled[count:].tolist())
    partitions = {
        "geometry_fit": np.asarray(sorted(geometry), dtype=np.int64),
        "score_calibration": np.asarray(sorted(calibration), dtype=np.int64),
        "development_eval": np.flatnonzero(np.isin(dev_labels, known_classes)),
        "unknown_eval": np.flatnonzero(np.isin(dev_labels, unknown_classes)),
    }
    identifiers = {
        "geometry_fit": [f"train:{index}" for index in partitions["geometry_fit"]],
        "score_calibration": [
            f"train:{index}" for index in partitions["score_calibration"]
        ],
        "development_eval": [
            f"dev:{index}" for index in partitions["development_eval"]
        ],
        "unknown_eval": [f"dev:{index}" for index in partitions["unknown_eval"]],
    }
    validate_disjoint_partitions(identifiers)
    return partitions


def _metric_pairs(values: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple((name, float(values[name])) for name in STRATA)


def _count_pairs(values: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple((name, int(values[name])) for name in STRATA)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _evaluate_variant(
    *,
    seed: int,
    variant: str,
    component_hash: str,
    representation_hash: str,
    classes: np.ndarray,
    geometry_labels: np.ndarray,
    development_labels: np.ndarray,
    geometry_scores: np.ndarray,
    development_scores: np.ndarray,
    unknown_scores: np.ndarray,
    normalized_geometry_scores: np.ndarray,
    metric_geometry_scores: np.ndarray,
    quantiles: np.ndarray,
    equal_mass_fraction: float,
    partition_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for column, class_label in enumerate(classes):
        own_geometry = geometry_scores[geometry_labels == class_label, column]
        meaningful_negative_interior = True
        try:
            bands = deterministic_equal_mass_bands(
                own_geometry, fraction=equal_mass_fraction
            )
            known_strata = assign_strata(development_scores[:, column], bands)
            unknown_strata = assign_strata(unknown_scores[:, column], bands)
        except ValueError:
            meaningful_negative_interior = False
            bands = {"mass_count": 0, "deep_upper": 0.0, "near_lower": 0.0}
            known_strata = np.full(
                len(development_scores), "middle_interior", dtype=object
            )
            unknown_strata = np.full(
                len(unknown_scores), "middle_interior", dtype=object
            )
            known_strata[development_scores[:, column] >= 0.0] = "exterior"
            unknown_strata[unknown_scores[:, column] >= 0.0] = "exterior"
        own_mask = development_labels == class_label
        competing_mask = ~own_mask
        counts: dict[str, int] = {}
        precisions: dict[str, float] = {}
        competing_occupancies: dict[str, float] = {}
        unknown_occupancies: dict[str, float] = {}
        own_concentrations: dict[str, float] = {}
        selected_ids: list[str] = []
        for stratum in STRATA:
            stratum_mask = known_strata == stratum
            counts[stratum] = int(np.sum(stratum_mask))
            precisions[stratum] = _safe_ratio(
                int(np.sum(stratum_mask & own_mask)), counts[stratum]
            )
            competing_occupancies[stratum] = _safe_ratio(
                int(np.sum(stratum_mask & competing_mask)),
                int(np.sum(competing_mask)),
            )
            unknown_occupancies[stratum] = _safe_ratio(
                int(np.sum(unknown_strata == stratum)), len(unknown_strata)
            )
            own_concentrations[stratum] = _safe_ratio(
                int(np.sum(stratum_mask & own_mask)), int(np.sum(own_mask))
            )
            selected_ids.extend(
                f"development_eval:{index}"
                for index in np.flatnonzero(stratum_mask)
            )
            selected_ids.extend(
                f"unknown_eval:{index}"
                for index in np.flatnonzero(unknown_strata == stratum)
            )
        normalized_quantiles = tuple(
            float(value)
            for value in np.quantile(
                normalized_geometry_scores[geometry_labels == class_label, column],
                quantiles,
                method="linear",
            )
        )
        metric_quantiles = tuple(
            float(value)
            for value in np.quantile(
                metric_geometry_scores[geometry_labels == class_label, column],
                quantiles,
                method="linear",
            )
        )
        replay_payload = {
            "seed": seed,
            "variant": variant,
            "class_label": int(class_label),
            "bands": bands,
            "selected_ids": selected_ids,
            "partition_hash": partition_hash,
        }
        record = SurfaceSupportDiagnostic(
            component_hash=component_hash,
            representation_hash=representation_hash,
            score_variant=variant,
            score_direction="lower_is_stronger_support",
            class_label=int(class_label),
            seed=seed,
            partition_id=partition_hash,
            normalized_signed_depth_quantiles=normalized_quantiles,
            metric_signed_depth_quantiles=metric_quantiles,
            stratum_counts=_count_pairs(counts),
            own_class_precision=_metric_pairs(precisions),
            competing_class_occupancy=_metric_pairs(competing_occupancies),
            unknown_occupancy=_metric_pairs(unknown_occupancies),
            width_selection_provenance=(
                (
                    "geometry_fit_no_meaningful_negative_interior"
                    if not meaningful_negative_interior
                    else "geometry_fit_negative_interior_equal_mass_fraction="
                    f"{equal_mass_fraction}"
                )
            ),
            selected_ids=tuple(selected_ids),
            replay_hash=replay_digest(replay_payload),
        )
        records.append(record.to_dict())
        summaries.append(
            {
                "class_label": int(class_label),
                "meaningful_negative_interior": meaningful_negative_interior,
                "bands": bands,
                "own_concentration": own_concentrations,
                "near_minus_deep_concentration": (
                    own_concentrations["near_surface"]
                    - own_concentrations["deep_interior"]
                ),
                "near_minus_deep_precision": (
                    precisions["near_surface"] - precisions["deep_interior"]
                ),
                "deep_minus_near_competing_occupancy": (
                    competing_occupancies["deep_interior"]
                    - competing_occupancies["near_surface"]
                ),
            }
        )
    return records, summaries


def _control_summary(
    *,
    candidates: list[SubspacePrimitive],
    component_labels: np.ndarray,
    classes: np.ndarray,
    geometry_features: np.ndarray,
    geometry_labels: np.ndarray,
    development_features: np.ndarray,
    development_labels: np.ndarray,
    fraction: float,
    seed: int,
) -> dict[str, Any]:
    oriented = [
        random_orientation(candidate, seed=seed * 1000 + index)
        for index, candidate in enumerate(candidates)
    ]
    oriented_fields = primitive_field_matrix(
        oriented,
        np.vstack([geometry_features, development_features]),
        primitive="subspace_r32",
        score="normalized_radial",
    )
    split = len(geometry_features)

    def concentration_fraction(
        fields: np.ndarray, labels: np.ndarray
    ) -> float:
        scores = class_minimum_fields(fields, labels, classes)
        geometry_scores = scores[:split]
        development_scores = scores[split:]
        directions = []
        for column, class_label in enumerate(classes):
            own_geometry = geometry_scores[geometry_labels == class_label, column]
            try:
                bands = deterministic_equal_mass_bands(
                    own_geometry, fraction=fraction
                )
            except ValueError:
                directions.append(False)
                continue
            strata = assign_strata(development_scores[:, column], bands)
            own = development_labels == class_label
            near = np.sum((strata == "near_surface") & own)
            deep = np.sum((strata == "deep_interior") & own)
            directions.append(bool(near > deep))
        return float(np.mean(directions))

    base_fields = primitive_field_matrix(
        candidates,
        np.vstack([geometry_features, development_features]),
        primitive="subspace_r32",
        score="normalized_radial",
    )
    permuted = permuted_labels(component_labels, seed=seed)
    payload = {
        "random_orientation_class_fraction": concentration_fraction(
            oriented_fields, component_labels
        ),
        "label_permutation_class_fraction": concentration_fraction(
            base_fields, permuted
        ),
    }
    return {**payload, "replay_hash": replay_digest(payload)}


def run_evaluation(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_config_path = _verify_artifact(config["source_config"])
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    for name in ("parent_evidence", "v7_gaussian_evidence", "v8_evidence"):
        _verify_artifact(config[name])
    known_classes = np.asarray(config["known_classes"], dtype=np.int64)
    unknown_classes = np.asarray(config["proxy_unknown_classes"], dtype=np.int64)
    all_records: list[dict[str, Any]] = []
    seed_results: list[dict[str, Any]] = []
    partition_manifest: dict[str, Any] = {}
    for seed in config["seeds"]:
        seed_input = source_config["seed_inputs"][str(seed)]
        loaded = _load_seed_data(seed_input)
        train_features, train_labels = loaded["datasets"]["train"]
        dev_features, dev_labels = loaded["datasets"]["dev"]
        partitions = _partition_seed(
            train_labels,
            dev_labels,
            seed=seed,
            known_classes=known_classes,
            unknown_classes=unknown_classes,
            geometry_fraction=float(config["geometry_fraction"]),
        )
        partition_ids = {
            "geometry_fit": [
                f"train:{index}" for index in partitions["geometry_fit"]
            ],
            "score_calibration": [
                f"train:{index}" for index in partitions["score_calibration"]
            ],
            "development_eval": [
                f"dev:{index}" for index in partitions["development_eval"]
            ],
            "unknown_eval": [
                f"dev:{index}" for index in partitions["unknown_eval"]
            ],
        }
        partition_hashes = {
            name: payload_hash(identifiers)
            for name, identifiers in partition_ids.items()
        }
        partition_hash = payload_hash(partition_hashes)
        partition_manifest[str(seed)] = {
            "hashes": partition_hashes,
            "counts": {name: len(values) for name, values in partition_ids.items()},
            "combined_hash": partition_hash,
        }
        parent_path = _verify_artifact(config["parent_students"][str(seed)])
        student = json.loads(parent_path.read_text(encoding="utf-8"))
        if student["parent_representation_hash"] != seed_input["parent_representation_hash"]:
            raise ValueError("A2 representation lineage mismatch")
        candidates = [
            SubspacePrimitive.from_dict(item["payload"])
            for item in student["selected_candidates"]
        ]
        component_labels = np.asarray(
            [int(candidate.class_label) for candidate in candidates], dtype=np.int64
        )
        classes = np.asarray(student["classes"], dtype=np.int64)
        eligible = np.isin(component_labels, known_classes)
        candidates = [
            candidate for candidate, keep in zip(candidates, eligible, strict=True) if keep
        ]
        component_labels = component_labels[eligible]
        classes = known_classes
        component_hash = payload_hash(
            [candidate.to_dict() for candidate in candidates]
        )
        geometry_indices = partitions["geometry_fit"]
        development_indices = partitions["development_eval"]
        unknown_indices = partitions["unknown_eval"]
        geometry_features = train_features[geometry_indices]
        geometry_labels = train_labels[geometry_indices]
        development_features = dev_features[development_indices]
        development_labels = dev_labels[development_indices]
        unknown_features = dev_features[unknown_indices]
        combined = np.vstack(
            [geometry_features, development_features, unknown_features]
        )
        normalized_fields = primitive_field_matrix(
            candidates,
            combined,
            primitive="subspace_r32",
            score="normalized_radial",
        )
        metric_fields = metric_field_matrix(
            candidates, combined, eta=float(config["metric_eta"])
        )
        normalized_scores = class_minimum_fields(
            normalized_fields, component_labels, classes
        )
        metric_scores = class_minimum_fields(
            metric_fields, component_labels, classes
        )
        geometry_end = len(geometry_features)
        development_end = geometry_end + len(development_features)
        variant_results: dict[str, Any] = {}
        for variant, fields in (
            ("normalized", normalized_scores),
            ("metric_corrected", metric_scores),
        ):
            records, summaries = _evaluate_variant(
                seed=seed,
                variant=variant,
                component_hash=component_hash,
                representation_hash=student["parent_representation_hash"],
                classes=classes,
                geometry_labels=geometry_labels,
                development_labels=development_labels,
                geometry_scores=fields[:geometry_end],
                development_scores=fields[geometry_end:development_end],
                unknown_scores=fields[development_end:],
                normalized_geometry_scores=normalized_scores[:geometry_end],
                metric_geometry_scores=metric_scores[:geometry_end],
                quantiles=np.asarray(config["quantiles"], dtype=np.float64),
                equal_mass_fraction=float(config["equal_mass_fraction"]),
                partition_hash=partition_hash,
            )
            all_records.extend(records)
            variant_results[variant] = summaries
        controls = _control_summary(
            candidates=candidates,
            component_labels=component_labels,
            classes=classes,
            geometry_features=geometry_features,
            geometry_labels=geometry_labels,
            development_features=development_features,
            development_labels=development_labels,
            fraction=float(config["equal_mass_fraction"]),
            seed=seed,
        )
        seed_results.append(
            {
                "seed": seed,
                "component_hash": component_hash,
                "representation_hash": student["parent_representation_hash"],
                "partition_hash": partition_hash,
                "variants": variant_results,
                "negative_controls": controls,
            }
        )
    metric_summaries = [
        (result["seed"], summary)
        for result in seed_results
        for summary in result["variants"]["metric_corrected"]
    ]
    class_fraction_per_seed = {}
    consistency_cells: list[dict[str, Any]] = []
    for result in seed_results:
        summaries = result["variants"]["metric_corrected"]
        class_fraction = float(
            np.mean(
                [
                    summary["near_minus_deep_concentration"] > 0.0
                    for summary in summaries
                ]
            )
        )
        class_fraction_per_seed[str(result["seed"])] = class_fraction
        consistency_cells.extend(
            [
                {
                    "seed": result["seed"],
                    "diagnostic": "concentration",
                    "passed": class_fraction > 0.5,
                },
                {
                    "seed": result["seed"],
                    "diagnostic": "precision",
                    "passed": float(
                        np.mean(
                            [
                                item["near_minus_deep_precision"]
                                for item in summaries
                            ]
                        )
                    )
                    > 0.0,
                },
                {
                    "seed": result["seed"],
                    "diagnostic": "competing_occupancy",
                    "passed": float(
                        np.mean(
                            [
                                item["deep_minus_near_competing_occupancy"]
                                for item in summaries
                            ]
                        )
                    )
                    > 0.0,
                },
            ]
        )
    mean_precision_difference = float(
        np.mean(
            [
                summary["near_minus_deep_precision"]
                for _, summary in metric_summaries
            ]
        )
    )
    mean_occupancy_difference = float(
        np.mean(
            [
                summary["deep_minus_near_competing_occupancy"]
                for _, summary in metric_summaries
            ]
        )
    )
    gate_config = config["gate"]
    gate = {
        "class_fraction_per_seed": class_fraction_per_seed,
        "class_fraction_passed": all(
            value >= float(gate_config["minimum_class_fraction_per_seed"])
            for value in class_fraction_per_seed.values()
        ),
        "mean_near_minus_deep_precision": mean_precision_difference,
        "mean_deep_minus_near_competing_occupancy": mean_occupancy_difference,
        "precision_or_occupancy_passed": (
            mean_precision_difference
            >= float(gate_config["minimum_precision_or_occupancy_difference"])
            or mean_occupancy_difference
            >= float(gate_config["minimum_precision_or_occupancy_difference"])
        ),
        "consistent_cells": int(sum(item["passed"] for item in consistency_cells)),
        "consistency_cells": consistency_cells,
    }
    gate["consistency_passed"] = gate["consistent_cells"] >= int(
        gate_config["minimum_consistent_seed_diagnostic_cells"]
    )
    gate["m51_passed"] = bool(
        gate["class_fraction_passed"]
        and gate["precision_or_occupancy_passed"]
        and gate["consistency_passed"]
    )
    evidence = {
        "schema_version": 1,
        "milestone": "M51",
        "configuration_hash": sha256_file(config_path),
        "parent_locks_verified": True,
        "final_labels_opened": False,
        "partition_manifest": partition_manifest,
        "diagnostics": all_records,
        "seed_results": seed_results,
        "gate": gate,
        "branch_decision": {
            "m52_shell_scores_open": gate["m51_passed"],
            "m53_fitted_shell_open": gate["m51_passed"],
            "m53_bounded_tube_open": True,
        },
    }
    write_canonical_json(output_dir / "partition_manifest.json", partition_manifest)
    write_canonical_json(output_dir / "evidence.json", evidence)
    verification = {
        "schema_version": 1,
        "milestone": "M51",
        "evidence_sha256": sha256_file(output_dir / "evidence.json"),
        "record_count": len(all_records),
        "all_record_roundtrips_valid": all(
            SurfaceSupportDiagnostic.from_dict(record).to_dict() == record
            for record in all_records
        ),
        "advance_to_m52": gate["m51_passed"],
        "bounded_tube_open": True,
    }
    write_canonical_json(output_dir / "verification.json", verification)
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
