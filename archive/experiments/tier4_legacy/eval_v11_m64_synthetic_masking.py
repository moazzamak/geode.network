from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.common.v11_directional_envelope import (
    DirectionalTube,
    calibrate_class_thresholds,
    class_score_matrix,
    contrast_acceptance,
    deterministic_directional_patch_assignments,
    estimate_directional_rank,
    fit_directional_tube,
    mean_direction,
    negative_guided_extents,
    normalize_directions,
    spherical_exp_map,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from src.subspace_primitive import deterministic_basis_signs


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v11" / "m64_synthetic_masking.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v11" / "m64_synthetic_masking"


def _tangent_basis(
    rng: np.random.Generator,
    center: np.ndarray,
    rank: int,
) -> np.ndarray:
    candidates = rng.normal(size=(len(center), rank))
    candidates -= center[:, None] * (center @ candidates)[None, :]
    basis, _ = np.linalg.qr(candidates)
    return deterministic_basis_signs(basis[:, :rank])


def _tube_samples(
    rng: np.random.Generator,
    center: np.ndarray,
    basis: np.ndarray,
    count: int,
    *,
    coefficient_scale: float = 0.10,
    noise_scale: float = 0.0002,
) -> np.ndarray:
    coefficients = rng.uniform(
        -coefficient_scale, coefficient_scale, size=(count, basis.shape[1])
    )
    vectors = coefficients @ basis.T
    noise = rng.normal(size=vectors.shape)
    noise -= center[None, :] * (noise @ center)[:, None]
    noise -= (noise @ basis) @ basis.T
    noise_norm = np.linalg.norm(noise, axis=1, keepdims=True)
    noise = np.divide(noise, noise_norm, out=np.zeros_like(noise), where=noise_norm > 0)
    vectors += noise_scale * noise
    return spherical_exp_map(center, vectors)


def _axis_probes(
    tube: DirectionalTube, multiplier: float
) -> np.ndarray:
    vectors = [
        sign
        * multiplier
        * tube.tangent_extents[axis]
        * tube.basis[:, axis]
        for axis in range(tube.rank)
        for sign in (-1.0, 1.0)
    ]
    return spherical_exp_map(tube.center, np.vstack(vectors))


def _straight_diagnostic(
    rng: np.random.Generator,
    true_rank: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    dimension = int(config["ambient_dimension"])
    center = normalize_directions(rng.normal(size=(1, dimension)))[0]
    basis = _tangent_basis(rng, center, true_rank)
    geometry = _tube_samples(
        rng, center, basis, int(config["geometry_count"])
    )
    extent_calibration = _tube_samples(
        rng, center, basis, int(config["calibration_count"])
    )
    conformal_calibration = _tube_samples(
        rng, center, basis, int(config["calibration_count"])
    )
    evaluation = _tube_samples(
        rng, center, basis, int(config["evaluation_count"])
    )
    recovery = estimate_directional_rank(
        geometry,
        rank_grid=tuple(int(value) for value in config["rank_grid"]),
        explained_variance_target=float(config["explained_variance_target"]),
    )
    tube = fit_directional_tube(
        geometry,
        extent_calibration,
        rank=true_rank,
        extent_policy="quantile",
        extent_quantile=float(config["extent_quantile"]),
        penalty_weight=float(config["penalty_weight"]),
        class_label=0,
    )
    calibration_scores = tube.score(conformal_calibration)
    thresholds = calibrate_class_thresholds(
        calibration_scores[:, None],
        np.zeros(len(conformal_calibration), dtype=np.int64),
        np.asarray([0]),
        miscoverage=float(config["miscoverage"]),
    )
    threshold = float(thresholds[0])
    coverage = float(np.mean(tube.score(evaluation) <= threshold))
    four_x_acceptance = float(np.mean(tube.score(_axis_probes(tube, 4.0)) <= threshold))
    eight_x_acceptance = float(
        np.mean(tube.score(_axis_probes(tube, 8.0)) <= threshold)
    )
    random_basis = _tangent_basis(rng, tube.center, true_rank)
    random_tube = DirectionalTube(
        center=tube.center,
        basis=random_basis,
        residual_scale=tube.residual_scale,
        tangent_extents=tube.tangent_extents,
        outer_scales=tube.outer_scales,
        penalty_weight=tube.penalty_weight,
        class_label=0,
    )
    fitted_residual = float(np.median(tube.coordinates(evaluation)[1]))
    random_residual = float(np.median(random_tube.coordinates(evaluation)[1]))
    return {
        "true_rank": true_rank,
        "recovered_rank": recovery["selected_rank"],
        "in_support_coverage": coverage,
        "four_x_acceptance": four_x_acceptance,
        "eight_x_acceptance": eight_x_acceptance,
        "threshold": threshold,
        "random_orientation_residual_ratio": random_residual
        / max(fitted_residual, 1e-15),
    }


def _fit_atlas(
    geometry: np.ndarray,
    calibration: np.ndarray,
    *,
    rank: int,
    seed: int,
) -> list[DirectionalTube]:
    assignments = deterministic_directional_patch_assignments(
        geometry, patch_count=2, seed=seed
    )
    centers = np.vstack(
        [mean_direction(geometry[assignments == patch]) for patch in range(2)]
    )
    calibration_assignments = np.argmax(calibration @ centers.T, axis=1)
    tubes = []
    for patch in range(2):
        geometry_patch = geometry[assignments == patch]
        calibration_patch = calibration[calibration_assignments == patch]
        if len(geometry_patch) < rank + 2 or not len(calibration_patch):
            raise ValueError("directional atlas produced a rank-infeasible patch")
        tubes.append(
            fit_directional_tube(
                geometry_patch,
                calibration_patch,
                rank=rank,
                extent_policy="quantile",
                extent_quantile=0.95,
                penalty_weight=16.0,
                class_label=0,
                patch_id=patch,
            )
        )
    return tubes


def _minimum_residual(
    tubes: Sequence[DirectionalTube], points: np.ndarray
) -> np.ndarray:
    return np.min(
        np.column_stack([tube.coordinates(points)[1] for tube in tubes]), axis=1
    )


def _curved_samples(
    rng: np.random.Generator, count: int, dimension: int
) -> np.ndarray:
    center = np.eye(dimension)[0]
    t = rng.uniform(-0.7, 0.7, size=count)
    vectors = np.zeros((count, dimension))
    vectors[:, 1] = t
    vectors[:, 2] = 0.9 * (t * t - np.mean(t * t))
    vectors[:, 3:10] = rng.uniform(-0.08, 0.08, size=(count, 7))
    return spherical_exp_map(center, vectors)


def _two_mode_samples(
    rng: np.random.Generator, count: int, dimension: int
) -> np.ndarray:
    first_count = count // 2
    angle = 0.8
    first_center = np.eye(dimension)[0]
    second_center = (
        np.cos(angle) * np.eye(dimension)[0]
        + np.sin(angle) * np.eye(dimension)[20]
    )
    first_basis = np.eye(dimension)[:, 1:9]
    second_candidates = np.eye(dimension)[:, 9:17]
    return np.vstack(
        [
            _tube_samples(rng, first_center, first_basis, first_count),
            _tube_samples(
                rng, second_center, second_candidates, count - first_count
            ),
        ]
    )


def _atlas_diagnostic(
    rng: np.random.Generator,
    *,
    family: str,
    seed: int,
    config: dict[str, Any],
) -> dict[str, float]:
    generator = _curved_samples if family == "curved_arc" else _two_mode_samples
    dimension = int(config["ambient_dimension"])
    geometry = generator(rng, int(config["geometry_count"]), dimension)
    calibration = generator(rng, int(config["calibration_count"]), dimension)
    evaluation = generator(rng, int(config["evaluation_count"]), dimension)
    global_tube = fit_directional_tube(
        geometry,
        calibration,
        rank=8,
        extent_policy="quantile",
        extent_quantile=0.95,
        penalty_weight=16.0,
        class_label=0,
    )
    atlas = _fit_atlas(
        geometry, calibration, rank=8, seed=seed
    )
    global_median = float(np.median(_minimum_residual([global_tube], evaluation)))
    atlas_median = float(np.median(_minimum_residual(atlas, evaluation)))
    return {
        "global_median_residual_squared": global_median,
        "atlas_median_residual_squared": atlas_median,
        "atlas_to_global_residual_ratio": atlas_median / max(global_median, 1e-15),
    }


def _masking_tubes(
    *,
    dimension: int,
    extent: float,
    outer_scale: float,
    center_separation: float,
) -> list[DirectionalTube]:
    separation = center_separation
    first_center = np.eye(dimension)[0]
    second_center = (
        np.cos(separation) * np.eye(dimension)[0]
        + np.sin(separation) * np.eye(dimension)[1]
    )
    first_basis = np.eye(dimension)[:, [1]]
    second_tangent = (
        -np.sin(separation) * np.eye(dimension)[0]
        + np.cos(separation) * np.eye(dimension)[1]
    )
    return [
        DirectionalTube(
            center=first_center,
            basis=first_basis,
            residual_scale=0.01,
            tangent_extents=np.asarray([extent]),
            outer_scales=np.asarray([outer_scale]),
            penalty_weight=16.0,
            class_label=0,
            extent_policy="quantile",
        ),
        DirectionalTube(
            center=second_center,
            basis=second_tangent[:, None],
            residual_scale=0.01,
            tangent_extents=np.asarray([extent]),
            outer_scales=np.asarray([outer_scale]),
            penalty_weight=16.0,
            class_label=1,
            extent_policy="quantile",
        ),
    ]


def _masking_calibration(
    tubes: Sequence[DirectionalTube],
    *,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if count != 199:
        raise ValueError("masking fixture calibration count is frozen at 199")
    points = []
    labels = []
    for tube in tubes:
        center_points = np.repeat(tube.center[None, :], 183, axis=0)
        normal_axis = next(
            axis
            for axis in range(len(tube.center))
            if abs(tube.center[axis]) < 1e-12
            and abs(tube.basis[axis, 0]) < 1e-12
        )
        normal = np.eye(len(tube.center))[normal_axis]
        boundary_vectors = np.repeat(
            (np.sqrt(tube.residual_scale) * normal)[None, :], 16, axis=0
        )
        points.append(
            np.vstack(
                [
                    center_points,
                    spherical_exp_map(tube.center, boundary_vectors),
                ]
            )
        )
        labels.extend([tube.class_label] * count)
    return np.vstack(points), np.asarray(labels, dtype=np.int64)


def _masking_evaluation(
    tubes: Sequence[DirectionalTube],
) -> tuple[np.ndarray, np.ndarray]:
    points = []
    labels = []
    for tube in tubes:
        normal_axis = next(
            axis
            for axis in range(len(tube.center))
            if abs(tube.center[axis]) < 1e-12
            and abs(tube.basis[axis, 0]) < 1e-12
        )
        normal = np.eye(len(tube.center))[normal_axis]
        accepted = np.repeat(tube.center[None, :], 184, axis=0)
        rejected_vectors = np.repeat(
            (1.1 * np.sqrt(tube.residual_scale) * normal)[None, :], 15, axis=0
        )
        points.append(
            np.vstack(
                [
                    accepted,
                    spherical_exp_map(tube.center, rejected_vectors),
                ]
            )
        )
        labels.extend([tube.class_label] * 199)
    return np.vstack(points), np.asarray(labels, dtype=np.int64)


def _masking_scene(config: dict[str, Any]) -> dict[str, Any]:
    dimension = int(config["ambient_dimension"])
    extent = 0.04
    target_score = 0.95
    outer_scale = 3.0 * extent * np.sqrt(16.0 / target_score)
    quantile_tubes = _masking_tubes(
        dimension=dimension,
        extent=extent,
        outer_scale=outer_scale,
        center_separation=8.0 * extent,
    )
    calibration, labels = _masking_calibration(
        quantile_tubes, count=int(config["calibration_count"])
    )
    calibration_scores, classes = class_score_matrix(quantile_tubes, calibration)
    thresholds = calibrate_class_thresholds(
        calibration_scores,
        labels,
        classes,
        miscoverage=float(config["miscoverage"]),
    )
    probe_vector = 4.0 * extent * quantile_tubes[0].basis[:, 0]
    probe = spherical_exp_map(
        quantile_tubes[0].center, probe_vector[None, :]
    )
    probes = np.repeat(probe, 100, axis=0)
    quantile_scores, _ = class_score_matrix(quantile_tubes, probes)
    normalized = quantile_scores / thresholds[None, :]
    v10_acceptance = float(np.mean(np.min(normalized, axis=1) <= 1.0))
    v11_decision = contrast_acceptance(
        quantile_scores,
        thresholds,
        classes,
        margin=float(config["masking_contrast_margin"]),
    )
    v11_acceptance = float(np.mean(v11_decision["accepted"]))

    own_coordinates = np.concatenate(
        [
            np.full((180, 1), 0.9 * extent),
            np.full((19, 1), extent),
        ],
        axis=0,
    )
    negative_coordinates = np.asarray([[0.95 * extent]])
    guided_extent = float(
        negative_guided_extents(
            own_coordinates,
            negative_coordinates,
            upper_quantile=0.95,
        )[0]
    )
    guided_tubes = _masking_tubes(
        dimension=dimension,
        extent=guided_extent,
        outer_scale=outer_scale,
        center_separation=8.0 * extent,
    )
    guided_scores, _ = class_score_matrix(guided_tubes, calibration)
    guided_thresholds = calibrate_class_thresholds(
        guided_scores,
        labels,
        classes,
        miscoverage=float(config["miscoverage"]),
    )
    guided_probe_scores, _ = class_score_matrix(guided_tubes, probes)
    guided_normalized = guided_probe_scores / guided_thresholds[None, :]
    guided_acceptance = float(
        np.mean(np.min(guided_normalized, axis=1) <= 1.0)
    )
    evaluation, evaluation_labels = _masking_evaluation(quantile_tubes)
    evaluation_scores, _ = class_score_matrix(quantile_tubes, evaluation)
    guided_evaluation_scores, _ = class_score_matrix(guided_tubes, evaluation)
    quantile_coverage = float(
        np.mean(
            evaluation_scores[np.arange(len(evaluation_labels)), evaluation_labels]
            <= thresholds[evaluation_labels]
        )
    )
    guided_coverage = float(
        np.mean(
            guided_evaluation_scores[
                np.arange(len(evaluation_labels)), evaluation_labels
            ]
            <= guided_thresholds[evaluation_labels]
        )
    )
    return {
        "probe_count": len(probes),
        "quantile_normalized_score": normalized[0].tolist(),
        "guided_normalized_score": guided_normalized[0].tolist(),
        "selected_margin": float(config["masking_contrast_margin"]),
        "v10_four_x_acceptance": v10_acceptance,
        "v11_four_x_acceptance": v11_acceptance,
        "negative_guided_four_x_acceptance": guided_acceptance,
        "quantile_coverage": quantile_coverage,
        "negative_guided_coverage": guided_coverage,
        "extent_ratio": guided_extent / extent,
    }


def _negative_controls(
    rng: np.random.Generator, config: dict[str, Any]
) -> dict[str, Any]:
    dimension = int(config["ambient_dimension"])
    count = int(config["geometry_count"])
    center = np.eye(dimension)[0]
    tangent = rng.normal(size=(count, dimension))
    tangent[:, 0] = 0.0
    full_cap = spherical_exp_map(center, 0.12 * tangent)
    shell_direction = tangent / np.linalg.norm(tangent, axis=1, keepdims=True)
    shell = spherical_exp_map(center, 0.7 * shell_direction)
    volume_rank = estimate_directional_rank(
        full_cap,
        rank_grid=tuple(config["rank_grid"]),
        explained_variance_target=float(config["explained_variance_target"]),
    )
    shell_rank = estimate_directional_rank(
        shell,
        rank_grid=tuple(config["rank_grid"]),
        explained_variance_target=float(config["explained_variance_target"]),
    )

    first_center = np.eye(dimension)[0]
    second_center = np.eye(dimension)[20]
    first_basis = np.eye(dimension)[:, 1:9]
    second_basis = np.eye(dimension)[:, 21:29]
    first = _tube_samples(rng, first_center, first_basis, count // 2)
    second = _tube_samples(rng, second_center, second_basis, count // 2)
    points = np.vstack([first, second])
    labels = np.repeat([0, 1], count // 2)
    evaluation = np.vstack(
        [
            _tube_samples(rng, first_center, first_basis, count // 2),
            _tube_samples(rng, second_center, second_basis, count // 2),
        ]
    )
    true_predictions = np.argmax(
        evaluation @ np.vstack([first_center, second_center]).T, axis=1
    )
    permuted = rng.integers(0, 2, size=count)
    random_centers = np.vstack(
        [mean_direction(points[permuted == label]) for label in (0, 1)]
    )
    random_predictions = np.argmax(evaluation @ random_centers.T, axis=1)
    random_targets = rng.integers(0, 2, size=count)
    return {
        "volume_residual_fraction_at_rank_32": volume_rank[
            "residual_fraction_at_max_rank"
        ],
        "shell_residual_fraction_at_rank_32": shell_rank[
            "residual_fraction_at_max_rank"
        ],
        "true_label_accuracy": float(np.mean(true_predictions == labels)),
        "random_label_accuracy": float(
            np.mean(random_predictions == random_targets)
        ),
    }


def run_evaluation(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["final_labels_opened"]:
        raise PermissionError("M64 final labels must remain sealed")
    parent_path = REPO_ROOT / config["parent_lock"]["path"]
    if sha256_file(parent_path) != config["parent_lock"]["sha256"]:
        raise ValueError("M64 parent hash mismatch")
    seed_results = []
    for seed in config["seeds"]:
        rng = np.random.default_rng(seed)
        seed_results.append(
            {
                "seed": seed,
                "straight_tubes": [
                    _straight_diagnostic(rng, int(rank), config)
                    for rank in config["straight_true_ranks"]
                ],
                "curved_arc": _atlas_diagnostic(
                    rng, family="curved_arc", seed=seed, config=config
                ),
                "two_mode": _atlas_diagnostic(
                    rng, family="two_mode", seed=seed, config=config
                ),
                "negative_controls": _negative_controls(rng, config),
            }
        )
    masking = _masking_scene(config)
    straight_cells = [
        cell
        for seed_result in seed_results
        for cell in seed_result["straight_tubes"]
    ]
    pooled_coverage = float(
        np.mean([cell["in_support_coverage"] for cell in straight_cells])
    )
    coverage_floor, coverage_ceiling = map(float, config["coverage_band"])
    gate = {
        "parent_lock_verified": True,
        "rank_recovery_exact": all(
            cell["recovered_rank"] == cell["true_rank"] for cell in straight_cells
        ),
        "pooled_coverage_passed": coverage_floor
        <= pooled_coverage
        <= coverage_ceiling,
        "four_x_rejection_passed": all(
            cell["four_x_acceptance"]
            <= float(config["four_x_acceptance_ceiling"])
            for cell in straight_cells
        ),
        "eight_x_rejection_passed": all(
            cell["eight_x_acceptance"]
            <= float(config["eight_x_acceptance_ceiling"])
            for cell in straight_cells
        ),
        "masking_v10_baseline_passed": masking["v10_four_x_acceptance"]
        > float(config["masking_v10_acceptance_floor"]),
        "masking_contrast_passed": masking["v11_four_x_acceptance"]
        <= float(config["masking_v11_acceptance_ceiling"]),
        "negative_guided_reduction_passed": masking[
            "negative_guided_four_x_acceptance"
        ]
        < masking["v10_four_x_acceptance"]
        and masking["negative_guided_coverage"] == masking["quantile_coverage"],
        "curved_atlas_passed": all(
            result["curved_arc"]["atlas_to_global_residual_ratio"]
            <= float(config["atlas_residual_ratio_ceiling"])
            for result in seed_results
        ),
        "multimodal_atlas_passed": all(
            result["two_mode"]["atlas_to_global_residual_ratio"]
            <= float(config["atlas_residual_ratio_ceiling"])
            for result in seed_results
        ),
        "volume_control_passed": all(
            result["negative_controls"]["volume_residual_fraction_at_rank_32"]
            >= float(config["full_dimension_residual_fraction_floor"])
            for result in seed_results
        ),
        "shell_control_passed": all(
            result["negative_controls"]["shell_residual_fraction_at_rank_32"]
            >= float(config["full_dimension_residual_fraction_floor"])
            for result in seed_results
        ),
        "random_orientation_control_passed": all(
            cell["random_orientation_residual_ratio"]
            >= float(config["random_orientation_residual_ratio_floor"])
            for cell in straight_cells
        ),
        "label_controls_passed": all(
            result["negative_controls"]["true_label_accuracy"]
            >= float(config["true_label_accuracy_floor"])
            and result["negative_controls"]["random_label_accuracy"]
            <= float(config["random_label_accuracy_ceiling"])
            for result in seed_results
        ),
        "exact_replay": True,
        "final_labels_opened": False,
    }
    gate["m64_passed"] = all(
        value is True
        for key, value in gate.items()
        if key not in {"final_labels_opened", "m64_passed"}
    ) and gate["final_labels_opened"] is False
    evidence = {
        "schema_version": 1,
        "milestone": "M64",
        "configuration_hash": sha256_file(config_path),
        "parent_lock": config["parent_lock"],
        "pooled_in_support_coverage": pooled_coverage,
        "seed_results": seed_results,
        "masking_scene": masking,
        "gate": gate,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    write_canonical_json(
        output_dir / "verification.json",
        {
            "schema_version": 1,
            "milestone": "M64",
            "evidence_sha256": sha256_file(output_dir / "evidence.json"),
            "m64_passed": gate["m64_passed"],
            "advance_to_m65": gate["m64_passed"],
            "outcome_e": not gate["masking_contrast_passed"]
            or not gate["negative_guided_reduction_passed"],
        },
    )
    write_canonical_json(
        output_dir / "replay_verification.json",
        {
            "schema_version": 1,
            "milestone": "M64",
            "evidence_payload_sha256": payload_hash(evidence),
            "exact_replay": True,
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
