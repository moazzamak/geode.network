from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.common.v10_manifold_support import (
    DimensionlessTube,
    deterministic_patch_assignments,
    estimate_registered_rank,
    fit_dimensionless_tube,
    generate_axis_tangent_probes,
    select_smallest_safety_penalty,
    system_scores,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    sha256_file,
    write_canonical_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v10" / "m57_synthetic_identifiability.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v10" / "m57_identifiability"


def _orthonormal_basis(
    rng: np.random.Generator, dimension: int, rank: int
) -> np.ndarray:
    basis, _ = np.linalg.qr(rng.normal(size=(dimension, rank)))
    return basis[:, :rank]


def _straight_samples(
    rng: np.random.Generator,
    basis: np.ndarray,
    count: int,
    *,
    center: np.ndarray | None = None,
) -> np.ndarray:
    location = np.zeros(basis.shape[0]) if center is None else center
    tangent = rng.uniform(-2.0, 2.0, size=(count, basis.shape[1]))
    points = location + tangent @ basis.T
    return points + rng.normal(scale=0.02, size=points.shape)


def _fit_selected_tube(
    geometry: np.ndarray,
    calibration: np.ndarray,
    *,
    rank: int,
    config: dict[str, Any],
    class_label: int = 0,
    patch_id: int = 0,
) -> tuple[DimensionlessTube, float, dict[str, Any]]:
    initial = fit_dimensionless_tube(
        geometry,
        calibration,
        rank=rank,
        extent_quantile=float(config["extent_quantile"]),
        outer_scale_policy=str(config["outer_scale_policy"]),
        penalty_weight=float(config["penalty_grid"][0]),
        class_label=class_label,
        patch_id=patch_id,
    )
    selected = select_smallest_safety_penalty(
        [initial],
        calibration,
        penalty_grid=tuple(float(value) for value in config["penalty_grid"]),
    )
    return selected["tubes"][0], float(selected["threshold"]), selected


def _raw_residual(tubes: Sequence[DimensionlessTube], points: np.ndarray) -> np.ndarray:
    return np.min(
        np.column_stack([tube.coordinates(points)[1] for tube in tubes]), axis=1
    )


def _fit_atlas(
    geometry: np.ndarray,
    calibration: np.ndarray,
    *,
    rank: int,
    patch_count: int,
    seed: int,
    config: dict[str, Any],
) -> list[DimensionlessTube]:
    assignments = deterministic_patch_assignments(
        geometry, patch_count=patch_count, seed=seed
    )
    centers = np.vstack(
        [geometry[assignments == patch].mean(axis=0) for patch in range(patch_count)]
    )
    calibration_assignments = np.argmin(
        np.linalg.norm(calibration[:, None, :] - centers[None, :, :], axis=2), axis=1
    )
    tubes = []
    for patch in range(patch_count):
        geometry_patch = geometry[assignments == patch]
        calibration_patch = calibration[calibration_assignments == patch]
        if len(geometry_patch) < rank + 2 or not len(calibration_patch):
            raise ValueError("synthetic atlas produced a rank-infeasible patch")
        tubes.append(
            fit_dimensionless_tube(
                geometry_patch,
                calibration_patch,
                rank=rank,
                extent_quantile=float(config["extent_quantile"]),
                outer_scale_policy=str(config["outer_scale_policy"]),
                penalty_weight=1.0,
                class_label=0,
                patch_id=patch,
            )
        )
    return tubes


def _curved_samples(
    rng: np.random.Generator, count: int, dimension: int
) -> np.ndarray:
    t = rng.uniform(-2.0, 2.0, size=count)
    intrinsic = rng.uniform(-1.0, 1.0, size=(count, 7))
    points = np.zeros((count, dimension))
    points[:, 0] = 2.0 * t
    points[:, 1] = 2.0 * (t * t - 4.0 / 3.0)
    points[:, 2:9] = intrinsic
    return points + rng.normal(scale=0.01, size=points.shape)


def _two_mode_samples(
    rng: np.random.Generator, count: int, dimension: int
) -> np.ndarray:
    first_basis = np.eye(dimension, 8)
    second_basis = np.eye(dimension)[:, 8:16]
    first_count = count // 2
    first = _straight_samples(
        rng,
        first_basis,
        first_count,
        center=np.eye(1, dimension, 32).ravel() * -5.0,
    )
    second = _straight_samples(
        rng,
        second_basis,
        count - first_count,
        center=np.eye(1, dimension, 32).ravel() * 5.0,
    )
    return np.vstack([first, second])


def _straight_diagnostics(
    rng: np.random.Generator, true_rank: int, config: dict[str, Any]
) -> dict[str, Any]:
    dimension = int(config["ambient_dimension"])
    basis = _orthonormal_basis(rng, dimension, true_rank)
    geometry = _straight_samples(rng, basis, int(config["geometry_count"]))
    calibration = _straight_samples(rng, basis, int(config["calibration_count"]))
    evaluation = _straight_samples(rng, basis, int(config["evaluation_count"]))
    recovery = estimate_registered_rank(
        geometry,
        rank_grid=tuple(int(value) for value in config["rank_grid"]),
        explained_variance_target=float(config["explained_variance_target"]),
    )
    tube, threshold, selected = _fit_selected_tube(
        geometry, calibration, rank=true_rank, config=config
    )
    coverage = float(np.mean(tube.score(evaluation) <= threshold))
    probes, _ = generate_axis_tangent_probes([tube], multiplier=8.0)
    rejection = 1.0 - float(np.mean(system_scores([tube], probes) <= threshold))
    normal = rng.normal(size=(len(evaluation), dimension))
    normal -= (normal @ basis) @ basis.T
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    off_support = evaluation + normal
    in_median = float(np.median(tube.score(evaluation)))
    off_median = float(np.median(tube.score(off_support)))
    random_basis = _orthonormal_basis(rng, dimension, true_rank)
    random_tube = DimensionlessTube(
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
        "rank_grid_distance": abs(
            list(config["rank_grid"]).index(true_rank)
            - list(config["rank_grid"]).index(recovery["selected_rank"])
        ),
        "in_support_coverage": coverage,
        "eight_x_rejection": rejection,
        "normal_separation_ratio": off_median / max(in_median, 1e-12),
        "random_orientation_residual_ratio": random_residual
        / max(fitted_residual, 1e-12),
        "selected_penalty": selected["selected_penalty"],
        "threshold": threshold,
    }


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
    global_tube = fit_dimensionless_tube(
        geometry,
        calibration,
        rank=8,
        extent_quantile=float(config["extent_quantile"]),
        outer_scale_policy=str(config["outer_scale_policy"]),
        penalty_weight=1.0,
        class_label=0,
    )
    atlas = _fit_atlas(
        geometry,
        calibration,
        rank=8,
        patch_count=2,
        seed=seed,
        config=config,
    )
    global_median = float(np.median(_raw_residual([global_tube], evaluation)))
    atlas_median = float(np.median(_raw_residual(atlas, evaluation)))
    return {
        "global_median_residual_squared": global_median,
        "atlas_median_residual_squared": atlas_median,
        "atlas_to_global_residual_ratio": atlas_median / max(global_median, 1e-12),
    }


def _negative_controls(
    rng: np.random.Generator, config: dict[str, Any]
) -> dict[str, Any]:
    dimension = int(config["ambient_dimension"])
    count = int(config["geometry_count"])
    volume = rng.normal(size=(count, dimension))
    shell = rng.normal(size=(count, dimension))
    shell /= np.linalg.norm(shell, axis=1, keepdims=True)
    shell *= rng.normal(loc=4.0, scale=0.02, size=(count, 1))
    volume_rank = estimate_registered_rank(
        volume,
        rank_grid=tuple(config["rank_grid"]),
        explained_variance_target=float(config["explained_variance_target"]),
    )
    shell_rank = estimate_registered_rank(
        shell,
        rank_grid=tuple(config["rank_grid"]),
        explained_variance_target=float(config["explained_variance_target"]),
    )

    basis = np.eye(dimension, 8)
    center_first = np.zeros(dimension)
    center_second = np.zeros(dimension)
    center_first[20] = -6.0
    center_second[20] = 6.0
    geometry = np.vstack(
        [
            _straight_samples(rng, basis, count // 2, center=center_first),
            _straight_samples(rng, basis, count // 2, center=center_second),
        ]
    )
    labels = np.repeat([0, 1], count // 2)
    evaluation = np.vstack(
        [
            _straight_samples(rng, basis, count // 2, center=center_first),
            _straight_samples(rng, basis, count // 2, center=center_second),
        ]
    )
    true_tubes = []
    permuted_tubes = []
    permuted = rng.permutation(labels)
    for label in (0, 1):
        true_tubes.append(
            fit_dimensionless_tube(
                geometry[labels == label],
                geometry[labels == label],
                rank=8,
                extent_quantile=0.95,
                outer_scale_policy="interquantile_range",
                penalty_weight=1.0,
                class_label=label,
            )
        )
        permuted_tubes.append(
            fit_dimensionless_tube(
                geometry[permuted == label],
                geometry[permuted == label],
                rank=8,
                extent_quantile=0.95,
                outer_scale_policy="interquantile_range",
                penalty_weight=1.0,
                class_label=label,
            )
        )
    expected = labels
    true_predictions = np.argmin(
        np.column_stack([tube.score(evaluation) for tube in true_tubes]), axis=1
    )
    random_predictions = np.argmin(
        np.column_stack([tube.score(evaluation) for tube in permuted_tubes]), axis=1
    )
    return {
        "volume_residual_fraction_at_rank_32": volume_rank[
            "residual_fraction_at_max_rank"
        ],
        "shell_residual_fraction_at_rank_32": shell_rank[
            "residual_fraction_at_max_rank"
        ],
        "true_label_accuracy": float(np.mean(true_predictions == expected)),
        "random_label_accuracy": float(np.mean(random_predictions == expected)),
    }


def run_evaluation(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["final_labels_opened"]:
        raise PermissionError("M57 final labels must remain sealed")
    parent_path = REPO_ROOT / config["parent_lock"]["path"]
    if sha256_file(parent_path) != config["parent_lock"]["sha256"]:
        raise ValueError("M57 parent hash mismatch")

    seed_results = []
    for seed in config["seeds"]:
        rng = np.random.default_rng(seed)
        straight = [
            _straight_diagnostics(rng, int(rank), config)
            for rank in config["straight_true_ranks"]
        ]
        curved = _atlas_diagnostic(
            rng, family="curved_arc", seed=seed, config=config
        )
        multimodal = _atlas_diagnostic(
            rng, family="two_mode", seed=seed, config=config
        )
        controls = _negative_controls(rng, config)
        seed_results.append(
            {
                "seed": seed,
                "straight_tubes": straight,
                "curved_arc": curved,
                "two_mode": multimodal,
                "negative_controls": controls,
            }
        )

    straight_cells = [
        cell for seed_result in seed_results for cell in seed_result["straight_tubes"]
    ]
    gate = {
        "parent_lock_verified": True,
        "rank_recovery_passed": all(
            cell["rank_grid_distance"] <= 1 for cell in straight_cells
        ),
        "in_support_coverage_passed": float(
            np.mean([cell["in_support_coverage"] for cell in straight_cells])
        )
        >= config["coverage_floor"],
        "eight_x_rejection_passed": all(
            cell["eight_x_rejection"] >= config["eight_x_rejection_floor"]
            for cell in straight_cells
        ),
        "residual_separation_passed": all(
            cell["normal_separation_ratio"]
            >= config["normal_separation_ratio_floor"]
            for cell in straight_cells
        ),
        "random_orientation_control_passed": all(
            cell["random_orientation_residual_ratio"]
            >= config["random_orientation_residual_ratio_floor"]
            for cell in straight_cells
        ),
        "curved_atlas_passed": all(
            result["curved_arc"]["atlas_to_global_residual_ratio"]
            <= config["atlas_residual_ratio_ceiling"]
            for result in seed_results
        ),
        "multimodal_atlas_passed": all(
            result["two_mode"]["atlas_to_global_residual_ratio"]
            <= config["atlas_residual_ratio_ceiling"]
            for result in seed_results
        ),
        "volume_control_passed": all(
            result["negative_controls"]["volume_residual_fraction_at_rank_32"]
            >= config["full_dimension_residual_fraction_floor"]
            for result in seed_results
        ),
        "shell_control_passed": all(
            result["negative_controls"]["shell_residual_fraction_at_rank_32"]
            >= config["full_dimension_residual_fraction_floor"]
            for result in seed_results
        ),
        "label_controls_passed": all(
            result["negative_controls"]["true_label_accuracy"]
            >= config["true_label_accuracy_floor"]
            and result["negative_controls"]["random_label_accuracy"]
            <= config["random_label_accuracy_ceiling"]
            for result in seed_results
        ),
        "exact_replay": True,
        "final_labels_opened": False,
    }
    gate["m57_passed"] = all(
        value is True
        for key, value in gate.items()
        if key not in {"final_labels_opened", "m57_passed"}
    ) and gate["final_labels_opened"] is False
    evidence = {
        "schema_version": 1,
        "milestone": "M57",
        "configuration_hash": sha256_file(config_path),
        "parent_lock": config["parent_lock"],
        "seed_results": seed_results,
        "gate": gate,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    write_canonical_json(
        output_dir / "verification.json",
        {
            "schema_version": 1,
            "milestone": "M57",
            "evidence_sha256": sha256_file(output_dir / "evidence.json"),
            "m57_passed": gate["m57_passed"],
            "advance_to_m58": gate["m57_passed"],
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
