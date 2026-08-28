from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from experiments.common.v11_directional_envelope import (
    PROBE_FAMILIES,
    DirectionalTube,
    calibrate_class_thresholds,
    class_score_matrix,
    contrast_acceptance,
    deterministic_directional_patch_assignments,
    generate_geodesic_probes,
    mean_direction,
    negative_guided_extents,
    normalize_directions,
    spherical_log_map,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.tier4.eval_v9_m51_surface_diagnostics import _partition_seed
from src.subspace_primitive import SubspacePrimitive, deterministic_basis_signs


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "configs"
    / "v11"
    / "m65_directional_envelope_screen.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "logs" / "results" / "v11" / "m65_directional_envelope_screen"
)


class CellCalibrationError(ValueError):
    def __init__(self, message: str, audit: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.audit = dict(audit)


def _resolve(path: str) -> Path:
    return REPO_ROOT / Path(path)


def _verify(specification: Mapping[str, str]) -> Path:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"M65 immutable artifact hash mismatch: {path}")
    return path


def _calibration_split(
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    known_classes: np.ndarray,
    seed: int,
    extent_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 65_000)
    extent: list[int] = []
    conformal: list[int] = []
    for class_label in known_classes:
        class_indices = indices[labels[indices] == class_label]
        shuffled = rng.permutation(class_indices)
        count = int(np.floor(len(shuffled) * extent_fraction))
        extent.extend(shuffled[:count].tolist())
        conformal.extend(shuffled[count:].tolist())
    return (
        np.asarray(sorted(extent), dtype=np.int64),
        np.asarray(sorted(conformal), dtype=np.int64),
    )


def _geometry_states(
    geometry_x: np.ndarray,
    geometry_y: np.ndarray,
    *,
    known_classes: np.ndarray,
    patch_count: int,
    seed: int,
    maximum_rank: int,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    states: list[dict[str, Any]] = []
    assignments_by_class: dict[int, np.ndarray] = {}
    for class_label in known_classes:
        points = normalize_directions(geometry_x[geometry_y == class_label])
        assignments = deterministic_directional_patch_assignments(
            points,
            patch_count=patch_count,
            seed=seed + int(class_label),
        )
        assignments_by_class[int(class_label)] = assignments
        for patch_id in range(patch_count):
            patch = points[assignments == patch_id]
            if len(patch) < maximum_rank + 2:
                raise ValueError("atlas patch is rank infeasible")
            center = mean_direction(patch)
            logarithms = spherical_log_map(center, patch)
            _, singular_values, right_vectors = np.linalg.svd(
                logarithms, full_matrices=False
            )
            if singular_values[maximum_rank - 1] <= np.finfo(np.float64).eps:
                raise ValueError("atlas patch is rank deficient")
            basis = deterministic_basis_signs(
                right_vectors[:maximum_rank].T
            )
            basis -= center[:, None] * (center @ basis)[None, :]
            basis, _ = np.linalg.qr(basis)
            states.append(
                {
                    "class_label": int(class_label),
                    "patch_id": patch_id,
                    "center": center,
                    "basis": deterministic_basis_signs(
                        basis[:, :maximum_rank]
                    ),
                    "geometry_count": int(len(patch)),
                }
            )
    return states, assignments_by_class


def _assign_to_states(
    points: np.ndarray,
    labels: np.ndarray,
    states: Sequence[dict[str, Any]],
) -> np.ndarray:
    directions = normalize_directions(points)
    result = np.empty(len(directions), dtype=np.int64)
    for class_label in sorted(set(int(value) for value in labels)):
        rows = np.flatnonzero(labels == class_label)
        candidates = [
            index
            for index, state in enumerate(states)
            if state["class_label"] == class_label
        ]
        centers = np.vstack([states[index]["center"] for index in candidates])
        nearest = np.argmax(directions[rows] @ centers.T, axis=1)
        result[rows] = np.asarray(candidates, dtype=np.int64)[nearest]
    return result


def _quantile_tube(
    state: dict[str, Any],
    extent_points: np.ndarray,
    *,
    rank: int,
    penalty_weight: float,
) -> DirectionalTube:
    center = state["center"]
    basis = state["basis"][:, :rank]
    logarithms = spherical_log_map(center, extent_points)
    coordinates = logarithms @ basis
    residual = logarithms - coordinates @ basis.T
    floor = max(
        float(np.finfo(np.float64).eps),
        float(np.var(logarithms)) * 1e-12,
    )
    residual_scale = max(
        float(
            np.quantile(
                np.sum(residual * residual, axis=1),
                0.95,
                method="higher",
            )
        ),
        floor,
    )
    absolute = np.abs(coordinates)
    extents = np.maximum(
        np.quantile(absolute, 0.95, axis=0, method="higher"),
        floor,
    )
    q90 = np.quantile(absolute, 0.90, axis=0, method="higher")
    return DirectionalTube(
        center=center,
        basis=basis,
        residual_scale=residual_scale,
        tangent_extents=extents,
        outer_scales=np.maximum(extents - q90, floor),
        penalty_weight=penalty_weight,
        class_label=state["class_label"],
        patch_id=state["patch_id"],
        extent_policy="quantile",
    )


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
    from experiments.common.v11_directional_envelope import spherical_exp_map

    return spherical_exp_map(tube.center, np.vstack(vectors))


def _policy_tube(
    quantile: DirectionalTube,
    own_extent_points: np.ndarray,
    negative_points: np.ndarray,
    *,
    policy: str,
) -> DirectionalTube:
    if policy == "quantile":
        return quantile
    own_coordinates = spherical_log_map(
        quantile.center, own_extent_points
    ) @ quantile.basis
    negative_coordinates = spherical_log_map(
        quantile.center, negative_points
    ) @ quantile.basis
    extents = negative_guided_extents(
        own_coordinates,
        negative_coordinates,
        upper_quantile=0.95,
    )
    floor = max(
        float(np.finfo(np.float64).eps),
        float(np.var(own_coordinates)) * 1e-12,
    )
    absolute = np.abs(own_coordinates)
    if policy == "negative_guided_iqr":
        scales = np.quantile(absolute, 0.75, axis=0) - np.quantile(
            absolute, 0.25, axis=0
        )
    else:
        q90 = np.quantile(absolute, 0.90, axis=0, method="higher")
        scales = extents - q90
    return DirectionalTube(
        center=quantile.center,
        basis=quantile.basis,
        residual_scale=quantile.residual_scale,
        tangent_extents=np.maximum(extents, floor),
        outer_scales=np.maximum(scales, floor),
        penalty_weight=quantile.penalty_weight,
        class_label=quantile.class_label,
        patch_id=quantile.patch_id,
        extent_policy=policy,
    )


def _fit_cell_tubes(
    states: Sequence[dict[str, Any]],
    extent_x: np.ndarray,
    extent_y: np.ndarray,
    *,
    rank: int,
    policy: str,
    penalty_weight: float,
) -> list[DirectionalTube]:
    assignments = _assign_to_states(extent_x, extent_y, states)
    quantile_tubes = []
    own_points: list[np.ndarray] = []
    for index, state in enumerate(states):
        points = extent_x[assignments == index]
        if not len(points):
            raise ValueError("atlas patch has no extent-calibration observations")
        own_points.append(points)
        quantile_tubes.append(
            _quantile_tube(
                state,
                points,
                rank=rank,
                penalty_weight=penalty_weight,
            )
        )
    if policy == "quantile":
        return quantile_tubes
    result = []
    for index, tube in enumerate(quantile_tubes):
        other_class = extent_x[extent_y != tube.class_label]
        four_x = _axis_probes(tube, 4.0)
        negative = np.vstack([other_class, four_x])
        try:
            result.append(
                _policy_tube(
                    tube,
                    own_points[index],
                    negative,
                    policy=policy,
                )
            )
        except ValueError as error:
            raise CellCalibrationError(
                str(error),
                {
                    "stage": "negative_guided_extent",
                    "class_label": tube.class_label,
                    "patch_id": tube.patch_id,
                    "own_extent_count": int(len(own_points[index])),
                    "other_class_negative_count": int(len(other_class)),
                    "four_x_probe_count": int(len(four_x)),
                    "quantile_extents": tube.tangent_extents.tolist(),
                },
            ) from error
    return result


def _probe_acceptance(
    tubes: Sequence[DirectionalTube],
    thresholds: np.ndarray,
    classes: np.ndarray,
    probes: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    margin: float,
) -> dict[str, Any]:
    class_columns = {int(value): index for index, value in enumerate(classes)}
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
        source_decisions = np.asarray(
            [
                tubes[int(owner)].score(points[row : row + 1])[0]
                <= thresholds[
                    class_columns[tubes[int(owner)].class_label]
                ]
                for row, owner in enumerate(owners)
            ],
            dtype=bool,
        )
        scores, observed_classes = class_score_matrix(tubes, points)
        if not np.array_equal(observed_classes, classes):
            raise ValueError("probe class order changed")
        system_decisions = contrast_acceptance(
            scores, thresholds, classes, margin=margin
        )["accepted"]
        source[name] = float(np.mean(source_decisions))
        system[name] = float(np.mean(system_decisions))
        by_multiplier[name] = {
            str(value).rstrip("0").rstrip("."): float(
                np.mean(system_decisions[multipliers == value])
            )
            for value in sorted(set(multipliers.tolist()))
        }
    return {
        "source_patch": source,
        "system": system,
        "system_by_multiplier": by_multiplier,
    }


def _safety_passes(
    acceptance: Mapping[str, Any],
    gate: Mapping[str, float],
) -> bool:
    multiplier = acceptance["system_by_multiplier"]
    system = acceptance["system"]
    return bool(
        multiplier["axis_tangent"].get("8", 0.0)
        <= gate["maximum_8x_tangent_acceptance"]
        and multiplier["axis_tangent"].get("4", 0.0)
        <= gate["maximum_4x_tangent_acceptance"]
        and multiplier["masking"].get("4", 0.0)
        <= gate["maximum_4x_masking_acceptance"]
        and max(
            system["bridge"],
            system["cross_class_bridge"],
            system["mixed"],
            system["random_direction"],
        )
        <= gate["maximum_other_probe_acceptance"]
    )


def _composite_metrics(
    decision: Mapping[str, np.ndarray],
    head_predictions: np.ndarray,
    labels: np.ndarray,
    *,
    known_classes: np.ndarray,
    unknown_classes: np.ndarray,
    novelty: np.ndarray,
) -> dict[str, Any]:
    accepted = np.asarray(decision["accepted"], dtype=bool)
    known = np.isin(labels, known_classes)
    unknown = np.isin(labels, unknown_classes)
    correct = head_predictions == labels
    class_composite = [
        float(np.mean((accepted & correct)[labels == class_label]))
        for class_label in known_classes
    ]
    accepted_known = accepted & known
    accepted_accuracy = (
        float(
            balanced_accuracy_score(
                labels[accepted_known],
                head_predictions[accepted_known],
            )
        )
        if len(np.unique(labels[accepted_known])) == len(known_classes)
        else 0.0
    )
    return {
        "known_coverage": float(np.mean(accepted[known])),
        "unknown_recall": float(np.mean(~accepted[unknown])),
        "composite_known_balanced_accuracy": float(
            np.mean(class_composite)
        ),
        "accepted_known_balanced_accuracy": accepted_accuracy,
        "accepted_class_count": int(
            sum(
                np.any(accepted & (labels == class_label))
                for class_label in known_classes
            )
        ),
        "auroc": float(roc_auc_score(unknown.astype(np.int64), novelty)),
        "fpr95": float(
            np.mean(
                novelty[known]
                >= np.quantile(novelty[unknown], 0.05, method="higher")
            )
        ),
    }


def _evaluate_cell(
    states: Sequence[dict[str, Any]],
    extent_x: np.ndarray,
    extent_y: np.ndarray,
    conformal_x: np.ndarray,
    conformal_y: np.ndarray,
    evaluation_x: np.ndarray,
    evaluation_y: np.ndarray,
    head_predictions: np.ndarray,
    *,
    rank: int,
    patch_count: int,
    policy: str,
    config: dict[str, Any],
    parent_parameter_count: int,
    geometry_count: int,
) -> dict[str, Any]:
    tubes = _fit_cell_tubes(
        states,
        extent_x,
        extent_y,
        rank=rank,
        policy=policy,
        penalty_weight=float(config["penalty_weight"]),
    )
    conformal_scores, classes = class_score_matrix(tubes, conformal_x)
    thresholds = calibrate_class_thresholds(
        conformal_scores,
        conformal_y,
        classes,
        miscoverage=float(config["miscoverage"]),
    )
    probes = generate_geodesic_probes(tubes, seed=int(config["seed"]))
    attempts = []
    selected_margin = None
    selected_acceptance = None
    for margin in tuple(float(value) for value in config["contrast_margin_grid"]):
        acceptance = _probe_acceptance(
            tubes,
            thresholds,
            classes,
            probes,
            margin=margin,
        )
        feasible = _safety_passes(acceptance, config["gate"])
        attempts.append(
            {
                "margin": margin,
                "acceptance": acceptance,
                "feasible": feasible,
            }
        )
        if feasible:
            selected_margin = margin
            selected_acceptance = acceptance
            break
    if selected_margin is None or selected_acceptance is None:
        raise CellCalibrationError(
            "no registered contrast margin passes calibration safety",
            {
                "stage": "contrast_selection",
                "thresholds": {
                    str(label): float(threshold)
                    for label, threshold in zip(
                        classes, thresholds, strict=True
                    )
                },
                "attempts": attempts,
            },
        )
    evaluation_scores, observed_classes = class_score_matrix(
        tubes, evaluation_x
    )
    if not np.array_equal(observed_classes, classes):
        raise ValueError("evaluation class order changed")
    decision = contrast_acceptance(
        evaluation_scores,
        thresholds,
        classes,
        margin=selected_margin,
    )
    normalized = evaluation_scores / thresholds[None, :]
    metrics = _composite_metrics(
        decision,
        head_predictions,
        evaluation_y,
        known_classes=np.asarray(config["known_classes"], dtype=np.int64),
        unknown_classes=np.asarray(
            config["proxy_unknown_classes"], dtype=np.int64
        ),
        novelty=np.min(normalized, axis=1),
    )
    parameter_count = sum(tube.parameter_count for tube in tubes)
    fit_work_units = int(geometry_count * evaluation_x.shape[1] * rank)
    parent_fit_work_units = int(parent_parameter_count * geometry_count)
    state_hash = payload_hash(
        {
            "tubes": [tube.to_dict() for tube in tubes],
            "thresholds": thresholds.tolist(),
            "margin": selected_margin,
        }
    )
    return {
        "rank": rank,
        "patch_count": patch_count,
        "extent_policy": policy,
        "calibration_feasible": True,
        "thresholds": {
            str(label): float(threshold)
            for label, threshold in zip(classes, thresholds, strict=True)
        },
        "conformal_counts": {
            str(label): int(np.sum(conformal_y == label))
            for label in classes
        },
        "selected_margin": selected_margin,
        "contrast_selection_attempts": attempts,
        "metrics": metrics,
        "probe_acceptance": selected_acceptance,
        "parameter_count": parameter_count,
        "parent_parameter_count": parent_parameter_count,
        "parameter_ratio": float(parameter_count / parent_parameter_count),
        "fit_work_units": fit_work_units,
        "parent_fit_work_units": parent_fit_work_units,
        "fit_work_ratio": float(fit_work_units / parent_fit_work_units),
        "state_hash": state_hash,
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
        raise PermissionError("M65 final labels must remain sealed")
    verified_locks = {
        name: {
            "path": specification["path"],
            "sha256": sha256_file(_verify(specification)),
        }
        for name, specification in sorted(config["parent_locks"].items())
    }
    v9_config = json.loads(
        _resolve(config["parent_locks"]["v9_m53_config"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    v9_evidence = json.loads(
        _resolve(config["parent_locks"]["v9_m53_evidence"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    m51_config = json.loads(_verify(v9_config["m51_config"]).read_text())
    source = json.loads(_verify(m51_config["source_config"]).read_text())
    parent = json.loads(_verify(v9_config["parent_student"]).read_text())
    seed = int(config["seed"])
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
        geometry_fraction=float(m51_config["geometry_fraction"]),
    )
    observed_hashes = {
        name: payload_hash(values.tolist())
        for name, values in partitions.items()
    }
    if observed_hashes != v9_evidence["partition_hashes"]:
        raise ValueError("M65 partitions differ from frozen v9 lineage")
    extent_indices, conformal_indices = _calibration_split(
        partitions["score_calibration"],
        train_labels,
        known_classes=known_classes,
        seed=seed,
        extent_fraction=float(config["calibration_extent_fraction"]),
    )
    extent_x = train_features[extent_indices]
    extent_y = train_labels[extent_indices]
    conformal_x = train_features[conformal_indices]
    conformal_y = train_labels[conformal_indices]
    minimum_count = int(config["minimum_conformal_class_count"])
    if any(
        np.sum(conformal_y == class_label) < minimum_count
        for class_label in known_classes
    ):
        raise ValueError("M65 conformal class count is below the frozen minimum")
    evaluation_indices = np.concatenate(
        [partitions["development_eval"], partitions["unknown_eval"]]
    )
    evaluation_x = dev_features[evaluation_indices]
    evaluation_y = dev_labels[evaluation_indices]
    m63 = json.loads(
        _resolve(config["parent_locks"]["m63_evidence"]["path"]).read_text()
    )
    head_metadata = m63["delegated_head"]
    head_predictions_all = np.load(
        _resolve(head_metadata["predictions_path"]),
        allow_pickle=False,
    )
    calibration_count = head_metadata["query_layout"][
        "score_calibration_count"
    ]
    dev_predictions = head_predictions_all[calibration_count:]
    head_predictions = dev_predictions[evaluation_indices]
    if sha256_file(_resolve(head_metadata["predictions_path"])) != head_metadata[
        "predictions_sha256"
    ]:
        raise ValueError("M65 delegated-head prediction hash mismatch")
    geometry_x = train_features[partitions["geometry_fit"]]
    geometry_y = train_labels[partitions["geometry_fit"]]
    parent_parameter_count = sum(
        SubspacePrimitive.from_dict(item["payload"]).parameter_count
        for item in parent["selected_candidates"]
    )
    state_cache = {}
    for patch_count in config["patch_counts"]:
        states, _ = _geometry_states(
            geometry_x,
            geometry_y,
            known_classes=known_classes,
            patch_count=int(patch_count),
            seed=seed,
            maximum_rank=max(int(value) for value in config["ranks"]),
        )
        state_cache[int(patch_count)] = states
    cells = []
    for rank in config["ranks"]:
        for patch_count in config["patch_counts"]:
            states = state_cache[int(patch_count)]
            for policy in config["extent_policies"]:
                try:
                    cell = _evaluate_cell(
                        states,
                        extent_x,
                        extent_y,
                        conformal_x,
                        conformal_y,
                        evaluation_x,
                        evaluation_y,
                        head_predictions,
                        rank=int(rank),
                        patch_count=int(patch_count),
                        policy=str(policy),
                        config=config,
                        parent_parameter_count=parent_parameter_count,
                        geometry_count=len(geometry_x),
                    )
                except CellCalibrationError as error:
                    cell = {
                        "rank": int(rank),
                        "patch_count": int(patch_count),
                        "extent_policy": str(policy),
                        "calibration_feasible": False,
                        "stop_reason": str(error),
                        "calibration_audit": error.audit,
                        "screen_passed": False,
                    }
                except ValueError as error:
                    cell = {
                        "rank": int(rank),
                        "patch_count": int(patch_count),
                        "extent_policy": str(policy),
                        "calibration_feasible": False,
                        "stop_reason": str(error),
                        "calibration_audit": {
                            "stage": "geometry_or_conformal_validation"
                        },
                        "screen_passed": False,
                    }
                cells.append(cell)
    gaussian = v9_evidence["controls"]["low_rank_gaussian"]
    a2 = v9_evidence["controls"]["signed_volume_a2"]
    head_only_accuracy = float(
        balanced_accuracy_score(
            evaluation_y[np.isin(evaluation_y, known_classes)],
            head_predictions[np.isin(evaluation_y, known_classes)],
        )
    )
    gate = config["gate"]
    for cell in cells:
        if not cell["calibration_feasible"]:
            continue
        metrics = cell["metrics"]
        acceptance = cell["probe_acceptance"]
        multiplier = acceptance["system_by_multiplier"]
        system = acceptance["system"]
        operands = {
            "unknown_recall_gain": float(
                metrics["unknown_recall"] - gaussian["unknown_recall"]
            ),
            "composite_accuracy_loss": float(
                a2["known_balanced_accuracy"]
                - metrics["composite_known_balanced_accuracy"]
            ),
            "accepted_known_accuracy_loss": float(
                head_only_accuracy
                - metrics["accepted_known_balanced_accuracy"]
            ),
            "eight_x_tangent_acceptance": multiplier[
                "axis_tangent"
            ].get("8", 0.0),
            "four_x_tangent_acceptance": multiplier[
                "axis_tangent"
            ].get("4", 0.0),
            "four_x_masking_acceptance": multiplier["masking"].get(
                "4", 0.0
            ),
            "other_probe_acceptance": max(
                system["bridge"],
                system["cross_class_bridge"],
                system["mixed"],
                system["random_direction"],
            ),
            "accepted_class_count": metrics["accepted_class_count"],
            "parameter_ratio": cell["parameter_ratio"],
            "fit_work_ratio": cell["fit_work_ratio"],
            "exact_replay": cell["exact_replay"],
            "atlas_stability_recorded": cell["patch_count"] == 1,
        }
        preliminary = bool(
            operands["unknown_recall_gain"]
            >= gate["minimum_unknown_recall_gain"]
            and operands["composite_accuracy_loss"]
            <= gate["maximum_composite_accuracy_loss"]
            and operands["accepted_known_accuracy_loss"]
            <= gate["maximum_accepted_known_accuracy_loss"]
            and operands["eight_x_tangent_acceptance"]
            <= gate["maximum_8x_tangent_acceptance"]
            and operands["four_x_tangent_acceptance"]
            <= gate["maximum_4x_tangent_acceptance"]
            and operands["four_x_masking_acceptance"]
            <= gate["maximum_4x_masking_acceptance"]
            and operands["other_probe_acceptance"]
            <= gate["maximum_other_probe_acceptance"]
            and operands["accepted_class_count"]
            >= gate["minimum_accepted_class_count"]
            and operands["parameter_ratio"] <= gate["maximum_parameter_ratio"]
            and operands["fit_work_ratio"] <= gate["maximum_fit_work_ratio"]
            and operands["exact_replay"]
        )
        if cell["patch_count"] > 1:
            cell["atlas_stability"] = {
                "assignment_adjusted_rand_index": None,
                "maximum_principal_angle_degrees": None,
                "maximum_extent_variation": None,
                "calibration_patch_change_fraction": None,
                "status": (
                    "not_run_cell_failed_other_operands"
                    if not preliminary
                    else "required_before_retention"
                ),
            }
            operands["atlas_stability_recorded"] = not preliminary
        cell["gate_operands"] = operands
        cell["screen_passed"] = preliminary and operands[
            "atlas_stability_recorded"
        ]
    eligible = [
        index for index, cell in enumerate(cells) if cell["screen_passed"]
    ]
    policy_order = {
        "negative_guided": 0,
        "negative_guided_iqr": 1,
        "quantile": 2,
    }
    retained_index = (
        min(
            eligible,
            key=lambda index: (
                cells[index]["patch_count"],
                cells[index]["rank"],
                policy_order[cells[index]["extent_policy"]],
                cells[index]["selected_margin"],
            ),
        )
        if eligible
        else None
    )
    evidence = {
        "schema_version": 1,
        "milestone": "M65",
        "configuration_hash": sha256_file(config_path),
        "verified_parent_locks": verified_locks,
        "partition_hashes": observed_hashes,
        "calibration_split": {
            "extent_count": int(len(extent_indices)),
            "conformal_count": int(len(conformal_indices)),
            "extent_hash": payload_hash(extent_indices.tolist()),
            "conformal_hash": payload_hash(conformal_indices.tolist()),
            "independent": not bool(
                np.intersect1d(extent_indices, conformal_indices).size
            ),
        },
        "delegated_head": {
            **head_metadata,
            "head_only_known_balanced_accuracy": head_only_accuracy,
        },
        "controls": v9_evidence["controls"],
        "resource_accounting": {
            "fit_work_unit_definition": "geometry_rows_x_ambient_dimension_x_rank",
            "a2_fit_work_budget_definition": "a2_parameter_count_x_geometry_rows",
        },
        "cells": cells,
        "eligible_cell_indices": eligible,
        "retained_cell_index": retained_index,
        "retained_cell": (
            cells[retained_index] if retained_index is not None else None
        ),
        "advance_to_m66": retained_index is not None,
        "final_labels_opened": False,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    write_canonical_json(
        output_dir / "verification.json",
        {
            "schema_version": 1,
            "milestone": "M65",
            "evidence_sha256": sha256_file(output_dir / "evidence.json"),
            "cell_count": len(cells),
            "calibration_feasible_cell_count": sum(
                cell["calibration_feasible"] for cell in cells
            ),
            "eligible_cell_indices": eligible,
            "retained_cell_index": retained_index,
            "advance_to_m66": evidence["advance_to_m66"],
        },
    )
    write_canonical_json(
        output_dir / "replay_verification.json",
        {
            "schema_version": 1,
            "milestone": "M65",
            "evidence_payload_sha256": payload_hash(evidence),
            "exact_replay": all(
                cell.get("exact_replay", True) for cell in cells
            ),
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
    print(
        json.dumps(
            {
                "cell_count": len(evidence["cells"]),
                "calibration_feasible_cell_count": sum(
                    cell["calibration_feasible"]
                    for cell in evidence["cells"]
                ),
                "eligible_cell_indices": evidence[
                    "eligible_cell_indices"
                ],
                "retained_cell_index": evidence["retained_cell_index"],
                "advance_to_m66": evidence["advance_to_m66"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
