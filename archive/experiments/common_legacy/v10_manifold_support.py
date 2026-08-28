from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.cluster import KMeans

from experiments.common.v9_surface_support import replay_digest
from src.subspace_primitive import deterministic_basis_signs


PROBE_FAMILIES = (
    "axis_tangent",
    "corner_tangent",
    "normal",
    "mixed",
    "bridge",
    "cross_class_bridge",
    "random_direction",
)
TANGENT_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0, 8.0)


class SafetyPenaltySelectionError(ValueError):
    def __init__(self, attempts: Sequence[dict[str, Any]]) -> None:
        super().__init__("no registered penalty satisfies calibration safety")
        self.attempts = tuple(dict(attempt) for attempt in attempts)


@dataclass(frozen=True)
class DimensionlessTube:
    center: np.ndarray
    basis: np.ndarray
    residual_scale: float
    tangent_extents: np.ndarray
    outer_scales: np.ndarray
    penalty_weight: float
    class_label: int
    patch_id: int = 0

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        basis = deterministic_basis_signs(np.asarray(self.basis, dtype=np.float64))
        extents = np.asarray(self.tangent_extents, dtype=np.float64)
        scales = np.asarray(self.outer_scales, dtype=np.float64)
        if center.ndim != 1 or not np.all(np.isfinite(center)):
            raise ValueError("center must be a finite vector")
        if (
            basis.ndim != 2
            or basis.shape[0] != len(center)
            or basis.shape[1] < 1
            or not np.all(np.isfinite(basis))
        ):
            raise ValueError("basis must have finite shape (dimension, positive rank)")
        if not np.allclose(
            basis.T @ basis, np.eye(basis.shape[1]), rtol=0.0, atol=1e-8
        ):
            raise ValueError("basis columns must be orthonormal")
        if extents.shape != (basis.shape[1],) or np.any(
            ~np.isfinite(extents) | (extents <= 0.0)
        ):
            raise ValueError("tangent_extents must be finite, positive, and match rank")
        if scales.shape != extents.shape or np.any(
            ~np.isfinite(scales) | (scales <= 0.0)
        ):
            raise ValueError("outer_scales must be finite, positive, and match rank")
        if not np.isfinite(self.residual_scale) or self.residual_scale <= 0.0:
            raise ValueError("residual_scale must be finite and positive")
        if not np.isfinite(self.penalty_weight) or self.penalty_weight <= 0.0:
            raise ValueError("penalty_weight must be finite and positive")
        if (
            isinstance(self.class_label, bool)
            or self.class_label < 0
            or isinstance(self.patch_id, bool)
            or self.patch_id < 0
        ):
            raise ValueError("class_label and patch_id must be nonnegative")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "tangent_extents", extents)
        object.__setattr__(self, "outer_scales", scales)

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    @property
    def parameter_count(self) -> int:
        return int(
            self.center.size
            + self.basis.size
            + self.tangent_extents.size
            + self.outer_scales.size
            + 2
        )

    def coordinates(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.center)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("points must be a finite matrix of the ambient dimension")
        deltas = values - self.center
        tangent = deltas @ self.basis
        residual = deltas - tangent @ self.basis.T
        return tangent, np.sum(residual * residual, axis=1)

    def score_terms(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tangent, residual_squared = self.coordinates(points)
        residual_term = residual_squared / self.residual_scale
        overshoot = np.maximum(
            (np.abs(tangent) - self.tangent_extents[None, :])
            / self.outer_scales[None, :],
            0.0,
        )
        tangent_term = np.mean(overshoot * overshoot, axis=1)
        return residual_term, tangent_term

    def score(self, points: np.ndarray) -> np.ndarray:
        residual, tangent = self.score_terms(points)
        return residual + self.penalty_weight * tangent

    def with_penalty(self, penalty_weight: float) -> "DimensionlessTube":
        return DimensionlessTube(
            center=self.center,
            basis=self.basis,
            residual_scale=self.residual_scale,
            tangent_extents=self.tangent_extents,
            outer_scales=self.outer_scales,
            penalty_weight=penalty_weight,
            class_label=self.class_label,
            patch_id=self.patch_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "center": self.center.tolist(),
            "basis": self.basis.tolist(),
            "residual_scale": float(self.residual_scale),
            "tangent_extents": self.tangent_extents.tolist(),
            "outer_scales": self.outer_scales.tolist(),
            "penalty_weight": float(self.penalty_weight),
            "class_label": self.class_label,
            "patch_id": self.patch_id,
        }


def fit_dimensionless_tube(
    geometry_points: np.ndarray,
    calibration_points: np.ndarray,
    *,
    rank: int,
    extent_quantile: float,
    outer_scale_policy: str,
    penalty_weight: float,
    class_label: int,
    patch_id: int = 0,
) -> DimensionlessTube:
    geometry = np.asarray(geometry_points, dtype=np.float64)
    calibration = np.asarray(calibration_points, dtype=np.float64)
    if (
        geometry.ndim != 2
        or not np.all(np.isfinite(geometry))
        or len(geometry) < rank + 2
        or rank >= geometry.shape[1]
    ):
        raise ValueError("geometry_points do not satisfy the rank contract")
    if (
        calibration.ndim != 2
        or calibration.shape[1] != geometry.shape[1]
        or not len(calibration)
        or not np.all(np.isfinite(calibration))
    ):
        raise ValueError("calibration_points have the wrong shape")
    if extent_quantile not in {0.90, 0.95, 0.99}:
        raise ValueError("extent_quantile is outside the registered grid")
    if outer_scale_policy not in {"median_overshoot", "interquantile_range"}:
        raise ValueError("unsupported outer_scale_policy")
    center = geometry.mean(axis=0)
    centered = geometry - center
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    basis = deterministic_basis_signs(right_vectors[:rank].T)
    deltas = calibration - center
    tangent = np.abs(deltas @ basis)
    residual = deltas - (deltas @ basis) @ basis.T
    residual_squared = np.sum(residual * residual, axis=1)
    floor = max(float(np.finfo(np.float64).eps), float(np.var(geometry)) * 1e-12)
    residual_scale = max(
        float(np.quantile(residual_squared, 0.95, method="higher")), floor
    )
    extents = np.maximum(
        np.quantile(tangent, extent_quantile, axis=0, method="higher"), floor
    )
    if outer_scale_policy == "interquantile_range":
        outer_scales = np.quantile(tangent, 0.75, axis=0) - np.quantile(
            tangent, 0.25, axis=0
        )
    else:
        overshoot = np.maximum(tangent - extents[None, :], 0.0)
        outer_scales = np.median(overshoot, axis=0)
    outer_scales = np.maximum(outer_scales, floor)
    return DimensionlessTube(
        center=center,
        basis=basis,
        residual_scale=residual_scale,
        tangent_extents=extents,
        outer_scales=outer_scales,
        penalty_weight=penalty_weight,
        class_label=class_label,
        patch_id=patch_id,
    )


def system_scores(tubes: Sequence[DimensionlessTube], points: np.ndarray) -> np.ndarray:
    if not tubes:
        raise ValueError("at least one tube is required")
    return np.min(np.column_stack([tube.score(points) for tube in tubes]), axis=1)


def deterministic_patch_assignments(
    points: np.ndarray,
    *,
    patch_count: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if (
        values.ndim != 2
        or not np.all(np.isfinite(values))
        or patch_count not in {1, 2, 4}
        or len(values) < patch_count
    ):
        raise ValueError("points or patch_count violate the v10 atlas contract")
    if patch_count == 1:
        return np.zeros(len(values), dtype=np.int64)
    return KMeans(
        n_clusters=patch_count,
        random_state=seed,
        n_init=10,
        algorithm="lloyd",
    ).fit_predict(values).astype(np.int64)


def estimate_registered_rank(
    points: np.ndarray,
    *,
    rank_grid: Sequence[int],
    explained_variance_target: float,
) -> dict[str, Any]:
    values = np.asarray(points, dtype=np.float64)
    ranks = tuple(int(rank) for rank in rank_grid)
    if (
        values.ndim != 2
        or len(values) < 2
        or not np.all(np.isfinite(values))
        or not ranks
        or tuple(sorted(set(ranks))) != ranks
        or ranks[0] < 1
        or ranks[-1] >= min(values.shape)
        or not 0.0 < explained_variance_target < 1.0
    ):
        raise ValueError("points or rank-recovery settings violate the contract")
    singular_values = np.linalg.svd(
        values - values.mean(axis=0), full_matrices=False, compute_uv=False
    )
    variances = singular_values * singular_values
    total = float(np.sum(variances))
    if total <= 0.0:
        raise ValueError("rank recovery requires nonconstant points")
    explained = {
        rank: float(np.sum(variances[:rank]) / total) for rank in ranks
    }
    selected = next(
        (rank for rank in ranks if explained[rank] >= explained_variance_target),
        ranks[-1],
    )
    return {
        "selected_rank": selected,
        "explained_variance_by_rank": explained,
        "residual_fraction_at_max_rank": 1.0 - explained[ranks[-1]],
    }


def generate_axis_tangent_probes(
    tubes: Sequence[DimensionlessTube],
    *,
    multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not tubes or multiplier <= 0.0 or not np.isfinite(multiplier):
        raise ValueError("tubes and a positive finite multiplier are required")
    probes = []
    owners = []
    for owner, tube in enumerate(tubes):
        for axis in range(tube.rank):
            for sign in (-1.0, 1.0):
                probes.append(
                    tube.center
                    + sign
                    * multiplier
                    * tube.tangent_extents[axis]
                    * tube.basis[:, axis]
                )
                owners.append(owner)
    return np.vstack(probes), np.asarray(owners, dtype=np.int64)


def generate_safety_probes(
    tubes: Sequence[DimensionlessTube],
    *,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not tubes:
        raise ValueError("at least one tube is required")
    rng = np.random.default_rng(seed)
    families: dict[str, list[np.ndarray]] = {name: [] for name in PROBE_FAMILIES}
    owners: dict[str, list[int]] = {name: [] for name in PROBE_FAMILIES}

    def add(name: str, owner: int, point: np.ndarray) -> None:
        families[name].append(np.asarray(point, dtype=np.float64))
        owners[name].append(owner)

    for owner, tube in enumerate(tubes):
        for multiplier in TANGENT_MULTIPLIERS:
            points, point_owners = generate_axis_tangent_probes(
                [tube], multiplier=multiplier
            )
            for point, _ in zip(points, point_owners, strict=True):
                add("axis_tangent", owner, point)
        corner = tube.center + (
            4.0 * tube.tangent_extents[None, :] * tube.basis
        ).sum(axis=1)
        add("corner_tangent", owner, corner)
        normal = rng.normal(size=len(tube.center))
        normal -= tube.basis @ (tube.basis.T @ normal)
        normal /= max(np.linalg.norm(normal), np.finfo(np.float64).eps)
        normal_scale = np.sqrt(tube.residual_scale)
        add("normal", owner, tube.center + 4.0 * normal_scale * normal)
        add(
            "mixed",
            owner,
            tube.center
            + 4.0 * tube.tangent_extents[0] * tube.basis[:, 0]
            + normal_scale * normal,
        )
        random_direction = rng.normal(size=len(tube.center))
        random_direction /= max(
            np.linalg.norm(random_direction), np.finfo(np.float64).eps
        )
        add(
            "random_direction",
            owner,
            tube.center + 8.0 * np.linalg.norm(tube.tangent_extents) * random_direction,
        )
    for first in range(len(tubes)):
        same = [
            index
            for index, tube in enumerate(tubes)
            if index != first and tube.class_label == tubes[first].class_label
        ]
        other = [
            index
            for index, tube in enumerate(tubes)
            if tube.class_label != tubes[first].class_label
        ]
        if same:
            second = min(
                same,
                key=lambda index: float(
                    np.linalg.norm(tubes[index].center - tubes[first].center)
                ),
            )
            add("bridge", first, 0.5 * (tubes[first].center + tubes[second].center))
        if other:
            second = min(
                other,
                key=lambda index: float(
                    np.linalg.norm(tubes[index].center - tubes[first].center)
                ),
            )
            add(
                "cross_class_bridge",
                first,
                0.5 * (tubes[first].center + tubes[second].center),
            )
    dimension = len(tubes[0].center)
    result = {}
    for name in PROBE_FAMILIES:
        points = (
            np.vstack(families[name])
            if families[name]
            else np.empty((0, dimension), dtype=np.float64)
        )
        result[name] = (points, np.asarray(owners[name], dtype=np.int64))
    return result


def probe_acceptance(
    tubes: Sequence[DimensionlessTube],
    probes: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    threshold: float,
) -> dict[str, dict[str, float]]:
    source: dict[str, float] = {}
    system: dict[str, float] = {}
    for name in PROBE_FAMILIES:
        points, owners = probes[name]
        if not len(points):
            source[name] = 0.0
            system[name] = 0.0
            continue
        source_scores = np.asarray(
            [
                tubes[int(owner)].score(points[index : index + 1])[0]
                for index, owner in enumerate(owners)
            ]
        )
        source[name] = float(np.mean(source_scores <= threshold))
        system[name] = float(np.mean(system_scores(tubes, points) <= threshold))
    return {"source_patch": source, "system": system}


def calibration_lineage_hash(
    tubes: Sequence[DimensionlessTube],
    *,
    penalty_grid: Sequence[float],
    threshold: float,
) -> str:
    return replay_digest(
        {
            "tubes": [tube.to_dict() for tube in tubes],
            "penalty_grid": list(penalty_grid),
            "threshold": threshold,
        }
    )


def select_smallest_safety_penalty(
    tubes: Sequence[DimensionlessTube],
    calibration_points: np.ndarray,
    *,
    penalty_grid: Sequence[float],
    known_coverage_target: float = 0.92,
) -> dict[str, Any]:
    if known_coverage_target != 0.92:
        raise ValueError("known_coverage_target is frozen at 0.92")
    penalties = tuple(float(value) for value in penalty_grid)
    if (
        not penalties
        or tuple(sorted(set(penalties))) != penalties
        or any(not np.isfinite(value) or value <= 0.0 for value in penalties)
    ):
        raise ValueError("penalty_grid must be finite, positive, and increasing")
    calibration = np.asarray(calibration_points, dtype=np.float64)
    attempts = []
    for penalty in penalties:
        candidates = [tube.with_penalty(penalty) for tube in tubes]
        calibration_scores = system_scores(candidates, calibration)
        threshold = float(
            np.quantile(
                calibration_scores, known_coverage_target, method="higher"
            )
        )
        coverage = float(np.mean(calibration_scores <= threshold))
        acceptance = {}
        for multiplier in TANGENT_MULTIPLIERS:
            probes, _ = generate_axis_tangent_probes(
                candidates, multiplier=multiplier
            )
            acceptance[str(multiplier).rstrip("0").rstrip(".")] = float(
                np.mean(system_scores(candidates, probes) <= threshold)
            )
        feasible = bool(
            coverage >= 0.90
            and acceptance["8"] == 0.0
            and acceptance["4"] <= 0.01
            and acceptance["0.5"] >= acceptance["1"]
        )
        attempts.append(
            {
                "penalty": penalty,
                "threshold": threshold,
                "coverage": coverage,
                "tangent_acceptance": acceptance,
                "feasible": feasible,
            }
        )
        if feasible:
            return {
                "selected_penalty": penalty,
                "threshold": threshold,
                "coverage": coverage,
                "tangent_acceptance": acceptance,
                "attempts": attempts,
                "tubes": candidates,
                "lineage_hash": calibration_lineage_hash(
                    candidates,
                    penalty_grid=penalties,
                    threshold=threshold,
                ),
            }
    raise SafetyPenaltySelectionError(attempts)
