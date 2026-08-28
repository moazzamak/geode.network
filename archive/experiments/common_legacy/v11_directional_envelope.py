from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.frozen import FrozenEstimator
from sklearn.svm import SVC

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
    "masking",
)
TANGENT_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0, 8.0)
MASKING_MULTIPLIERS = (2.0, 4.0, 8.0)
EXTENT_POLICIES = ("quantile", "negative_guided", "negative_guided_iqr")
CONTRAST_MARGINS = (0.0, 0.05, 0.1, 0.2)


def verify_delegated_head_lineage(
    metadata: Mapping[str, Any],
    *,
    representation_hash: str,
    training_split_hash: str,
    predictions_hash: str,
) -> None:
    expected = {
        "family": "rbf_svm",
        "representation_hash": representation_hash,
        "training_split_hash": training_split_hash,
        "predictions_sha256": predictions_hash,
    }
    mismatches = {
        name: (metadata.get(name), value)
        for name, value in expected.items()
        if metadata.get(name) != value
    }
    if mismatches:
        raise ValueError(f"delegated-head lineage mismatch: {mismatches}")


def fit_delegated_rbf_head(
    geometry_points: np.ndarray,
    geometry_labels: np.ndarray,
    calibration_points: np.ndarray,
    calibration_labels: np.ndarray,
    query_points: np.ndarray,
    *,
    known_classes: Sequence[int],
    seed: int,
    c_value: float = 1.0,
    gamma: str = "scale",
) -> dict[str, Any]:
    geometry = np.asarray(geometry_points, dtype=np.float64)
    geometry_targets = np.asarray(geometry_labels, dtype=np.int64)
    calibration = np.asarray(calibration_points, dtype=np.float64)
    calibration_targets = np.asarray(calibration_labels, dtype=np.int64)
    query = np.asarray(query_points, dtype=np.float64)
    classes = np.asarray(tuple(int(value) for value in known_classes), dtype=np.int64)
    if (
        geometry.ndim != 2
        or calibration.ndim != 2
        or query.ndim != 2
        or geometry.shape[1] != calibration.shape[1]
        or geometry.shape[1] != query.shape[1]
        or geometry_targets.shape != (len(geometry),)
        or calibration_targets.shape != (len(calibration),)
        or not np.all(np.isfinite(geometry))
        or not np.all(np.isfinite(calibration))
        or not np.all(np.isfinite(query))
        or tuple(np.unique(geometry_targets)) != tuple(classes)
        or tuple(np.unique(calibration_targets)) != tuple(classes)
        or c_value <= 0.0
        or gamma != "scale"
    ):
        raise ValueError("delegated-head inputs violate the frozen v11 contract")
    estimator = SVC(
        C=c_value,
        gamma=gamma,
        kernel="rbf",
        random_state=seed,
    ).fit(geometry, geometry_targets)
    support_before = estimator.support_vectors_.copy()
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(estimator),
        method="sigmoid",
    ).fit(calibration, calibration_targets)
    if not np.array_equal(estimator.support_vectors_, support_before):
        raise RuntimeError("delegated-head calibration retrained the frozen estimator")
    probabilities = calibrated.predict_proba(query)
    predictions = calibrated.classes_[np.argmax(probabilities, axis=1)]
    if not set(np.unique(predictions)).issubset(set(classes)):
        raise RuntimeError("delegated head predicted outside the known-class set")
    return {
        "predictions": predictions.astype(np.int64),
        "probabilities": probabilities.astype(np.float64),
        "classes": calibrated.classes_.astype(np.int64),
        "support_vector_count": int(len(estimator.support_vectors_)),
        "support_vectors_unchanged_by_calibration": True,
        "fit_class_count": int(len(np.unique(geometry_targets))),
        "calibration_class_count": int(len(np.unique(calibration_targets))),
    }


def normalize_directions(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("points must be a nonempty finite matrix")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= np.finfo(np.float64).tiny):
        raise ValueError("directions must have nonzero norm")
    return values / norms[:, None]


def mean_direction(points: np.ndarray) -> np.ndarray:
    directions = normalize_directions(points)
    resultant = directions.sum(axis=0)
    norm = float(np.linalg.norm(resultant))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("mean direction is undefined for an antipodal patch")
    return resultant / norm


def estimate_directional_rank(
    points: np.ndarray,
    *,
    rank_grid: Sequence[int],
    explained_variance_target: float,
) -> dict[str, Any]:
    directions = normalize_directions(points)
    ranks = tuple(int(rank) for rank in rank_grid)
    if (
        not ranks
        or tuple(sorted(set(ranks))) != ranks
        or ranks[0] < 1
        or ranks[-1] >= min(directions.shape)
        or not 0.0 < explained_variance_target < 1.0
    ):
        raise ValueError("directional rank settings violate the registered contract")
    logarithms = spherical_log_map(mean_direction(directions), directions)
    singular_values = np.linalg.svd(
        logarithms, full_matrices=False, compute_uv=False
    )
    variances = singular_values * singular_values
    total = float(np.sum(variances))
    if total <= 0.0:
        raise ValueError("directional rank recovery requires nonconstant points")
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


def deterministic_directional_patch_assignments(
    points: np.ndarray,
    *,
    patch_count: int,
    seed: int,
) -> np.ndarray:
    directions = normalize_directions(points)
    if patch_count not in {1, 2, 4} or len(directions) < patch_count:
        raise ValueError("points or patch_count violate the v11 atlas contract")
    if patch_count == 1:
        return np.zeros(len(directions), dtype=np.int64)
    return KMeans(
        n_clusters=patch_count,
        random_state=seed,
        n_init=10,
        algorithm="lloyd",
    ).fit_predict(directions).astype(np.int64)


def spherical_log_map(center: np.ndarray, points: np.ndarray) -> np.ndarray:
    origin = np.asarray(center, dtype=np.float64)
    if (
        origin.ndim != 1
        or not np.all(np.isfinite(origin))
        or not np.isclose(np.linalg.norm(origin), 1.0, rtol=0.0, atol=1e-10)
    ):
        raise ValueError("center must be a finite unit direction")
    directions = normalize_directions(points)
    cosine = np.clip(directions @ origin, -1.0, 1.0)
    angles = np.arccos(cosine)
    tangent = directions - cosine[:, None] * origin[None, :]
    tangent_norms = np.linalg.norm(tangent, axis=1)
    if np.any((np.pi - angles < 1e-10) & (tangent_norms < 1e-10)):
        raise ValueError("log map is undefined at the antipode")
    scale = np.divide(
        angles,
        tangent_norms,
        out=np.ones_like(angles),
        where=tangent_norms > np.finfo(np.float64).eps,
    )
    result = tangent * scale[:, None]
    result[tangent_norms <= np.finfo(np.float64).eps] = 0.0
    return result


def spherical_exp_map(center: np.ndarray, tangent_vectors: np.ndarray) -> np.ndarray:
    origin = np.asarray(center, dtype=np.float64)
    vectors = np.asarray(tangent_vectors, dtype=np.float64)
    if (
        origin.ndim != 1
        or vectors.ndim != 2
        or vectors.shape[1] != len(origin)
        or not np.all(np.isfinite(vectors))
        or not np.isclose(np.linalg.norm(origin), 1.0, rtol=0.0, atol=1e-10)
        or not np.allclose(vectors @ origin, 0.0, rtol=0.0, atol=1e-8)
    ):
        raise ValueError("exp map requires finite tangent vectors and a unit center")
    angles = np.linalg.norm(vectors, axis=1)
    unit = np.divide(
        vectors,
        angles[:, None],
        out=np.zeros_like(vectors),
        where=angles[:, None] > np.finfo(np.float64).eps,
    )
    result = np.cos(angles)[:, None] * origin + np.sin(angles)[:, None] * unit
    result[angles <= np.finfo(np.float64).eps] = origin
    return normalize_directions(result)


def split_conformal_quantile(scores: np.ndarray, *, miscoverage: float = 0.08) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if (
        values.ndim != 1
        or not len(values)
        or not np.all(np.isfinite(values))
        or not 0.0 < miscoverage < 1.0
    ):
        raise ValueError("scores and miscoverage violate the conformal contract")
    rank = min(len(values), int(np.ceil((len(values) + 1) * (1.0 - miscoverage))))
    return float(np.sort(values, kind="stable")[rank - 1])


def negative_guided_extents(
    own_coordinates: np.ndarray,
    negative_coordinates: np.ndarray,
    *,
    upper_quantile: float,
) -> np.ndarray:
    own = np.abs(np.asarray(own_coordinates, dtype=np.float64))
    negative = np.abs(np.asarray(negative_coordinates, dtype=np.float64))
    if (
        own.ndim != 2
        or not len(own)
        or negative.ndim != 2
        or negative.shape[1] != own.shape[1]
        or not len(negative)
        or not np.all(np.isfinite(own))
        or not np.all(np.isfinite(negative))
        or upper_quantile not in {0.95, 0.99}
    ):
        raise ValueError("negative-guided extent inputs violate the registered policy")
    floor = np.quantile(own, 0.90, axis=0, method="higher")
    upper = np.quantile(own, upper_quantile, axis=0, method="higher")
    extents = upper.copy()
    for axis in range(own.shape[1]):
        candidates = np.unique(
            np.concatenate(([floor[axis]], own[:, axis], [upper[axis]]))
        )
        candidates = candidates[
            (candidates >= floor[axis]) & (candidates <= upper[axis])
        ][::-1]
        selected = None
        for candidate in candidates:
            trial = extents.copy()
            trial[axis] = candidate
            if not np.any(np.all(negative <= trial[None, :], axis=1)):
                selected = float(candidate)
                break
        if selected is None:
            raise ValueError("negative-guided extent is infeasible above the 0.90 floor")
        extents[axis] = selected
    return extents


@dataclass(frozen=True)
class DirectionalTube:
    center: np.ndarray
    basis: np.ndarray
    residual_scale: float
    tangent_extents: np.ndarray
    outer_scales: np.ndarray
    penalty_weight: float
    class_label: int
    patch_id: int = 0
    extent_policy: str = "quantile"

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        basis = deterministic_basis_signs(np.asarray(self.basis, dtype=np.float64))
        extents = np.asarray(self.tangent_extents, dtype=np.float64)
        scales = np.asarray(self.outer_scales, dtype=np.float64)
        if (
            center.ndim != 1
            or not np.all(np.isfinite(center))
            or not np.isclose(np.linalg.norm(center), 1.0, rtol=0.0, atol=1e-10)
        ):
            raise ValueError("center must be a finite unit direction")
        if (
            basis.ndim != 2
            or basis.shape[0] != len(center)
            or basis.shape[1] < 1
            or not np.all(np.isfinite(basis))
            or not np.allclose(
                basis.T @ basis, np.eye(basis.shape[1]), rtol=0.0, atol=1e-8
            )
            or not np.allclose(basis.T @ center, 0.0, rtol=0.0, atol=1e-8)
        ):
            raise ValueError("basis must be orthonormal in the center tangent plane")
        if (
            extents.shape != (basis.shape[1],)
            or scales.shape != extents.shape
            or np.any(~np.isfinite(extents) | (extents <= 0.0))
            or np.any(~np.isfinite(scales) | (scales <= 0.0))
        ):
            raise ValueError("extents and scales must be finite, positive rank vectors")
        if not np.isfinite(self.residual_scale) or self.residual_scale <= 0.0:
            raise ValueError("residual_scale must be finite and positive")
        if not np.isfinite(self.penalty_weight) or self.penalty_weight <= 0.0:
            raise ValueError("penalty_weight must be finite and positive")
        if self.extent_policy not in EXTENT_POLICIES:
            raise ValueError("unsupported v11 extent policy")
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
        logarithms = spherical_log_map(self.center, points)
        tangent = logarithms @ self.basis
        residual = logarithms - tangent @ self.basis.T
        return tangent, np.sum(residual * residual, axis=1)

    def score_terms(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tangent, residual_squared = self.coordinates(points)
        residual = residual_squared / self.residual_scale
        overshoot = np.maximum(
            (np.abs(tangent) - self.tangent_extents[None, :])
            / self.outer_scales[None, :],
            0.0,
        )
        return residual, np.mean(overshoot * overshoot, axis=1)

    def score(self, points: np.ndarray) -> np.ndarray:
        residual, tangent = self.score_terms(points)
        return residual + self.penalty_weight * tangent

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
            "extent_policy": self.extent_policy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DirectionalTube":
        required = {
            "schema_version",
            "center",
            "basis",
            "residual_scale",
            "tangent_extents",
            "outer_scales",
            "penalty_weight",
            "class_label",
            "patch_id",
            "extent_policy",
        }
        if set(payload) != required or payload["schema_version"] != 1:
            raise ValueError("invalid directional-tube payload")
        return cls(
            center=np.asarray(payload["center"], dtype=np.float64),
            basis=np.asarray(payload["basis"], dtype=np.float64),
            residual_scale=float(payload["residual_scale"]),
            tangent_extents=np.asarray(payload["tangent_extents"], dtype=np.float64),
            outer_scales=np.asarray(payload["outer_scales"], dtype=np.float64),
            penalty_weight=float(payload["penalty_weight"]),
            class_label=int(payload["class_label"]),
            patch_id=int(payload["patch_id"]),
            extent_policy=str(payload["extent_policy"]),
        )


def fit_directional_tube(
    geometry_points: np.ndarray,
    own_calibration_points: np.ndarray,
    *,
    rank: int,
    extent_policy: str,
    extent_quantile: float,
    penalty_weight: float,
    class_label: int,
    negative_points: np.ndarray | None = None,
    patch_id: int = 0,
) -> DirectionalTube:
    geometry = normalize_directions(geometry_points)
    own_calibration = normalize_directions(own_calibration_points)
    if (
        rank < 1
        or rank >= geometry.shape[1]
        or len(geometry) < rank + 2
        or own_calibration.shape[1] != geometry.shape[1]
        or extent_policy not in EXTENT_POLICIES
        or extent_quantile not in {0.95, 0.99}
    ):
        raise ValueError("directional tube inputs violate the registered grid")
    center = mean_direction(geometry)
    logarithms = spherical_log_map(center, geometry)
    _, singular_values, right_vectors = np.linalg.svd(logarithms, full_matrices=False)
    if singular_values[rank - 1] <= np.finfo(np.float64).eps:
        raise ValueError("directional patch is rank deficient")
    basis = deterministic_basis_signs(right_vectors[:rank].T)
    basis -= center[:, None] * (center @ basis)[None, :]
    basis, _ = np.linalg.qr(basis)
    basis = deterministic_basis_signs(basis[:, :rank])
    own_logarithms = spherical_log_map(center, own_calibration)
    own_coordinates = own_logarithms @ basis
    own_residual = own_logarithms - own_coordinates @ basis.T
    floor = max(float(np.finfo(np.float64).eps), float(np.var(logarithms)) * 1e-12)
    residual_scale = max(
        float(
            np.quantile(
                np.sum(own_residual * own_residual, axis=1), 0.95, method="higher"
            )
        ),
        floor,
    )
    if extent_policy == "quantile":
        extents = np.quantile(
            np.abs(own_coordinates), extent_quantile, axis=0, method="higher"
        )
    else:
        if negative_points is None:
            raise ValueError("negative-guided policies require negative points")
        negative_coordinates = spherical_log_map(center, negative_points) @ basis
        extents = negative_guided_extents(
            own_coordinates,
            negative_coordinates,
            upper_quantile=extent_quantile,
        )
    if extent_policy == "negative_guided_iqr":
        scales = np.quantile(np.abs(own_coordinates), 0.75, axis=0) - np.quantile(
            np.abs(own_coordinates), 0.25, axis=0
        )
    else:
        q90 = np.quantile(np.abs(own_coordinates), 0.90, axis=0, method="higher")
        scales = extents - q90
    return DirectionalTube(
        center=center,
        basis=basis,
        residual_scale=residual_scale,
        tangent_extents=np.maximum(extents, floor),
        outer_scales=np.maximum(scales, floor),
        penalty_weight=penalty_weight,
        class_label=class_label,
        patch_id=patch_id,
        extent_policy=extent_policy,
    )


def class_score_matrix(
    tubes: Sequence[DirectionalTube], points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not tubes:
        raise ValueError("at least one directional tube is required")
    classes = np.asarray(sorted({tube.class_label for tube in tubes}), dtype=np.int64)
    scores = np.empty((len(np.asarray(points)), len(classes)), dtype=np.float64)
    for column, class_label in enumerate(classes):
        members = [tube for tube in tubes if tube.class_label == class_label]
        scores[:, column] = np.min(
            np.column_stack([tube.score(points) for tube in members]), axis=1
        )
    return scores, classes


def calibrate_class_thresholds(
    score_matrix: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    *,
    miscoverage: float = 0.08,
) -> np.ndarray:
    scores = np.asarray(score_matrix, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    class_values = np.asarray(classes, dtype=np.int64)
    if (
        scores.ndim != 2
        or scores.shape != (len(targets), len(class_values))
        or not np.all(np.isfinite(scores))
    ):
        raise ValueError("class calibration scores have incompatible shapes")
    thresholds = []
    for column, class_label in enumerate(class_values):
        own = scores[targets == class_label, column]
        if not len(own):
            raise ValueError("every class requires conformal calibration observations")
        thresholds.append(split_conformal_quantile(own, miscoverage=miscoverage))
    result = np.asarray(thresholds, dtype=np.float64)
    if np.any(result <= 0.0):
        raise ValueError("conformal thresholds must be positive")
    return result


def contrast_acceptance(
    score_matrix: np.ndarray,
    thresholds: np.ndarray,
    classes: np.ndarray,
    *,
    margin: float,
) -> dict[str, np.ndarray]:
    scores = np.asarray(score_matrix, dtype=np.float64)
    limits = np.asarray(thresholds, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.int64)
    if (
        scores.ndim != 2
        or scores.shape[1] != len(limits)
        or len(limits) != len(class_values)
        or len(limits) < 2
        or np.any(~np.isfinite(scores))
        or np.any(~np.isfinite(limits) | (limits <= 0.0))
        or margin not in CONTRAST_MARGINS
    ):
        raise ValueError("contrast inputs violate the registered contract")
    normalized = scores / limits[None, :]
    selected_columns = np.argmin(normalized, axis=1)
    rows = np.arange(len(scores))
    selected_scores = normalized[rows, selected_columns]
    competing = normalized.copy()
    competing[rows, selected_columns] = np.inf
    gaps = np.min(competing, axis=1) - selected_scores
    conformal = selected_scores <= 1.0
    contrast = (gaps >= margin) | (selected_scores <= 1.0 - margin)
    return {
        "selected_class": class_values[selected_columns],
        "normalized_score": selected_scores,
        "contrast_gap": gaps,
        "conformal_accepted": conformal,
        "contrast_accepted": contrast,
        "accepted": conformal & contrast,
    }


def composite_endpoint_records(
    envelope: Mapping[str, np.ndarray],
    head_predictions: np.ndarray,
    labels: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    head = np.asarray(head_predictions, dtype=np.int64)
    accepted = np.asarray(envelope["accepted"], dtype=bool)
    selected = np.asarray(envelope["selected_class"], dtype=np.int64)
    if head.shape != accepted.shape or selected.shape != accepted.shape:
        raise ValueError("head and envelope decisions must align")
    targets = None if labels is None else np.asarray(labels, dtype=np.int64)
    if targets is not None and targets.shape != head.shape:
        raise ValueError("labels must align with decisions")
    records = []
    for index in range(len(head)):
        record: dict[str, Any] = {
            "observation_index": index,
            "envelope_accepted": bool(accepted[index]),
            "envelope_class": int(selected[index]),
            "head_class": int(head[index]),
            "composite_class": int(head[index]) if accepted[index] else None,
        }
        if targets is not None:
            record["label"] = int(targets[index])
            record["head_correct"] = bool(head[index] == targets[index])
            record["composite_correct"] = bool(
                accepted[index] and head[index] == targets[index]
            )
        records.append(record)
    return records


def _normal_direction(tube: DirectionalTube, rng: np.random.Generator) -> np.ndarray:
    direction = rng.normal(size=len(tube.center))
    direction -= tube.center * (tube.center @ direction)
    direction -= tube.basis @ (tube.basis.T @ direction)
    norm = float(np.linalg.norm(direction))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("no normal direction is available")
    return direction / norm


def generate_geodesic_probes(
    tubes: Sequence[DirectionalTube], *, seed: int
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not tubes:
        raise ValueError("at least one directional tube is required")
    dimension = len(tubes[0].center)
    if any(len(tube.center) != dimension for tube in tubes):
        raise ValueError("all tubes must share an ambient dimension")
    rng = np.random.default_rng(seed)
    points: dict[str, list[np.ndarray]] = {name: [] for name in PROBE_FAMILIES}
    owners: dict[str, list[int]] = {name: [] for name in PROBE_FAMILIES}
    multipliers: dict[str, list[float]] = {name: [] for name in PROBE_FAMILIES}

    def add(name: str, owner: int, multiplier: float, vector: np.ndarray) -> None:
        point = spherical_exp_map(tubes[owner].center, vector[None, :])[0]
        points[name].append(point)
        owners[name].append(owner)
        multipliers[name].append(multiplier)

    for owner, tube in enumerate(tubes):
        normal = _normal_direction(tube, rng)
        normal_scale = float(np.sqrt(tube.residual_scale))
        for multiplier in TANGENT_MULTIPLIERS:
            for axis in range(tube.rank):
                for sign in (-1.0, 1.0):
                    add(
                        "axis_tangent",
                        owner,
                        multiplier,
                        sign
                        * multiplier
                        * tube.tangent_extents[axis]
                        * tube.basis[:, axis],
                    )
        add(
            "corner_tangent",
            owner,
            4.0,
            tube.basis @ (4.0 * tube.tangent_extents),
        )
        add("normal", owner, 4.0, 4.0 * normal_scale * normal)
        add(
            "mixed",
            owner,
            4.0,
            4.0 * tube.tangent_extents[0] * tube.basis[:, 0]
            + normal_scale * normal,
        )
        random_direction = rng.normal(size=dimension)
        random_direction -= tube.center * (tube.center @ random_direction)
        random_direction /= np.linalg.norm(random_direction)
        add(
            "random_direction",
            owner,
            8.0,
            8.0 * np.linalg.norm(tube.tangent_extents) * random_direction,
        )
        competitors = [
            index
            for index, candidate in enumerate(tubes)
            if candidate.class_label != tube.class_label
        ]
        if competitors:
            nearest = max(
                competitors,
                key=lambda index: float(tube.center @ tubes[index].center),
            )
            toward = spherical_log_map(
                tube.center, tubes[nearest].center[None, :]
            )[0]
            projected = tube.basis @ (tube.basis.T @ toward)
            if np.linalg.norm(projected) > np.finfo(np.float64).eps:
                axis = int(np.argmax(np.abs(tube.basis.T @ projected)))
                sign = float(np.sign(tube.basis[:, axis] @ projected))
                for multiplier in MASKING_MULTIPLIERS:
                    add(
                        "masking",
                        owner,
                        multiplier,
                        sign
                        * multiplier
                        * tube.tangent_extents[axis]
                        * tube.basis[:, axis],
                    )
    for owner, tube in enumerate(tubes):
        same = [
            index
            for index, candidate in enumerate(tubes)
            if index != owner and candidate.class_label == tube.class_label
        ]
        other = [
            index
            for index, candidate in enumerate(tubes)
            if candidate.class_label != tube.class_label
        ]
        for name, candidates in (("bridge", same), ("cross_class_bridge", other)):
            if candidates:
                nearest = max(
                    candidates,
                    key=lambda index: float(tube.center @ tubes[index].center),
                )
                toward = spherical_log_map(
                    tube.center, tubes[nearest].center[None, :]
                )[0]
                add(name, owner, 0.5, 0.5 * toward)
    return {
        name: (
            np.vstack(points[name])
            if points[name]
            else np.empty((0, dimension), dtype=np.float64),
            np.asarray(owners[name], dtype=np.int64),
            np.asarray(multipliers[name], dtype=np.float64),
        )
        for name in PROBE_FAMILIES
    }


def directional_replay_hash(
    tubes: Sequence[DirectionalTube],
    thresholds: np.ndarray,
    *,
    miscoverage: float,
    contrast_margin: float,
) -> str:
    return replay_digest(
        {
            "tubes": [tube.to_dict() for tube in tubes],
            "thresholds": np.asarray(thresholds, dtype=np.float64).tolist(),
            "miscoverage": miscoverage,
            "contrast_margin": contrast_margin,
        }
    )
