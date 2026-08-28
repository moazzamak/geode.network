from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from src.subspace_primitive import SubspacePrimitive, fit_subspace_primitive


@dataclass(frozen=True)
class MetricFieldState:
    classes: np.ndarray
    centers: np.ndarray
    bases: np.ndarray
    tangent_scales: np.ndarray
    residual_scales: np.ndarray

    def __post_init__(self) -> None:
        classes = np.asarray(self.classes, dtype=np.int64)
        centers = np.asarray(self.centers, dtype=np.float64)
        bases = np.asarray(self.bases, dtype=np.float64)
        tangent = np.asarray(self.tangent_scales, dtype=np.float64)
        residual = np.asarray(self.residual_scales, dtype=np.float64)
        if (
            classes.ndim != 1
            or centers.ndim != 2
            or bases.ndim != 3
            or tangent.ndim != 2
            or residual.shape != classes.shape
            or centers.shape[0] != len(classes)
            or bases.shape[0] != len(classes)
            or bases.shape[1] != centers.shape[1]
            or bases.shape[2] != tangent.shape[1]
            or tangent.shape[0] != len(classes)
            or len(np.unique(classes)) != len(classes)
            or np.any(tangent <= 0.0)
            or np.any(residual <= 0.0)
            or not all(
                np.all(np.isfinite(value))
                for value in (centers, bases, tangent, residual)
            )
        ):
            raise ValueError("metric field state violates its shape or value contract")
        for basis in bases:
            if not np.allclose(
                basis.T @ basis,
                np.eye(basis.shape[1]),
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError("metric field bases must be orthonormal")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "bases", bases)
        object.__setattr__(self, "tangent_scales", tangent)
        object.__setattr__(self, "residual_scales", residual)

    @property
    def rank(self) -> int:
        return int(self.bases.shape[2])

    @property
    def dimension(self) -> int:
        return int(self.centers.shape[1])

    @property
    def parameter_count(self) -> int:
        return int(
            self.centers.size
            + self.bases.size
            + self.tangent_scales.size
            + self.residual_scales.size
        )

    @property
    def array_bytes(self) -> int:
        return int(
            self.classes.nbytes
            + self.centers.nbytes
            + self.bases.nbytes
            + self.tangent_scales.nbytes
            + self.residual_scales.nbytes
        )

    def score_terms(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("metric field points have the wrong shape")
        deltas = values[:, None, :] - self.centers[None, :, :]
        coordinates = np.einsum("nkd,kdr->nkr", deltas, self.bases)
        tangent_terms = (
            coordinates * coordinates
            / (self.tangent_scales[None, :, :] ** 2)
        )
        tangent_projection = np.einsum(
            "nkr,kdr->nkd", coordinates, self.bases
        )
        residual = deltas - tangent_projection
        residual_terms = np.sum(residual * residual, axis=2) / (
            self.residual_scales[None, :] ** 2
        )
        return tangent_terms, residual_terms

    def scores(self, points: np.ndarray) -> np.ndarray:
        tangent, residual = self.score_terms(points)
        return np.sqrt(
            np.maximum(np.sum(tangent, axis=2) + residual, 0.0)
            + np.finfo(np.float64).eps
        )

    def score_gradients(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        deltas = values[:, None, :] - self.centers[None, :, :]
        coordinates = np.einsum("nkd,kdr->nkr", deltas, self.bases)
        tangent_projection = np.einsum(
            "nkr,kdr->nkd", coordinates, self.bases
        )
        residual = deltas - tangent_projection
        inverse_delta = np.einsum(
            "nkr,kdr->nkd",
            coordinates / (self.tangent_scales[None, :, :] ** 2),
            self.bases,
        ) + residual / (self.residual_scales[None, :, None] ** 2)
        return inverse_delta / self.scores(values)[:, :, None]

    def predict(self, points: np.ndarray) -> np.ndarray:
        return self.classes[np.argmin(self.scores(points), axis=1)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "classes": self.classes.tolist(),
            "centers": self.centers.tolist(),
            "bases": self.bases.tolist(),
            "tangent_scales": self.tangent_scales.tolist(),
            "residual_scales": self.residual_scales.tolist(),
        }


@dataclass(frozen=True)
class ProjectedMetricFieldState:
    projection_mean: np.ndarray
    projection: np.ndarray
    fields: MetricFieldState

    def __post_init__(self) -> None:
        mean = np.asarray(self.projection_mean, dtype=np.float64)
        projection = np.asarray(self.projection, dtype=np.float64)
        if (
            mean.ndim != 1
            or projection.ndim != 2
            or projection.shape[1] != len(mean)
            or projection.shape[0] != self.fields.dimension
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(projection))
        ):
            raise ValueError("projected metric field state has invalid projection")
        object.__setattr__(self, "projection_mean", mean)
        object.__setattr__(self, "projection", projection)

    @property
    def parameter_count(self) -> int:
        return int(
            self.projection_mean.size
            + self.projection.size
            + self.fields.parameter_count
        )

    @property
    def array_bytes(self) -> int:
        return int(
            self.projection_mean.nbytes
            + self.projection.nbytes
            + self.fields.array_bytes
        )

    def transform(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.projection_mean):
            raise ValueError("projection inputs have the wrong shape")
        return (values - self.projection_mean) @ self.projection.T

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "projection_mean": self.projection_mean.tolist(),
            "projection": self.projection.tolist(),
            "fields": self.fields.to_dict(),
        }


def initialize_projected_metric_fields(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    output_dimension: int,
    rank: int,
) -> ProjectedMetricFieldState:
    values = np.asarray(features, dtype=np.float64)
    if (
        values.ndim != 2
        or not 1 <= output_dimension < values.shape[1]
        or rank >= output_dimension
    ):
        raise ValueError("projected metric field dimensions are invalid")
    mean = np.mean(values, axis=0)
    _, _, right_vectors = np.linalg.svd(values - mean, full_matrices=False)
    projection = right_vectors[:output_dimension].copy()
    for row in range(len(projection)):
        pivot = int(np.argmax(np.abs(projection[row])))
        if projection[row, pivot] < 0.0:
            projection[row] *= -1.0
    projected = (values - mean) @ projection.T
    return ProjectedMetricFieldState(
        projection_mean=mean,
        projection=projection,
        fields=initialize_metric_fields(projected, labels, rank=rank),
    )


def initialize_metric_fields(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    rank: int,
) -> MetricFieldState:
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    classes = np.unique(targets)
    primitives: list[SubspacePrimitive] = [
        fit_subspace_primitive(
            values[targets == class_label],
            min(rank, values.shape[1] - 1, int(np.sum(targets == class_label)) - 2),
            class_label=int(class_label),
        )
        for class_label in classes
    ]
    return MetricFieldState(
        classes=classes,
        centers=np.vstack([item.center for item in primitives]),
        bases=np.stack([item.basis for item in primitives]),
        tangent_scales=np.sqrt(
            np.vstack([item.tangent_variances for item in primitives])
        ),
        residual_scales=np.sqrt(
            np.asarray([item.residual_variance for item in primitives])
        ),
    )


def _torch_scores(
    points: Tensor,
    centers: Tensor,
    bases: Tensor,
    tangent_scales: Tensor,
    residual_scales: Tensor,
) -> tuple[Tensor, Tensor]:
    deltas = points[:, None, :] - centers[None, :, :]
    coordinates = torch.einsum("nkd,kdr->nkr", deltas, bases)
    tangent_projection = torch.einsum("nkr,kdr->nkd", coordinates, bases)
    residual = deltas - tangent_projection
    tangent = torch.sum(
        coordinates.square() / tangent_scales[None, :, :].square(), dim=2
    )
    residual_term = torch.sum(residual.square(), dim=2) / residual_scales[
        None, :
    ].square()
    scores = torch.sqrt(tangent + residual_term + torch.finfo(points.dtype).eps)
    inverse_delta = torch.einsum(
        "nkr,kdr->nkd",
        coordinates / tangent_scales[None, :, :].square(),
        bases,
    ) + residual / residual_scales[None, :, None].square()
    gradients = inverse_delta / scores[:, :, None]
    return scores, gradients


def _torch_probes(
    centers: Tensor,
    bases: Tensor,
    tangent_scales: Tensor,
    residual_scales: Tensor,
    *,
    families: Sequence[str],
    seed: int,
) -> Tensor:
    generator = torch.Generator(device=centers.device)
    generator.manual_seed(seed)
    points = []
    class_count, dimension = centers.shape
    rank = bases.shape[2]
    for class_index in range(class_count):
        center = centers[class_index]
        basis = bases[class_index]
        normal = torch.randn(
            dimension,
            generator=generator,
            dtype=centers.dtype,
            device=centers.device,
        )
        normal = normal - basis @ (basis.T @ normal)
        normal = normal / torch.linalg.vector_norm(normal).clamp_min(1e-12)
        if "axis_tangent" in families:
            for axis in range(rank):
                for sign in (-1.0, 1.0):
                    points.append(
                        center
                        + sign
                        * 4.0
                        * tangent_scales[class_index, axis]
                        * basis[:, axis]
                    )
        if "normal" in families:
            points.append(
                center + 4.0 * residual_scales[class_index] * normal
            )
        if "masking" in families and class_count > 1:
            distances = torch.linalg.vector_norm(
                centers.detach() - center.detach(), dim=1
            )
            distances[class_index] = torch.inf
            competitor = int(torch.argmin(distances).item())
            toward = centers[competitor].detach() - center.detach()
            coordinates = basis.T @ toward
            axis = int(torch.argmax(torch.abs(coordinates)).item())
            sign = torch.sign(coordinates[axis])
            if float(sign) == 0.0:
                sign = torch.ones_like(sign)
            points.append(
                center
                + sign
                * 4.0
                * tangent_scales[class_index, axis]
                * basis[:, axis]
            )
        if "random_direction" in families:
            direction = torch.randn(
                dimension,
                generator=generator,
                dtype=centers.dtype,
                device=centers.device,
            )
            direction = direction / torch.linalg.vector_norm(direction).clamp_min(
                1e-12
            )
            points.append(
                center
                + 8.0
                * torch.linalg.vector_norm(tangent_scales[class_index])
                * direction
            )
    if not points:
        raise ValueError("at least one probe-training family is required")
    return torch.stack(points)


def train_metric_fields(
    initial: MetricFieldState,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    classification_temperature: float,
    target_score: float,
    separation_margin: float,
    probe_margin: float,
    loss_weights: dict[str, float],
    probe_families: Sequence[str],
    seed: int,
) -> tuple[MetricFieldState, list[dict[str, float]]]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("metric field optimizer settings are invalid")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dtype = torch.float64
    values = torch.as_tensor(np.asarray(features), dtype=dtype)
    targets_numpy = np.asarray(labels, dtype=np.int64)
    class_lookup = {
        int(class_label): index
        for index, class_label in enumerate(initial.classes.tolist())
    }
    targets = torch.as_tensor(
        [class_lookup[int(label)] for label in targets_numpy],
        dtype=torch.int64,
    )
    centers = torch.nn.Parameter(torch.tensor(initial.centers, dtype=dtype))
    log_tangent = torch.nn.Parameter(
        torch.log(torch.tensor(initial.tangent_scales, dtype=dtype))
    )
    log_residual = torch.nn.Parameter(
        torch.log(torch.tensor(initial.residual_scales, dtype=dtype))
    )
    bases = torch.as_tensor(initial.bases, dtype=dtype)
    optimizer = torch.optim.Adam(
        [centers, log_tangent, log_residual], lr=learning_rate
    )
    history = []
    generator = torch.Generator()
    generator.manual_seed(seed + 1)
    for epoch in range(epochs):
        order = torch.randperm(len(values), generator=generator)
        totals = {
            "classification": 0.0,
            "eikonal": 0.0,
            "probe": 0.0,
            "distribution": 0.0,
            "separation": 0.0,
            "total": 0.0,
        }
        batches = 0
        for start in range(0, len(values), batch_size):
            rows = order[start : start + batch_size]
            batch = values[rows]
            batch_targets = targets[rows]
            tangent_scales = torch.exp(log_tangent).clamp_min(1e-6)
            residual_scales = torch.exp(log_residual).clamp_min(1e-6)
            scores, gradients = _torch_scores(
                batch, centers, bases, tangent_scales, residual_scales
            )
            own_scores = scores[
                torch.arange(len(batch_targets)), batch_targets
            ]
            own_gradients = gradients[
                torch.arange(len(batch_targets)), batch_targets
            ]
            classification = torch.nn.functional.cross_entropy(
                -classification_temperature * scores, batch_targets
            )
            distribution = torch.mean((own_scores - target_score) ** 2)
            eikonal = torch.mean(
                (torch.linalg.vector_norm(own_gradients, dim=1) - 1.0) ** 2
            )
            masked = scores.clone()
            masked[
                torch.arange(len(batch_targets)), batch_targets
            ] = torch.inf
            separation = torch.mean(
                torch.relu(
                    separation_margin
                    - torch.min(masked, dim=1).values
                    + own_scores
                )
            )
            probes = _torch_probes(
                centers,
                bases,
                tangent_scales,
                residual_scales,
                families=probe_families,
                seed=seed + epoch,
            )
            probe_scores, probe_gradients = _torch_scores(
                probes, centers, bases, tangent_scales, residual_scales
            )
            interpolants = 0.5 * (batch + torch.roll(batch, shifts=1, dims=0))
            _, interpolant_gradients = _torch_scores(
                interpolants,
                centers,
                bases,
                tangent_scales,
                residual_scales,
            )
            probe = torch.mean(
                torch.relu(probe_margin - torch.min(probe_scores, dim=1).values)
            )
            eikonal = eikonal + torch.mean(
                (
                    torch.linalg.vector_norm(probe_gradients, dim=2)
                    - 1.0
                )
                ** 2
            ) + torch.mean(
                (
                    torch.linalg.vector_norm(interpolant_gradients, dim=2)
                    - 1.0
                )
                ** 2
            )
            losses = {
                "classification": classification,
                "eikonal": eikonal,
                "probe": probe,
                "distribution": distribution,
                "separation": separation,
            }
            total = sum(loss_weights[name] * loss for name, loss in losses.items())
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            for name, loss in losses.items():
                totals[name] += float(loss.detach())
            totals["total"] += float(total.detach())
            batches += 1
        history.append(
            {
                "epoch": float(epoch + 1),
                **{name: value / batches for name, value in totals.items()},
            }
        )
    state = MetricFieldState(
        classes=initial.classes,
        centers=centers.detach().numpy(),
        bases=initial.bases,
        tangent_scales=torch.exp(log_tangent).detach().numpy(),
        residual_scales=torch.exp(log_residual).detach().numpy(),
    )
    return state, history


def train_projected_metric_fields(
    initial: ProjectedMetricFieldState,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    classification_temperature: float,
    target_score: float,
    separation_margin: float,
    probe_margin_multiplier: float,
    loss_weights: dict[str, float],
    collapse_weight: float,
    probe_families: Sequence[str],
    seed: int,
) -> tuple[ProjectedMetricFieldState, list[dict[str, float]]]:
    if (
        epochs < 1
        or batch_size < 2
        or learning_rate <= 0.0
        or probe_margin_multiplier <= 1.0
        or collapse_weight < 0.0
    ):
        raise ValueError("projected metric field optimizer settings are invalid")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dtype = torch.float64
    raw_values = torch.as_tensor(np.asarray(features), dtype=dtype)
    mean = torch.as_tensor(initial.projection_mean, dtype=dtype)
    targets_numpy = np.asarray(labels, dtype=np.int64)
    class_lookup = {
        int(class_label): index
        for index, class_label in enumerate(initial.fields.classes.tolist())
    }
    targets = torch.as_tensor(
        [class_lookup[int(label)] for label in targets_numpy], dtype=torch.int64
    )
    projection = torch.nn.Parameter(
        torch.tensor(initial.projection, dtype=dtype)
    )
    reference_projection = torch.tensor(initial.projection, dtype=dtype)
    centers = torch.nn.Parameter(
        torch.tensor(initial.fields.centers, dtype=dtype)
    )
    log_tangent = torch.nn.Parameter(
        torch.log(torch.tensor(initial.fields.tangent_scales, dtype=dtype))
    )
    log_residual = torch.nn.Parameter(
        torch.log(torch.tensor(initial.fields.residual_scales, dtype=dtype))
    )
    bases = torch.as_tensor(initial.fields.bases, dtype=dtype)
    optimizer = torch.optim.Adam(
        [projection, centers, log_tangent, log_residual], lr=learning_rate
    )
    identity = torch.eye(projection.shape[0], dtype=dtype)
    history = []
    generator = torch.Generator()
    generator.manual_seed(seed + 1)
    for epoch in range(epochs):
        order = torch.randperm(len(raw_values), generator=generator)
        totals = {
            "classification": 0.0,
            "eikonal": 0.0,
            "probe": 0.0,
            "distribution": 0.0,
            "separation": 0.0,
            "collapse": 0.0,
            "orthogonality": 0.0,
            "distance_preservation": 0.0,
            "total": 0.0,
        }
        batches = 0
        for start in range(0, len(raw_values), batch_size):
            rows = order[start : start + batch_size]
            raw_batch = raw_values[rows] - mean
            batch = raw_batch @ projection.T
            with torch.no_grad():
                reference_batch = raw_batch @ reference_projection.T
            batch_targets = targets[rows]
            tangent_scales = torch.exp(log_tangent).clamp_min(1e-6)
            residual_scales = torch.exp(log_residual).clamp_min(1e-6)
            scores, gradients = _torch_scores(
                batch, centers, bases, tangent_scales, residual_scales
            )
            row_indices = torch.arange(len(batch_targets))
            own_scores = scores[row_indices, batch_targets]
            own_gradients = gradients[row_indices, batch_targets]
            classification = torch.nn.functional.cross_entropy(
                -classification_temperature * scores, batch_targets
            )
            distribution = torch.mean((own_scores - target_score) ** 2)
            masked = scores.clone()
            masked[row_indices, batch_targets] = torch.inf
            separation = torch.mean(
                torch.relu(
                    separation_margin
                    - torch.min(masked, dim=1).values
                    + own_scores
                )
            )
            probes = _torch_probes(
                centers,
                bases,
                tangent_scales,
                residual_scales,
                families=probe_families,
                seed=seed + epoch,
            )
            probe_scores, probe_gradients = _torch_scores(
                probes, centers, bases, tangent_scales, residual_scales
            )
            interpolants = 0.5 * (batch + torch.roll(batch, shifts=1, dims=0))
            _, interpolant_gradients = _torch_scores(
                interpolants,
                centers,
                bases,
                tangent_scales,
                residual_scales,
            )
            eikonal = (
                torch.mean(
                    (
                        torch.linalg.vector_norm(own_gradients, dim=1)
                        - 1.0
                    )
                    ** 2
                )
                + torch.mean(
                    (
                        torch.linalg.vector_norm(probe_gradients, dim=2)
                        - 1.0
                    )
                    ** 2
                )
                + torch.mean(
                    (
                        torch.linalg.vector_norm(
                            interpolant_gradients, dim=2
                        )
                        - 1.0
                    )
                    ** 2
                )
            )
            probe_target = (
                probe_margin_multiplier * torch.median(own_scores).detach()
            )
            probe = torch.mean(
                torch.relu(
                    probe_target - torch.min(probe_scores, dim=1).values
                )
            )
            orthogonality = torch.mean(
                (projection @ projection.T - identity) ** 2
            )
            current_distances = torch.linalg.vector_norm(
                batch - torch.roll(batch, shifts=1, dims=0), dim=1
            )
            reference_distances = torch.linalg.vector_norm(
                reference_batch
                - torch.roll(reference_batch, shifts=1, dims=0),
                dim=1,
            ).clamp_min(1e-12)
            distance_preservation = torch.mean(
                ((current_distances - reference_distances) / reference_distances)
                ** 2
            )
            collapse = orthogonality + distance_preservation
            losses = {
                "classification": classification,
                "eikonal": eikonal,
                "probe": probe,
                "distribution": distribution,
                "separation": separation,
            }
            total = (
                sum(loss_weights[name] * loss for name, loss in losses.items())
                + collapse_weight * collapse
            )
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            for name, loss in losses.items():
                totals[name] += float(loss.detach())
            totals["collapse"] += float(collapse.detach())
            totals["orthogonality"] += float(orthogonality.detach())
            totals["distance_preservation"] += float(
                distance_preservation.detach()
            )
            totals["total"] += float(total.detach())
            batches += 1
        history.append(
            {
                "epoch": float(epoch + 1),
                **{name: value / batches for name, value in totals.items()},
            }
        )
    state = ProjectedMetricFieldState(
        projection_mean=initial.projection_mean,
        projection=projection.detach().numpy(),
        fields=MetricFieldState(
            classes=initial.fields.classes,
            centers=centers.detach().numpy(),
            bases=initial.fields.bases,
            tangent_scales=torch.exp(log_tangent).detach().numpy(),
            residual_scales=torch.exp(log_residual).detach().numpy(),
        ),
    )
    return state, history


def projection_diagnostics(
    state: ProjectedMetricFieldState,
    reference: ProjectedMetricFieldState,
    points: np.ndarray,
) -> dict[str, float | int]:
    singular_values = np.linalg.svd(
        state.projection, compute_uv=False
    )
    current = state.transform(points)
    baseline = reference.transform(points)
    current_distances = np.linalg.norm(
        current - np.roll(current, shift=1, axis=0), axis=1
    )
    reference_distances = np.maximum(
        np.linalg.norm(
            baseline - np.roll(baseline, shift=1, axis=0), axis=1
        ),
        1e-12,
    )
    return {
        "minimum_singular_value": float(np.min(singular_values)),
        "maximum_singular_value": float(np.max(singular_values)),
        "effective_rank": int(np.sum(singular_values > 1e-6)),
        "row_orthogonality_error": float(
            np.mean(
                (
                    state.projection @ state.projection.T
                    - np.eye(state.projection.shape[0])
                )
                ** 2
            )
        ),
        "mean_relative_distance_drift": float(
            np.mean(np.abs(current_distances - reference_distances) / reference_distances)
        ),
    }
