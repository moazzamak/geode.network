"""M77 probe-degeneracy forensics for GEODE v13.

This module instruments the v12 projected metric-field training loop so the
probe hinge can be decomposed while training proceeds. The instrumented loop is
a faithful copy of
``experiments.common.v12_metric_fields.train_projected_metric_fields``; the
runner verifies that it reproduces the v12 optimizer history exactly before any
diagnostic is interpreted.

Three registered operands are supported:

``O77.1``
    Decomposition of the probe hinge into the fraction of probe points whose
    minimising class is their own source class, and the mean own-class probe
    score per probe family.
``O77.2``
    Analytic invariance: own-class probe scores for the scale-relative families
    are unchanged when the fitted extents are rescaled across three orders of
    magnitude.
``O77.3``
    Gradient of the probe term with respect to ``log_tangent`` and
    ``log_residual``, recorded per epoch.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from experiments.common.v12_metric_fields import (
    MetricFieldState,
    ProjectedMetricFieldState,
    _torch_probes,
    _torch_scores,
)


#: Probe families whose displacement is expressed as a multiple of the fitted
#: extent along the direction they displace. For these the own-class score is
#: algebraically independent of that extent.
SCALE_RELATIVE_FAMILIES = ("axis_tangent", "normal", "masking")

#: Probe families whose displacement is not aligned with a single fitted axis,
#: so the own-class score retains a dependence on the fitted extents.
SCALE_COUPLED_FAMILIES = ("random_direction",)


def _probe_source_classes(
    class_count: int,
    rank: int,
    families: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the source class and family index for each generated probe.

    The ordering mirrors ``_torch_probes`` exactly: the outer loop is over
    classes and the inner blocks append in the order ``axis_tangent``,
    ``normal``, ``masking``, ``random_direction``.
    """
    sources: list[int] = []
    labels: list[str] = []
    for class_index in range(class_count):
        if "axis_tangent" in families:
            for _ in range(rank):
                for _ in (-1.0, 1.0):
                    sources.append(class_index)
                    labels.append("axis_tangent")
        if "normal" in families:
            sources.append(class_index)
            labels.append("normal")
        if "masking" in families and class_count > 1:
            sources.append(class_index)
            labels.append("masking")
        if "random_direction" in families:
            sources.append(class_index)
            labels.append("random_direction")
    return np.asarray(sources, dtype=np.int64), np.asarray(labels, dtype=object)


def probe_scale_invariance(
    fields: MetricFieldState,
    *,
    families: Sequence[str],
    scale_factors: Sequence[float],
    seed: int,
) -> dict[str, Any]:
    """O77.2 — recompute own-class probe scores under rescaled extents.

    Every fitted extent is multiplied by a common factor. A probe family whose
    displacement is defined as a multiple of the extent it displaces along will
    return an identical own-class score for every factor.
    """
    dtype = torch.float64
    centers = torch.as_tensor(fields.centers, dtype=dtype)
    bases = torch.as_tensor(fields.bases, dtype=dtype)
    base_tangent = torch.as_tensor(fields.tangent_scales, dtype=dtype)
    base_residual = torch.as_tensor(fields.residual_scales, dtype=dtype)
    sources, labels = _probe_source_classes(
        len(fields.classes), fields.rank, families
    )
    rows = torch.arange(len(sources))
    source_index = torch.as_tensor(sources, dtype=torch.int64)

    per_factor: dict[str, dict[str, float]] = {}
    for factor in scale_factors:
        tangent = base_tangent * float(factor)
        residual = base_residual * float(factor)
        probes = _torch_probes(
            centers,
            bases,
            tangent,
            residual,
            families=families,
            seed=seed,
        )
        scores, _ = _torch_scores(probes, centers, bases, tangent, residual)
        own = scores[rows, source_index].detach().numpy()
        per_factor[f"{float(factor):.6g}"] = {
            family: float(np.mean(own[labels == family]))
            for family in sorted(set(labels.tolist()))
        }

    families_seen = sorted(set(labels.tolist()))
    spreads = {
        family: float(
            max(entry[family] for entry in per_factor.values())
            - min(entry[family] for entry in per_factor.values())
        )
        for family in families_seen
    }
    return {
        "scale_factors": [float(value) for value in scale_factors],
        "mean_own_class_score_by_factor": per_factor,
        "own_class_score_spread_by_family": spreads,
        "invariant_families": sorted(
            family for family, spread in spreads.items() if spread < 1e-9
        ),
        "scale_coupled_families": sorted(
            family for family, spread in spreads.items() if spread >= 1e-9
        ),
    }


def _probe_diagnostics(
    probe_scores: Tensor,
    probe_target: Tensor,
    source_index: Tensor,
    labels: np.ndarray,
) -> dict[str, float]:
    rows = torch.arange(len(source_index))
    own_scores = probe_scores[rows, source_index]
    minimum_scores, minimum_index = torch.min(probe_scores, dim=1)
    own_is_minimiser = minimum_index == source_index
    hinge = torch.relu(probe_target - minimum_scores)
    own_hinge = torch.relu(probe_target - own_scores)

    diagnostics: dict[str, float] = {
        "probe_target": float(probe_target),
        "mean_own_class_score": float(torch.mean(own_scores)),
        "mean_minimum_score": float(torch.mean(minimum_scores)),
        "own_class_is_minimiser_fraction": float(
            torch.mean(own_is_minimiser.to(probe_scores.dtype))
        ),
        "mean_hinge": float(torch.mean(hinge)),
        "mean_own_class_hinge": float(torch.mean(own_hinge)),
        "hinge_active_fraction": float(
            torch.mean((hinge > 0.0).to(probe_scores.dtype))
        ),
    }
    for family in sorted(set(labels.tolist())):
        mask = torch.as_tensor(labels == family)
        diagnostics[f"own_score__{family}"] = float(
            torch.mean(own_scores[mask])
        )
        diagnostics[f"own_hinge__{family}"] = float(
            torch.mean(own_hinge[mask])
        )
    return diagnostics


def train_projected_metric_fields_instrumented(
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
) -> tuple[ProjectedMetricFieldState, list[dict[str, float]], list[dict[str, Any]]]:
    """Faithful copy of the v12 projected trainer with probe instrumentation.

    Diagnostics are computed without mutating any gradient state: probe-term
    gradients are obtained through ``torch.autograd.grad``, which does not
    accumulate into ``.grad``, and every other diagnostic is detached. The
    returned ``history`` must equal the v12 function's history exactly.
    """
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
    projection = torch.nn.Parameter(torch.tensor(initial.projection, dtype=dtype))
    reference_projection = torch.tensor(initial.projection, dtype=dtype)
    centers = torch.nn.Parameter(torch.tensor(initial.fields.centers, dtype=dtype))
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

    probe_sources, probe_labels = _probe_source_classes(
        len(initial.fields.classes), initial.fields.rank, probe_families
    )
    source_index = torch.as_tensor(probe_sources, dtype=torch.int64)

    history: list[dict[str, float]] = []
    diagnostics: list[dict[str, Any]] = []
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
        probe_totals: dict[str, float] = {}
        gradient_totals = {
            "probe_grad_norm_log_tangent": 0.0,
            "probe_grad_norm_log_residual": 0.0,
            "probe_grad_norm_centers": 0.0,
            "total_grad_norm_log_tangent": 0.0,
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
                    separation_margin - torch.min(masked, dim=1).values + own_scores
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
                interpolants, centers, bases, tangent_scales, residual_scales
            )
            eikonal = (
                torch.mean(
                    (torch.linalg.vector_norm(own_gradients, dim=1) - 1.0) ** 2
                )
                + torch.mean(
                    (torch.linalg.vector_norm(probe_gradients, dim=2) - 1.0) ** 2
                )
                + torch.mean(
                    (torch.linalg.vector_norm(interpolant_gradients, dim=2) - 1.0)
                    ** 2
                )
            )
            probe_target = (
                probe_margin_multiplier * torch.median(own_scores).detach()
            )
            probe = torch.mean(
                torch.relu(probe_target - torch.min(probe_scores, dim=1).values)
            )
            orthogonality = torch.mean((projection @ projection.T - identity) ** 2)
            current_distances = torch.linalg.vector_norm(
                batch - torch.roll(batch, shifts=1, dims=0), dim=1
            )
            reference_distances = torch.linalg.vector_norm(
                reference_batch - torch.roll(reference_batch, shifts=1, dims=0),
                dim=1,
            ).clamp_min(1e-12)
            distance_preservation = torch.mean(
                ((current_distances - reference_distances) / reference_distances) ** 2
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

            # --- O77.1 / O77.3 instrumentation (no gradient-state mutation) ---
            with torch.no_grad():
                batch_probe = _probe_diagnostics(
                    probe_scores.detach(),
                    probe_target.detach(),
                    source_index,
                    probe_labels,
                )
            for key, value in batch_probe.items():
                probe_totals[key] = probe_totals.get(key, 0.0) + value
            probe_grads = torch.autograd.grad(
                probe,
                [log_tangent, log_residual, centers],
                retain_graph=True,
                allow_unused=True,
            )
            names = (
                "probe_grad_norm_log_tangent",
                "probe_grad_norm_log_residual",
                "probe_grad_norm_centers",
            )
            for name, grad in zip(names, probe_grads):
                gradient_totals[name] += (
                    0.0 if grad is None else float(torch.linalg.vector_norm(grad))
                )
            total_grads = torch.autograd.grad(
                total, [log_tangent], retain_graph=True, allow_unused=True
            )
            gradient_totals["total_grad_norm_log_tangent"] += (
                0.0
                if total_grads[0] is None
                else float(torch.linalg.vector_norm(total_grads[0]))
            )
            # --- end instrumentation ---

            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            for name, loss in losses.items():
                totals[name] += float(loss.detach())
            totals["collapse"] += float(collapse.detach())
            totals["orthogonality"] += float(orthogonality.detach())
            totals["distance_preservation"] += float(distance_preservation.detach())
            totals["total"] += float(total.detach())
            batches += 1
        history.append(
            {
                "epoch": float(epoch + 1),
                **{name: value / batches for name, value in totals.items()},
            }
        )
        diagnostics.append(
            {
                "epoch": float(epoch + 1),
                **{name: value / batches for name, value in probe_totals.items()},
                **{
                    name: value / batches
                    for name, value in gradient_totals.items()
                },
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
    return state, history, diagnostics
