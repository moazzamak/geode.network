from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.inference_engine import InferenceEngine
from src.sdf_engine import Expert


@dataclass(frozen=True)
class CandidateRoutingResult:
    predictions: np.ndarray
    winning_scores: np.ndarray
    candidate_counts: np.ndarray
    primitive_evaluation_counts: np.ndarray


@dataclass(frozen=True)
class CertifiedRoutingResult(CandidateRoutingResult):
    proposal_counts: np.ndarray
    fallback_mask: np.ndarray


class CertifiedTopKRouter:
    def __init__(
        self,
        models: dict[int, list[Expert]],
        candidate_budget: int,
    ) -> None:
        if not models:
            raise ValueError("At least one class model is required.")
        if not 1 <= candidate_budget <= len(models):
            raise ValueError("candidate_budget must be between 1 and class count.")
        self.models = models
        self.class_ids = np.asarray(sorted(models))
        self.candidate_budget = candidate_budget
        centroids = np.asarray([
            np.mean([
                expert.bounding_sphere()[0] for expert in models[int(class_id)]
            ], axis=0)
            for class_id in self.class_ids
        ])
        self.index = NearestNeighbors(
            n_neighbors=candidate_budget,
            algorithm="auto",
        ).fit(centroids)

    def route(self, points: np.ndarray) -> CertifiedRoutingResult:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError("points must have shape (samples, dimensions).")
        proposals = self.index.kneighbors(points, return_distance=False)
        predictions = np.full(
            len(points), self.class_ids[0], dtype=self.class_ids.dtype,
        )
        winning_scores = np.full(len(points), np.inf, dtype=np.float64)
        candidate_counts = np.zeros(len(points), dtype=np.int64)
        primitive_counts = np.zeros(len(points), dtype=np.int64)

        def evaluate(class_index: int, selected: np.ndarray) -> None:
            class_id = int(self.class_ids[class_index])
            experts = self.models[class_id]
            scores = InferenceEngine(experts, alpha=2.0).get_fused_sdf(
                points[selected]
            )
            improved = scores < winning_scores[selected]
            winning_scores[selected] = np.minimum(winning_scores[selected], scores)
            predictions[selected[improved]] = self.class_ids[class_index]
            candidate_counts[selected] += 1
            primitive_counts[selected] += sum(
                len(expert.ellipsoids) for expert in experts
            )

        for class_index in np.unique(proposals):
            evaluate(
                int(class_index),
                np.flatnonzero(np.any(proposals == class_index, axis=1)),
            )

        bounds = _class_lower_bounds(self.models, self.class_ids, points)
        proposed = np.zeros(bounds.shape, dtype=bool)
        proposed[np.arange(len(points))[:, None], proposals] = True
        omitted_bounds = np.where(proposed, np.inf, bounds)
        fallback_mask = np.min(omitted_bounds, axis=1) <= winning_scores
        for class_index in range(len(self.class_ids)):
            selected = np.flatnonzero(fallback_mask & ~proposed[:, class_index])
            if len(selected):
                evaluate(class_index, selected)

        return CertifiedRoutingResult(
            predictions=predictions,
            winning_scores=winning_scores,
            candidate_counts=candidate_counts,
            primitive_evaluation_counts=primitive_counts,
            proposal_counts=np.full(len(points), self.candidate_budget),
            fallback_mask=fallback_mask,
        )


def _class_lower_bounds(
    models: dict[int, list[Expert]],
    class_ids: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    return np.column_stack([
        np.min(np.column_stack([
            np.linalg.norm(points - center, axis=1) / max(radius, 1e-12) - 1.0
            for center, radius in (
                expert.bounding_sphere() for expert in models[int(class_id)]
            )
        ]), axis=1)
        for class_id in class_ids
    ])


def exact_bound_routing(
    models: dict[int, list[Expert]],
    points: np.ndarray,
) -> CandidateRoutingResult:
    if not models:
        raise ValueError("At least one class model is required.")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must have shape (samples, dimensions).")
    class_ids = np.asarray(sorted(models))
    predictions = np.empty(len(points), dtype=class_ids.dtype)
    winning_scores = np.empty(len(points), dtype=np.float64)
    candidate_counts = np.zeros(len(points), dtype=np.int64)
    primitive_counts = np.zeros(len(points), dtype=np.int64)

    bounds = _class_lower_bounds(models, class_ids, points)
    for point_index, point in enumerate(points):
        best_score = np.inf
        best_class = class_ids[0]
        for class_index in np.argsort(bounds[point_index], kind="stable"):
            if bounds[point_index, class_index] > best_score:
                break
            class_id = int(class_ids[class_index])
            experts = models[class_id]
            score = float(InferenceEngine(experts, alpha=2.0).get_fused_sdf(
                point.reshape(1, -1),
            )[0])
            candidate_counts[point_index] += 1
            primitive_counts[point_index] += sum(
                len(expert.ellipsoids) for expert in experts
            )
            if score < best_score:
                best_score = score
                best_class = class_ids[class_index]
        predictions[point_index] = best_class
        winning_scores[point_index] = best_score
    return CandidateRoutingResult(
        predictions=predictions,
        winning_scores=winning_scores,
        candidate_counts=candidate_counts,
        primitive_evaluation_counts=primitive_counts,
    )


def batched_exact_bound_routing(
    models: dict[int, list[Expert]],
    points: np.ndarray,
) -> CandidateRoutingResult:
    if not models:
        raise ValueError("At least one class model is required.")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must have shape (samples, dimensions).")
    class_ids = np.asarray(sorted(models))
    bounds = _class_lower_bounds(models, class_ids, points)
    orders = np.argsort(bounds, axis=1, kind="stable")
    positions = np.zeros(len(points), dtype=np.int64)
    predictions = np.full(len(points), class_ids[0], dtype=class_ids.dtype)
    winning_scores = np.full(len(points), np.inf, dtype=np.float64)
    candidate_counts = np.zeros(len(points), dtype=np.int64)
    primitive_counts = np.zeros(len(points), dtype=np.int64)
    sample_indices = np.arange(len(points))

    while True:
        unfinished = positions < len(class_ids)
        next_classes = np.zeros(len(points), dtype=np.int64)
        next_classes[unfinished] = orders[
            sample_indices[unfinished], positions[unfinished]
        ]
        active = unfinished & (
            bounds[sample_indices, next_classes] <= winning_scores
        )
        if not np.any(active):
            break
        for class_index in np.unique(next_classes[active]):
            selected = np.flatnonzero(active & (next_classes == class_index))
            class_id = int(class_ids[class_index])
            experts = models[class_id]
            scores = InferenceEngine(experts, alpha=2.0).get_fused_sdf(
                points[selected]
            )
            improved = scores < winning_scores[selected]
            winning_scores[selected] = np.minimum(
                winning_scores[selected], scores
            )
            predictions[selected[improved]] = class_ids[class_index]
            candidate_counts[selected] += 1
            primitive_counts[selected] += sum(
                len(expert.ellipsoids) for expert in experts
            )
            positions[selected] += 1
    return CandidateRoutingResult(
        predictions=predictions,
        winning_scores=winning_scores,
        candidate_counts=candidate_counts,
        primitive_evaluation_counts=primitive_counts,
    )


def class_major_exact_bound_routing(
    models: dict[int, list[Expert]],
    points: np.ndarray,
) -> CandidateRoutingResult:
    if not models:
        raise ValueError("At least one class model is required.")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must have shape (samples, dimensions).")
    class_ids = np.asarray(sorted(models))
    bounds = _class_lower_bounds(models, class_ids, points)
    initial_classes = np.argmin(bounds, axis=1)
    predictions = np.full(len(points), class_ids[0], dtype=class_ids.dtype)
    winning_scores = np.full(len(points), np.inf, dtype=np.float64)
    candidate_counts = np.zeros(len(points), dtype=np.int64)
    primitive_counts = np.zeros(len(points), dtype=np.int64)

    def evaluate(class_index: int, selected: np.ndarray) -> None:
        class_id = int(class_ids[class_index])
        experts = models[class_id]
        scores = InferenceEngine(experts, alpha=2.0).get_fused_sdf(points[selected])
        improved = scores < winning_scores[selected]
        winning_scores[selected] = np.minimum(winning_scores[selected], scores)
        predictions[selected[improved]] = class_ids[class_index]
        candidate_counts[selected] += 1
        primitive_counts[selected] += sum(
            len(expert.ellipsoids) for expert in experts
        )

    for class_index in np.unique(initial_classes):
        evaluate(class_index, np.flatnonzero(initial_classes == class_index))

    ranks = np.argsort(np.argsort(bounds, axis=1, kind="stable"), axis=1)
    for class_index in np.argsort(np.mean(ranks, axis=0), kind="stable"):
        selected = np.flatnonzero(
            (initial_classes != class_index)
            & (bounds[:, class_index] <= winning_scores)
        )
        if len(selected):
            evaluate(class_index, selected)
    return CandidateRoutingResult(
        predictions=predictions,
        winning_scores=winning_scores,
        candidate_counts=candidate_counts,
        primitive_evaluation_counts=primitive_counts,
    )