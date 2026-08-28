"""Measurement-only candidate routing beside authoritative exhaustive inference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

import numpy as np

from src.candidate_routing import CandidateRoutingResult, CertifiedRoutingResult
from src.inference_engine import InferenceEngine
from src.open_set import RoutingStageCounters
from src.sdf_engine import Expert


CandidateRouter = Callable[
    [dict[int, list[Expert]], np.ndarray], CandidateRoutingResult
]


def _percentiles(values: list[float]) -> dict[str, float]:
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def exhaustive_scores(
    models: dict[int, list[Expert]], points: np.ndarray, *, alpha: float,
    score_scales: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, RoutingStageCounters]:
    if not models:
        raise ValueError("At least one class model is required.")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must have shape (samples, dimensions).")
    class_ids = np.asarray(sorted(models))
    scores = np.column_stack([
        InferenceEngine(models[int(class_id)], alpha=alpha).get_fused_sdf(points)
        / (1.0 if score_scales is None else score_scales[int(class_id)])
        for class_id in class_ids
    ])
    pair_count = len(points) * len(class_ids)
    primitive_count = len(points) * sum(
        len(expert.ellipsoids)
        for experts in models.values()
        for expert in experts
    )
    return class_ids, scores, RoutingStageCounters(
        sample_count=len(points),
        nodes_executed=1,
        compatible_candidate_pairs=pair_count,
        shortlisted_candidate_pairs=pair_count,
        exact_class_sdf_pairs=pair_count,
        primitive_sdf_pairs=primitive_count,
        score_values_materialized=pair_count,
    )


@dataclass(frozen=True)
class ShadowRoutingObservation:
    router_name: str
    authoritative_predictions: np.ndarray
    candidate_predictions: np.ndarray
    oracle_counters: RoutingStageCounters
    candidate_counters: RoutingStageCounters
    agreement: float
    maximum_winning_score_error: float
    bound_distance_pairs: int
    fallback_samples: int
    oracle_latency_seconds: dict[str, float]
    candidate_latency_seconds: dict[str, float]
    quality_gate_passed: bool
    latency_gate_passed: bool
    candidate_controls_outputs: bool = False

    @property
    def promotion_eligible(self) -> bool:
        return self.quality_gate_passed and self.latency_gate_passed


def run_shadow_router(
    models: dict[int, list[Expert]],
    points: np.ndarray,
    router: CandidateRouter,
    *,
    router_name: str,
    alpha: float = 2.0,
    timing_repeats: int = 5,
    score_tolerance: float = 1e-12,
    score_scales: dict[int, float] | None = None,
) -> ShadowRoutingObservation:
    """Measure a candidate while oracle predictions remain authoritative."""
    if timing_repeats < 1:
        raise ValueError("timing_repeats must be positive.")
    points = np.asarray(points, dtype=np.float64)
    class_ids, oracle_matrix, oracle_counters = exhaustive_scores(
        models, points, alpha=alpha, score_scales=score_scales,
    )
    oracle_columns = np.argmin(oracle_matrix, axis=1)
    oracle_predictions = class_ids[oracle_columns]
    oracle_winners = oracle_matrix[np.arange(len(points)), oracle_columns]
    candidate = router(models, points)

    oracle_latencies = []
    candidate_latencies = []
    for _ in range(timing_repeats):
        started = time.perf_counter()
        exhaustive_scores(
            models, points, alpha=alpha, score_scales=score_scales,
        )
        oracle_latencies.append(time.perf_counter() - started)
        started = time.perf_counter()
        router(models, points)
        candidate_latencies.append(time.perf_counter() - started)

    candidate_pairs = int(np.sum(candidate.candidate_counts))
    candidate_counters = RoutingStageCounters(
        sample_count=len(points),
        nodes_executed=1,
        compatible_candidate_pairs=len(points) * len(models),
        shortlisted_candidate_pairs=candidate_pairs,
        exact_class_sdf_pairs=candidate_pairs,
        primitive_sdf_pairs=int(np.sum(candidate.primitive_evaluation_counts)),
        score_values_materialized=candidate_pairs,
    )
    agreement = float(np.mean(candidate.predictions == oracle_predictions))
    maximum_error = float(np.max(np.abs(
        candidate.winning_scores - oracle_winners
    ))) if len(points) else 0.0
    fallback_samples = (
        int(np.sum(candidate.fallback_mask))
        if isinstance(candidate, CertifiedRoutingResult) else 0
    )
    oracle_timing = _percentiles(oracle_latencies)
    candidate_timing = _percentiles(candidate_latencies)
    return ShadowRoutingObservation(
        router_name=router_name,
        authoritative_predictions=oracle_predictions,
        candidate_predictions=candidate.predictions,
        oracle_counters=oracle_counters,
        candidate_counters=candidate_counters,
        agreement=agreement,
        maximum_winning_score_error=maximum_error,
        bound_distance_pairs=len(points) * len(models),
        fallback_samples=fallback_samples,
        oracle_latency_seconds=oracle_timing,
        candidate_latency_seconds=candidate_timing,
        quality_gate_passed=(agreement == 1.0 and maximum_error <= score_tolerance),
        latency_gate_passed=(candidate_timing["p95"] < oracle_timing["p95"]),
    )