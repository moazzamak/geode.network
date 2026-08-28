from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.tier5.eval_editability_scaling import (
    _models,
    _predictions,
    _routing_counts,
    _scores,
)
from src.candidate_routing import exact_bound_routing
from src.open_set import RoutingStageCounters


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def _slope(x: list[int], y: list[float]) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def run_exact_bound_routing(config: dict) -> dict:
    records = []
    for class_count in config["class_counts"]:
        for seed in config["seeds"]:
            condition = {
                "class_count": int(class_count),
                "dimensions": int(config["dimensions"]),
                "primitives_per_class": int(config["primitives_per_class"]),
            }
            models = _models(condition, int(seed))
            points = np.random.default_rng(seed).normal(
                scale=3.0,
                size=(config["batch_size"], config["dimensions"]),
            )
            exhaustive_scores, _ = _scores(models, points)
            exhaustive_predictions = _predictions(exhaustive_scores)
            exhaustive_winners = np.min(np.column_stack([
                exhaustive_scores[class_id] for class_id in sorted(models)
            ]), axis=1)
            exact_bound_routing(models, points)
            exhaustive_latencies = []
            shortlist_latencies = []
            result = None
            for _ in range(config["timing_repeats"]):
                started = time.perf_counter()
                _scores(models, points)
                exhaustive_latencies.append(time.perf_counter() - started)
                started = time.perf_counter()
                result = exact_bound_routing(models, points)
                shortlist_latencies.append(time.perf_counter() - started)
            assert result is not None
            agreement = float(np.mean(
                result.predictions == exhaustive_predictions
            ))
            maximum_score_error = float(np.max(np.abs(
                result.winning_scores - exhaustive_winners
            )))
            if agreement != 1.0 or maximum_score_error > 1e-12:
                raise AssertionError(
                    f"Exact bound routing diverged: {agreement}, {maximum_score_error}"
                )
            candidate_pairs = int(np.sum(result.candidate_counts))
            primitive_pairs = int(np.sum(
                result.primitive_evaluation_counts
            ))
            counters = RoutingStageCounters(
                sample_count=len(points),
                nodes_executed=1,
                compatible_candidate_pairs=len(points) * class_count,
                shortlisted_candidate_pairs=candidate_pairs,
                exact_class_sdf_pairs=candidate_pairs,
                primitive_sdf_pairs=primitive_pairs,
                score_values_materialized=candidate_pairs,
            )
            records.append({
                "class_count": class_count,
                "seed": seed,
                "agreement": agreement,
                "maximum_winning_score_error": maximum_score_error,
                "mean_candidates_per_sample": float(np.mean(
                    result.candidate_counts
                )),
                "p95_candidates_per_sample": float(np.quantile(
                    result.candidate_counts, 0.95,
                )),
                "mean_primitives_per_sample": float(np.mean(
                    result.primitive_evaluation_counts
                )),
                "candidate_fraction": candidate_pairs / (len(points) * class_count),
                "certificate_fallback_rate": 0.0,
                "exhaustive_latency_seconds": _percentiles(exhaustive_latencies),
                "shortlist_latency_seconds": _percentiles(shortlist_latencies),
                "routing_counters": asdict(counters),
                "exhaustive_routing_counters": asdict(
                    _routing_counts(models, len(points))
                ),
            })

    summaries = []
    for class_count in config["class_counts"]:
        cells = [record for record in records if record["class_count"] == class_count]
        summaries.append({
            "class_count": class_count,
            "agreement": float(np.mean([cell["agreement"] for cell in cells])),
            "maximum_winning_score_error": float(np.max([
                cell["maximum_winning_score_error"] for cell in cells
            ])),
            "mean_candidates_per_sample": float(np.mean([
                cell["mean_candidates_per_sample"] for cell in cells
            ])),
            "candidate_fraction": float(np.mean([
                cell["candidate_fraction"] for cell in cells
            ])),
            "exhaustive_latency_p50_seconds": float(np.mean([
                cell["exhaustive_latency_seconds"]["p50"] for cell in cells
            ])),
            "shortlist_latency_p50_seconds": float(np.mean([
                cell["shortlist_latency_seconds"]["p50"] for cell in cells
            ])),
        })
    class_counts = [item["class_count"] for item in summaries]
    return {
        "milestone": "M12.1",
        "protocol": {
            "routing": "conservative_bounding_sphere_lower_bound",
            "candidate_order": "ascending_lower_bound",
            "stop_rule": "next_lower_bound_exceeds_best_exact_score",
            "exhaustive_authoritative": True,
            "approximate_index_used": False,
            "minimum_required_agreement": 1.0,
        },
        "summary": {
            "minimum_agreement": float(min(
                item["agreement"] for item in summaries
            )),
            "maximum_winning_score_error": float(max(
                item["maximum_winning_score_error"] for item in summaries
            )),
            "candidate_count_log_log_slope": _slope(
                class_counts,
                [item["mean_candidates_per_sample"] for item in summaries],
            ),
            "shortlist_latency_log_log_slope": _slope(
                class_counts,
                [item["shortlist_latency_p50_seconds"] for item in summaries],
            ),
            "exhaustive_latency_log_log_slope": _slope(
                class_counts,
                [item["exhaustive_latency_p50_seconds"] for item in summaries],
            ),
        },
        "class_count_summaries": summaries,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_exact_bound_routing(config)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
