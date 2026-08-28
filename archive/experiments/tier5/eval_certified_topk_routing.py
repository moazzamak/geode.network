from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.tier5.eval_editability_scaling import _models, _predictions, _scores
from experiments.tier5.eval_exact_bound_routing import _percentiles, _slope
from src.candidate_routing import CertifiedTopKRouter


def run_certified_topk_routing(config: dict) -> dict:
    records = []
    for class_count in config["class_counts"]:
        budgets = [budget for budget in config["candidate_budgets"] if budget < class_count]
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
            exhaustive_latencies = []
            for _ in range(config["timing_repeats"]):
                started = time.perf_counter()
                _scores(models, points)
                exhaustive_latencies.append(time.perf_counter() - started)

            for budget in budgets:
                started = time.perf_counter()
                router = CertifiedTopKRouter(models, budget)
                build_seconds = time.perf_counter() - started
                result = router.route(points)
                if not (
                    np.array_equal(result.predictions, exhaustive_predictions)
                    and np.allclose(result.winning_scores, exhaustive_winners)
                ):
                    raise AssertionError("Certified top-k routing diverged.")
                routed_latencies = []
                for _ in range(config["timing_repeats"]):
                    started = time.perf_counter()
                    router.route(points)
                    routed_latencies.append(time.perf_counter() - started)
                records.append({
                    "class_count": class_count,
                    "candidate_budget": budget,
                    "seed": seed,
                    "agreement": 1.0,
                    "maximum_winning_score_error": float(np.max(np.abs(
                        result.winning_scores - exhaustive_winners
                    ))),
                    "fallback_fraction": float(np.mean(result.fallback_mask)),
                    "mean_candidates_per_sample": float(np.mean(
                        result.candidate_counts
                    )),
                    "candidate_fraction": float(np.sum(result.candidate_counts)) / (
                        len(points) * class_count
                    ),
                    "index_build_seconds": build_seconds,
                    "exhaustive_latency_seconds": _percentiles(exhaustive_latencies),
                    "routed_latency_seconds": _percentiles(routed_latencies),
                })

    summaries = []
    for class_count in config["class_counts"]:
        for budget in config["candidate_budgets"]:
            cells = [
                record for record in records
                if record["class_count"] == class_count
                and record["candidate_budget"] == budget
            ]
            if not cells:
                continue
            exhaustive_p50 = float(np.mean([
                cell["exhaustive_latency_seconds"]["p50"] for cell in cells
            ]))
            routed_p50 = float(np.mean([
                cell["routed_latency_seconds"]["p50"] for cell in cells
            ]))
            summaries.append({
                "class_count": class_count,
                "candidate_budget": budget,
                "agreement": 1.0,
                "maximum_winning_score_error": float(max(
                    cell["maximum_winning_score_error"] for cell in cells
                )),
                "fallback_fraction": float(np.mean([
                    cell["fallback_fraction"] for cell in cells
                ])),
                "mean_candidates_per_sample": float(np.mean([
                    cell["mean_candidates_per_sample"] for cell in cells
                ])),
                "candidate_fraction": float(np.mean([
                    cell["candidate_fraction"] for cell in cells
                ])),
                "mean_index_build_seconds": float(np.mean([
                    cell["index_build_seconds"] for cell in cells
                ])),
                "exhaustive_latency_p50_seconds": exhaustive_p50,
                "routed_latency_p50_seconds": routed_p50,
                "speedup_over_exhaustive": exhaustive_p50 / routed_p50,
            })

    class_counts = config["class_counts"]
    budget_slopes = {}
    for budget in config["candidate_budgets"]:
        selected = [
            item for item in summaries if item["candidate_budget"] == budget
        ]
        if len(selected) >= 2:
            budget_slopes[str(budget)] = {
                "candidate_count": _slope(
                    [item["class_count"] for item in selected],
                    [item["mean_candidates_per_sample"] for item in selected],
                ),
                "routed_latency": _slope(
                    [item["class_count"] for item in selected],
                    [item["routed_latency_p50_seconds"] for item in selected],
                ),
            }
    return {
        "milestone": "M12.4",
        "protocol": {
            "index": "sklearn_nearest_neighbors_on_class_support_centroids",
            "certificate": "all_omitted_support_sphere_lower_bounds",
            "fallback": "score_all_omitted_classes_if_any_bound_is_competitive",
            "index_build_excluded_from_query_latency": True,
            "exhaustive_authoritative": True,
        },
        "summary": {
            "minimum_agreement": 1.0,
            "maximum_winning_score_error": float(max(
                item["maximum_winning_score_error"] for item in summaries
            )),
            "budget_slopes": budget_slopes,
            "break_even_conditions": [
                {
                    "class_count": item["class_count"],
                    "candidate_budget": item["candidate_budget"],
                }
                for item in summaries if item["speedup_over_exhaustive"] > 1.0
            ],
            "maximum_speedup": float(max(
                item["speedup_over_exhaustive"] for item in summaries
            )),
        },
        "class_budget_summaries": summaries,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_certified_topk_routing(config)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
