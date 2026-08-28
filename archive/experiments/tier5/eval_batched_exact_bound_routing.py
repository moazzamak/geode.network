from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.tier5.eval_editability_scaling import _models, _predictions, _scores
from experiments.tier5.eval_exact_bound_routing import _percentiles, _slope
from src.candidate_routing import (
    batched_exact_bound_routing,
    exact_bound_routing,
)


def run_batched_exact_bound_routing(config: dict) -> dict:
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
            scalar = exact_bound_routing(models, points)
            batched = batched_exact_bound_routing(models, points)
            if not (
                np.array_equal(batched.predictions, exhaustive_predictions)
                and np.array_equal(batched.predictions, scalar.predictions)
                and np.allclose(batched.winning_scores, exhaustive_winners)
                and np.allclose(batched.winning_scores, scalar.winning_scores)
                and np.array_equal(batched.candidate_counts, scalar.candidate_counts)
            ):
                raise AssertionError("Batched exact routing diverged.")

            latencies = {"exhaustive": [], "scalar": [], "batched": []}
            for _ in range(config["timing_repeats"]):
                started = time.perf_counter()
                _scores(models, points)
                latencies["exhaustive"].append(time.perf_counter() - started)
                started = time.perf_counter()
                exact_bound_routing(models, points)
                latencies["scalar"].append(time.perf_counter() - started)
                started = time.perf_counter()
                batched_exact_bound_routing(models, points)
                latencies["batched"].append(time.perf_counter() - started)
            records.append({
                "class_count": class_count,
                "seed": seed,
                "agreement": 1.0,
                "maximum_winning_score_error": float(np.max(np.abs(
                    batched.winning_scores - exhaustive_winners
                ))),
                "mean_candidates_per_sample": float(np.mean(
                    batched.candidate_counts
                )),
                "candidate_fraction": float(np.sum(batched.candidate_counts)) / (
                    len(points) * class_count
                ),
                "latency_seconds": {
                    name: _percentiles(values)
                    for name, values in latencies.items()
                },
            })

    summaries = []
    for class_count in config["class_counts"]:
        cells = [record for record in records if record["class_count"] == class_count]
        summary = {
            "class_count": class_count,
            "agreement": 1.0,
            "maximum_winning_score_error": float(max(
                cell["maximum_winning_score_error"] for cell in cells
            )),
            "mean_candidates_per_sample": float(np.mean([
                cell["mean_candidates_per_sample"] for cell in cells
            ])),
            "candidate_fraction": float(np.mean([
                cell["candidate_fraction"] for cell in cells
            ])),
        }
        for variant in ("exhaustive", "scalar", "batched"):
            summary[f"{variant}_latency_p50_seconds"] = float(np.mean([
                cell["latency_seconds"][variant]["p50"] for cell in cells
            ]))
            summary[f"{variant}_latency_p95_seconds"] = float(np.mean([
                cell["latency_seconds"][variant]["p95"] for cell in cells
            ]))
        summary["batched_speedup_over_scalar"] = (
            summary["scalar_latency_p50_seconds"]
            / summary["batched_latency_p50_seconds"]
        )
        summary["batched_speedup_over_exhaustive"] = (
            summary["exhaustive_latency_p50_seconds"]
            / summary["batched_latency_p50_seconds"]
        )
        summaries.append(summary)

    break_even = [
        item["class_count"] for item in summaries
        if item["batched_speedup_over_exhaustive"] > 1.0
    ]
    class_counts = [item["class_count"] for item in summaries]
    return {
        "milestone": "M12.2",
        "protocol": {
            "certificate_identical_to_m12_1": True,
            "active_pairs_grouped_by_class": True,
            "exhaustive_authoritative": True,
            "approximate_index_used": False,
        },
        "summary": {
            "minimum_agreement": 1.0,
            "maximum_winning_score_error": float(max(
                item["maximum_winning_score_error"] for item in summaries
            )),
            "candidate_count_log_log_slope": _slope(
                class_counts,
                [item["mean_candidates_per_sample"] for item in summaries],
            ),
            "batched_latency_log_log_slope": _slope(
                class_counts,
                [item["batched_latency_p50_seconds"] for item in summaries],
            ),
            "exhaustive_latency_log_log_slope": _slope(
                class_counts,
                [item["exhaustive_latency_p50_seconds"] for item in summaries],
            ),
            "break_even_class_count": min(break_even) if break_even else None,
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
    result = run_batched_exact_bound_routing(config)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
