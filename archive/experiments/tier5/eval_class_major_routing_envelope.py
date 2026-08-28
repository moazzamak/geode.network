from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.tier5.eval_editability_scaling import _models, _predictions, _scores
from experiments.tier5.eval_exact_bound_routing import _percentiles
from experiments.tier5.eval_exhaustive_routing_cost import AXES, _conditions
from src.candidate_routing import class_major_exact_bound_routing


def _slope(records: list[dict], axis: str, key: str) -> float:
    selected = [
        record for record in records
        if record["condition"]["axis"] in {"baseline", axis}
    ]
    values = {}
    for record in selected:
        values[record["condition"][axis]] = record[key]
    x = np.asarray(sorted(values), dtype=np.float64)
    y = np.asarray([values[value] for value in x], dtype=np.float64)
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def run_class_major_routing_envelope(config: dict) -> dict:
    cell_records = []
    condition_config = {
        "baseline": config["baseline"],
        "sweeps": config["sweeps"],
    }
    for condition in _conditions(condition_config):
        for seed in config["seeds"]:
            models = _models(condition, int(seed))
            points = np.random.default_rng(seed).normal(
                scale=3.0,
                size=(condition["batch_size"], condition["dimensions"]),
            )
            exhaustive_scores, _ = _scores(models, points)
            exhaustive_predictions = _predictions(exhaustive_scores)
            exhaustive_winners = np.min(np.column_stack([
                exhaustive_scores[class_id] for class_id in sorted(models)
            ]), axis=1)
            result = class_major_exact_bound_routing(models, points)
            agreement = float(np.mean(result.predictions == exhaustive_predictions))
            score_error = float(np.max(np.abs(
                result.winning_scores - exhaustive_winners
            )))
            if agreement != 1.0 or score_error > 1e-12:
                raise AssertionError("Class-major routing diverged from exhaustive.")

            exhaustive_latencies = []
            routed_latencies = []
            for _ in range(config["timing_repeats"]):
                started = time.perf_counter()
                _scores(models, points)
                exhaustive_latencies.append(time.perf_counter() - started)
                started = time.perf_counter()
                class_major_exact_bound_routing(models, points)
                routed_latencies.append(time.perf_counter() - started)
            cell_records.append({
                "condition": condition,
                "seed": seed,
                "agreement": agreement,
                "maximum_winning_score_error": score_error,
                "mean_candidates_per_sample": float(np.mean(
                    result.candidate_counts
                )),
                "candidate_fraction": float(np.sum(result.candidate_counts)) / (
                    len(points) * condition["class_count"]
                ),
                "mean_primitives_per_sample": float(np.mean(
                    result.primitive_evaluation_counts
                )),
                "exhaustive_latency_seconds": _percentiles(exhaustive_latencies),
                "routed_latency_seconds": _percentiles(routed_latencies),
            })

    records = []
    for condition in _conditions(condition_config):
        cells = [
            cell for cell in cell_records
            if cell["condition"]["name"] == condition["name"]
        ]
        exhaustive_p50 = float(np.mean([
            cell["exhaustive_latency_seconds"]["p50"] for cell in cells
        ]))
        routed_p50 = float(np.mean([
            cell["routed_latency_seconds"]["p50"] for cell in cells
        ]))
        records.append({
            "condition": condition,
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
            "mean_primitives_per_sample": float(np.mean([
                cell["mean_primitives_per_sample"] for cell in cells
            ])),
            "exhaustive_latency_p50_seconds": exhaustive_p50,
            "routed_latency_p50_seconds": routed_p50,
            "speedup_over_exhaustive": exhaustive_p50 / routed_p50,
        })

    return {
        "milestone": "M12.3",
        "protocol": {
            "design": "one_factor_at_a_time",
            "schedule": "class_major_at_most_one_exact_call_per_class",
            "exhaustive_authoritative": True,
            "approximate_index_used": False,
        },
        "summary": {
            "minimum_agreement": 1.0,
            "maximum_winning_score_error": float(max(
                record["maximum_winning_score_error"] for record in records
            )),
            "slopes": {
                axis: {
                    "candidate_count": _slope(
                        records, axis, "mean_candidates_per_sample",
                    ),
                    "routed_latency": _slope(
                        records, axis, "routed_latency_p50_seconds",
                    ),
                    "exhaustive_latency": _slope(
                        records, axis, "exhaustive_latency_p50_seconds",
                    ),
                }
                for axis in AXES
            },
            "break_even_conditions": [
                record["condition"]["name"] for record in records
                if record["speedup_over_exhaustive"] > 1.0
            ],
            "maximum_speedup": float(max(
                record["speedup_over_exhaustive"] for record in records
            )),
        },
        "condition_summaries": records,
        "cells": cell_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_class_major_routing_envelope(config)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
