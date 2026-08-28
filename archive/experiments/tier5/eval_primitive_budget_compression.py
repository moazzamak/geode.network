from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from experiments.tier5.eval_editability_scaling import _models, _predictions, _scores
from experiments.tier5.eval_exact_bound_routing import _percentiles
from src.model_compression import compress_primitive_budget


def run_primitive_budget_compression(config: dict) -> dict:
    records = []
    for primitive_count in config["primitives_per_class"]:
        budgets = sorted(set(max(1, int(np.ceil(
            primitive_count * ratio
        ))) for ratio in config["budget_ratios"]))
        for budget in budgets:
            for seed in config["seeds"]:
                condition = {
                    "class_count": config["class_count"],
                    "dimensions": config["dimensions"],
                    "primitives_per_class": primitive_count,
                }
                models = _models(condition, seed)
                rng = np.random.default_rng(seed)
                calibration_points = rng.normal(
                    scale=3.0,
                    size=(config["calibration_count"], config["dimensions"]),
                )
                confirmation_count = int(config.get("confirmation_count", 0))
                confirmation_points = (
                    rng.normal(
                        scale=3.0,
                        size=(confirmation_count, config["dimensions"]),
                    )
                    if confirmation_count else None
                )
                evaluation_points = rng.normal(
                    scale=3.0,
                    size=(config["evaluation_count"], config["dimensions"]),
                )
                baseline_scores, _ = _scores(models, evaluation_points)
                baseline_predictions = _predictions(baseline_scores)
                baseline_matrix = np.column_stack([
                    baseline_scores[class_id] for class_id in sorted(models)
                ])
                started = time.perf_counter()
                result = compress_primitive_budget(
                    models,
                    calibration_points,
                    budget,
                    minimum_prediction_agreement=config["minimum_calibration_agreement"],
                    maximum_score_drift=config["maximum_calibration_score_drift"],
                    confirmation_points=confirmation_points,
                )
                compression_seconds = time.perf_counter() - started
                compressed_scores, _ = _scores(result.models, evaluation_points)
                compressed_predictions = _predictions(compressed_scores)
                compressed_matrix = np.column_stack([
                    compressed_scores[class_id] for class_id in sorted(result.models)
                ])
                baseline_latencies = []
                compressed_latencies = []
                for _ in range(config["timing_repeats"]):
                    started = time.perf_counter()
                    _scores(models, evaluation_points)
                    baseline_latencies.append(time.perf_counter() - started)
                    started = time.perf_counter()
                    _scores(result.models, evaluation_points)
                    compressed_latencies.append(time.perf_counter() - started)
                initial_bytes = len(pickle.dumps(models, protocol=5))
                final_bytes = len(pickle.dumps(result.models, protocol=5))
                records.append({
                    "primitives_per_class": primitive_count,
                    "budget_per_class": budget,
                    "seed": seed,
                    "calibration_prediction_agreement": result.prediction_agreement,
                    "calibration_maximum_score_drift": result.maximum_score_drift,
                    "confirmation_prediction_agreement": (
                        result.confirmation_prediction_agreement
                    ),
                    "confirmation_maximum_score_drift": (
                        result.confirmation_maximum_score_drift
                    ),
                    "held_out_prediction_agreement": float(np.mean(
                        compressed_predictions == baseline_predictions
                    )),
                    "held_out_maximum_score_drift": float(np.max(np.abs(
                        compressed_matrix - baseline_matrix
                    ))),
                    "initial_primitive_count": result.initial_primitive_count,
                    "final_primitive_count": result.final_primitive_count,
                    "removals": result.removals,
                    "rollbacks": result.rollbacks,
                    "initial_snapshot_bytes": initial_bytes,
                    "final_snapshot_bytes": final_bytes,
                    "compression_seconds": compression_seconds,
                    "baseline_latency_seconds": _percentiles(baseline_latencies),
                    "compressed_latency_seconds": _percentiles(compressed_latencies),
                })

    summaries = []
    for primitive_count in config["primitives_per_class"]:
        for budget in sorted(set(max(1, int(np.ceil(
            primitive_count * ratio
        ))) for ratio in config["budget_ratios"])):
            cells = [
                record for record in records
                if record["primitives_per_class"] == primitive_count
                and record["budget_per_class"] == budget
            ]
            baseline_p50 = float(np.mean([
                cell["baseline_latency_seconds"]["p50"] for cell in cells
            ]))
            compressed_p50 = float(np.mean([
                cell["compressed_latency_seconds"]["p50"] for cell in cells
            ]))
            summaries.append({
                "primitives_per_class": primitive_count,
                "budget_per_class": budget,
                "minimum_held_out_agreement": float(min(
                    cell["held_out_prediction_agreement"] for cell in cells
                )),
                "maximum_held_out_score_drift": float(max(
                    cell["held_out_maximum_score_drift"] for cell in cells
                )),
                "mean_primitive_fraction": float(np.mean([
                    cell["final_primitive_count"] / cell["initial_primitive_count"]
                    for cell in cells
                ])),
                "mean_snapshot_fraction": float(np.mean([
                    cell["final_snapshot_bytes"] / cell["initial_snapshot_bytes"]
                    for cell in cells
                ])),
                "mean_rollback_count": float(np.mean([
                    cell["rollbacks"] for cell in cells
                ])),
                "mean_compression_seconds": float(np.mean([
                    cell["compression_seconds"] for cell in cells
                ])),
                "baseline_latency_p50_seconds": baseline_p50,
                "compressed_latency_p50_seconds": compressed_p50,
                "speedup_over_baseline": baseline_p50 / compressed_p50,
            })

    accepted = [
        item for item in summaries
        if item["minimum_held_out_agreement"] >= config["exit_minimum_agreement"]
        and item["maximum_held_out_score_drift"] <= config["exit_maximum_score_drift"]
        and (
            item["speedup_over_baseline"] > 1.0
            or item["mean_snapshot_fraction"] < 1.0
        )
    ]
    return {
        "milestone": "M12.6" if config.get("confirmation_count", 0) else "M12.5",
        "protocol": {
            "selection": "minimum held-out class-score contribution",
            "labels_used": False,
            "rollback_on_gate_failure": True,
            "calibration_and_evaluation_disjoint": True,
            "independent_confirmation_gate": bool(
                config.get("confirmation_count", 0)
            ),
            "source_models_mutated": False,
        },
        "summary": {
            "accepted_conditions": [
                {
                    "primitives_per_class": item["primitives_per_class"],
                    "budget_per_class": item["budget_per_class"],
                }
                for item in accepted
            ],
            "best_snapshot_reduction": float(max(
                1.0 - item["mean_snapshot_fraction"] for item in accepted
            )) if accepted else 0.0,
            "best_latency_speedup": float(max(
                item["speedup_over_baseline"] for item in accepted
            )) if accepted else 0.0,
        },
        "condition_summaries": summaries,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_primitive_budget_compression(config)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
