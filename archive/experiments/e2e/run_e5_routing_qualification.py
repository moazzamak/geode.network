"""Measure shadow routers against the deployed E4 exhaustive oracle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from experiments.common.experiment_manifest import array_fingerprint
from experiments.e2e.e4_cifar_protocol import (
    build_id_partitions,
    load_config,
    load_e4_data,
)
from experiments.e2e.e5_bundle_loader import load_e4_candidate
from src.candidate_routing import (
    CertifiedTopKRouter,
    batched_exact_bound_routing,
    class_major_exact_bound_routing,
    exact_bound_routing,
)
from src.shadow_routing import run_shadow_router


def _percentiles(values: list[float]) -> dict[str, float]:
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def _stage_timings(model, raw_features: np.ndarray, repeats: int) -> dict:
    transform_times = []
    exhaustive_times = []
    readout_times = []
    end_to_end_times = []
    for _ in range(repeats):
        started = time.perf_counter()
        transformed = model.transform.transform(raw_features)
        transform_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        scores = model.raw_scores(transformed)
        exhaustive_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        model.readout.predict_proba(scores)
        readout_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        transformed = model.transform.transform(raw_features)
        scores = model.raw_scores(transformed)
        model.readout.predict_proba(scores)
        end_to_end_times.append(time.perf_counter() - started)
    return {
        "transform": _percentiles(transform_times),
        "exhaustive_class_fields": _percentiles(exhaustive_times),
        "multinomial_readout": _percentiles(readout_times),
        "end_to_end": _percentiles(end_to_end_times),
    }


def _router_factories(models):
    return {
        "exact_bound": exact_bound_routing,
        "batched_exact_bound": batched_exact_bound_routing,
        "class_major_exact_bound": class_major_exact_bound_routing,
        "certified_top5": lambda current_models, points: CertifiedTopKRouter(
            current_models, candidate_budget=min(5, len(current_models)),
        ).route(points),
    }


def run_qualification(
    *,
    config_path: Path,
    bundle_root: Path,
    batch_sizes: tuple[int, ...] = (1, 32, 256, 1024),
    timing_repeats: int = 5,
) -> dict:
    config = load_config(config_path)
    data = load_e4_data(config)
    partitions, _ = build_id_partitions(data, config, config["deployment_seed"])
    final_indices = partitions["final_test"]
    model = load_e4_candidate(bundle_root)
    expected = json.loads(
        (bundle_root / "bundles" / model.manifest.bundle_id / "components"
         / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    expected_record = next(
        record for record in expected["records"]
        if record["seed"] == config["deployment_seed"]
    )
    actual_prediction_hash = array_fingerprint(
        model.predict(data.id_features[final_indices])
    )
    expected_prediction_hash = expected_record["prediction_hashes"]["geode_multinomial"]
    if actual_prediction_hash != expected_prediction_hash:
        raise ValueError("loaded candidate does not reproduce its E4 prediction hash")

    records = []
    transformed_final = model.transform.transform(data.id_features[final_indices])
    primitive_count = sum(
        len(expert.ellipsoids)
        for experts in model.class_models.values()
        for expert in experts
    )
    for batch_size in batch_sizes:
        points = transformed_final[:batch_size]
        raw_features = data.id_features[final_indices[:batch_size]]
        stage_timings = _stage_timings(model, raw_features, timing_repeats)
        for router_name, router in _router_factories(model.class_models).items():
            observation = run_shadow_router(
                model.class_models,
                points,
                router,
                router_name=router_name,
                alpha=model.alpha,
                timing_repeats=timing_repeats,
                score_scales=model.score_scales,
            )
            per_class_agreement = {}
            for class_id in model.class_ids:
                selected = observation.authoritative_predictions == class_id
                if np.any(selected):
                    per_class_agreement[str(int(class_id))] = float(np.mean(
                        observation.candidate_predictions[selected]
                        == observation.authoritative_predictions[selected]
                    ))
            candidate_fraction = (
                observation.candidate_counters.exact_class_sdf_pairs
                / observation.oracle_counters.exact_class_sdf_pairs
            )
            primitive_fraction = (
                observation.candidate_counters.primitive_sdf_pairs
                / observation.oracle_counters.primitive_sdf_pairs
            )
            latency_ratio = (
                observation.candidate_latency_seconds["p95"]
                / observation.oracle_latency_seconds["p95"]
            )
            records.append({
                "batch_size": batch_size,
                "router": router_name,
                "oracle_agreement": observation.agreement,
                "maximum_winning_score_error": observation.maximum_winning_score_error,
                "minimum_class_agreement": min(per_class_agreement.values()),
                "worst_classes": [
                    class_id for class_id, agreement in per_class_agreement.items()
                    if agreement == min(per_class_agreement.values())
                ],
                "candidate_fraction": candidate_fraction,
                "primitive_fraction": primitive_fraction,
                "fallback_rate": observation.fallback_samples / batch_size,
                "bound_distance_pairs": observation.bound_distance_pairs,
                "oracle_counters": asdict(observation.oracle_counters),
                "candidate_counters": asdict(observation.candidate_counters),
                "oracle_latency_seconds": observation.oracle_latency_seconds,
                "candidate_latency_seconds": observation.candidate_latency_seconds,
                "candidate_to_oracle_p95_latency_ratio": latency_ratio,
                "complete_pipeline_stage_latency_seconds": stage_timings,
                "quality_gate_passed": observation.quality_gate_passed,
                "latency_gate_passed": observation.latency_gate_passed,
                "calibration_ood_gate_passed": False,
                "calibration_ood_gate_reason": (
                    "E4 multinomial readout requires the complete class-score vector"
                ),
                "candidate_controls_outputs": observation.candidate_controls_outputs,
                "promotion_eligible": False,
            })

    oracle_reported_every_run = all(
        "oracle_agreement" in record and "oracle_counters" in record
        for record in records
    )
    no_candidate_controls_outputs = all(
        not record["candidate_controls_outputs"] for record in records
    )
    return {
        "schema_version": 1,
        "milestone": "E5",
        "bundle_id": model.manifest.bundle_id,
        "bundle_prediction_hash_verified": True,
        "model": {
            "class_count": len(model.class_models),
            "primitive_count": primitive_count,
            "dimensions": transformed_final.shape[1],
            "score_normalization": "deployed per-class score scales",
            "readout": "multinomial complete-score-vector",
        },
        "protocol": {
            "authoritative_route": "exhaustive_vectorized_class_fields",
            "shadow_only": True,
            "batch_sizes": list(batch_sizes),
            "timing_repeats": timing_repeats,
            "latency_includes_candidate_lookup_bounds_and_fallback": True,
            "candidate_score_scales_modified": False,
        },
        "summary": {
            "oracle_agreement_reported_every_run": oracle_reported_every_run,
            "no_candidate_controls_outputs": no_candidate_controls_outputs,
            "candidate_count": len(records),
            "quality_pass_count": sum(
                record["quality_gate_passed"] for record in records
            ),
            "latency_pass_count": sum(
                record["latency_gate_passed"] for record in records
            ),
            "promotion_eligible_count": 0,
            "authoritative_route_retained": True,
            "gate_passed": (
                oracle_reported_every_run and no_candidate_controls_outputs
            ),
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("experiments/configs/e4_cifar_qualification.json"),
    )
    parser.add_argument(
        "--bundle-root", type=Path,
        default=Path("logs/results/e4_model_registry"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("logs/results/e5_routing_qualification.json"),
    )
    parser.add_argument("--timing-repeats", type=int, default=5)
    arguments = parser.parse_args()
    result = run_qualification(
        config_path=arguments.config,
        bundle_root=arguments.bundle_root,
        timing_repeats=arguments.timing_repeats,
    )
    if not result["summary"]["gate_passed"]:
        raise RuntimeError(f"E5 routing gate failed: {result['summary']}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()