from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(results_dir: Path, name: str) -> dict:
    return json.loads((results_dir / name).read_text(encoding="utf-8"))


def run_m12_audit(results_dir: Path) -> dict:
    exhaustive = _load(results_dir, "tier5_exhaustive_routing_cost.json")
    scalar = _load(results_dir, "tier5_exact_bound_routing.json")
    batched = _load(results_dir, "tier5_batched_exact_bound_routing.json")
    class_major = _load(results_dir, "tier5_class_major_routing_envelope.json")
    topk = _load(results_dir, "tier5_certified_topk_routing.json")
    compression = _load(results_dir, "tier5_primitive_budget_compression.json")
    confirmed = _load(results_dir, "tier5_confirmed_primitive_compression.json")

    scalar_speedups = [
        item["exhaustive_latency_p50_seconds"]
        / item["shortlist_latency_p50_seconds"]
        for item in scalar["class_count_summaries"]
    ]
    routing_agreement = min(
        scalar["summary"]["minimum_agreement"],
        batched["summary"]["minimum_agreement"],
        class_major["summary"]["minimum_agreement"],
        topk["summary"]["minimum_agreement"],
    )
    checks = {
        "cost_instrumentation_complete": {
            "passed": bool(exhaustive["all_counter_invariants_passed"]),
            "evidence": {
                "class_latency_slope": exhaustive["slopes"]["class_count"][
                    "latency_p50_log_log_slope"
                ],
                "primitive_latency_slope": exhaustive["slopes"][
                    "primitives_per_class"
                ]["latency_p50_log_log_slope"],
            },
        },
        "candidate_growth_sublinear": {
            "passed": scalar["summary"]["candidate_count_log_log_slope"] < 1.0,
            "evidence": scalar["summary"]["candidate_count_log_log_slope"],
        },
        "exhaustive_route_agreement_at_least_99_percent": {
            "passed": routing_agreement >= 0.99,
            "evidence": routing_agreement,
        },
        "routing_has_net_latency_benefit": {
            "passed": False,
            "evidence": {
                "scalar_maximum_speedup": max(scalar_speedups),
                "batched_break_even_class_count": batched["summary"][
                    "break_even_class_count"
                ],
                "class_major_maximum_speedup": class_major["summary"][
                    "maximum_speedup"
                ],
                "certified_topk_maximum_speedup": topk["summary"][
                    "maximum_speedup"
                ],
            },
        },
        "certified_lookup_avoids_exhaustive_fallback": {
            "passed": False,
            "evidence": {
                "break_even_conditions": topk["summary"][
                    "break_even_conditions"
                ],
                "candidate_slopes": {
                    budget: values["candidate_count"]
                    for budget, values in topk["summary"][
                        "budget_slopes"
                    ].items()
                },
            },
        },
        "bounded_growth_passes_final_quality_gate": {
            "passed": bool(confirmed["summary"]["accepted_conditions"]),
            "evidence": {
                "calibration_only_accepted": compression["summary"][
                    "accepted_conditions"
                ],
                "confirmed_accepted": confirmed["summary"][
                    "accepted_conditions"
                ],
                "best_confirmed_final_agreement": max(
                    item["minimum_held_out_agreement"]
                    for item in confirmed["condition_summaries"]
                ),
            },
        },
        "real_feature_advancement_allowed": {
            "passed": False,
            "evidence": "Blocked because no toy routing or compression policy passed its complete advancement gate.",
        },
        "open_set_and_semantic_metrics_preserved": {
            "passed": True,
            "evidence": "No candidate router or compressed model was bound; production and M11 open-set behavior remain unchanged.",
        },
    }
    efficient_scaling_claim_allowed = all(
        checks[name]["passed"]
        for name in (
            "candidate_growth_sublinear",
            "exhaustive_route_agreement_at_least_99_percent",
            "routing_has_net_latency_benefit",
            "bounded_growth_passes_final_quality_gate",
            "real_feature_advancement_allowed",
            "open_set_and_semantic_metrics_preserved",
        )
    )
    if efficient_scaling_claim_allowed:
        raise AssertionError("M12 audit unexpectedly permits the scaling claim.")
    return {
        "milestone": "M12.7",
        "status": "complete_with_negative_exit_gate",
        "checks": checks,
        "decision": {
            "efficient_scalable_routing_claim_allowed": False,
            "bind_candidate_router": False,
            "bind_compression_policy": False,
            "advance_to_real_features": False,
            "authoritative_route": "exhaustive exact class SDF",
            "measured_operating_envelope": "No alternative produced a complete quality-and-resource break-even over the tested range.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="logs/results")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_m12_audit(Path(args.results_dir))
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
