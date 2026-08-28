from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.tier4.eval_pairwise_partition_refinement import (
    METRICS,
    _constraints_from_review,
    _mean_metrics,
)
from experiments.tier4.eval_pairwise_query_budget import (
    _normalized_proxy_embeddings,
)
from experiments.tier4.eval_real_feature_event_review import (
    evaluate_event_review_payload,
)
from experiments.tier4.eval_real_feature_ood_transfer import (
    run_real_feature_ood_episode,
)
from src.feedback_constraints import (
    PairwiseConstraint,
    select_constraint_queries,
    validate_pairwise_constraints,
)


def _opposite(relation: str) -> str:
    return "cannot_link" if relation == "must_link" else "must_link"


def _perturb_constraints(
    constraints: tuple[PairwiseConstraint, ...],
    *,
    mode: str,
    rate: float,
    random_state: int,
) -> tuple[PairwiseConstraint, ...]:
    if mode not in {"missing", "flipped", "contradictory"}:
        raise ValueError("mode must be missing, flipped, or contradictory.")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between zero and one.")
    if not constraints or rate == 0.0:
        return constraints

    count = min(len(constraints), max(1, int(round(len(constraints) * rate))))
    selected_indices = set(
        int(index) for index in np.random.default_rng(random_state).permutation(
            len(constraints)
        )[:count]
    )
    if mode == "missing":
        return tuple(
            constraint for index, constraint in enumerate(constraints)
            if index not in selected_indices
        )
    changed = tuple(
        PairwiseConstraint(
            constraint.left_record_id,
            constraint.right_record_id,
            _opposite(constraint.relation),
        ) if index in selected_indices else constraint
        for index, constraint in enumerate(constraints)
    )
    if mode == "flipped":
        return changed
    contradictions = tuple(
        changed[index] for index in sorted(selected_indices)
    )
    return constraints + contradictions


def run_feedback_robustness_study(
    *,
    dataset_path: str,
    episodes: list[dict],
    seeds: list[int],
    query_budget: int,
    missing_rates: list[float],
    flip_rates: list[float],
    contradiction_rates: list[float],
    flag_fraction: float = 0.3,
    hdbscan_minimum_cluster_size: int = 3,
    hdbscan_minimum_samples: int | None = 3,
    samples_per_slice: int = 100,
    pca_components: int = 8,
    representation: str = "mobilenetv2",
) -> dict:
    scenarios = [
        (mode, rate)
        for mode, rates in (
            ("missing", missing_rates),
            ("flipped", flip_rates),
            ("contradictory", contradiction_rates),
        )
        for rate in rates
    ]
    cells = []
    for seed in seeds:
        for episode_config in episodes:
            episode = run_real_feature_ood_episode(
                dataset_path=dataset_path,
                known_classes=tuple(episode_config["known_classes"]),
                proxy_unknown_classes=tuple(episode_config["proxy_unknown_classes"]),
                final_unknown_classes=tuple(episode_config["final_unknown_classes"]),
                seed=seed,
                samples_per_slice=samples_per_slice,
                pca_components=pca_components,
                representation=representation,
                include_score_payload=True,
            )
            payload = episode["score_payload"]
            common = {
                "flag_fraction": flag_fraction,
                "embedding_space": "representation_l2",
                "clustering_method": "hdbscan",
                "hdbscan_minimum_cluster_size": hdbscan_minimum_cluster_size,
                "hdbscan_minimum_samples": hdbscan_minimum_samples,
            }
            baseline = evaluate_event_review_payload(
                payload,
                known_split="id_validation",
                unknown_split="proxy_unknown",
                **common,
            )
            all_constraints = _constraints_from_review(payload, baseline)
            selected = select_constraint_queries(
                all_constraints,
                _normalized_proxy_embeddings(payload),
                tuple(tuple(group) for group in baseline[
                    "partition_source_sample_ids"
                ]),
                budget=query_budget,
                strategy="active",
                random_state=seed,
            )
            clean = evaluate_event_review_payload(
                payload,
                known_split="id_validation",
                unknown_split="proxy_unknown",
                partition_constraints=selected,
                **common,
            )
            variants = {}
            for scenario_index, (mode, rate) in enumerate(scenarios):
                constraints = _perturb_constraints(
                    selected,
                    mode=mode,
                    rate=rate,
                    random_state=seed * 1000 + scenario_index,
                )
                consistency = validate_pairwise_constraints(constraints)
                naive = evaluate_event_review_payload(
                    payload,
                    known_split="id_validation",
                    unknown_split="proxy_unknown",
                    partition_constraints=constraints,
                    **common,
                )
                guarded_constraints = constraints if consistency.is_consistent else ()
                guarded = evaluate_event_review_payload(
                    payload,
                    known_split="id_validation",
                    unknown_split="proxy_unknown",
                    partition_constraints=guarded_constraints,
                    **common,
                )
                variants[f"{mode}:{rate:g}"] = {
                    "returned_answer_count": len(constraints),
                    "quarantined": not consistency.is_consistent,
                    "direct_conflict_count": consistency.direct_conflict_count,
                    "transitive_conflict_count": consistency.transitive_conflict_count,
                    "naive": naive["metrics"],
                    "guarded": guarded["metrics"],
                }
            final_reference = evaluate_event_review_payload(
                payload,
                known_split="id_test",
                unknown_split="final_unknown",
                **common,
            )
            cells.append({
                "seed": seed,
                "known_classes": episode_config["known_classes"],
                "query_count": len(selected),
                "baseline": baseline["metrics"],
                "clean_active": clean["metrics"],
                "variants": variants,
                "final_reference": final_reference["metrics"],
            })

    summaries = {}
    for variant_name in cells[0]["variants"]:
        variant_cells = [cell["variants"][variant_name] for cell in cells]
        summaries[variant_name] = {
            "mean_returned_answer_count": float(np.mean([
                variant["returned_answer_count"] for variant in variant_cells
            ])),
            "quarantine_rate": float(np.mean([
                variant["quarantined"] for variant in variant_cells
            ])),
            "mean_direct_conflict_count": float(np.mean([
                variant["direct_conflict_count"] for variant in variant_cells
            ])),
            "mean_transitive_conflict_count": float(np.mean([
                variant["transitive_conflict_count"] for variant in variant_cells
            ])),
            "naive": _mean_metrics([
                variant["naive"] for variant in variant_cells
            ]),
            "guarded": _mean_metrics([
                variant["guarded"] for variant in variant_cells
            ]),
        }
    return {
        "protocol": {
            "query_budget": query_budget,
            "query_selection_uses_constraint_answers": False,
            "perturbation_axes_varied_independently": True,
            "constraints_from_proxy_reviews_only": True,
            "guard_policy": "quarantine_entire_inconsistent_batch",
            "experimental_constraints_persisted": False,
            "final_partition_refined": False,
            "final_labels_used_for_selection_or_constraints": False,
            "mutation_published": False,
        },
        "proxy_baseline_summary": _mean_metrics([
            cell["baseline"] for cell in cells
        ]),
        "clean_active_summary": _mean_metrics([
            cell["clean_active"] for cell in cells
        ]),
        "variant_summaries": summaries,
        "final_unconstrained_reference": _mean_metrics([
            cell["final_reference"] for cell in cells
        ]),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_feedback_robustness_study(**config)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
