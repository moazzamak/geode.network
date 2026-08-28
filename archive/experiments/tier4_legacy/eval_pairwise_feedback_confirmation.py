from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.tier4.eval_pairwise_feedback_robustness import (
    _perturb_constraints,
)
from experiments.tier4.eval_pairwise_partition_refinement import (
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
    confirm_pairwise_constraints,
    select_constraint_queries,
    validate_pairwise_constraints,
)


def _relation_by_pair(constraints: tuple) -> dict[tuple[int, int], str]:
    return {
        tuple(sorted((item.left_record_id, item.right_record_id))): item.relation
        for item in constraints
    }


def run_feedback_confirmation_study(
    *,
    dataset_path: str,
    episodes: list[dict],
    seeds: list[int],
    query_budget: int,
    answer_error_rates: list[float],
    flag_fraction: float = 0.3,
    hdbscan_minimum_cluster_size: int = 3,
    hdbscan_minimum_samples: int | None = 3,
    samples_per_slice: int = 100,
    pca_components: int = 8,
    representation: str = "mobilenetv2",
) -> dict:
    cells = []
    for seed in seeds:
        for episode_index, episode_config in enumerate(episodes):
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
            selected = select_constraint_queries(
                _constraints_from_review(payload, baseline),
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
            true_relations = _relation_by_pair(selected)
            variants = {}
            for rate_index, error_rate in enumerate(answer_error_rates):
                base_state = seed * 1000 + episode_index * 100 + rate_index
                first_answers = _perturb_constraints(
                    selected,
                    mode="flipped",
                    rate=error_rate,
                    random_state=base_state,
                )
                second_answers = _perturb_constraints(
                    selected,
                    mode="flipped",
                    rate=error_rate,
                    random_state=base_state + 100_000,
                )
                single_consistency = validate_pairwise_constraints(first_answers)
                single_constraints = (
                    first_answers if single_consistency.is_consistent else ()
                )
                confirmation = confirm_pairwise_constraints(
                    first_answers,
                    second_answers,
                )
                confirmed_consistency = validate_pairwise_constraints(
                    confirmation.accepted
                )
                confirmed_constraints = (
                    confirmation.accepted
                    if confirmed_consistency.is_consistent else ()
                )
                single = evaluate_event_review_payload(
                    payload,
                    known_split="id_validation",
                    unknown_split="proxy_unknown",
                    partition_constraints=single_constraints,
                    **common,
                )
                confirmed = evaluate_event_review_payload(
                    payload,
                    known_split="id_validation",
                    unknown_split="proxy_unknown",
                    partition_constraints=confirmed_constraints,
                    **common,
                )
                harmful_accepted = sum(
                    true_relations[pair] != relation
                    for pair, relation in _relation_by_pair(
                        confirmation.accepted
                    ).items()
                )
                variants[f"error:{error_rate:g}"] = {
                    "initial_query_count": len(selected),
                    "total_response_count": 2 * len(selected),
                    "accepted_constraint_count": len(confirmation.accepted),
                    "disagreement_count": confirmation.disagreement_count,
                    "harmful_accepted_count": harmful_accepted,
                    "single_quarantined": not single_consistency.is_consistent,
                    "confirmed_quarantined": not confirmed_consistency.is_consistent,
                    "single": single["metrics"],
                    "confirmed": confirmed["metrics"],
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
                "baseline": baseline["metrics"],
                "clean_active": clean["metrics"],
                "variants": variants,
                "final_reference": final_reference["metrics"],
            })

    summaries = {}
    for variant_name in cells[0]["variants"]:
        variants = [cell["variants"][variant_name] for cell in cells]
        summaries[variant_name] = {
            "mean_initial_query_count": float(np.mean([
                item["initial_query_count"] for item in variants
            ])),
            "mean_total_response_count": float(np.mean([
                item["total_response_count"] for item in variants
            ])),
            "mean_accepted_constraint_count": float(np.mean([
                item["accepted_constraint_count"] for item in variants
            ])),
            "mean_disagreement_count": float(np.mean([
                item["disagreement_count"] for item in variants
            ])),
            "abstention_rate": float(np.mean([
                item["disagreement_count"] / item["initial_query_count"]
                for item in variants
            ])),
            "mean_harmful_accepted_count": float(np.mean([
                item["harmful_accepted_count"] for item in variants
            ])),
            "harmful_acceptance_cell_rate": float(np.mean([
                item["harmful_accepted_count"] > 0 for item in variants
            ])),
            "single_quarantine_rate": float(np.mean([
                item["single_quarantined"] for item in variants
            ])),
            "confirmed_quarantine_rate": float(np.mean([
                item["confirmed_quarantined"] for item in variants
            ])),
            "single": _mean_metrics([item["single"] for item in variants]),
            "confirmed": _mean_metrics([
                item["confirmed"] for item in variants
            ]),
        }
    return {
        "protocol": {
            "query_budget": query_budget,
            "independent_confirmation": True,
            "acceptance_rule": "accept_only_matching_answers",
            "graph_consistency_checked_after_confirmation": True,
            "constraints_from_proxy_reviews_only": True,
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
    result = run_feedback_confirmation_study(**config)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
