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
    confirm_pairwise_constraints,
    select_constraint_queries,
    validate_pairwise_constraints,
)


def _corrupt_answers(
    constraints: tuple[PairwiseConstraint, ...],
    *,
    error_rate: float,
    random_state: int,
) -> tuple[PairwiseConstraint, ...]:
    if not 0.0 <= error_rate <= 1.0:
        raise ValueError("error_rate must be between zero and one.")
    flips = np.random.default_rng(random_state).random(len(constraints)) < error_rate
    return tuple(
        PairwiseConstraint(
            constraint.left_record_id,
            constraint.right_record_id,
            "cannot_link" if constraint.relation == "must_link" else "must_link",
        ) if flips[index] else constraint
        for index, constraint in enumerate(constraints)
    )


def _relation_by_pair(constraints: tuple) -> dict[tuple[int, int], str]:
    return {
        tuple(sorted((item.left_record_id, item.right_record_id))): item.relation
        for item in constraints
    }


def _distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
    }


def run_repeated_confirmation_study(
    *,
    dataset_path: str,
    episodes: list[dict],
    seeds: list[int],
    query_budget: int,
    answer_error_rates: list[float],
    simulation_repeats: int,
    simulation_seed: int,
    flag_fraction: float = 0.3,
    hdbscan_minimum_cluster_size: int = 3,
    hdbscan_minimum_samples: int | None = 3,
    samples_per_slice: int = 100,
    pca_components: int = 8,
    representation: str = "mobilenetv2",
) -> dict:
    if simulation_repeats <= 0:
        raise ValueError("simulation_repeats must be positive.")
    cells = []
    observations = {f"error:{rate:g}": [] for rate in answer_error_rates}
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
            for rate_index, error_rate in enumerate(answer_error_rates):
                variant_name = f"error:{error_rate:g}"
                for repeat in range(simulation_repeats):
                    base_state = (
                        simulation_seed
                        + seed * 1_000_000
                        + episode_index * 10_000
                        + rate_index * 1_000
                        + repeat * 2
                    )
                    first = _corrupt_answers(
                        selected,
                        error_rate=error_rate,
                        random_state=base_state,
                    )
                    second = _corrupt_answers(
                        selected,
                        error_rate=error_rate,
                        random_state=base_state + 1,
                    )
                    single_consistency = validate_pairwise_constraints(first)
                    single_constraints = first if single_consistency.is_consistent else ()
                    confirmation = confirm_pairwise_constraints(first, second)
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
                    )["metrics"]
                    confirmed = evaluate_event_review_payload(
                        payload,
                        known_split="id_validation",
                        unknown_split="proxy_unknown",
                        partition_constraints=confirmed_constraints,
                        **common,
                    )["metrics"]
                    harmful_accepted = sum(
                        true_relations[pair] != relation
                        for pair, relation in _relation_by_pair(
                            confirmation.accepted
                        ).items()
                    )
                    observations[variant_name].append({
                        "baseline": baseline["metrics"],
                        "single": single,
                        "confirmed": confirmed,
                        "accepted_constraint_count": len(confirmation.accepted),
                        "disagreement_count": confirmation.disagreement_count,
                        "harmful_accepted_count": harmful_accepted,
                        "single_quarantined": not single_consistency.is_consistent,
                        "confirmed_quarantined": not confirmed_consistency.is_consistent,
                    })
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
                "final_reference": final_reference["metrics"],
            })

    summaries = {}
    for variant_name, runs in observations.items():
        summaries[variant_name] = {
            "observation_count": len(runs),
            "mean_total_response_count": float(2 * query_budget),
            "mean_accepted_constraint_count": float(np.mean([
                run["accepted_constraint_count"] for run in runs
            ])),
            "mean_abstention_rate": float(np.mean([
                run["disagreement_count"] / query_budget for run in runs
            ])),
            "harmful_acceptance_observation_rate": float(np.mean([
                run["harmful_accepted_count"] > 0 for run in runs
            ])),
            "mean_harmful_accepted_count": float(np.mean([
                run["harmful_accepted_count"] for run in runs
            ])),
            "single_quarantine_rate": float(np.mean([
                run["single_quarantined"] for run in runs
            ])),
            "confirmed_quarantine_rate": float(np.mean([
                run["confirmed_quarantined"] for run in runs
            ])),
            "single_mean_metrics": _mean_metrics([
                run["single"] for run in runs
            ]),
            "confirmed_mean_metrics": _mean_metrics([
                run["confirmed"] for run in runs
            ]),
            "single_delta_distributions": {
                metric: _distribution([
                    run["single"][metric] - run["baseline"][metric]
                    for run in runs
                ])
                for metric in METRICS
            },
            "confirmed_delta_distributions": {
                metric: _distribution([
                    run["confirmed"][metric] - run["baseline"][metric]
                    for run in runs
                ])
                for metric in METRICS
            },
        }
    return {
        "protocol": {
            "query_budget": query_budget,
            "simulation_repeats_per_cell": simulation_repeats,
            "simulation_seed": simulation_seed,
            "independent_bernoulli_answer_errors": True,
            "acceptance_rule": "accept_only_matching_answers",
            "paired_deltas_against_cell_baseline": True,
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
    result = run_repeated_confirmation_study(**config)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
