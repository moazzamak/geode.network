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
from experiments.tier4.eval_real_feature_event_review import (
    evaluate_event_review_payload,
)
from experiments.tier4.eval_real_feature_ood_transfer import (
    run_real_feature_ood_episode,
)
from src.feedback_constraints import select_constraint_queries


def _normalized_proxy_embeddings(payload: dict) -> dict[int, np.ndarray]:
    embeddings = np.vstack((
        np.asarray(payload["id_validation_representation_embeddings"]),
        np.asarray(payload["proxy_unknown_representation_embeddings"]),
    )).astype(np.float64)
    embeddings /= np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        np.finfo(float).eps,
    )
    return {index: embedding for index, embedding in enumerate(embeddings)}


def run_query_budget_study(
    *,
    dataset_path: str,
    episodes: list[dict],
    seeds: list[int],
    query_budgets: list[int],
    flag_fraction: float = 0.3,
    hdbscan_minimum_cluster_size: int = 3,
    hdbscan_minimum_samples: int | None = 3,
    samples_per_slice: int = 100,
    pca_components: int = 8,
    representation: str = "mobilenetv2",
) -> dict:
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
            constraints = _constraints_from_review(payload, baseline)
            embeddings = _normalized_proxy_embeddings(payload)
            variants = {}
            for strategy in ("active", "random"):
                for budget in query_budgets:
                    selected = select_constraint_queries(
                        constraints,
                        embeddings,
                        tuple(tuple(group) for group in baseline[
                            "partition_source_sample_ids"
                        ]),
                        budget=budget,
                        strategy=strategy,
                        random_state=seed,
                    )
                    result = evaluate_event_review_payload(
                        payload,
                        known_split="id_validation",
                        unknown_split="proxy_unknown",
                        partition_constraints=selected,
                        **common,
                    )
                    variants[f"{strategy}:{budget}"] = {
                        "query_count": len(selected),
                        "metrics": result["metrics"],
                    }
            dense = evaluate_event_review_payload(
                payload,
                known_split="id_validation",
                unknown_split="proxy_unknown",
                partition_constraints=constraints,
                **common,
            )
            variants["dense"] = {
                "query_count": len(constraints),
                "metrics": dense["metrics"],
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
                "variants": variants,
                "final_reference": final_reference["metrics"],
            })

    baseline_summary = _mean_metrics([cell["baseline"] for cell in cells])
    variant_names = list(cells[0]["variants"])
    summaries = {}
    for variant_name in variant_names:
        summary = _mean_metrics([
            cell["variants"][variant_name]["metrics"] for cell in cells
        ])
        mean_queries = float(np.mean([
            cell["variants"][variant_name]["query_count"] for cell in cells
        ]))
        summaries[variant_name] = {
            "mean_query_count": mean_queries,
            **summary,
            "ari_gain_per_query": (
                (summary["unknown_group_ari"]
                 - baseline_summary["unknown_group_ari"])
                / mean_queries
            ),
            "duplicate_reduction_per_query": (
                (baseline_summary["duplicate_review_rate"]
                 - summary["duplicate_review_rate"])
                / mean_queries
            ),
        }
    return {
        "protocol": {
            "query_selection_uses_constraint_answers": False,
            "constraints_from_proxy_reviews_only": True,
            "final_partition_refined": False,
            "final_labels_used_for_selection_or_constraints": False,
            "review_only": True,
            "mutation_published": False,
        },
        "proxy_baseline_summary": baseline_summary,
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
    result = run_query_budget_study(**config)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()