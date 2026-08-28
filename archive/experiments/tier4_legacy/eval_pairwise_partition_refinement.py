from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.tier4.eval_real_feature_event_review import (
    evaluate_event_review_payload,
)
from experiments.tier4.eval_real_feature_ood_transfer import (
    run_real_feature_ood_episode,
)
from src.feedback_constraints import build_pairwise_constraints


METRICS = (
    "accumulated_event_recall",
    "distinct_group_recall",
    "unknown_group_ari",
    "mean_cluster_purity",
    "reviews_per_1000",
    "duplicate_review_rate",
    "partition_group_count",
)


def _mean_metrics(runs: list[dict]) -> dict:
    return {
        name: float(np.mean([run[name] for run in runs]))
        for name in METRICS
    }


def _constraints_from_review(payload: dict, review: dict) -> tuple:
    reviewed_source_ids = np.asarray(sorted({
        source_id
        for group in review["reviews"]
        for source_id in group["source_sample_ids"]
    }), dtype=np.int64)
    labels = np.concatenate((
        np.asarray(payload["id_validation_labels"], dtype=np.int64),
        np.asarray(payload["proxy_unknown_labels"], dtype=np.int64),
    ))
    return build_pairwise_constraints(
        reviewed_source_ids,
        labels[reviewed_source_ids],
    )


def run_pairwise_partition_refinement(
    *,
    dataset_path: str,
    episodes: list[dict],
    seeds: list[int],
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
            refined = evaluate_event_review_payload(
                payload,
                known_split="id_validation",
                unknown_split="proxy_unknown",
                partition_constraints=constraints,
                **common,
            )
            final_reference = evaluate_event_review_payload(
                payload,
                known_split="id_test",
                unknown_split="final_unknown",
                **common,
            )
            cells.append({
                "seed": seed,
                "known_classes": episode_config["known_classes"],
                "must_link_count": sum(
                    constraint.relation == "must_link"
                    for constraint in constraints
                ),
                "cannot_link_count": sum(
                    constraint.relation == "cannot_link"
                    for constraint in constraints
                ),
                "baseline": baseline["metrics"],
                "refined": refined["metrics"],
                "final_reference": final_reference["metrics"],
            })

    baseline_summary = _mean_metrics([cell["baseline"] for cell in cells])
    refined_summary = _mean_metrics([cell["refined"] for cell in cells])
    return {
        "protocol": {
            "representation": representation,
            "embedding_space": "representation_l2",
            "clustering_method": "hdbscan",
            "flag_fraction": flag_fraction,
            "feedback_after_initial_review": True,
            "constraints_from_proxy_reviews_only": True,
            "final_partition_refined": False,
            "final_labels_used_for_selection_or_constraints": False,
            "review_only": True,
            "temporary_semantic_ids_emitted": 0,
            "mutation_published": False,
        },
        "proxy_baseline_summary": baseline_summary,
        "proxy_refined_summary": refined_summary,
        "proxy_refined_minus_baseline": {
            name: refined_summary[name] - baseline_summary[name]
            for name in METRICS
        },
        "final_unconstrained_reference": _mean_metrics([
            cell["final_reference"] for cell in cells
        ]),
        "mean_must_link_count": float(np.mean([
            cell["must_link_count"] for cell in cells
        ])),
        "mean_cannot_link_count": float(np.mean([
            cell["cannot_link_count"] for cell in cells
        ])),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_pairwise_partition_refinement(**config)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()