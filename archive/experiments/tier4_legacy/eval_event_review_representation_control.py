from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.tier4.eval_pairwise_partition_refinement import _mean_metrics
from experiments.tier4.eval_real_feature_event_review import (
    evaluate_event_review_payload,
)
from experiments.tier4.eval_real_feature_ood_transfer import (
    run_real_feature_ood_episode,
)


def run_event_review_representation_control(
    *,
    dataset_path: str,
    episodes: list[dict],
    seeds: list[int],
    representations: list[str],
    flag_fraction: float = 0.3,
    hdbscan_minimum_cluster_size: int = 3,
    hdbscan_minimum_samples: int | None = 3,
    samples_per_slice: int = 100,
    pca_components: int = 8,
) -> dict:
    if not representations or len(representations) != len(set(representations)):
        raise ValueError("representations must be a non-empty unique list.")
    cells = []
    for representation in representations:
        for seed in seeds:
            for episode_config in episodes:
                episode = run_real_feature_ood_episode(
                    dataset_path=dataset_path,
                    known_classes=tuple(episode_config["known_classes"]),
                    proxy_unknown_classes=tuple(
                        episode_config["proxy_unknown_classes"]
                    ),
                    final_unknown_classes=tuple(
                        episode_config["final_unknown_classes"]
                    ),
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
                proxy = evaluate_event_review_payload(
                    payload,
                    known_split="id_validation",
                    unknown_split="proxy_unknown",
                    **common,
                )
                final = evaluate_event_review_payload(
                    payload,
                    known_split="id_test",
                    unknown_split="final_unknown",
                    **common,
                )
                cells.append({
                    "representation": representation,
                    "seed": seed,
                    "known_classes": episode_config["known_classes"],
                    "proxy": proxy["metrics"],
                    "final": final["metrics"],
                })

    summaries = {}
    for representation in representations:
        representation_cells = [
            cell for cell in cells
            if cell["representation"] == representation
        ]
        summaries[representation] = {
            "proxy": _mean_metrics([
                cell["proxy"] for cell in representation_cells
            ]),
            "final_observational": _mean_metrics([
                cell["final"] for cell in representation_cells
            ]),
        }
    retained = representations[0]
    candidates = representations[1:]
    passing_candidates = [
        candidate for candidate in candidates
        if (
            summaries[candidate]["proxy"]["unknown_group_ari"]
            > summaries[retained]["proxy"]["unknown_group_ari"]
            and summaries[candidate]["proxy"]["distinct_group_recall"]
            >= summaries[retained]["proxy"]["distinct_group_recall"]
            and summaries[candidate]["proxy"]["duplicate_review_rate"]
            <= summaries[retained]["proxy"]["duplicate_review_rate"]
        )
    ]
    return {
        "protocol": {
            "retained_representation": retained,
            "candidate_representations": candidates,
            "selection_uses_proxy_only": True,
            "final_labels_used_for_selection": False,
            "pairwise_feedback_used": False,
            "clustering_policy_frozen_from_m11_29": True,
            "feature_provenance": {
                "mobilenetv2": "torchvision MobileNet_V2_Weights.IMAGENET1K_V1",
                "resnet18_imagenet": "torchvision ResNet18_Weights.IMAGENET1K_V1",
            },
            "mutation_published": False,
        },
        "representation_summaries": summaries,
        "proxy_gate": {
            "rule": "higher_ari_noninferior_distinct_recall_noninferior_duplicates",
            "passing_candidates": passing_candidates,
            "selected_representation": (
                passing_candidates[0] if passing_candidates else retained
            ),
        },
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_event_review_representation_control(**config)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
