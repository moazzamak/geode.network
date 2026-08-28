from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.tier4.eval_adaptation_actions import _candidate_evidence
from experiments.tier4.eval_unknown_streaming import (
    UNSEEN_CLASS,
    _buffer_rejections,
    generate_class_incremental_stream,
)
from src.adaptation_policy import (
    AdaptationAction,
    AdaptationGatePolicy,
    ConfirmationKind,
    select_adaptation_action,
)
from src.discovery_clustering import dbscan_rejections
from src.discovery_policy import ClusterProposalPolicy, evaluate_cluster_proposal


def run_ambiguity_resolution(
    *,
    seed: int,
    stream_family: str,
    rejection_threshold: float = 1.0,
    dbscan_epsilon: float = 0.9,
    dbscan_minimum_samples: int = 4,
    frozen_separation: float = 2.5,
) -> dict:
    fixture = generate_class_incremental_stream(
        seed=seed, stream_family=stream_family,
    )
    records = _buffer_rejections(
        fixture, rejection_threshold=rejection_threshold,
    ).snapshot()
    proposal_policy = ClusterProposalPolicy(
        minimum_support=8,
        minimum_windows=2,
        maximum_rms_radius=1.2,
        minimum_known_separation=frozen_separation,
    )
    gate_policy = AdaptationGatePolicy(
        minimum_proposal_gain=0.5,
        maximum_replay_accuracy_drop=0.02,
        maximum_ood_recall_drop=0.02,
    )
    oracle_types = np.asarray(fixture.oracle.event_types)
    resolutions = []
    for cluster in dbscan_rejections(
        records,
        epsilon=dbscan_epsilon,
        minimum_samples=dbscan_minimum_samples,
    ):
        review = evaluate_cluster_proposal(
            cluster, fixture.known_centroids, proposal_policy,
        )
        if not review.review_required:
            continue
        sample_ids = np.asarray(
            [record.source_sample_id for record in cluster], dtype=np.int64,
        )
        event_types = oracle_types[sample_ids]
        names, counts = np.unique(event_types, return_counts=True)
        majority_event = str(names[int(np.argmax(counts))])
        if counts.max() <= len(sample_ids) / 2:
            confirmation = None
            target_class_id = None
            expected_action = AdaptationAction.QUARANTINE
        elif majority_event == UNSEEN_CLASS:
            confirmation = ConfirmationKind.NEW_CLASS
            target_class_id = None
            expected_action = AdaptationAction.CREATE_NEW
        else:
            confirmation = ConfirmationKind.EXISTING_CLASS
            target_labels = fixture.oracle.class_ids[sample_ids]
            target_class_id = int(np.bincount(target_labels).argmax())
            expected_action = AdaptationAction.UPDATE_EXISTING

        proposal_embeddings = fixture.observable.embeddings[sample_ids]
        nearest_candidate = int(cluster[0].nearest_candidates[0])
        candidates = (
            _candidate_evidence(
                fixture,
                proposal_embeddings,
                action=AdaptationAction.UPDATE_EXISTING,
                target_class_id=(
                    target_class_id
                    if target_class_id is not None
                    else nearest_candidate
                ),
                rejection_threshold=rejection_threshold,
            ),
            _candidate_evidence(
                fixture,
                proposal_embeddings,
                action=AdaptationAction.CREATE_NEW,
                target_class_id=None,
                rejection_threshold=rejection_threshold,
            ),
        )
        decision = select_adaptation_action(
            candidates, confirmation=confirmation, policy=gate_policy,
        )
        resolutions.append({
            "review_id": review.review_id,
            "nearest_known_distance": review.nearest_known_distance,
            "oracle_confirmation": (
                confirmation.value if confirmation is not None else None
            ),
            "expected_action": expected_action.value,
            "decision": {
                "action": decision.action.value,
                "target_class_id": decision.target_class_id,
                "failed_gates": list(decision.failed_gates),
            },
            "candidates": [
                {**asdict(candidate), "action": candidate.action.value}
                for candidate in candidates
            ],
            "mutation_published": False,
        })
    return {
        "protocol": {
            "seed": seed,
            "stream_family": stream_family,
            "frozen_separation": frozen_separation,
            "oracle_used_only_after_review": True,
            "automatic_mutation_enabled": False,
        },
        "review_count": len(resolutions),
        "resolutions": resolutions,
    }


def run_ambiguity_resolution_matrix(
    *,
    seeds: tuple[int, ...],
    stream_families: tuple[str, ...],
    **study_kwargs,
) -> dict:
    runs = [
        run_ambiguity_resolution(
            seed=seed, stream_family=family, **study_kwargs,
        )
        for family in stream_families
        for seed in seeds
    ]
    resolutions = [
        resolution for run in runs for resolution in run["resolutions"]
    ]
    return {
        "protocol": {
            "seeds": list(seeds),
            "stream_families": list(stream_families),
            "oracle_used_only_after_review": True,
            "automatic_mutation_enabled": False,
        },
        "summary": {
            "review_count": len(resolutions),
            "confirmation_counts": {
                kind.value: sum(
                    item["oracle_confirmation"] == kind.value
                    for item in resolutions
                )
                for kind in ConfirmationKind
            },
            "decision_counts": {
                action.value: sum(
                    item["decision"]["action"] == action.value
                    for item in resolutions
                )
                for action in AdaptationAction
            },
            "confirmation_action_match": float(np.mean([
                item["decision"]["action"] == item["expected_action"]
                for item in resolutions
            ])) if resolutions else 0.0,
            "published_mutations": sum(
                item["mutation_published"] for item in resolutions
            ),
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M11 ambiguity resolution study")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact_path = Path(config.pop("artifact_path"))
    result = run_ambiguity_resolution_matrix(
        seeds=tuple(config.pop("seeds")),
        stream_families=tuple(config.pop("stream_families")),
        **config,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()