from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.tier4.eval_unknown_streaming import (
    NEW_KNOWN_MODE,
    SHIFTED_KNOWN,
    UNSEEN_CLASS,
    _buffer_rejections,
    generate_class_incremental_stream,
)
from src.adaptation_policy import (
    AdaptationAction,
    AdaptationCandidateEvidence,
    AdaptationGatePolicy,
    ConfirmationKind,
    select_adaptation_action,
)
from src.discovery_clustering import dbscan_rejections
from src.discovery_policy import ClusterProposalPolicy, evaluate_cluster_proposal


def _predict_modes(
    modes: dict[int, tuple[np.ndarray, ...]],
    features: np.ndarray,
    rejection_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    class_ids = np.asarray(sorted(modes), dtype=np.int64)
    class_distances = np.column_stack([
        np.min(np.linalg.norm(
            features[:, None, :] - np.asarray(modes[int(class_id)])[None, :, :],
            axis=2,
        ), axis=1)
        for class_id in class_ids
    ])
    nearest = np.argmin(class_distances, axis=1)
    return class_ids[nearest], class_distances[np.arange(len(features)), nearest] < rejection_threshold


def _candidate_evidence(
    fixture,
    proposal_embeddings: np.ndarray,
    *,
    action: AdaptationAction,
    target_class_id: int | None,
    rejection_threshold: float,
) -> AdaptationCandidateEvidence:
    baseline_modes = {
        class_id: (center.copy(),)
        for class_id, center in enumerate(fixture.known_centroids)
    }
    candidate_modes = {
        class_id: tuple(mode.copy() for mode in modes)
        for class_id, modes in baseline_modes.items()
    }
    proposal_center = proposal_embeddings.mean(axis=0)
    if action == AdaptationAction.UPDATE_EXISTING:
        if target_class_id is None:
            raise ValueError("update evidence requires a target class.")
        candidate_modes[target_class_id] += (proposal_center,)
        intended_class_id = target_class_id
    else:
        intended_class_id = max(candidate_modes) + 1
        candidate_modes[intended_class_id] = (proposal_center,)

    baseline_proposal_labels, baseline_proposal_accept = _predict_modes(
        baseline_modes, proposal_embeddings, rejection_threshold,
    )
    candidate_proposal_labels, candidate_proposal_accept = _predict_modes(
        candidate_modes, proposal_embeddings, rejection_threshold,
    )
    baseline_replay_labels, baseline_replay_accept = _predict_modes(
        baseline_modes, fixture.replay.known_embeddings, rejection_threshold,
    )
    candidate_replay_labels, candidate_replay_accept = _predict_modes(
        candidate_modes, fixture.replay.known_embeddings, rejection_threshold,
    )
    _, baseline_ood_accept = _predict_modes(
        baseline_modes, fixture.replay.ood_embeddings, rejection_threshold,
    )
    _, candidate_ood_accept = _predict_modes(
        candidate_modes, fixture.replay.ood_embeddings, rejection_threshold,
    )
    baseline_proposal_success = np.mean(
        baseline_proposal_accept & (baseline_proposal_labels == intended_class_id),
    )
    candidate_proposal_success = np.mean(
        candidate_proposal_accept & (candidate_proposal_labels == intended_class_id),
    )
    replay_truth = fixture.replay.known_class_ids
    baseline_replay_accuracy = np.mean(
        baseline_replay_accept & (baseline_replay_labels == replay_truth),
    )
    candidate_replay_accuracy = np.mean(
        candidate_replay_accept & (candidate_replay_labels == replay_truth),
    )
    original_unchanged = all(
        len(modes) == 1 and np.array_equal(modes[0], fixture.known_centroids[class_id])
        for class_id, modes in baseline_modes.items()
    )
    return AdaptationCandidateEvidence(
        action=action,
        target_class_id=target_class_id,
        proposal_gain=float(candidate_proposal_success - baseline_proposal_success),
        replay_accuracy_before=float(baseline_replay_accuracy),
        replay_accuracy_after=float(candidate_replay_accuracy),
        ood_unknown_recall_before=float(np.mean(~baseline_ood_accept)),
        ood_unknown_recall_after=float(np.mean(~candidate_ood_accept)),
        transaction_validated=original_unchanged,
    )


def run_adaptation_action_study(
    *,
    seed: int = 42,
    rejection_threshold: float = 1.0,
    dbscan_epsilon: float = 0.9,
    minimum_known_separation: float = 1.5,
    gate_policy: AdaptationGatePolicy | None = None,
) -> dict:
    fixture = generate_class_incremental_stream(seed=seed)
    records = _buffer_rejections(
        fixture, rejection_threshold=rejection_threshold,
    ).snapshot()
    proposal_policy = ClusterProposalPolicy(
        minimum_support=8,
        minimum_windows=2,
        maximum_rms_radius=1.2,
        minimum_known_separation=minimum_known_separation,
    )
    action_policy = gate_policy or AdaptationGatePolicy(
        minimum_proposal_gain=0.5,
        maximum_replay_accuracy_drop=0.02,
        maximum_ood_recall_drop=0.02,
    )
    oracle_types = np.asarray(fixture.oracle.event_types)
    proposals = []
    for cluster in dbscan_rejections(
        records, epsilon=dbscan_epsilon, minimum_samples=4,
    ):
        cluster_decision = evaluate_cluster_proposal(
            cluster, fixture.known_centroids, proposal_policy,
        )
        if not cluster_decision.eligible:
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
        elif majority_event in {NEW_KNOWN_MODE, SHIFTED_KNOWN}:
            confirmation = ConfirmationKind.EXISTING_CLASS
            target_labels = fixture.oracle.class_ids[sample_ids]
            target_class_id = int(np.bincount(target_labels).argmax())
            expected_action = AdaptationAction.UPDATE_EXISTING
        else:
            confirmation = None
            target_class_id = None
            expected_action = AdaptationAction.QUARANTINE

        proposal_embeddings = fixture.observable.embeddings[sample_ids]
        candidates = (
            _candidate_evidence(
                fixture,
                proposal_embeddings,
                action=AdaptationAction.UPDATE_EXISTING,
                target_class_id=(
                    target_class_id
                    if target_class_id is not None
                    else int(cluster[0].nearest_candidates[0])
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
            candidates,
            confirmation=confirmation,
            policy=action_policy,
        )
        proposals.append({
            "temporary_unknown_id": cluster_decision.temporary_unknown_id,
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
    correct = sum(
        proposal["decision"]["action"] == proposal["expected_action"]
        for proposal in proposals
    )
    return {
        "protocol": {
            "seed": seed,
            "oracle_used_only_for_confirmation_and_evaluation": True,
            "candidate_models_isolated": True,
            "candidate_model_family": "nearest_mode_surrogate",
            "automatic_mutation_enabled": False,
        },
        "proposal_count": len(proposals),
        "action_decision_accuracy": correct / len(proposals) if proposals else 0.0,
        "actions": {
            action.value: sum(
                proposal["decision"]["action"] == action.value
                for proposal in proposals
            )
            for action in AdaptationAction
        },
        "proposals": proposals,
    }


def run_adaptation_action_multiseed(
    seeds: tuple[int, ...],
    **study_kwargs,
) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique.")
    runs = [
        run_adaptation_action_study(seed=seed, **study_kwargs)
        for seed in seeds
    ]
    proposals = [proposal for run in runs for proposal in run["proposals"]]
    expected_actions = [proposal["expected_action"] for proposal in proposals]
    return {
        "protocol": {
            "seeds": list(seeds),
            "candidate_model_family": "nearest_mode_surrogate",
            "automatic_mutation_enabled": False,
        },
        "summary": {
            "proposal_count": len(proposals),
            "action_decision_accuracy": float(np.mean([
                proposal["decision"]["action"] == proposal["expected_action"]
                for proposal in proposals
            ])) if proposals else 0.0,
            "actions": {
                action.value: sum(
                    proposal["decision"]["action"] == action.value
                    for proposal in proposals
                )
                for action in AdaptationAction
            },
            "confirmation_burden": sum(
                proposal["oracle_confirmation"] is not None
                for proposal in proposals
            ),
            "constant_action_controls": {
                action.value: float(np.mean([
                    expected == action.value for expected in expected_actions
                ])) if expected_actions else 0.0
                for action in AdaptationAction
            },
            "failed_gate_counts": {
                gate: sum(
                    gate in proposal["decision"]["failed_gates"]
                    for proposal in proposals
                )
                for gate in (
                    "confirmation_required",
                    "insufficient_proposal_gain",
                    "replay_regression",
                    "ood_regression",
                    "transaction_validation_failed",
                )
            },
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M11 replay-gated action study")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact_path = Path(config.pop("artifact_path"))
    result = run_adaptation_action_multiseed(
        tuple(config.pop("seeds")), **config,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()