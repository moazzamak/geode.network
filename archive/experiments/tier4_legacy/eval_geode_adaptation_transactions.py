from __future__ import annotations

import argparse
import copy
import json
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
from src.model_editor import ModelEditor
from src.model_fingerprint import InputSpec, ModelFingerprint, OutputSpec
from src.model_migration import dry_run_add_class_migration
from src.model_network import FittedModel, ModelNetwork
from src.replay_constrained_fitter import fit_replay_constrained_expert
from src.sdf_engine import EllipsoidExpert, Expert


def _expert_from_points(points: np.ndarray) -> Expert:
    center = points.mean(axis=0)
    radii = np.maximum(np.max(np.abs(points - center), axis=0) * 1.1, 0.5)
    expert = Expert(alpha=2.0)
    expert.add_ellipsoid(EllipsoidExpert(center=center, radii=radii))
    return expert


def _build_model(known_centroids: np.ndarray) -> FittedModel:
    class_models = {}
    for class_id, center in enumerate(known_centroids):
        expert = Expert(alpha=2.0)
        expert.add_ellipsoid(EllipsoidExpert(
            center=center.copy(), radii=np.full(len(center), 1.25),
        ))
        class_models[class_id] = [expert]
    classes = tuple(sorted(class_models))
    return FittedModel(
        ModelFingerprint(
            task_name="stream_geode",
            input_spec=InputSpec("passthrough", dim=known_centroids.shape[1]),
            output_spec=OutputSpec("sdf_scores", classes),
        ),
        class_models,
        {class_id: 1.0 for class_id in classes},
    )


def _metrics(
    model: FittedModel,
    fixture,
    proposal_embeddings: np.ndarray,
    intended_class: int,
) -> tuple[float, float, float]:
    proposal_scores = model.sdf_scores(proposal_embeddings)
    proposal_labels = model._predict_from_scores(proposal_scores)
    proposal_success = np.mean(
        (proposal_scores.min(axis=1) < 0.0)
        & (proposal_labels == intended_class)
    )
    replay_scores = model.sdf_scores(fixture.replay.known_embeddings)
    replay_labels = model._predict_from_scores(replay_scores)
    replay_accuracy = np.mean(
        (replay_scores.min(axis=1) < 0.0)
        & (replay_labels == fixture.replay.known_class_ids)
    )
    ood_scores = model.sdf_scores(fixture.replay.ood_embeddings)
    ood_recall = np.mean(ood_scores.min(axis=1) >= 0.0)
    return float(proposal_success), float(replay_accuracy), float(ood_recall)


def _evidence(
    baseline: FittedModel,
    candidate: FittedModel,
    fixture,
    proposal_embeddings: np.ndarray,
    *,
    action: AdaptationAction,
    intended_class: int,
    target_class_id: int | None,
    transaction_validated: bool,
) -> AdaptationCandidateEvidence:
    before = _metrics(baseline, fixture, proposal_embeddings, intended_class)
    after = _metrics(candidate, fixture, proposal_embeddings, intended_class)
    return AdaptationCandidateEvidence(
        action=action,
        target_class_id=target_class_id,
        proposal_gain=after[0] - before[0],
        replay_accuracy_before=before[1],
        replay_accuracy_after=after[1],
        ood_unknown_recall_before=before[2],
        ood_unknown_recall_after=after[2],
        transaction_validated=transaction_validated,
    )


def run_geode_transaction_study(
    *,
    seed: int,
    minimum_known_separation: float = 1.5,
    stream_family: str = "baseline",
    review_only: bool = False,
    replay_constrained_updates: bool = False,
) -> dict:
    fixture = generate_class_incremental_stream(
        seed=seed, stream_family=stream_family,
    )
    records = _buffer_rejections(fixture, rejection_threshold=1.0).snapshot()
    proposal_policy = ClusterProposalPolicy(8, 2, 1.2, minimum_known_separation)
    gate_policy = AdaptationGatePolicy(0.5, 0.02, 0.02)
    oracle_types = np.asarray(fixture.oracle.event_types)
    outcomes = []
    for cluster in dbscan_rejections(records, epsilon=0.9, minimum_samples=4):
        proposal = evaluate_cluster_proposal(
            cluster, fixture.known_centroids, proposal_policy,
        )
        selected = proposal.review_required if review_only else proposal.eligible
        if not selected:
            continue
        sample_ids = np.asarray(
            [record.source_sample_id for record in cluster], dtype=np.int64,
        )
        event_names, event_counts = np.unique(
            oracle_types[sample_ids], return_counts=True,
        )
        majority_event = str(event_names[int(np.argmax(event_counts))])
        if event_counts.max() <= len(sample_ids) / 2:
            confirmation = None
            expected = AdaptationAction.QUARANTINE
            target_class = None
        elif majority_event == UNSEEN_CLASS:
            confirmation = ConfirmationKind.NEW_CLASS
            expected = AdaptationAction.CREATE_NEW
            target_class = None
        elif majority_event in {NEW_KNOWN_MODE, SHIFTED_KNOWN}:
            confirmation = ConfirmationKind.EXISTING_CLASS
            expected = AdaptationAction.UPDATE_EXISTING
            target_class = int(np.bincount(
                fixture.oracle.class_ids[sample_ids],
            ).argmax())
        else:
            confirmation = None
            expected = AdaptationAction.QUARANTINE
            target_class = None

        baseline = _build_model(fixture.known_centroids)
        proposal_points = fixture.observable.embeddings[sample_ids]
        fit_diagnostics = None
        if confirmation == ConfirmationKind.EXISTING_CLASS:
            if replay_constrained_updates:
                replay_exclusions = fixture.replay.known_embeddings[
                    fixture.replay.known_class_ids != target_class
                ]
                constrained_fit = fit_replay_constrained_expert(
                    proposal_points,
                    replay_exclusions,
                    exclusion_margin=0.1,
                )
                candidate_expert = constrained_fit.expert
                fit_diagnostics = {
                    "radius_scale": constrained_fit.radius_scale,
                    "positive_coverage": constrained_fit.positive_coverage,
                    "exclusion_violations": constrained_fit.exclusion_violations,
                    "minimum_exclusion_sdf": constrained_fit.minimum_exclusion_sdf,
                }
            else:
                candidate_expert = _expert_from_points(proposal_points)
            candidate = copy.deepcopy(baseline)
            if replay_constrained_updates:
                candidate.class_fusion_modes[target_class] = "hard_min"
            editor = ModelEditor(candidate.class_models)
            captured_evidence = None

            def validate(_models) -> bool:
                nonlocal captured_evidence
                captured_evidence = _evidence(
                    baseline,
                    candidate,
                    fixture,
                    proposal_points,
                    action=AdaptationAction.UPDATE_EXISTING,
                    intended_class=target_class,
                    target_class_id=target_class,
                    transaction_validated=True,
                )
                return select_adaptation_action(
                    (captured_evidence,),
                    confirmation=confirmation,
                    policy=gate_policy,
                ).action == AdaptationAction.UPDATE_EXISTING

            transaction = editor.apply_transaction(
                lambda: candidate.class_models[target_class].append(
                    copy.deepcopy(candidate_expert),
                ),
                validate,
                operation_name="confirmed_existing_class_update",
                class_id=target_class,
            )
            evidence = captured_evidence
            decision = select_adaptation_action(
                (evidence,), confirmation=confirmation, policy=gate_policy,
            )
            transaction_valid = transaction["accepted"]
        elif confirmation == ConfirmationKind.NEW_CLASS:
            candidate_expert = _expert_from_points(proposal_points)
            network = ModelNetwork()
            network.add_node("source", baseline)
            new_class_id = max(baseline.class_ids) + 1
            migration = dry_run_add_class_migration(
                network,
                source_node="source",
                new_class_id=new_class_id,
                new_class_models=[candidate_expert],
                score_scale=1.0,
            )
            candidate = migration.candidate_network._nodes["source"].model
            evidence = _evidence(
                baseline,
                candidate,
                fixture,
                proposal_points,
                action=AdaptationAction.CREATE_NEW,
                intended_class=new_class_id,
                target_class_id=None,
                transaction_validated=migration.valid,
            )
            decision = select_adaptation_action(
                (evidence,), confirmation=confirmation, policy=gate_policy,
            )
            transaction_valid = migration.valid
        else:
            evidence = None
            decision = select_adaptation_action(
                (), confirmation=None, policy=gate_policy,
            )
            transaction_valid = False

        outcomes.append({
            "temporary_unknown_id": proposal.temporary_unknown_id,
            "review_id": proposal.review_id,
            "expected_action": expected.value,
            "decision_action": decision.action.value,
            "failed_gates": list(decision.failed_gates),
            "transaction_validated": transaction_valid,
            "mutation_published": False,
            "fit_diagnostics": fit_diagnostics,
            "evidence": None if evidence is None else {
                "proposal_gain": evidence.proposal_gain,
                "replay_accuracy_before": evidence.replay_accuracy_before,
                "replay_accuracy_after": evidence.replay_accuracy_after,
                "ood_unknown_recall_before": evidence.ood_unknown_recall_before,
                "ood_unknown_recall_after": evidence.ood_unknown_recall_after,
            },
        })
    return {
        "protocol": {
            "seed": seed,
            "stream_family": stream_family,
            "review_only": review_only,
            "replay_constrained_updates": replay_constrained_updates,
            "oracle_used_only_after_review": review_only,
            "candidate_model_family": "geode_ellipsoid",
            "automatic_mutation_enabled": False,
        },
        "outcomes": outcomes,
    }


def run_geode_transaction_multiseed(seeds: tuple[int, ...]) -> dict:
    runs = [run_geode_transaction_study(seed=seed) for seed in seeds]
    outcomes = [outcome for run in runs for outcome in run["outcomes"]]
    return {
        "protocol": {
            "seeds": list(seeds),
            "automatic_mutation_enabled": False,
        },
        "summary": {
            "proposal_count": len(outcomes),
            "decision_accuracy": float(np.mean([
                outcome["decision_action"] == outcome["expected_action"]
                for outcome in outcomes
            ])) if outcomes else 0.0,
            "validated_transactions": sum(
                outcome["transaction_validated"] for outcome in outcomes
            ),
            "published_mutations": sum(
                outcome["mutation_published"] for outcome in outcomes
            ),
            "quarantined": sum(
                outcome["decision_action"] == AdaptationAction.QUARANTINE.value
                for outcome in outcomes
            ),
        },
        "runs": runs,
    }


def run_geode_review_transaction_matrix(
    *,
    seeds: tuple[int, ...],
    stream_families: tuple[str, ...],
    minimum_known_separation: float = 2.5,
    replay_constrained_updates: bool = False,
) -> dict:
    runs = [
        run_geode_transaction_study(
            seed=seed,
            stream_family=family,
            minimum_known_separation=minimum_known_separation,
            review_only=True,
            replay_constrained_updates=replay_constrained_updates,
        )
        for family in stream_families
        for seed in seeds
    ]
    outcomes = [outcome for run in runs for outcome in run["outcomes"]]
    update_outcomes = [
        outcome for outcome in outcomes
        if outcome["expected_action"] == AdaptationAction.UPDATE_EXISTING.value
    ]
    accepted_updates = [
        outcome for outcome in update_outcomes
        if outcome["decision_action"] == AdaptationAction.UPDATE_EXISTING.value
    ]
    return {
        "protocol": {
            "seeds": list(seeds),
            "stream_families": list(stream_families),
            "minimum_known_separation": minimum_known_separation,
            "replay_constrained_updates": replay_constrained_updates,
            "oracle_used_only_after_review": True,
            "automatic_mutation_enabled": False,
        },
        "summary": {
            "review_count": len(outcomes),
            "accepted_transactions": sum(
                outcome["transaction_validated"] for outcome in outcomes
            ),
            "decision_counts": {
                action.value: sum(
                    outcome["decision_action"] == action.value
                    for outcome in outcomes
                )
                for action in AdaptationAction
            },
            "decision_accuracy": float(np.mean([
                outcome["decision_action"] == outcome["expected_action"]
                for outcome in outcomes
            ])) if outcomes else 0.0,
            "published_mutations": sum(
                outcome["mutation_published"] for outcome in outcomes
            ),
            "reviewed_update_count": len(update_outcomes),
            "accepted_reviewed_updates": len(accepted_updates),
            "accepted_update_replay_drop_maximum": max((
                outcome["evidence"]["replay_accuracy_before"]
                - outcome["evidence"]["replay_accuracy_after"]
                for outcome in accepted_updates
            ), default=0.0),
            "failed_gate_counts": {
                gate: sum(
                    gate in outcome["failed_gates"] for outcome in outcomes
                )
                for gate in (
                    "insufficient_proposal_gain",
                    "replay_regression",
                    "ood_regression",
                    "transaction_validation_failed",
                )
            },
            "exclusion_violations": sum(
                outcome["fit_diagnostics"]["exclusion_violations"]
                for outcome in update_outcomes
                if outcome["fit_diagnostics"] is not None
            ),
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Transactional GEODE adaptation study")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact_path = Path(config.pop("artifact_path"))
    if "stream_families" in config:
        result = run_geode_review_transaction_matrix(
            seeds=tuple(config.pop("seeds")),
            stream_families=tuple(config.pop("stream_families")),
            **config,
        )
    else:
        result = run_geode_transaction_multiseed(tuple(config["seeds"]))
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()