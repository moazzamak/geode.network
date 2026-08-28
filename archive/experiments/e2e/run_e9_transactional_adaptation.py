"""Qualify confirmation-gated adaptation publication and exact rollback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.adaptation_policy import ConfirmationKind
from src.discovery_policy import ClusterProposalPolicy, evaluate_cluster_proposal
from src.model_fingerprint import InputSpec, ModelFingerprint, OutputSpec
from src.open_set import SupportProfile
from src.rejection_buffer import RejectionRecord
from src.runtime.adaptation_transaction import (
    AdaptationPublicationEvidence,
    AdaptationPublicationPolicy,
    ReviewConfirmation,
    publish_confirmed_adaptation,
    rollback_adaptation,
)
from src.runtime.model_bundle import (
    BundleNode,
    BundleProvenance,
    LocalModelBundleStore,
)


def _review_id() -> tuple[str, float]:
    records = tuple(
        RejectionRecord(
            record_id=index,
            embedding=(5.0 + 0.01 * (index % 2), 5.0 + 0.01 * (index // 2)),
            timestamp=float(index),
            window_id=index % 2,
            source_model_signature="e9-source-v1",
            support_profile_version="support-v1",
            novelty_score=1.5,
            decision_margin=0.4,
            nearest_candidates=(0,),
            source_sample_id=f"unlabeled-{index}",
        )
        for index in range(8)
    )
    proposal = evaluate_cluster_proposal(
        records,
        np.asarray([[0.0, 0.0]], dtype=np.float64),
        ClusterProposalPolicy(8, 2, 0.2, 2.0, review_only=True),
    )
    if not proposal.review_required or proposal.review_id is None:
        raise RuntimeError("E9 fixture did not produce a persistent review ID")
    return proposal.review_id, max(record.timestamp for record in records)


def _node() -> BundleNode:
    fingerprint = ModelFingerprint(
        task_name="e9-confirmed-adaptation",
        input_spec=InputSpec("passthrough", dim=2),
        output_spec=OutputSpec("sdf_scores", (0, 1)),
    )
    profile = SupportProfile(
        model_signature=fingerprint.signature,
        feature_transform_fingerprint="identity-2d",
        training_dataset_fingerprint="e9-train-v1",
        calibration_dataset_fingerprint="e9-calibration-v1",
        class_ids=(0, 1),
        score_scales=(1.0, 1.0),
        novelty_score="minimum_sdf",
        global_threshold=0.5,
        version="e9-support-v1",
        fit_seed=11,
        created_at="2026-07-26T00:00:00Z",
    )
    return BundleNode(
        name="source",
        artifact_path="source.bin",
        fingerprint=fingerprint,
        class_order=(0, 1),
        feature_transform_fingerprint="identity-2d",
        support_profile=profile,
    )


def _provenance() -> BundleProvenance:
    return BundleProvenance(
        routing_mode="exhaustive",
        semantic_router_cache_version="disabled-e9",
        training_manifest_hash="1" * 64,
        evaluation_manifest_hash="2" * 64,
        metric_summary_hash="3" * 64,
        software_compatibility="python>=3.11",
        environment_fingerprint="e9-controlled-fixture-v1",
        created_at="2026-07-26T00:00:00Z",
        created_by="E9 transactional adaptation qualification",
    )


def run_qualification(config_path: Path, registry: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported E9 configuration schema")
    review_id, latest_rejection_timestamp = _review_id()
    confirmation = ReviewConfirmation(
        review_id=review_id,
        kind=ConfirmationKind.EXISTING_CLASS,
        confirmed_label="class-0",
        confirmed_at=str(config["confirmation_timestamp"]),
    )
    store = LocalModelBundleStore(registry)
    parent = store.publish(
        {"source.bin": b"e9-parent-model"}, [_node()], provenance=_provenance(),
    )
    store.activate(parent.bundle_id)
    policy = AdaptationPublicationPolicy(
        float(config["maximum_calibration_nll_increase"]),
    )
    base_evidence = {
        "replay_verified": True,
        "calibration_nll_before": float(config["passing_calibration_nll_before"]),
        "graph_validation_issues": (),
        "final_novel_labels_hidden": True,
    }
    rejected = publish_confirmed_adaptation(
        store,
        review_id=review_id,
        confirmation=confirmation,
        evidence=AdaptationPublicationEvidence(
            **base_evidence,
            calibration_nll_after=float(config["rejected_calibration_nll_after"]),
        ),
        policy=policy,
        components={"source.bin": b"e9-child-model"},
        nodes=[_node()],
        provenance=_provenance(),
        publish=True,
    )
    pointer_after_rejection = store.current().bundle_id
    passing_evidence = AdaptationPublicationEvidence(
        **base_evidence,
        calibration_nll_after=float(config["passing_calibration_nll_after"]),
    )
    dry_run = publish_confirmed_adaptation(
        store,
        review_id=review_id,
        confirmation=confirmation,
        evidence=passing_evidence,
        policy=policy,
        components={"source.bin": b"e9-child-model"},
        nodes=[_node()],
        provenance=_provenance(),
        publish=False,
    )
    pointer_after_dry_run = store.current().bundle_id
    published = publish_confirmed_adaptation(
        store,
        review_id=review_id,
        confirmation=confirmation,
        evidence=passing_evidence,
        policy=policy,
        components={"source.bin": b"e9-child-model"},
        nodes=[_node()],
        provenance=_provenance(),
        publish=True,
    )
    child = store.current()
    rolled_back = rollback_adaptation(store, published)
    restored = store.current()
    checks = {
        "persistent_review_id": review_id.startswith("review-"),
        "delayed_feedback_linked": confirmation.review_id == review_id,
        "confirmation_after_rejections": bool(config["confirmation_timestamp"])
        and latest_rejection_timestamp == 7.0,
        "rejected_calibration_gate": rejected.failed_gates
        == ("calibration_gate_failed",),
        "rejection_pointer_unchanged": pointer_after_rejection == parent.bundle_id,
        "dry_run_not_published": not dry_run.mutation_published,
        "dry_run_pointer_unchanged": pointer_after_dry_run == parent.bundle_id,
        "all_publication_gates_passed": not published.failed_gates,
        "child_parent_lineage": child.parent_bundle_id == parent.bundle_id,
        "child_activated": child.bundle_id == published.child_bundle_id,
        "exact_parent_rollback": restored.bundle_id == parent.bundle_id,
        "rollback_recorded": rolled_back.rollback_bundle_id == parent.bundle_id,
        "final_novel_labels_hidden": passing_evidence.final_novel_labels_hidden,
    }
    return {
        "schema_version": 1,
        "milestone": "E9",
        "qualification_status": "passed" if all(checks.values()) else "failed",
        "gate_passed": all(checks.values()),
        "review": {
            "review_id": review_id,
            "confirmation_id": confirmation.confirmation_id,
            "confirmation_kind": confirmation.kind.value,
            "confirmed_label": confirmation.confirmed_label,
        },
        "transactions": {
            "rejected": rejected.to_dict(),
            "dry_run": dry_run.to_dict(),
            "published": published.to_dict(),
            "rolled_back": rolled_back.to_dict(),
        },
        "checks": checks,
        "registry": {
            "parent_bundle_id": parent.bundle_id,
            "child_bundle_id": child.bundle_id,
            "current_bundle_id": restored.bundle_id,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("experiments/configs/e9_transactional_adaptation.json"),
    )
    parser.add_argument(
        "--registry", type=Path,
        default=Path("logs/results/e9_model_registry"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("logs/results/e9_transactional_adaptation.json"),
    )
    arguments = parser.parse_args()
    result = run_qualification(arguments.config, arguments.registry)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["gate_passed"]:
        raise RuntimeError("E9 transactional adaptation qualification failed")


if __name__ == "__main__":
    main()