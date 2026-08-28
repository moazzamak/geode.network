"""Rehearse replicated serving, failed canary rollback, and coordinator recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from src.model_fingerprint import InputSpec, ModelFingerprint, OutputSpec
from src.open_set import SupportProfile
from src.runtime.model_bundle import (
    BundleNode,
    BundleProvenance,
    LocalModelBundleStore,
)
from src.runtime.production_service import (
    ProductionPromotionCoordinator,
    ReplicatedBundleService,
)


def _node() -> BundleNode:
    fingerprint = ModelFingerprint(
        task_name="e10-production-classifier",
        input_spec=InputSpec("passthrough", dim=2),
        output_spec=OutputSpec("labels", (0, 1)),
    )
    profile = SupportProfile(
        model_signature=fingerprint.signature,
        feature_transform_fingerprint="identity-2d",
        training_dataset_fingerprint="e10-train-v1",
        calibration_dataset_fingerprint="e10-calibration-v1",
        class_ids=(0, 1),
        score_scales=(1.0, 1.0),
        novelty_score="maximum_probability",
        global_threshold=0.5,
        version="e10-support-v1",
        fit_seed=11,
        created_at="2026-07-26T00:00:00Z",
    )
    return BundleNode(
        name="classifier",
        artifact_path="model.json",
        fingerprint=fingerprint,
        class_order=(0, 1),
        feature_transform_fingerprint="identity-2d",
        support_profile=profile,
    )


def _provenance() -> BundleProvenance:
    return BundleProvenance(
        routing_mode="exhaustive",
        semantic_router_cache_version="shadow-e10",
        training_manifest_hash="1" * 64,
        evaluation_manifest_hash="2" * 64,
        metric_summary_hash="3" * 64,
        software_compatibility="python>=3.11,numpy>=2",
        environment_fingerprint="e10-controlled-service-v1",
        created_at="2026-07-26T00:00:00Z",
        created_by="E10 production rehearsal",
    )


def _component(direction: int, version: str) -> bytes:
    return (json.dumps(
        {"direction": direction, "version": version},
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n").encode("utf-8")


def run_qualification(config_path: Path, registry: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported E10 configuration schema")
    store = LocalModelBundleStore(registry)
    parent = store.publish(
        {"model.json": _component(1, "production")},
        [_node()],
        provenance=_provenance(),
    )
    bad_canary = store.publish(
        {"model.json": _component(-1, "bad-canary")},
        [_node()],
        provenance=_provenance(),
        parent_bundle_id=parent.bundle_id,
    )
    interrupted_canary = store.publish(
        {"model.json": _component(1, "interrupted-canary")},
        [_node()],
        provenance=_provenance(),
        parent_bundle_id=parent.bundle_id,
    )
    store.activate(parent.bundle_id)

    def loader(bundle_id: str):
        manifest = store.load(bundle_id)
        component_path = (
            registry / "bundles" / manifest.bundle_id / "components" / "model.json"
        )
        state = json.loads(component_path.read_text(encoding="utf-8"))
        direction = int(state["direction"])
        return lambda values: (
            direction * np.asarray(values)[:, 0] >= 0.0
        ).astype(np.int32)

    service = ReplicatedBundleService(
        store, loader, replica_count=int(config["replica_count"]),
    )
    count = int(config["shadow_samples"])
    coordinates = np.linspace(-2.0, 2.0, count, dtype=np.float64)
    coordinates[coordinates == 0.0] = 0.01
    requests = np.column_stack([coordinates, np.zeros(count)])
    for batch in np.array_split(requests, 4):
        service.predict(batch)
    shadow = service.shadow(bad_canary.bundle_id, requests)
    shadow_gate_passed = shadow.agreement >= float(config["minimum_shadow_agreement"])

    coordinator = ProductionPromotionCoordinator(store)
    started = time.perf_counter()
    bad_canary_state = coordinator.promote_or_rollback(
        bad_canary.bundle_id,
        canary_gate_passed=shadow_gate_passed,
    )
    bad_canary_recovery_seconds = time.perf_counter() - started
    service.reload_current()
    pointer_after_bad_canary = store.current().bundle_id

    coordinator.begin_promotion(interrupted_canary.bundle_id)
    pointer_during_interruption = store.current().bundle_id
    del coordinator
    started = time.perf_counter()
    restarted = ProductionPromotionCoordinator(store)
    coordinator_recovery_state = restarted.recover()
    coordinator_recovery_seconds = time.perf_counter() - started
    service.reload_current()
    pointer_after_coordinator_recovery = store.current().bundle_id

    rto = float(config["recovery_time_objective_seconds"])
    rpo = int(config["recovery_point_objective_requests"])
    partials = [path.name for path in registry.rglob("*.partial")]
    checks = {
        "replicas_loaded": service.serving_bundle_ids
        == (parent.bundle_id,) * int(config["replica_count"]),
        "shadow_candidate_never_authoritative": not shadow.candidate_controls_outputs,
        "bad_canary_detected": not shadow_gate_passed,
        "bad_canary_parent_restored": pointer_after_bad_canary == parent.bundle_id,
        "bad_canary_within_rto": bad_canary_recovery_seconds <= rto,
        "coordinator_loss_injected_after_activation": pointer_during_interruption
        == interrupted_canary.bundle_id,
        "coordinator_parent_restored": pointer_after_coordinator_recovery
        == parent.bundle_id,
        "coordinator_recovery_within_rto": coordinator_recovery_seconds <= rto,
        "recovery_point_objective_met": rpo == 0,
        "no_partial_artifacts": not partials,
        "journal_stable": coordinator_recovery_state["phase"] == "stable",
    }
    return {
        "schema_version": 1,
        "milestone": "E10",
        "qualification_status": "passed" if all(checks.values()) else "failed",
        "gate_passed": all(checks.values()),
        "recovery_objectives": {
            "rto_seconds": rto,
            "rpo_requests": rpo,
        },
        "bundles": {
            "production": parent.bundle_id,
            "bad_canary": bad_canary.bundle_id,
            "interrupted_canary": interrupted_canary.bundle_id,
            "current": pointer_after_coordinator_recovery,
        },
        "shadow": shadow.to_dict(),
        "bad_canary": {
            "recovery_seconds": bad_canary_recovery_seconds,
            "coordinator_state": bad_canary_state,
        },
        "coordinator_loss": {
            "fault_injection": "discard coordinator after candidate activation before journal completion",
            "recovery_seconds": coordinator_recovery_seconds,
            "recovered_state": coordinator_recovery_state,
        },
        "telemetry": service.telemetry(),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("experiments/configs/e10_production_rehearsal.json"),
    )
    parser.add_argument(
        "--registry", type=Path,
        default=Path("logs/results/e10_model_registry"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("logs/results/e10_production_rehearsal.json"),
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
        raise RuntimeError("E10 production rehearsal failed")


if __name__ == "__main__":
    main()