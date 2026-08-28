"""V8 protocol validation and deterministic episode-replay fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.common.v5_artifacts import sha256_file
from src.runtime.schemas import (
    AdaptationUtilityEndpoint,
    EpisodeReplayContract,
    InterfaceContractAudit,
    V8_EPISODE_PARTITIONS,
    V8_INTERFACES,
)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_parent_locks(
    locks: Sequence[Mapping[str, Any]], repository_root: Path
) -> tuple[dict[str, str], ...]:
    validated = []
    for lock in locks:
        path = repository_root / str(lock["path"])
        expected = str(lock["sha256"])
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"immutable parent drift for {lock['id']}: {actual} != {expected}")
        validated.append({"id": str(lock["id"]), "path": str(lock["path"]), "sha256": actual})
    return tuple(validated)


def endpoint_from_config(config: Mapping[str, Any]) -> AdaptationUtilityEndpoint:
    payload = {"schema_version": int(config["schema_version"]), **dict(config["endpoint"])}
    return AdaptationUtilityEndpoint.from_dict(payload)


def validate_m45_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1 or config.get("milestone") != "M45":
        raise ValueError("unsupported M45 protocol")
    endpoint_from_config(config)
    episodes = config.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 3:
        raise ValueError("M45 requires three ordered synthetic episodes")
    episode_ids = [episode["id"] for episode in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode IDs must be unique")
    class_order = config.get("initial_class_order")
    if not isinstance(class_order, list) or len(set(class_order)) != len(class_order):
        raise ValueError("initial class order must be unique")
    interfaces = config.get("interfaces")
    if not isinstance(interfaces, dict) or tuple(interfaces) != V8_INTERFACES:
        raise ValueError("all five interfaces must be registered in order")
    if any(not interfaces[name] for name in V8_INTERFACES):
        raise ValueError("interface sufficient statistics must be non-empty")
    expected_failures = {
        "partition_leakage",
        "stale_class_order",
        "threshold_lineage_mismatch",
        "review_budget_overflow",
        "missing_confirmation",
        "rollback_parent_drift",
    }
    failure_policy = config.get("failure_policy")
    if set(failure_policy or {}) != expected_failures:
        raise ValueError("all registered fail-closed cases must be present")
    if set(failure_policy.values()) != {"abort"}:
        raise ValueError("all M45 contract violations must abort")
    sealed = config.get("sealed_data")
    if sealed != {
        "training_data_loaded": False,
        "final_labels_opened": False,
        "final_test_sealed": True,
    }:
        raise PermissionError("M45 protocol lock must not open training or final-label data")


def assert_threshold_lineage(expected: str, observed: str) -> None:
    if expected != observed:
        raise ValueError("threshold lineage mismatch")


def assert_review_budget(reviewed_sample_ids: Sequence[str], budget: int) -> None:
    if len(reviewed_sample_ids) > budget:
        raise ValueError("review budget overflow")


def require_confirmation(confirmation_id: str | None) -> None:
    if not confirmation_id:
        raise PermissionError("confirmed adaptation requires a confirmation ID")


def assert_rollback_parent(parent_bundle_id: str, rollback_bundle_id: str) -> None:
    if parent_bundle_id != rollback_bundle_id:
        raise ValueError("rollback parent drift")


def build_episode_contracts(
    config: Mapping[str, Any], parent_bundle_hash: str
) -> tuple[EpisodeReplayContract, ...]:
    class_order = tuple(str(value) for value in config["initial_class_order"])
    budget = int(config["endpoint"]["review_budget"])
    contracts = []
    for episode in config["episodes"]:
        arrival_class = str(episode["arrival_class"])
        episode_id = str(episode["id"])
        seed = int(episode["seed"])
        hashes = tuple(
            sorted(
                (
                    name,
                    canonical_hash(
                        {
                            "episode_id": episode_id,
                            "partition": name,
                            "seed": seed,
                            "class_order": class_order,
                        }
                    ),
                )
                for name in V8_EPISODE_PARTITIONS
            )
        )
        policy_hash = canonical_hash(
            {"episode_id": episode_id, "class_order": class_order, "policy": "gaussian_rank32"}
        )
        anchor_hash = dict(hashes)["anchor"]
        child_order = (*class_order, arrival_class)
        contracts.append(
            EpisodeReplayContract(
                episode_id=episode_id,
                seed=seed,
                arrival_class=arrival_class,
                parent_class_order=class_order,
                child_class_order=child_order,
                partition_hashes=hashes,
                parent_bundle_hash=parent_bundle_hash,
                acceptance_policy_hash=policy_hash,
                anchor_set_hash=anchor_hash,
                review_budget=budget,
                final_test_sealed=True,
            )
        )
        class_order = child_order
        parent_bundle_hash = canonical_hash(
            {"parent": parent_bundle_hash, "episode_id": episode_id, "class_order": class_order}
        )
    return tuple(contracts)


def build_interface_audits(
    config: Mapping[str, Any], producer_hash: str
) -> tuple[InterfaceContractAudit, ...]:
    audits = []
    for interface_name in V8_INTERFACES:
        statistics = tuple(str(value) for value in config["interfaces"][interface_name])
        audits.append(
            InterfaceContractAudit(
                interface_name=interface_name,
                producer_schema=interface_name.split("_to_")[0] + "_v1",
                consumer_schema=interface_name.split("_to_")[1] + "_v1",
                producer_artifact_hash=producer_hash,
                required_statistics=statistics,
                supplied_statistics=statistics,
                class_order_version="v8-m45-class-order",
                calibration_version="v8-m45-anchor-policy",
            )
        )
    return tuple(audits)


def synthetic_episode_replay(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_m45_config(config)
    endpoint = endpoint_from_config(config)
    parent_hash = canonical_hash({"fixture": "v8-m45-parent"})
    contracts = build_episode_contracts(config, parent_hash)
    audits = build_interface_audits(config, canonical_hash(config["interfaces"]))
    parent_scores = (0.675, 0.702, 0.724)
    child_scores = (0.731, 0.749, 0.778)
    utilities = tuple(
        endpoint.utility(parent, child)
        for parent, child in zip(parent_scores, child_scores, strict=True)
    )
    return {
        "endpoint": endpoint.to_dict(),
        "episode_contracts": [contract.to_dict() for contract in contracts],
        "interface_audits": [audit.to_dict() for audit in audits],
        "synthetic_utility": {
            "parent_balanced_accuracy": list(parent_scores),
            "child_balanced_accuracy": list(child_scores),
            "episode_utility": list(utilities),
            "cumulative_utility": sum(utilities),
        },
        "training_data_loaded": False,
        "final_labels_opened": False,
    }
