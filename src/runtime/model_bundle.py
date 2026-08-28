"""Immutable fingerprinted model bundles with fail-closed activation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any

from src.model_fingerprint import ModelFingerprint
from src.open_set import SupportProfile


MANIFEST_NAME = "bundle.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _safe_name(value: str, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise ValueError(f"{field} must be a safe identifier")
    return value


def _safe_artifact_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) is None for part in path.parts)
    ):
        raise ValueError("artifact path must be safe and relative")
    return path.as_posix()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ArtifactIdentity:
    path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        _safe_artifact_path(self.path)
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("artifact sha256 must be a lowercase SHA-256 digest")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("artifact size must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactIdentity":
        if set(payload) != {"path", "sha256", "size"}:
            raise ValueError("invalid artifact identity fields")
        return cls(str(payload["path"]), str(payload["sha256"]), int(payload["size"]))


@dataclass(frozen=True)
class BundleNode:
    name: str
    artifact_path: str
    fingerprint: ModelFingerprint
    class_order: tuple[Any, ...]
    feature_transform_fingerprint: str
    upstream: tuple[str, ...] = ()
    support_profile: SupportProfile | None = None

    def __post_init__(self) -> None:
        _safe_name(self.name, "node name")
        _safe_artifact_path(self.artifact_path)
        if not self.feature_transform_fingerprint:
            raise ValueError("feature_transform_fingerprint must be non-empty")
        if not self.class_order or len(set(self.class_order)) != len(self.class_order):
            raise ValueError("class_order must be non-empty and unique")
        for upstream_name in self.upstream:
            _safe_name(upstream_name, "upstream node")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "artifact_path": self.artifact_path,
            "fingerprint": self.fingerprint.to_dict(),
            "class_order": list(self.class_order),
            "feature_transform_fingerprint": self.feature_transform_fingerprint,
            "upstream": list(self.upstream),
            "support_profile": (
                None if self.support_profile is None else self.support_profile.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BundleNode":
        required = {
            "name", "artifact_path", "fingerprint", "class_order",
            "feature_transform_fingerprint", "upstream", "support_profile",
        }
        if set(payload) != required:
            raise ValueError("invalid bundle node fields")
        profile = payload["support_profile"]
        return cls(
            name=str(payload["name"]),
            artifact_path=str(payload["artifact_path"]),
            fingerprint=ModelFingerprint.from_dict(payload["fingerprint"]),
            class_order=tuple(payload["class_order"]),
            feature_transform_fingerprint=str(payload["feature_transform_fingerprint"]),
            upstream=tuple(str(value) for value in payload["upstream"]),
            support_profile=None if profile is None else SupportProfile.from_dict(profile),
        )


@dataclass(frozen=True)
class BundleProvenance:
    routing_mode: str
    semantic_router_cache_version: str
    training_manifest_hash: str
    evaluation_manifest_hash: str
    metric_summary_hash: str
    software_compatibility: str
    environment_fingerprint: str
    created_at: str
    created_by: str

    def __post_init__(self) -> None:
        required = {
            "routing_mode": self.routing_mode,
            "semantic_router_cache_version": self.semantic_router_cache_version,
            "software_compatibility": self.software_compatibility,
            "environment_fingerprint": self.environment_fingerprint,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"bundle provenance fields must be non-empty: {missing}")
        for name in (
            "training_manifest_hash", "evaluation_manifest_hash", "metric_summary_hash",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "routing_mode": self.routing_mode,
            "semantic_router_cache_version": self.semantic_router_cache_version,
            "training_manifest_hash": self.training_manifest_hash,
            "evaluation_manifest_hash": self.evaluation_manifest_hash,
            "metric_summary_hash": self.metric_summary_hash,
            "software_compatibility": self.software_compatibility,
            "environment_fingerprint": self.environment_fingerprint,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BundleProvenance":
        fields = {
            "routing_mode", "semantic_router_cache_version", "training_manifest_hash",
            "evaluation_manifest_hash", "metric_summary_hash", "software_compatibility",
            "environment_fingerprint", "created_at", "created_by",
        }
        if set(payload) != fields:
            raise ValueError("invalid bundle provenance fields")
        return cls(**{name: str(payload[name]) for name in fields})


@dataclass(frozen=True)
class ModelBundleManifest:
    bundle_id: str
    parent_bundle_id: str | None
    artifacts: tuple[ArtifactIdentity, ...]
    nodes: tuple[BundleNode, ...]
    provenance: BundleProvenance
    schema_version: int = 1

    def content_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "parent_bundle_id": self.parent_bundle_id,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "nodes": [node.to_dict() for node in self.nodes],
            "provenance": self.provenance.to_dict(),
        }

    @property
    def expected_bundle_id(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.content_payload()).encode("utf-8")
        ).hexdigest()[:20]

    def to_dict(self) -> dict[str, Any]:
        return {"bundle_id": self.bundle_id, **self.content_payload()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelBundleManifest":
        required = {
            "schema_version", "bundle_id", "parent_bundle_id", "artifacts", "nodes",
            "provenance",
        }
        if set(payload) != required or payload.get("schema_version") != 1:
            raise ValueError("unsupported model bundle schema")
        manifest = cls(
            bundle_id=str(payload["bundle_id"]),
            parent_bundle_id=(
                None if payload["parent_bundle_id"] is None
                else str(payload["parent_bundle_id"])
            ),
            artifacts=tuple(ArtifactIdentity.from_dict(item) for item in payload["artifacts"]),
            nodes=tuple(BundleNode.from_dict(item) for item in payload["nodes"]),
            provenance=BundleProvenance.from_dict(payload["provenance"]),
        )
        if manifest.bundle_id != manifest.expected_bundle_id:
            raise ValueError("bundle identity does not match manifest content")
        _validate_manifest(manifest)
        return manifest


def _validate_manifest(manifest: ModelBundleManifest) -> None:
    artifact_paths = [artifact.path for artifact in manifest.artifacts]
    if not artifact_paths or len(set(artifact_paths)) != len(artifact_paths):
        raise ValueError("bundle artifact paths must be non-empty and unique")
    nodes = {node.name: node for node in manifest.nodes}
    if not nodes or len(nodes) != len(manifest.nodes):
        raise ValueError("bundle node names must be non-empty and unique")
    for node in manifest.nodes:
        if node.artifact_path not in artifact_paths:
            raise ValueError(f"node {node.name!r} references a missing artifact")
        if tuple(node.fingerprint.output_spec.classes) != node.class_order:
            raise ValueError(f"node {node.name!r} class order does not match fingerprint")
        if node.support_profile is not None:
            node.support_profile.assert_compatible(
                model_signature=node.fingerprint.signature,
                class_ids=node.class_order,
                feature_transform_fingerprint=node.feature_transform_fingerprint,
            )
        for upstream_name in node.upstream:
            if upstream_name not in nodes:
                raise ValueError(f"node {node.name!r} references unknown upstream node")
        if node.upstream:
            if node.fingerprint.input_spec.source != "sdf_scores":
                raise ValueError(f"downstream node {node.name!r} must accept sdf_scores")
            width = sum(nodes[name].fingerprint.output_spec.dim for name in node.upstream)
            if node.fingerprint.input_spec.dim > 0 and node.fingerprint.input_spec.dim != width:
                raise ValueError(f"node {node.name!r} input dimension does not match graph")
            for upstream_name in node.upstream:
                if not node.fingerprint.accepts_from(nodes[upstream_name].fingerprint):
                    raise ValueError(f"edge {upstream_name!r} to {node.name!r} is incompatible")
        elif node.fingerprint.input_spec.source == "sdf_scores":
            raise ValueError(f"source node {node.name!r} cannot require sdf_scores")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError("bundle graph contains a cycle")
        if name in visited:
            return
        visiting.add(name)
        for upstream_name in nodes[name].upstream:
            visit(upstream_name)
        visiting.remove(name)
        visited.add(name)

    for node_name in nodes:
        visit(node_name)


def assert_node_replacement(existing: BundleNode, replacement: BundleNode) -> None:
    if not existing.fingerprint.is_swappable_with(replacement.fingerprint):
        raise ValueError("replacement fingerprint is not role-compatible")
    if existing.class_order != replacement.class_order:
        raise ValueError("replacement class order does not match")
    if existing.upstream != replacement.upstream:
        raise ValueError("replacement graph inputs do not match")
    if existing.feature_transform_fingerprint != replacement.feature_transform_fingerprint:
        raise ValueError("replacement transform fingerprint does not match")


class LocalModelBundleStore:
    """Publish, verify, activate, and roll back immutable local bundles."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.bundles = self.root / "bundles"
        self.current_pointer = self.root / "CURRENT"

    def publish(
        self,
        components: Mapping[str, bytes],
        nodes: Sequence[BundleNode],
        *,
        provenance: BundleProvenance,
        parent_bundle_id: str | None = None,
    ) -> ModelBundleManifest:
        normalized = {_safe_artifact_path(path): bytes(data) for path, data in components.items()}
        if parent_bundle_id is not None:
            self.load(parent_bundle_id)
        artifacts = tuple(
            ArtifactIdentity(path, _sha256(data), len(data))
            for path, data in sorted(normalized.items())
        )
        provisional = ModelBundleManifest(
            "", parent_bundle_id, artifacts, tuple(nodes), provenance,
        )
        manifest = ModelBundleManifest(
            provisional.expected_bundle_id, parent_bundle_id, artifacts, tuple(nodes), provenance,
        )
        _validate_manifest(manifest)
        final_path = self.bundles / manifest.bundle_id
        if final_path.exists():
            existing = self.load(manifest.bundle_id)
            if existing != manifest:
                raise ValueError("existing bundle content differs from publication")
            return existing
        self.bundles.mkdir(parents=True, exist_ok=True)
        partial = final_path.with_name(f"{manifest.bundle_id}.partial")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir()
        try:
            for path, data in normalized.items():
                destination = partial / "components" / Path(path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            (partial / MANIFEST_NAME).write_text(
                _canonical_json(manifest.to_dict()) + "\n", encoding="utf-8", newline="\n",
            )
            os.replace(partial, final_path)
        except Exception:
            if partial.exists():
                shutil.rmtree(partial)
            raise
        return self.load(manifest.bundle_id)

    def load(self, bundle_id: str) -> ModelBundleManifest:
        _safe_name(bundle_id, "bundle_id")
        path = self.bundles / bundle_id
        try:
            manifest = ModelBundleManifest.from_dict(json.loads(
                (path / MANIFEST_NAME).read_text(encoding="utf-8")
            ))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid model bundle {bundle_id}") from error
        if manifest.bundle_id != bundle_id:
            raise ValueError("bundle path does not match bundle identity")
        actual_paths = {
            item.relative_to(path / "components").as_posix()
            for item in (path / "components").rglob("*")
            if item.is_file()
        }
        expected_paths = {artifact.path for artifact in manifest.artifacts}
        if actual_paths != expected_paths:
            raise ValueError("bundle component set does not match manifest")
        for artifact in manifest.artifacts:
            component = path / "components" / Path(artifact.path)
            data = component.read_bytes()
            if len(data) != artifact.size or _sha256(data) != artifact.sha256:
                raise ValueError(f"bundle artifact verification failed: {artifact.path}")
        return manifest

    def activate(self, bundle_id: str) -> ModelBundleManifest:
        manifest = self.load(bundle_id)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.current_pointer.with_suffix(".partial")
        temporary.write_text(bundle_id + "\n", encoding="ascii", newline="\n")
        os.replace(temporary, self.current_pointer)
        return manifest

    def current(self) -> ModelBundleManifest | None:
        if not self.current_pointer.exists():
            return None
        return self.load(self.current_pointer.read_text(encoding="ascii").strip())

    def rollback(self) -> ModelBundleManifest:
        current = self.current()
        if current is None or current.parent_bundle_id is None:
            raise ValueError("current bundle has no rollback parent")
        return self.activate(current.parent_bundle_id)