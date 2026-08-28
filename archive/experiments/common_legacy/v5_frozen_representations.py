"""Immutable representation manifest for v5 frozen-backbone experiments.

Records backbone identity, upstream weights digest, preprocessing digest,
interface architecture/weights digest, objective hash, training split hash,
output dimension, checkpoint source/license, token pooling, parent artifact, and
a complete representation hash that changes for any lineage input. Feature
caches key on this hash. Downstream binding guards fail closed on mismatch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.experiment_manifest import canonical_json
from experiments.common.v5_protocol import require_sha256


# ---------------------------------------------------------------------------
# Representation manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepresentationManifest:
    """Complete provenance record for a frozen representation."""

    backbone_id: str
    upstream_weights_digest: str
    preprocessing_digest: str
    interface_architecture: str  # "identity", "linear", "low_rank"
    interface_weights_digest: str  # sha256 of interface weights or "none" for identity
    objective_hash: str  # sha256 of the objective configuration
    training_split_hash: str  # sha256 of train split indices
    output_dimension: int
    checkpoint_source: str | None = None
    checkpoint_license: str | None = None
    token_pooling_policy: str | None = None
    parent_artifact: str | None = None  # hash of parent if derived
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.backbone_id:
            raise ValueError("backbone_id is required.")
        require_sha256(self.upstream_weights_digest, "upstream_weights_digest")
        require_sha256(self.preprocessing_digest, "preprocessing_digest")
        if self.interface_architecture not in ("identity", "linear", "low_rank"):
            raise ValueError(f"Unsupported interface: {self.interface_architecture!r}")
        if self.interface_architecture == "identity":
            if self.interface_weights_digest != "none":
                raise ValueError("Identity interface must have weights_digest='none'.")
        else:
            require_sha256(self.interface_weights_digest, "interface_weights_digest")
        require_sha256(self.objective_hash, "objective_hash")
        require_sha256(self.training_split_hash, "training_split_hash")
        if self.output_dimension < 1:
            raise ValueError("output_dimension must be positive.")
        if self.schema_version not in (1, 2):
            raise ValueError(
                f"Unsupported representation manifest schema: {self.schema_version}."
            )
        if self.schema_version == 2:
            for value, field_name in (
                (self.checkpoint_source, "checkpoint_source"),
                (self.checkpoint_license, "checkpoint_license"),
                (self.token_pooling_policy, "token_pooling_policy"),
            ):
                if value is None or not value.strip():
                    raise ValueError(f"{field_name} is required for schema version 2.")
        if self.parent_artifact is not None:
            require_sha256(self.parent_artifact, "parent_artifact")

    @property
    def representation_hash(self) -> str:
        """Complete hash incorporating all lineage inputs."""
        payload = {
            "backbone_id": self.backbone_id,
            "upstream_weights_digest": self.upstream_weights_digest,
            "preprocessing_digest": self.preprocessing_digest,
            "interface_architecture": self.interface_architecture,
            "interface_weights_digest": self.interface_weights_digest,
            "objective_hash": self.objective_hash,
            "training_split_hash": self.training_split_hash,
            "output_dimension": self.output_dimension,
            "parent_artifact": self.parent_artifact,
        }
        if self.schema_version == 2:
            payload.update(
                {
                    "checkpoint_source": self.checkpoint_source,
                    "checkpoint_license": self.checkpoint_license,
                    "token_pooling_policy": self.token_pooling_policy,
                }
            )
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "backbone_id": self.backbone_id,
            "upstream_weights_digest": self.upstream_weights_digest,
            "preprocessing_digest": self.preprocessing_digest,
            "interface_architecture": self.interface_architecture,
            "interface_weights_digest": self.interface_weights_digest,
            "objective_hash": self.objective_hash,
            "training_split_hash": self.training_split_hash,
            "output_dimension": self.output_dimension,
            "parent_artifact": self.parent_artifact,
            "representation_hash": self.representation_hash,
        }
        if self.schema_version == 2:
            payload.update(
                {
                    "checkpoint_source": self.checkpoint_source,
                    "checkpoint_license": self.checkpoint_license,
                    "token_pooling_policy": self.token_pooling_policy,
                }
            )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RepresentationManifest":
        schema_version = payload.get("schema_version")
        if schema_version not in (1, 2):
            raise ValueError("Unsupported representation manifest schema.")
        instance = cls(
            backbone_id=payload["backbone_id"],
            upstream_weights_digest=payload["upstream_weights_digest"],
            preprocessing_digest=payload["preprocessing_digest"],
            interface_architecture=payload["interface_architecture"],
            interface_weights_digest=payload["interface_weights_digest"],
            objective_hash=payload["objective_hash"],
            training_split_hash=payload["training_split_hash"],
            output_dimension=payload["output_dimension"],
            checkpoint_source=payload.get("checkpoint_source"),
            checkpoint_license=payload.get("checkpoint_license"),
            token_pooling_policy=payload.get("token_pooling_policy"),
            parent_artifact=payload.get("parent_artifact"),
            schema_version=schema_version,
        )
        declared = payload.get("representation_hash")
        if declared is not None and declared != instance.representation_hash:
            raise ValueError(
                f"Representation hash mismatch: declared={declared}, "
                f"computed={instance.representation_hash}."
            )
        return instance


# ---------------------------------------------------------------------------
# Feature cache metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureCacheMetadata:
    """Metadata binding a feature cache to its representation."""

    representation_hash: str
    feature_file_hash: str  # sha256 of the .npz content
    n_samples: int
    feature_dimension: int
    split_name: str  # "train", "dev", "test"

    def __post_init__(self) -> None:
        require_sha256(self.representation_hash, "representation_hash")
        require_sha256(self.feature_file_hash, "feature_file_hash")
        if self.n_samples < 1:
            raise ValueError("n_samples must be positive.")
        if self.feature_dimension < 1:
            raise ValueError("feature_dimension must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "representation_hash": self.representation_hash,
            "feature_file_hash": self.feature_file_hash,
            "n_samples": self.n_samples,
            "feature_dimension": self.feature_dimension,
            "split_name": self.split_name,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureCacheMetadata":
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported feature cache metadata schema.")
        return cls(
            representation_hash=payload["representation_hash"],
            feature_file_hash=payload["feature_file_hash"],
            n_samples=payload["n_samples"],
            feature_dimension=payload["feature_dimension"],
            split_name=payload["split_name"],
        )


# ---------------------------------------------------------------------------
# Binding guards (fail closed on mismatch)
# ---------------------------------------------------------------------------


def require_cache_binding(
    cache_meta: FeatureCacheMetadata,
    representation: RepresentationManifest,
) -> None:
    """Fail closed if the cache does not match the active representation."""
    if cache_meta.representation_hash != representation.representation_hash:
        raise ValueError(
            f"Feature cache bound to representation "
            f"{cache_meta.representation_hash[:16]}... but active representation is "
            f"{representation.representation_hash[:16]}... - MISMATCH, refusing to proceed."
        )


def require_feature_dimension(
    cache_meta: FeatureCacheMetadata,
    expected_dim: int,
) -> None:
    """Fail closed if cached features have unexpected dimension."""
    if cache_meta.feature_dimension != expected_dim:
        raise ValueError(
            f"Feature cache dimension {cache_meta.feature_dimension} != "
            f"expected {expected_dim}."
        )


def verify_cache_file_integrity(
    cache_path: Path,
    expected_hash: str,
) -> None:
    """Verify SHA-256 of a cache file matches the declared hash."""
    digest = hashlib.sha256()
    with cache_path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected_hash:
        raise ValueError(
            f"Cache file integrity check failed: "
            f"expected {expected_hash[:16]}..., got {actual[:16]}..."
        )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def save_manifest(manifest: RepresentationManifest, path: Path) -> str:
    """Save manifest to JSON, return file hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json(manifest.to_dict()) + "\n"
    path.write_text(content, encoding="utf-8", newline="\n")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> RepresentationManifest:
    """Load manifest from JSON."""
    content = path.read_text(encoding="utf-8")
    return RepresentationManifest.from_dict(json.loads(content))


def compute_preprocessing_digest(config_path: Path) -> str:
    """Hash the preprocessor config file for lineage tracking."""
    content = config_path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def compute_objective_hash(lambdas: tuple[float, float, float], architecture: str) -> str:
    """Hash the objective configuration."""
    payload = {
        "lambda_compact": lambdas[0],
        "lambda_margin": lambdas[1],
        "lambda_complexity": lambdas[2],
        "architecture": architecture,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def compute_split_hash(indices: np.ndarray) -> str:
    """Hash split indices for reproducibility tracking."""
    contiguous = np.ascontiguousarray(indices, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(b"split_indices_v1:")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()
