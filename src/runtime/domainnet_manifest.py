"""Verified source manifest for DomainNet qualification inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


DOMAINNET_DOMAINS = (
    "clipart", "infograph", "painting", "quickdraw", "real", "sketch",
)


@dataclass(frozen=True)
class DomainNetFile:
    path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("DomainNet file path must be safe and relative")
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("DomainNet file hash must be SHA-256")
        if self.size < 1:
            raise ValueError("DomainNet file size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DomainNetFile":
        if set(payload) != {"path", "sha256", "size"}:
            raise ValueError("invalid DomainNet file fields")
        return cls(str(payload["path"]), str(payload["sha256"]), int(payload["size"]))


@dataclass(frozen=True)
class DomainNetShard:
    """Legacy schema-v1 physical domain/split shard."""

    domain: str
    split: str
    path: str
    sha256: str
    samples: int

    def __post_init__(self) -> None:
        if self.domain not in DOMAINNET_DOMAINS or self.split not in {"train", "test"}:
            raise ValueError("invalid DomainNet domain or split")
        DomainNetFile(self.path, self.sha256, 1)
        if self.samples < 1:
            raise ValueError("DomainNet shard samples must be positive")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DomainNetShard":
        if set(payload) != {"domain", "split", "path", "sha256", "samples"}:
            raise ValueError("invalid DomainNet shard fields")
        return cls(
            str(payload["domain"]), str(payload["split"]), str(payload["path"]),
            str(payload["sha256"]), int(payload["samples"]),
        )


@dataclass(frozen=True)
class DomainNetManifest:
    files: tuple[DomainNetFile, ...]
    class_count: int
    version: str
    source_repository: str
    source_revision: str
    split_samples: tuple[tuple[str, int], ...]
    domains: tuple[str, ...] = DOMAINNET_DOMAINS
    schema_version: int = 2

    @classmethod
    def load(cls, path: str | Path) -> "DomainNetManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("schema_version")
        if version == 1:
            return cls._from_legacy(payload)
        required = {
            "schema_version", "version", "class_count", "source_repository",
            "source_revision", "domains", "split_samples", "files",
        }
        if version != 2 or set(payload) != required:
            raise ValueError("unsupported DomainNet manifest schema")
        manifest = cls(
            files=tuple(DomainNetFile.from_dict(item) for item in payload["files"]),
            class_count=int(payload["class_count"]),
            version=str(payload["version"]),
            source_repository=str(payload["source_repository"]),
            source_revision=str(payload["source_revision"]),
            split_samples=tuple(
                (str(name), int(count)) for name, count in payload["split_samples"].items()
            ),
            domains=tuple(str(value) for value in payload["domains"]),
        )
        manifest._validate()
        return manifest

    @classmethod
    def _from_legacy(cls, payload: dict[str, Any]) -> "DomainNetManifest":
        if set(payload) != {"schema_version", "version", "class_count", "shards"}:
            raise ValueError("invalid legacy DomainNet manifest fields")
        shards = tuple(DomainNetShard.from_dict(item) for item in payload["shards"])
        pairs = {(shard.domain, shard.split) for shard in shards}
        expected = {
            (domain, split) for domain in DOMAINNET_DOMAINS for split in ("train", "test")
        }
        if pairs != expected or len(pairs) != len(shards):
            raise ValueError("legacy manifest requires one train/test shard per domain")
        files = tuple(DomainNetFile(shard.path, shard.sha256, 1) for shard in shards)
        split_samples = tuple(
            (split, sum(shard.samples for shard in shards if shard.split == split))
            for split in ("train", "test")
        )
        manifest = cls(
            files=files,
            class_count=int(payload["class_count"]),
            version=str(payload["version"]),
            source_repository="legacy-physical-shards",
            source_revision="unversioned",
            split_samples=split_samples,
            schema_version=1,
        )
        manifest._validate()
        return manifest

    def _validate(self) -> None:
        if self.class_count != 345:
            raise ValueError("full DomainNet qualification requires 345 classes")
        if self.domains != DOMAINNET_DOMAINS:
            raise ValueError("DomainNet manifest must declare all six canonical domains")
        if dict(self.split_samples).keys() != {"train", "test"}:
            raise ValueError("DomainNet manifest requires train and test counts")
        if any(count < 1 for _, count in self.split_samples):
            raise ValueError("DomainNet split counts must be positive")
        paths = [item.path for item in self.files]
        if not paths or len(paths) != len(set(paths)):
            raise ValueError("DomainNet files must be non-empty and unique")
        if not self.source_repository or not self.source_revision:
            raise ValueError("DomainNet source repository and revision are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "class_count": self.class_count,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "domains": list(self.domains),
            "split_samples": dict(self.split_samples),
            "files": [item.to_dict() for item in self.files],
        }

    def verify(self, root: str | Path) -> dict[str, Any]:
        root = Path(root)
        verified = []
        for item in self.files:
            path = root / item.path
            digest = hashlib.sha256()
            try:
                if self.schema_version == 2 and path.stat().st_size != item.size:
                    raise ValueError(f"DomainNet file size mismatch: {item.path}")
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
            except OSError as error:
                raise ValueError(f"missing DomainNet file: {item.path}") from error
            if digest.hexdigest() != item.sha256:
                raise ValueError(f"DomainNet file hash mismatch: {item.path}")
            verified.append(item.path)
        report = {
            "version": self.version,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "class_count": self.class_count,
            "domains": list(self.domains),
            "split_samples": dict(self.split_samples),
            "verified_files": verified,
            "total_declared_samples": sum(dict(self.split_samples).values()),
            "total_verified_bytes": sum(item.size for item in self.files),
        }
        if self.schema_version == 1:
            report["verified_shards"] = verified
        return report