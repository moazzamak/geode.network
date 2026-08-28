"""Verified source and derived-artifact manifest for ModelNet40."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class ModelNetFile:
    path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("ModelNet40 file path must be safe and relative")
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("ModelNet40 file hash must be SHA-256")
        if self.size < 1:
            raise ValueError("ModelNet40 file size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelNetFile":
        if set(payload) != {"path", "sha256", "size"}:
            raise ValueError("invalid ModelNet40 file fields")
        return cls(str(payload["path"]), str(payload["sha256"]), int(payload["size"]))


@dataclass(frozen=True)
class ModelNet40Manifest:
    source_repository: str
    source_revision: str
    source_files: tuple[ModelNetFile, ...]
    artifact: ModelNetFile
    split_samples: tuple[tuple[str, int], ...]
    class_count: int = 40
    points_per_shape: int = 2048
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.source_repository or not self.source_revision:
            raise ValueError("ModelNet40 source repository and revision are required")
        if self.class_count != 40:
            raise ValueError("ModelNet40 requires exactly 40 classes")
        if self.points_per_shape != 2048:
            raise ValueError("ModelNet40 artifact requires 2048 points per shape")
        if dict(self.split_samples).keys() != {"train", "test"}:
            raise ValueError("ModelNet40 requires train and test sample counts")
        if any(count < 1 for _, count in self.split_samples):
            raise ValueError("ModelNet40 split counts must be positive")
        paths = [item.path for item in self.source_files]
        if not paths or len(paths) != len(set(paths)):
            raise ValueError("ModelNet40 source files must be non-empty and unique")

    @classmethod
    def load(cls, path: str | Path) -> "ModelNet40Manifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {
            "schema_version", "source_repository", "source_revision",
            "source_files", "artifact", "split_samples", "class_count",
            "points_per_shape",
        }
        if payload.get("schema_version") != 1 or set(payload) != required:
            raise ValueError("unsupported ModelNet40 manifest schema")
        return cls(
            source_repository=str(payload["source_repository"]),
            source_revision=str(payload["source_revision"]),
            source_files=tuple(
                ModelNetFile.from_dict(item) for item in payload["source_files"]
            ),
            artifact=ModelNetFile.from_dict(payload["artifact"]),
            split_samples=tuple(
                (str(name), int(count)) for name, count in payload["split_samples"].items()
            ),
            class_count=int(payload["class_count"]),
            points_per_shape=int(payload["points_per_shape"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "source_files": [item.to_dict() for item in self.source_files],
            "artifact": self.artifact.to_dict(),
            "split_samples": dict(self.split_samples),
            "class_count": self.class_count,
            "points_per_shape": self.points_per_shape,
        }

    def verify(self, root: str | Path) -> dict[str, Any]:
        root = Path(root)
        verified = []
        for item in (*self.source_files, self.artifact):
            path = root / item.path
            try:
                if path.stat().st_size != item.size:
                    raise ValueError(f"ModelNet40 file size mismatch: {item.path}")
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
            except OSError as error:
                raise ValueError(f"missing ModelNet40 file: {item.path}") from error
            if digest.hexdigest() != item.sha256:
                raise ValueError(f"ModelNet40 file hash mismatch: {item.path}")
            verified.append(item.path)
        return {
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "class_count": self.class_count,
            "points_per_shape": self.points_per_shape,
            "split_samples": dict(self.split_samples),
            "total_samples": sum(dict(self.split_samples).values()),
            "verified_files": verified,
            "artifact_path": self.artifact.path,
            "artifact_sha256": self.artifact.sha256,
        }