"""GEODE artifact fetch/verify (v25 M258 cell 1) — content-addressed
component delivery.

Arms depend on frozen backbones and weights; deployment needs to
fetch them by CONTENT, not by location. An `ArtifactRef` names what
to fetch (digest, size, location) and the local store fetches and
VERIFIES the sha256 before the artifact is admitted — a mismatch
raises, never warns. Deterministic: the digest is the identity; no
RNG, no wall clocks. This is the M254-ready digest plumbing: the
same digests the public-chain anchor will later carry.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from geode.audit import sha256_file


@dataclass(frozen=True)
class ArtifactRef:
    """What to fetch, keyed by content."""
    name: str
    digest: str            # hex sha256 of the artifact bytes
    size: int              # expected byte size
    location: str = ""     # a subpath relative to the store root


class ArtifactStore:
    """A local content-addressed store rooted at a directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path_for(self, ref: ArtifactRef) -> Path:
        return self.root / ref.location

    def fetch_and_verify(self, ref: ArtifactRef,
                         dest: str | Path) -> Path:
        """Copy the stored artifact to `dest`, verifying its digest
        and size. Raises FileNotFoundError when absent, ValueError
        on a digest/size mismatch (a wrong artifact is NEVER
        admitted)."""
        source = self.path_for(ref)
        if not source.exists():
            raise FileNotFoundError(
                f"artifact {ref.name!r} not in store at {source}")
        size = source.stat().st_size
        if size != ref.size:
            raise ValueError(
                f"artifact {ref.name!r} size mismatch: {size} bytes "
                f"vs expected {ref.size}")
        digest = sha256_file(source)
        if digest != ref.digest:
            raise ValueError(
                f"artifact {ref.name!r} digest mismatch: {digest[:16]}"
                f"... vs expected {ref.digest[:16]}...")
        target = Path(dest)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return target

    def publish(self, path: str | Path, name: str,
                location: str = "") -> ArtifactRef:
        """Register an existing file in the store and return its
        content-addressed reference."""
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"{source} does not exist")
        digest = sha256_file(source)
        size = source.stat().st_size
        target = self.root / location / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return ArtifactRef(name=name, digest=digest, size=size,
                           location=str(Path(location) / name))


def verify_artifact(path: str | Path, digest: str) -> bool:
    """Standalone digest check (the CLI's `artifacts verify`)."""
    return sha256_file(Path(path)) == digest
