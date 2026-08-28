"""M320 - the versioned feature bus.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md``
§8.37 (27 Aug 2026, before any build). Every feature set is a
content-addressed, versioned artifact: (encoder version,
extraction version, preprocessing version) plus a content digest.
Consumers resolve by the version triple or are refused - there is
no silent default, and a digest mismatch is a refusal, not a
repair. Alignment artifacts (M301) are the first registered
consumers.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class FeatureVersionError(RuntimeError):
    """A requested feature version is unregistered or its digest
    does not match the sealed one."""


@dataclass(frozen=True)
class FeatureVersion:
    encoder: str
    extraction: str
    preprocessing: str

    def as_key(self) -> tuple[str, str, str]:
        return (self.encoder, self.extraction, self.preprocessing)


@dataclass(frozen=True)
class FeatureArtifact:
    version: FeatureVersion
    digest: str
    path: str


@dataclass
class FeatureBus:
    """Resolves version triples to sealed feature artifacts."""
    artifacts: dict[tuple[str, str, str], FeatureArtifact] = field(
        default_factory=dict)

    def register(self, artifact: FeatureArtifact) -> None:
        key = artifact.version.as_key()
        existing = self.artifacts.get(key)
        if existing is not None and existing.digest != artifact.digest:
            raise FeatureVersionError(
                f"feature version {key} is already sealed to digest "
                f"{existing.digest}; re-registering it under digest "
                f"{artifact.digest} is a version mutation, not a new "
                "version")
        self.artifacts[key] = artifact

    def resolve(self, version: FeatureVersion) -> FeatureArtifact:
        """A-G3/A-G4: the exact version triple resolves to its
        sealed artifact; anything else is a refusal."""
        key = version.as_key()
        artifact = self.artifacts.get(key)
        if artifact is None:
            raise FeatureVersionError(
                f"feature version {key} is not registered")
        return artifact

    def resolve_or_refuse(self, version: FeatureVersion
                          ) -> FeatureArtifact:
        return self.resolve(version)

    def registered_versions(self) -> list[tuple[str, str, str]]:
        return sorted(self.artifacts)
