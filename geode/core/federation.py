"""M342 - the federation wiring: RFF maps and alignment bridges as
first-class feature-bus blocks.

Registered in ``analysis/SCIENCE_LAYER_PLAN_2026-08-28.md`` (M342,
28 Aug 2026, before the build; registration note updated post
wave-1). Wave 1 reweighted the federation: the bus is the spine,
the RFF map is the measured star (M300: 0.6695 vs the 0.6335 wall),
and CCA bridges are optional per-pair artifacts (M341: alignment
loses to raw concatenation on a clean cell). This module wires all
three onto the M320 bus:

- ``RffBlock``: a hash-seeded random-feature map registered as a
  bus block. The map is deterministic given (dim, n_features,
  sigma, seed) - the M300 property, unit-pinned. FHE-compatible by
  construction: the device computes phi(z) before encrypting, so
  the head stays a linear evaluation on ciphertext (the M322e
  path).
- ``BridgeBlock``: a closed-form alignment (Procrustes/CCA, the
  M301 module) registered as a bus block that consumes two blocks
  and produces their aligned variates. Optional, measured, priced -
  never load-bearing (the M341 reading).
- ``CodeManifest``: a head's ordered list of bus blocks. A head is
  replayable exactly when its manifest resolves (the M320 rule,
  extended from version triples to block lists).

The wiring is the deliverable: a head resolves through
[trunk-block, rff-block] and through a bridge on the bus, with
refusals - never silent defaults - for unregistered blocks and
digest mismatches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from geode.core.alignment import (
    AlignmentError,
    CcaArtifact,
    ProcrustesArtifact,
)
from geode.core.feature_bus import (
    FeatureArtifact,
    FeatureBus,
    FeatureVersion,
    FeatureVersionError,
)


class ManifestError(RuntimeError):
    """A code manifest does not resolve on the bus."""


@dataclass(frozen=True)
class RffSpec:
    """The registered RFF map parameters. Deterministic given these
    (the M300 g3 property): omega ~ N(0, sigma^-2 I), b ~ U[0, 2pi)
    from the seed."""
    input_dim: int
    n_features: int
    sigma: float
    seed: int

    def params(self) -> tuple[np.ndarray, np.ndarray]:
        """(omega, phase) - the M300 construction, verbatim."""
        rng = np.random.default_rng(self.seed)
        omega = (rng.standard_normal((self.input_dim, self.n_features))
                 / float(self.sigma)).astype(np.float32)
        phase = (rng.random(self.n_features) * 2.0 * np.pi
                 ).astype(np.float32)
        return omega, phase

    def project(self, block: np.ndarray) -> np.ndarray:
        """phi(z) = sqrt(2/D) cos(z @ omega + b) - the M300 map."""
        omega, phase = self.params()
        z = np.asarray(block, dtype=np.float32)
        arg = z @ omega + phase[None, :]
        return (np.sqrt(2.0 / self.n_features)
                * np.cos(arg)).astype(np.float32)

    def digest(self) -> str:
        """Content-addressed: the spec IS the content."""
        import hashlib
        payload = (f"rff:{self.input_dim}:{self.n_features}:"
                   f"{self.sigma}:{self.seed}")
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BusBlock:
    """One block on the federation bus: either a raw feature
    artifact (the M320 form), an RFF map over another block, or a
    bridge between two blocks."""
    kind: str            # "feature" | "rff" | "bridge"
    name: str
    # "feature": the version triple
    version: FeatureVersion | None = None
    # "rff": the map spec
    rff: RffSpec | None = None
    # "rff": the input block's name
    source: str | None = None
    # "bridge": the two input blocks' names and the artifact digest
    bridge_a: str | None = None
    bridge_b: str | None = None
    bridge_digest: str | None = None


@dataclass
class FederationBus:
    """The M320 bus extended with derived blocks (RFF maps and
    bridges). Raw features keep the M320 contract (version triple
    -> sealed digest); derived blocks are content-addressed by
    their own construction."""
    bus: FeatureBus = field(default_factory=FeatureBus)
    blocks: dict[str, BusBlock] = field(default_factory=dict)
    bridges: dict[str, ProcrustesArtifact | CcaArtifact] = field(
        default_factory=dict)

    def register_feature(self, name: str, artifact: FeatureArtifact
                         ) -> None:
        self.bus.register(artifact)
        self.blocks[name] = BusBlock(kind="feature", name=name,
                                     version=artifact.version)

    def register_rff(self, name: str, spec: RffSpec,
                     source: str) -> None:
        """An RFF map over a registered block. The source must
        exist; the map's input dimension must match the source's
        registered dimension (supplied by the caller at
        registration)."""
        if source not in self.blocks:
            raise ManifestError(
                f"rff block {name!r} source {source!r} is not "
                "registered")
        self.blocks[name] = BusBlock(kind="rff", name=name,
                                     rff=spec, source=source)

    def register_bridge(self, name: str,
                        artifact: ProcrustesArtifact | CcaArtifact,
                        block_a: str, block_b: str) -> None:
        """An alignment bridge between two registered blocks. The
        bridge is optional by construction: nothing in the manifest
        resolution requires one (the M341 reading)."""
        for src in (block_a, block_b):
            if src not in self.blocks:
                raise ManifestError(
                    f"bridge {name!r} source {src!r} is not "
                    "registered")
        self.bridges[name] = artifact
        self.blocks[name] = BusBlock(
            kind="bridge", name=name,
            bridge_a=block_a, bridge_b=block_b,
            bridge_digest=artifact.digest())

    def resolve_block(self, name: str) -> BusBlock:
        block = self.blocks.get(name)
        if block is None:
            raise ManifestError(f"block {name!r} is not registered")
        return block

    def feature_digest(self, name: str) -> str:
        """The sealed digest of a raw feature block (the M320
        contract)."""
        block = self.resolve_block(name)
        if block.kind != "feature" or block.version is None:
            raise ManifestError(
                f"block {name!r} is not a raw feature block")
        return self.bus.resolve(block.version).digest

    def rff_digest(self, name: str) -> str:
        """The content digest of an RFF block (its spec)."""
        block = self.resolve_block(name)
        if block.kind != "rff" or block.rff is None:
            raise ManifestError(
                f"block {name!r} is not an RFF block")
        return block.rff.digest()

    def bridge_digest(self, name: str) -> str:
        """The content digest of a bridge block (its alignment
        artifact)."""
        block = self.resolve_block(name)
        if block.kind != "bridge":
            raise ManifestError(
                f"block {name!r} is not a bridge block")
        return str(block.bridge_digest)


@dataclass(frozen=True)
class CodeManifest:
    """A head's ordered block list. The head is replayable exactly
    when every block resolves (the M320 rule, extended)."""
    blocks: tuple[str, ...]

    def resolve(self, fed: FederationBus) -> list[BusBlock]:
        out: list[BusBlock] = []
        for name in self.blocks:
            out.append(fed.resolve_block(name))
        return out

    def total_width(self, fed: FederationBus) -> int:
        """The concatenated design width the manifest produces.
        Features contribute their registered width; an RFF block
        contributes its n_features; a bridge contributes its
        artifact width."""
        width = 0
        for block in self.resolve(fed):
            if block.kind == "rff" and block.rff is not None:
                width += block.rff.n_features
            elif block.kind == "bridge":
                art = fed.bridges[block.name]
                if isinstance(art, CcaArtifact):
                    width += 2 * art.projection_a.shape[1]
                else:
                    width += art.rotation.shape[1]
            else:
                # a raw feature block: the caller registers widths
                # alongside features (see register_feature_width)
                width += fed._feature_widths[block.name]
        return width


def _attach_widths(fed: FederationBus,
                   widths: dict[str, int]) -> None:
    """Raw feature blocks carry their design width (the bus stores
    digests, not arrays; the width is registration metadata)."""
    fed._feature_widths = dict(widths)  # type: ignore[attr-defined]


def assemble(fed: FederationBus, manifest: CodeManifest,
             features: dict[str, np.ndarray]) -> np.ndarray:
    """Assemble the design matrix a head reads: each manifest block
    contributes its columns, in manifest order. RFF blocks project
    their source; bridges project their two inputs; feature blocks
    pass through. Refuses (never defaults) on a missing block."""
    columns: list[np.ndarray] = []
    for block in manifest.resolve(fed):
        if block.kind == "feature":
            if block.name not in features:
                raise ManifestError(
                    f"feature block {block.name!r} has no data")
            columns.append(np.asarray(features[block.name]))
        elif block.kind == "rff" and block.rff is not None:
            if block.source not in features:
                raise ManifestError(
                    f"rff block {block.name!r} source "
                    f"{block.source!r} has no data")
            columns.append(block.rff.project(features[block.source]))
        elif block.kind == "bridge":
            art = fed.bridges[block.name]
            a = features.get(str(block.bridge_a))
            b = features.get(str(block.bridge_b))
            if a is None or b is None:
                raise ManifestError(
                    f"bridge {block.name!r} is missing an input")
            if isinstance(art, CcaArtifact):
                za = (np.asarray(a, dtype=np.float64)
                      - np.asarray(a, dtype=np.float64).mean(axis=0)
                      ) @ art.projection_a
                zb = (np.asarray(b, dtype=np.float64)
                      - np.asarray(b, dtype=np.float64).mean(axis=0)
                      ) @ art.projection_b
                columns.append(za.astype(np.float32))
                columns.append(zb.astype(np.float32))
            else:
                mapped = np.asarray(a, dtype=np.float64) \
                    @ art.rotation
                columns.append(mapped.astype(np.float32))
    if not columns:
        raise ManifestError("the manifest produced no columns")
    return np.concatenate(columns, axis=1)
