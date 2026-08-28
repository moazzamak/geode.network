"""Unit tests for the M342 federation wiring (registered 28 Aug
2026, before the build). Pins: RFF determinism and digest stability,
block registration refusals, manifest resolution, the assemble path
for feature/rff/bridge blocks, and the no-silent-default rule."""
from __future__ import annotations

import numpy as np
import pytest

from geode.core.alignment import cca_align
from geode.core.federation import (
    BusBlock,
    CodeManifest,
    FederationBus,
    ManifestError,
    RffSpec,
    assemble,
)
from geode.core.feature_bus import (
    FeatureArtifact,
    FeatureVersion,
    FeatureVersionError,
)


def _fed_with_features() -> tuple[FederationBus, dict[str, int]]:
    fed = FederationBus()
    fed.register_feature(
        "clip", FeatureArtifact(
            version=FeatureVersion("clip-l", "native224", "l2norm"),
            digest="d" * 64, path="clip.npy"))
    fed.register_feature(
        "dino", FeatureArtifact(
            version=FeatureVersion("dino-b", "native224", "l2norm"),
            digest="e" * 64, path="dino.npy"))
    _attach = {"clip": 768, "dino": 768}
    fed._feature_widths = _attach  # registration metadata
    return fed, _attach


def test_rff_spec_deterministic():
    spec = RffSpec(input_dim=64, n_features=128, sigma=1.0,
                   seed=20260828)
    rng = np.random.default_rng(0)
    block = rng.standard_normal((16, 64)).astype(np.float32)
    assert np.array_equal(spec.project(block), spec.project(block))


def test_rff_spec_seed_sensitive():
    a = RffSpec(64, 128, 1.0, 1)
    b = RffSpec(64, 128, 1.0, 2)
    assert a.digest() != b.digest()


def test_rff_digest_stable():
    spec = RffSpec(64, 128, 1.0, 20260828)
    assert spec.digest() == RffSpec(64, 128, 1.0,
                                    20260828).digest()


def test_register_rff_requires_source():
    fed, _ = _fed_with_features()
    with pytest.raises(ManifestError):
        fed.register_rff("phi", RffSpec(768, 512, 0.5, 1),
                         source="nonexistent")


def test_register_bridge_requires_sources():
    fed, _ = _fed_with_features()
    rng = np.random.default_rng(1)
    a = rng.standard_normal((32, 16))
    b = rng.standard_normal((32, 16))
    art = cca_align(a, b, 8)
    with pytest.raises(ManifestError):
        fed.register_bridge("br", art, "clip", "nonexistent")


def test_manifest_resolves_and_refuses():
    fed, _ = _fed_with_features()
    fed.register_rff("phi", RffSpec(768, 512, 0.5, 1),
                     source="clip")
    manifest = CodeManifest(("clip", "dino", "phi"))
    blocks = manifest.resolve(fed)
    assert [b.name for b in blocks] == ["clip", "dino", "phi"]
    with pytest.raises(ManifestError):
        CodeManifest(("clip", "missing")).resolve(fed)


def test_total_width():
    fed, _ = _fed_with_features()
    fed.register_rff("phi", RffSpec(768, 512, 0.5, 1),
                     source="clip")
    manifest = CodeManifest(("clip", "dino", "phi"))
    assert manifest.total_width(fed) == 768 + 768 + 512


def test_assemble_feature_and_rff():
    fed, _ = _fed_with_features()
    fed.register_rff("phi", RffSpec(32, 64, 0.5, 7),
                     source="clip")
    rng = np.random.default_rng(2)
    feats = {"clip": rng.standard_normal((8, 32)).astype(np.float32),
             "dino": rng.standard_normal((8, 32)).astype(np.float32)}
    design = assemble(fed, CodeManifest(("clip", "dino", "phi")),
                      feats)
    assert design.shape == (8, 32 + 32 + 64)
    # the RFF columns are the spec's own projection
    spec = RffSpec(32, 64, 0.5, 7)
    assert np.array_equal(design[:, 64:], spec.project(feats["clip"]))


def test_assemble_bridge():
    fed, _ = _fed_with_features()
    rng = np.random.default_rng(3)
    a = rng.standard_normal((64, 24))
    b = rng.standard_normal((64, 24))
    art = cca_align(a, b, 8)
    fed.register_bridge("br", art, "clip", "dino")
    feats = {"clip": a.astype(np.float32),
             "dino": b.astype(np.float32)}
    design = assemble(fed, CodeManifest(("br",)), feats)
    assert design.shape == (64, 16)   # 2 x 8 canonical variates


def test_assemble_refuses_missing_data():
    fed, _ = _fed_with_features()
    with pytest.raises(ManifestError):
        assemble(fed, CodeManifest(("clip",)), {})


def test_feature_digest_refusal_on_version_mutation():
    fed, _ = _fed_with_features()
    with pytest.raises(FeatureVersionError):
        fed.bus.register(FeatureArtifact(
            version=FeatureVersion("clip-l", "native224", "l2norm"),
            digest="f" * 64, path="other.npy"))


def test_bridge_is_optional():
    """The M341 reading: nothing in manifest resolution requires a
    bridge. A concat-only manifest resolves and assembles without
    any bridge registered."""
    fed, _ = _fed_with_features()
    rng = np.random.default_rng(4)
    feats = {"clip": rng.standard_normal((4, 8)).astype(np.float32),
             "dino": rng.standard_normal((4, 8)).astype(np.float32)}
    design = assemble(fed, CodeManifest(("clip", "dino")), feats)
    assert design.shape == (4, 16)
