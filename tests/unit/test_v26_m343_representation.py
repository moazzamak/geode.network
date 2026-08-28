"""Unit tests for the M343 representation-artifact registration
(registered 28 Aug 2026, before the build). Pins: the registration
form's validation; input-contract resolution; the frozen-weights
digest contract; the cost form; the leave-one-out attribution share;
the manifest integration (a head resolves through a registered
representation)."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from geode.core.federation import (
    CodeManifest,
    FederationBus,
    ManifestError,
)
from geode.core.representation import (
    RegistrationError,
    RepresentationArtifact,
    attribution_share,
    measure_ubar,
    register_representation,
    representation_cost,
    resolve_representation,
)


def _digest(weights: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(weights, dtype=np.float32).tobytes()).hexdigest()


def _bus_with_feature() -> FederationBus:
    """A minimal bus with one raw feature block registered (the
    M320-style version triple is not needed for the M343 contract -
    the representation kind only needs the block to EXIST)."""
    from geode.core.federation import BusBlock
    fed = FederationBus()
    fed.blocks["trunk"] = BusBlock(kind="feature", name="trunk")
    return fed


def _artifact(utility: float = 0.05) -> RepresentationArtifact:
    return RepresentationArtifact(
        input_contract=("trunk",),
        output_name="adapter_a",
        output_width=64,
        utility=utility,
        ubar=1.5,
        price_per_unit=2.0,
        weights_digest="a" * 64,
    )


def test_artifact_form_validation():
    # empty input contract refused
    with pytest.raises(RegistrationError):
        RepresentationArtifact(
            input_contract=(), output_name="x", output_width=8,
            utility=0.1, ubar=1.0, price_per_unit=1.0,
            weights_digest="a" * 64)
    # non-finite utility refused
    with pytest.raises(RegistrationError):
        RepresentationArtifact(
            input_contract=("trunk",), output_name="x",
            output_width=8, utility=float("nan"), ubar=1.0,
            price_per_unit=1.0, weights_digest="a" * 64)
    # zero ubar refused
    with pytest.raises(RegistrationError):
        RepresentationArtifact(
            input_contract=("trunk",), output_name="x",
            output_width=8, utility=0.1, ubar=0.0,
            price_per_unit=1.0, weights_digest="a" * 64)
    # negative price refused
    with pytest.raises(RegistrationError):
        RepresentationArtifact(
            input_contract=("trunk",), output_name="x",
            output_width=8, utility=0.1, ubar=1.0,
            price_per_unit=-1.0, weights_digest="a" * 64)
    # zero utility ADMITTED (a measured null is honest)
    art = RepresentationArtifact(
        input_contract=("trunk",), output_name="x", output_width=8,
        utility=0.0, ubar=1.0, price_per_unit=1.0,
        weights_digest="a" * 64)
    assert art.utility == 0.0


def test_register_requires_input_contract_resolution():
    fed = _bus_with_feature()
    art = _artifact()
    # the input contract resolves: registration succeeds
    register_representation(fed, art)
    assert "adapter_a" in fed.blocks
    # a second registration of the same output name is refused
    with pytest.raises(RegistrationError):
        register_representation(fed, _artifact())
    # an artifact whose input contract names an unknown block is
    # refused
    bad = RepresentationArtifact(
        input_contract=("missing",), output_name="adapter_b",
        output_width=8, utility=0.1, ubar=1.0, price_per_unit=1.0,
        weights_digest="b" * 64)
    with pytest.raises(RegistrationError):
        register_representation(fed, bad)


def test_register_weights_digest_contract():
    fed = _bus_with_feature()
    weights = np.ones((8, 4), dtype=np.float32)
    art = RepresentationArtifact(
        input_contract=("trunk",), output_name="adapter_c",
        output_width=4, utility=0.1, ubar=1.0, price_per_unit=1.0,
        weights_digest=_digest(weights))
    # matching digest: registration succeeds
    register_representation(fed, art, weights=weights)
    # mismatched weights: refused
    fed2 = _bus_with_feature()
    art2 = RepresentationArtifact(
        input_contract=("trunk",), output_name="adapter_c",
        output_width=4, utility=0.1, ubar=1.0, price_per_unit=1.0,
        weights_digest=_digest(weights))
    with pytest.raises(RegistrationError):
        register_representation(fed2, art2,
                                weights=weights * 2.0)


def test_resolve_representation():
    fed = _bus_with_feature()
    register_representation(fed, _artifact())
    art = resolve_representation(fed, "adapter_a")
    assert art.output_name == "adapter_a"
    with pytest.raises(ManifestError):
        resolve_representation(fed, "not_registered")


def test_representation_cost_form():
    art = _artifact()   # ubar 1.5, price 2.0
    assert representation_cost(art, 10) == pytest.approx(30.0)
    assert representation_cost(art, 0) == 0.0
    with pytest.raises(ValueError):
        representation_cost(art, -1)


def test_attribution_share_positive_loo():
    art = _artifact()
    shares = attribution_share(
        head_utility_with=0.60, head_utility_without=0.55,
        artifact=art, other_blocks=["trunk"])
    # the artifact's share is its leave-one-out utility
    assert shares["adapter_a"] == pytest.approx(0.05)
    # the trunk shares the remainder
    assert shares["trunk"] == pytest.approx(0.55)


def test_attribution_share_zero_or_negative_loo():
    art = _artifact()
    # a null LOO: the artifact earns nothing, recorded honestly;
    # the single other block carries the whole head utility
    shares = attribution_share(0.55, 0.55, art, ["trunk"])
    assert shares["adapter_a"] == 0.0
    assert shares["trunk"] == pytest.approx(0.55)
    # a negative LOO: the artifact pays nothing and is reported
    shares = attribution_share(0.50, 0.55, art, ["trunk"])
    assert shares["adapter_a"] == 0.0
    assert shares["trunk"] == pytest.approx(0.50)


def test_measure_ubar_is_the_m304_form():
    assert measure_ubar([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        measure_ubar([])
    with pytest.raises(ValueError):
        measure_ubar([1.0, 0.0])


def test_manifest_resolves_through_representation():
    fed = _bus_with_feature()
    register_representation(fed, _artifact())
    manifest = CodeManifest(blocks=("trunk", "adapter_a"))
    resolved = manifest.resolve(fed)
    kinds = [b.kind for b in resolved]
    assert kinds == ["feature", "representation"]
    # an unregistered block in the manifest still refuses
    with pytest.raises(ManifestError):
        CodeManifest(blocks=("trunk", "ghost")).resolve(fed)
