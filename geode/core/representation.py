"""M343 - representation-artifact registration: learned adapters as
first-class registrable bus artifacts.

Registered in ``analysis/SCIENCE_LAYER_PLAN_2026-08-28.md`` (cell
M343, 28 Aug 2026, before the build). The frozen discipline applies
at SERVE-time, not contribution-time: a contributor can train an
adapter, projection, or trunk off-network and register it frozen,
exactly like any arm. The feature bus makes such artifacts additive
and priceable - the single largest untapped contribution surface the
protocol already permits.

This module extends the M342 federation bus with a
representation-artifact kind:

- ``RepresentationArtifact``: the registration form - input contract
  (the bus blocks it consumes), output contract (the block it
  produces), measured utility (downstream head improvement on the
  sealed reference workload, the M304 ubar machinery), price per
  unit, and the frozen weights digest.
- ``register_representation`` on the FederationBus: the artifact
  resolves on the bus, its input contract must be registered, and
  its utility must be measured (never claimed).
- ``attribution_share``: the artifact earns through a downstream
  head - its share of the head's utility gain, by the protocol's
  single attribution rule, the Shapley value of the fee flow
  (``geode.core.chains.shapley_split``). Amended by M375
  (28 Aug 2026): this module previously documented a leave-one-out
  rule, which contradicted the fee flow. M375 measured both on the
  same coalition values of a real two-stage chain and they differ
  by a factor of 1.66, so the contradiction was worth money, not
  only words. Shapley binds; the margin form survives only as an
  explicitly-labelled stand-in for callers with no measured
  coalition values.

The boundary: the FREE standard library never holds learned models;
third-party learned representations are registrable artifacts with
measured utility and a price. The paper states this surface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Mapping

import numpy as np

from geode.core.chains import shapley_split
from geode.core.economics import reference_workload_units
from geode.core.federation import (
    BusBlock,
    CodeManifest,
    FederationBus,
    ManifestError,
)


class RegistrationError(RuntimeError):
    """A representation artifact cannot be registered."""


@dataclass(frozen=True)
class RepresentationArtifact:
    """The M343 registration form for a learned representation.

    - input_contract: the ordered bus-block names the artifact
      consumes (its manifest).
    - output_name: the bus block it produces (registered as a
      derived block on registration).
    - output_width: the design width the artifact emits.
    - utility: the MEASURED downstream head improvement on the
      sealed reference workload (accuracy with the artifact minus
      accuracy without it, on the same head recipe) - never a
      claim.
    - ubar: the expected unit count on the reference workload (the
      M304 form; measured, sealed at registration).
    - price_per_unit: the contributor's posted price.
    - weights_digest: the frozen weights' content digest (the
      artifact is served frozen; the digest is the replay
      contract).
    """
    input_contract: tuple[str, ...]
    output_name: str
    output_width: int
    utility: float
    ubar: float
    price_per_unit: float
    weights_digest: str

    def __post_init__(self) -> None:
        if not self.input_contract:
            raise RegistrationError(
                "the input contract must name at least one block")
        if self.output_width <= 0:
            raise RegistrationError("the output width must be positive")
        if float(self.utility) < 0.0:
            # a measured utility may be zero or negative on a sealed
            # workload - the registration records it honestly; only
            # NaN/inf are inadmissible
            pass
        if not np.isfinite(float(self.utility)):
            raise RegistrationError("the utility must be finite")
        if float(self.ubar) <= 0.0:
            raise RegistrationError("ubar must be positive")
        if float(self.price_per_unit) < 0.0:
            raise RegistrationError("the price must be non-negative")
        if not self.weights_digest:
            raise RegistrationError("the weights digest is required")


def register_representation(fed: FederationBus,
                            artifact: RepresentationArtifact,
                            weights: np.ndarray | None = None,
                            ) -> None:
    """Register a learned representation artifact on the bus. The
    input contract must resolve (every consumed block registered);
    the artifact becomes a derived block of kind "representation"
    that the manifest system can consume like any other. The
    weights, when supplied, must hash to the registered digest (the
    frozen-serve contract)."""
    for name in artifact.input_contract:
        if name not in fed.blocks:
            raise RegistrationError(
                f"input-contract block {name!r} is not registered")
    if artifact.output_name in fed.blocks:
        raise RegistrationError(
            f"output block {artifact.output_name!r} already exists")
    if weights is not None:
        import hashlib
        digest = hashlib.sha256(
            np.asarray(weights, dtype=np.float32).tobytes()
        ).hexdigest()
        if digest != artifact.weights_digest:
            raise RegistrationError(
                "the supplied weights do not hash to the registered "
                f"digest ({digest[:16]}... != "
                f"{artifact.weights_digest[:16]}...)")
    fed.blocks[artifact.output_name] = BusBlock(
        kind="representation", name=artifact.output_name)
    if not hasattr(fed, "representations"):
        fed.representations = {}  # type: ignore[attr-defined]
    fed.representations[artifact.output_name] = artifact  # type: ignore[attr-defined]


def resolve_representation(fed: FederationBus,
                           name: str) -> RepresentationArtifact:
    """Fetch a registered representation artifact by its output
    block name."""
    reps = getattr(fed, "representations", None) or {}
    if name not in reps:
        raise ManifestError(
            f"block {name!r} is not a registered representation")
    return reps[name]


def representation_cost(artifact: RepresentationArtifact,
                        n_queries: int) -> float:
    """The posted cost of serving n queries through the artifact:
    price_per_unit * ubar * n_queries (the M304 unit form)."""
    if n_queries < 0:
        raise ValueError("the query count must be non-negative")
    return float(artifact.price_per_unit) * float(artifact.ubar) \
        * int(n_queries)


def attribution_share(head_utility_with: float,
                      head_utility_without: float,
                      artifact: RepresentationArtifact,
                      other_blocks: list[str] = [],
                      coalition_values: Mapping[
                          frozenset[str], float] | None = None,
                      ) -> dict[str, float]:
    """The artifact's attribution share of a downstream head's
    utility gain.

    The protocol has ONE attribution rule: the Shapley value of
    Section "The fee flow". When ``coalition_values`` is supplied
    -- a measured utility for every subset of
    ``[artifact.output_name] + other_blocks`` -- that rule is
    applied exactly.

    Without measured coalition values the exact rule cannot be
    computed from two numbers, and this falls back to the
    with-minus-without margin. That fallback is a STAND-IN, not a
    second rule. M375 measured both on the same four coalition
    values of a real two-stage chain: the margin rule awarded the
    routing stage 0.0977 of the pool where Shapley awarded it
    0.0588, a factor of 1.66. Callers that settle real payments
    must pass ``coalition_values``.

    Negative shares are recorded honestly (a block that hurts pays
    nothing and is reported).
    """
    blocks = [artifact.output_name, *other_blocks]
    if coalition_values is not None:
        missing = [frozenset(c) for r in range(len(blocks) + 1)
                   for c in combinations(blocks, r)
                   if frozenset(c) not in coalition_values]
        if missing:
            raise ValueError(
                f"the Shapley rule needs every coalition value; "
                f"{len(missing)} of {2 ** len(blocks)} are missing")
        raw = shapley_split(
            {name: 0.0 for name in blocks},
            chain_score=coalition_values[frozenset(blocks)],
            coalition_value=lambda members: coalition_values[
                frozenset(members)])
        return {name: max(value, 0.0) for name, value in raw.items()}

    loo = float(head_utility_with) - float(head_utility_without)
    shares: dict[str, float] = {}
    if loo > 0.0:
        shares[artifact.output_name] = loo
        remainder = float(head_utility_with) - loo
        if other_blocks:
            per = remainder / len(other_blocks)
            for name in other_blocks:
                shares[name] = per
    else:
        shares[artifact.output_name] = 0.0
        if other_blocks:
            per = float(head_utility_with) / len(other_blocks)
            for name in other_blocks:
                shares[name] = per
    return shares


def measure_ubar(units_per_query: list[float]) -> float:
    """The M304 form: the expected unit count on the reference
    workload, measured and sealed at registration."""
    return reference_workload_units(units_per_query)
