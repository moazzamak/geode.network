"""M355 / M352 - declared-label-set scoring and the privacy tier.

Registered in ``analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md`` under
G20 (coverage multiplication inverts quality) and G5/G6 (the
privacy tier is an axis, not a price point).

This replaces the routing use of
``geode.core.coverage_adjusted``. That module multiplied the axis
metric by coverage, which was measured to invert the ranking: on
the sealed Open Images axis it scored a scoped arm reading 0.901
on the classes it serves at 0.0441, below a full-coverage arm
reading 0.1643 on everything, so ``argmax`` returned the worse
arm. ``coverage_adjusted`` is retained because M302 and M321
sealed evidence against it; it must not be used to rank.

The replacement separates two questions the product confused:

- **Qualification** is membership. A capability is offered for a
  declaration only if it covers every declared label and sits in
  the declared privacy tier. Covering less is exclusion, not a
  ranking penalty. This is what stops an arm winning by refusing
  almost everything -- the property the coverage factor was
  defending.
- **Score** is the axis metric restricted to the declared labels,
  oriented so higher is better. Coverage never multiplies it.

Cross-tier routing is a refusal rather than a ranking, because a
plaintext capability and a private one are not substitutes at any
price: they differ in what the buyer must hand over, which no
score expresses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PrivacyTier(str, Enum):
    """The privacy axis. Ordering is deliberately NOT defined:
    these are separate markets, not grades of one."""

    PLAINTEXT = "plaintext"
    DEVICE_ENCODER = "device_encoder"
    PRIVATE = "private"


class CrossTierRoute(RuntimeError):
    """A route across privacy tiers is refused, not ranked."""


class NotQualified(RuntimeError):
    """The capability does not cover the declared label set."""


@dataclass(frozen=True)
class Declaration:
    """What the buyer asks for: the labels it cares about and the
    privacy tier it requires."""

    labels: frozenset[str]
    privacy_tier: PrivacyTier = PrivacyTier.PLAINTEXT

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("a declaration must name at least one "
                             "label")


@dataclass(frozen=True)
class Capability:
    """A registered artifact: the labels it serves, its per-label
    held-out metric, and the tier it serves in."""

    name: str
    per_label_metric: dict[str, float]
    privacy_tier: PrivacyTier = PrivacyTier.PLAINTEXT
    label_rows: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in self.per_label_metric.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"metric for {label!r} must be in "
                                 "[0, 1]")

    @property
    def served(self) -> frozenset[str]:
        return frozenset(self.per_label_metric)


def qualifies(capability: Capability,
              declaration: Declaration) -> bool:
    """Membership, not ranking. Both clauses are hard."""
    if capability.privacy_tier is not declaration.privacy_tier:
        return False
    return declaration.labels <= capability.served


def declared_score(capability: Capability,
                   declaration: Declaration) -> float:
    """The axis metric restricted to the declared labels, weighted
    by held-out rows where they are known and equally otherwise.

    Raises rather than returning a penalised number: an
    unqualified capability has no score for this declaration, and
    returning 0.0 would invite it back into a ranking.
    """
    if capability.privacy_tier is not declaration.privacy_tier:
        raise CrossTierRoute(
            f"{capability.name!r} serves "
            f"{capability.privacy_tier.value!r}; the declaration "
            f"requires {declaration.privacy_tier.value!r}")
    if not qualifies(capability, declaration):
        missing = sorted(declaration.labels - capability.served)
        raise NotQualified(
            f"{capability.name!r} does not serve "
            f"{len(missing)} declared label(s), e.g. {missing[:3]}")

    labels = sorted(declaration.labels)
    weights = [capability.label_rows.get(label, 1) for label in labels]
    total = sum(weights)
    if total <= 0:
        raise ValueError("declared labels carry no held-out rows")
    return sum(capability.per_label_metric[label] * weight
               for label, weight in zip(labels, weights)) / total


def rank(capabilities: list[Capability],
         declaration: Declaration) -> list[tuple[str, float]]:
    """Qualified capabilities, best first. Unqualified ones are
    absent rather than last. An empty result means abstain."""
    scored = [(c.name, declared_score(c, declaration))
              for c in capabilities if qualifies(c, declaration)]
    return sorted(scored, key=lambda pair: (-pair[1], pair[0]))
