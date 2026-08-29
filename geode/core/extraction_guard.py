"""M332 - extraction minimization: bucketed confidence, metered
abstentions, per-payer query budgets (R-A2a/b/c).

Registered in ``analysis/FEASIBILITY_THREAT_REVIEW_2026-08-28.md``
(F3, R-F3) and the review's queue (M332). The gap: the black-box
output was the softmax margin kappa(x) - a smooth function of
s = W^T z - and abstentions cost nothing. The trunk is a public
checkpoint, so an attacker computes z locally for free and
recovering W from margin-annotated responses: a linear system in
d*C unknowns, a low six-figure query count at commodity prices.
The most informative queries - boundary-mapping near the margin
threshold - were exactly the free ones.

The registered minimum-viable set (R-A2a/b/c; R-A2d, the lottery
router, is shipped separately):

- **Bucketed confidence (R-A2a).** The answer carries the
  predicted label and a COARSE confidence bucket, never the raw
  margin and never the score vector. The bucket edges are part of
  the sealed artifact, so the bucketing itself replays.
- **Metered abstentions (R-A2b).** An abstention consumed compute;
  it is metered at a reduced but nonzero price - the registered
  fraction of the unit price. The free oracle is gone, and the
  over-abstention incentive (A7) is priced.
- **Per-payer query budgets (R-A2c).** Per payer, per axis, per
  epoch, enforced LOCALLY at the gateway.

M372 (G8, 29 Aug 2026) moved the budget's enforcement surface. The
R-A2c rate was framed as "ledger-visible" - a per-user telemetry
stream on an immutable ledger, which is exactly the audience-pointing
surface the "no control surface" principle denies. The budget is a
GATEWAY-LOCAL rule: the gateway enforces it, the gateway sees it, and
nothing per-payer is ever published. If a network-level bound is
needed, only the AXIS AGGREGATE (used/cap per axis per epoch, no payer
dimension) may be published. Duration-metered axes carry a registered
padding quantum so the meter itself does not leak durations (e.g.
audio metered in fixed blocks, never in exact seconds).

The gate (a separate harness): an extraction simulation
recovering W from bucketed, metered responses costs more than the
head's expected lifetime revenue on the axis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# M357 (G19, 29 Aug 2026): the abstention is metered at the MEASURED
# compute fraction actually consumed, per axis, not at a flat half
# price. For a single-pass classification head the abstention decision
# is a function of the final margin, so the whole forward pass (trunk
# + head) runs before the abstention is known: the consumed fraction
# is 1.0. A cascade that abstains before its expensive stage would
# register a genuinely lower fraction per axis. The half-price figure
# (pre-M357) was the defense-pessimistic adversary rate in M332; the
# full-cost figure is both honest cost recovery and a STRONGER
# extraction bound (the adversary's cheapest rate doubles).
ABSTENTION_PRICE_FRACTION = 1.0   # measured, single-pass heads

# Registered per-axis abstention compute fractions (M357). Only the
# single-pass family is built today; cascades would register their
# own value at axis creation.
ABSTENTION_COMPUTE_FRACTIONS = {
    "single_pass_head": 1.0,
}

# M372 (G8): the registered meter-side padding quantum for
# duration-metered axes (audio, video). Durations are metered in
# whole blocks of this many seconds, never in exact seconds, so the
# ledger-side aggregate cannot be turned into a per-session length
# stream.
DURATION_QUANTUM_SECONDS = 15


def pad_duration(seconds: float,
                 quantum: int = DURATION_QUANTUM_SECONDS) -> int:
    """Meter a duration in whole blocks of ``quantum`` seconds. The
    billed block count is ceil(seconds / quantum); the residual is
    not leaked (a 1 s and a 14 s clip both meter as one block)."""
    if seconds < 0.0:
        raise ValueError("duration must be non-negative")
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    return int(math.ceil(seconds / quantum))


class BudgetExhausted(RuntimeError):
    """A payer's per-axis per-epoch query budget is spent."""


@dataclass(frozen=True)
class BucketedAnswer:
    """The ONLY classification output form under R-A2a: the label
    (or an abstention) and a coarse confidence bucket. No raw
    margin, no score vector - structural, so the extraction oracle
    cannot leak through a field that is not there."""
    label: int | None      # None = abstained
    bucket: int | None     # None = abstained
    abstained: bool


def confidence_bucket(margin: float, bucket_edges: tuple[float, ...],
                      ) -> int:
    """The coarse confidence bucket for a margin, under the
    artifact's registered edges (sorted, positive). Bucket 0 is the
    lowest confidence (margin <= edges[0]); bucket k covers
    edges[k-1] < margin <= edges[k]; the top bucket covers margins
    above the last edge. Deterministic given (margin, edges) - the
    edges are part of the sealed artifact, so the bucketing
    replays."""
    if not bucket_edges:
        raise ValueError("bucket edges must be non-empty")
    if any(e <= 0.0 for e in bucket_edges):
        raise ValueError("bucket edges must be positive")
    if list(bucket_edges) != sorted(bucket_edges):
        raise ValueError("bucket edges must be sorted ascending")
    for k, edge in enumerate(bucket_edges):
        if margin <= edge:
            return k
    return len(bucket_edges)


def abstention_charge(unit_price: float,
                      fraction: float = ABSTENTION_PRICE_FRACTION,
                      ) -> float:
    """The metered price of one abstention: the registered fraction
    of the unit price (R-A2b). M357 (G19): the fraction is the
    MEASURED compute actually consumed by the abstention on the axis
    -- 1.0 for a single-pass head (the full forward pass runs before
    the abstention is decided), lower only for a cascade that
    abstains before its expensive stage. The charge is nonzero and
    never above the full unit price."""
    if unit_price <= 0.0:
        raise ValueError("the unit price must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("the abstention fraction must lie in "
                         "(0, 1]")
    return float(unit_price) * float(fraction)


class PayerBudgetLedger:
    """Per-payer, per-axis, per-epoch query budgets (R-A2c), enforced
    LOCALLY at the gateway (M372, G8). Exhaustion refuses further
    queries - never silent. Nothing per-payer is published: the only
    publishable view is the per-axis aggregate, which carries no payer
    dimension."""

    def __init__(self) -> None:
        self._grants: dict[tuple[str, str, int], int] = {}
        self._used: dict[tuple[str, str, int], int] = {}

    def grant(self, payer: str, axis: str, epoch: int, cap: int
              ) -> None:
        """Set the budget cap for one (payer, axis, epoch). A grant
        does not reset usage."""
        if cap <= 0:
            raise ValueError("the budget cap must be positive")
        key = (payer, axis, int(epoch))
        self._grants[key] = cap
        self._used.setdefault(key, 0)

    def consume(self, payer: str, axis: str, epoch: int,
                queries: int = 1) -> int:
        """Charge ``queries`` against the budget; raises
        BudgetExhausted when the cap is spent (the overage is
        refused, never silently allowed)."""
        if queries <= 0:
            raise ValueError("queries must be positive")
        key = (payer, axis, int(epoch))
        cap = self._grants.get(key)
        if cap is None:
            raise BudgetExhausted(
                f"no budget granted for {payer} on {axis} in "
                f"epoch {epoch}")
        used = self._used.get(key, 0)
        if used + queries > cap:
            raise BudgetExhausted(
                f"budget exhausted for {payer} on {axis} in epoch "
                f"{epoch}: {used}/{cap} used")
        self._used[key] = used + queries
        return self._used[key]

    def rate(self, payer: str, axis: str, epoch: int) -> float:
        """The LOCAL rate: used / cap. Never published per payer;
        this is the gateway's own view of one of its users. Raises
        on an ungranted key."""
        key = (payer, axis, int(epoch))
        if key not in self._grants:
            raise BudgetExhausted(
                f"no budget granted for {payer} on {axis} in "
                f"epoch {epoch}")
        return self._used.get(key, 0) / float(self._grants[key])

    def axis_rate(self, axis: str, epoch: int) -> float:
        """The ONLY publishable budget view (M372, G8): the axis
        aggregate used/cap across all payers for the epoch. No payer
        dimension - this is what a network-level bound may rest on,
        and nothing finer may be published."""
        epoch = int(epoch)
        total_cap = sum(cap for (p, a, e), cap in self._grants.items()
                        if a == axis and e == epoch)
        if total_cap == 0:
            raise BudgetExhausted(
                f"no budget granted on {axis} in epoch {epoch}")
        total_used = sum(u for (p, a, e), u in self._used.items()
                         if a == axis and e == epoch)
        return total_used / float(total_cap)

    def exhausted(self, payer: str, axis: str, epoch: int) -> bool:
        key = (payer, axis, int(epoch))
        if key not in self._grants:
            return True
        return self._used.get(key, 0) >= self._grants[key]
