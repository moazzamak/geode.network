"""M332 - extraction minimization: bucketed confidence, metered
abstentions, per-payer query budgets (R-A2a/b/c).

Registered in ``analysis/FEASIBILITY_THREAT_REVIEW_2026-08-28.md``
(F3, R-F3) and the review's queue (M332). The gap: the black-box
output was the softmax margin kappa(x) - a smooth function of
s = W^T z - and abstentions cost nothing. The trunk is a public
checkpoint, so an attacker computes z locally for free and
recovers W from margin-annotated responses: a linear system in
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
  epoch, with the ledger-visible used/cap rate as the enforcement
  surface.

The gate (a separate harness): an extraction simulation
recovering W from bucketed, metered responses costs more than the
head's expected lifetime revenue on the axis.
"""
from __future__ import annotations

from dataclasses import dataclass

# the registered abstention rate: half the unit price. An
# abstention consumed the trunk compute (the expensive part) and
# skipped only the head evaluation; a nonzero charge removes the
# free boundary-mapping oracle without making abstention
# punitive (the A7 balance).
ABSTENTION_PRICE_FRACTION = 0.5


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
    of the unit price, reduced but nonzero (R-A2b)."""
    if unit_price <= 0.0:
        raise ValueError("the unit price must be positive")
    if not 0.0 < fraction < 1.0:
        raise ValueError("the abstention fraction must lie in "
                         "(0, 1)")
    return float(unit_price) * float(fraction)


class PayerBudgetLedger:
    """Per-payer, per-axis, per-epoch query budgets (R-A2c). The
    enforcement surface is the ledger-visible rate: used / cap, per
    (payer, axis, epoch). Exhaustion refuses further queries -
    never silent."""

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
        """The ledger-visible rate: used / cap. Raises on an
        ungranted key (a rate for a missing budget is not a
        number)."""
        key = (payer, axis, int(epoch))
        if key not in self._grants:
            raise BudgetExhausted(
                f"no budget granted for {payer} on {axis} in "
                f"epoch {epoch}")
        return self._used.get(key, 0) / float(self._grants[key])

    def exhausted(self, payer: str, axis: str, epoch: int) -> bool:
        key = (payer, axis, int(epoch))
        if key not in self._grants:
            return True
        return self._used.get(key, 0) >= self._grants[key]
