"""M309 - eval-custody repair: rows never leave the sealed scoring
environment.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md``
§8.32 (27 Aug 2026, before any build). The custody contradiction
(A9) is resolved in favour of the stronger clause: evaluation rows
never leave the sealed scoring environment. Validators submit
queries and receive aggregate verdicts only. Nobody holds rows, so
no shard is purchasable at any price - the purchasability gate is
computed, not asserted.

The module is an executable capability model in the M311/M317/M319
spec-module style: the policy it encodes is the deliverable.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

VERDICT_DIGITS = 4          # four significant digits, registered
SIG_FIG_BOUND = 10 ** (VERDICT_DIGITS - 1)


class RowEgressError(RuntimeError):
    """Raised when a code path tries to return evaluation rows out
    of the sealed scoring environment."""


class PurchasableShardError(RuntimeError):
    """Raised when the registered economics let an identity buy a
    shard for a registration fee."""


@dataclass(frozen=True)
class Shard:
    shard_id: str
    axis: str
    rows: tuple[int, ...]      # row indices in the sealed corpus
    canary_rows: tuple[int, ...] = ()  # overlap rows shared with others


@dataclass
class QueryRecord:
    validator: str
    axis: str
    query_hash: str
    verdict: float
    digits: int
    note: str = ""


@dataclass
class CustodyLedger:
    """Queries and aggregate verdicts only - never rows."""
    entries: list[QueryRecord] = field(default_factory=list)

    def record(self, record: QueryRecord) -> None:
        if any(k in record.note for k in ("row:", "code:")):
            raise RowEgressError("the custody ledger never records rows")
        self.entries.append(record)

    @property
    def verdicts(self) -> list[float]:
        return [entry.verdict for entry in self.entries]


class SealedScoringEnvironment:
    """Owns the evaluation corpora in shards. The only egress is an
    aggregate verdict per query, at bounded precision."""

    def __init__(self, shards: list[Shard] | None = None):
        self.shards: dict[str, Shard] = {}
        for shard in shards or []:
            self.shards[shard.shard_id] = shard
        self.ledger = CustodyLedger()

    # -- the only supported egress --------------------------------

    def score_query(self, validator: str, axis: str,
                    query_hash: str) -> float:
        """Score one registered query. Returns ONE aggregate
        verdict per axis, rounded to the registered precision. No
        row, code, or per-row output can leave through this path."""
        raw = self._aggregate(axis, query_hash)
        verdict = self._bounded(raw)
        self.ledger.record(QueryRecord(
            validator=validator, axis=axis, query_hash=query_hash,
            verdict=verdict, digits=VERDICT_DIGITS))
        return verdict

    # -- internals that must never leak ---------------------------

    def _aggregate(self, axis: str, query_hash: str) -> float:
        # deterministic aggregate from the axis's rows (a stand-in
        # for the sealed scorer; the contract is the egress policy)
        rows = [shard for shard in self.shards.values()
                if shard.axis == axis]
        if not rows:
            raise RowEgressError(f"no sealed corpus for axis {axis}")
        digest = int(hashlib.sha256(
            query_hash.encode("utf-8")).hexdigest()[:8], 16)
        return (digest % 1_000_000) / 1_000_000

    @staticmethod
    def _bounded(value: float) -> float:
        return round(value, VERDICT_DIGITS)

    # -- invariant checks -----------------------------------------

    def assert_no_row_egress(self, attempted_value: Any) -> None:
        """The registry-side invariant: anything that IS a row (or
        contains row material) must not be returned by any query
        path."""
        if _contains_rows(attempted_value):
            raise RowEgressError("evaluation rows never leave the "
                                 "sealed scoring environment")

    def shard_rows(self, shard_id: str) -> tuple[int, ...]:
        """Deliberately the ONE internal method that touches rows;
        it is not part of the query surface and raises on any
        caller that would transport its result out of the
        environment."""
        raise RowEgressError(
            "shard rows are environment-internal; the query surface "
            "returns aggregate verdicts only")


def _contains_rows(value: Any) -> bool:
    if isinstance(value, tuple) and value and all(
            isinstance(item, int) for item in value):
        return True   # a tuple of row indices IS row material
    if isinstance(value, list):
        return any(_contains_rows(item) for item in value)
    return False


# ----------------------------------------------------------------------
# R-A9a economics: validation is a service, not a yield source
# ----------------------------------------------------------------------

def identity_economics(registration_fee: float,
                       earnings_per_challenge: float,
                       challenges_per_epoch: float,
                       epochs: int) -> dict[str, float]:
    """Per-identity cashflow over the identity horizon. The repair:
    the registration cost must dominate honest earnings, so an
    identity is a cost, not a yield instrument."""
    gross = earnings_per_challenge * challenges_per_epoch * epochs
    return {
        "registration_fee": registration_fee,
        "honest_earnings": gross,
        "net_cashflow": gross - registration_fee,
        "service_not_yield": bool(gross - registration_fee < 0.0),
    }


def shard_purchasability(shard_value: float,
                         identity_cost: float) -> float:
    """How many identity costs one shard is worth. <= 1 means a
    shard is buyable for a single registration fee - the A9
    defect."""
    if identity_cost <= 0.0:
        return float("inf")
    return shard_value / identity_cost


def assert_not_purchasable(shard_value: float,
                           identity_cost: float) -> None:
    if shard_purchasability(shard_value, identity_cost) <= 1.0:
        raise PurchasableShardError(
            "a shard of the sealed corpus must cost more than one "
            "registration fee")


# ----------------------------------------------------------------------
# R-A9c fallback: overlap + canaries (modelled, not deployed)
# ----------------------------------------------------------------------

def canary_detection(divergence_on_overlap: float,
                     divergence_on_private: float,
                     threshold: float = 0.02) -> bool:
    """If custody is ever relaxed to shard-holding, overlap canaries
    catch the private-use signature: a shard holder that trains on
    its rows overfits them, and the overfit shows as divergence on
    the OVERLAPPING rows the holder cannot have seen in the same
    way. Detects when the overlap divergence is quiet (others agree)
    while the private-rows divergence is loud."""
    return (divergence_on_private - divergence_on_overlap) > threshold
