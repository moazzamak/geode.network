"""M312 — librarian containment for finding A14 (26 Aug 2026).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M312
before any build. A14: one role appends every ledger entry, and the
paper models only rewriting. Three repairs:

- **R-A14a force-inclusion queue.** Any party can post an entry
  directly to the settlement contract; the librarian must
  incorporate it within a registered window or the chain is invalid.
  Withholding, reordering, and stopping become visible violations.
- **R-A14b executable replacement.** A recorded divergence reason
  plus endorsements from a registered fraction of validators replaces
  the librarian at the next epoch — a mechanism, not a "recorded
  reason".
- **R-A14c liveness statistics.** Anchor cadence and inclusion
  latency are registered, measured, publicly visible statistics.

M365 (G24, 29 Aug 2026) amends R-A14a. Every posted entry used to
oblige the librarian within one window, so an attacker could buy
chain-invalidity by posting faster than any operator can
incorporate. The obligation is now CAPPED at
``MAX_INCORPORATIONS_PER_EPOCH`` per epoch; a backlog beyond the cap
rolls forward in posting order instead of invalidating the chain. A
censored entry is still guaranteed inclusion within a finite,
computable bound — it is just not instantaneous. This mirrors
``infrastructure/evm/contracts/InclusionInbox.sol``.

M366 (G25, 29 Aug 2026) amends R-A14b. Replacing the librarian used
to need endorsements from half of the *registered validators* — an
unweighted headcount, which is the cheapest quantity in the system
to Sybil (M335: at fee 0.01 the per-identity recovery is 0.4). Every
other governance action in GEODE uses earned weight with the
pedigree gate, the 20% cap, the diversity floor and the two-thirds
bar. The single most powerful action escaped it. It no longer does:
``replacement`` takes externally-verified weight (G13's
``geode.core.voting_weight``) and an injected ratification predicate,
and there is no headcount path left in this module.

The predicate is injected rather than imported because
``geode.privacy.vote_machinery`` sits outside what ``geode.core`` may
import. Injection is the repo's established answer to that (see
``geode.core.fund_pacing``), and it is required rather than
defaulted: a caller cannot fall back to a local copy of the rule,
because this module no longer contains one.

M388 (M382 remainder, 29 Aug 2026) closes the two R-A14b gaps the
review registered: the deputy was prose ("deterministic successor
order") with no code, and ``replacement()`` returned ``fires: true``
without naming who takes over. ``successor_order``/``deputy`` compute
the deputy deterministically from the ledger (epoch + anchor hash),
and ``replacement()`` names it when the caller supplies the roster.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Collection, Mapping, Sequence

from geode.core.bootstrap_council import (
    VOTING_CAP_BPS,
    EarnedWeightLedger,
)

INCLUSION_WINDOW_EPOCHS = 1      # R-A14a: incorporate within 1 epoch
MAX_INCORPORATIONS_PER_EPOCH = 8  # M365: the capped obligation

# The ratification predicate's signature, injected by the caller:
#   (support_weight, total_weight, supporting_identities,
#    pool_size, responders) -> (verdict, reason)
RatifyFn = Callable[
    [float, float, Sequence[str], int, int], tuple[bool, str]]


def deadline_epoch(posted_epoch: int, backlog_ahead: int,
                   window: int = INCLUSION_WINDOW_EPOCHS,
                   cap: int = MAX_INCORPORATIONS_PER_EPOCH) -> int:
    """M365: the epoch by which an entry must be incorporated.

    ``backlog_ahead`` is the number of entries already open when this
    one was posted. Because the librarian owes at most ``cap``
    incorporations per epoch, a backlog of ``b`` rolls the deadline
    forward by ``b // cap`` epochs. The bound is finite and
    computable at posting time, so the censorship guarantee survives
    the cap.
    """
    if int(cap) <= 0:
        raise ValueError("the incorporation cap must be positive")
    if int(backlog_ahead) < 0:
        raise ValueError("backlog_ahead must be non-negative")
    return (int(posted_epoch) + int(window)
            + int(backlog_ahead) // int(cap))


def post(queue: list[dict[str, Any]], entry_id: str,
         epoch: int,
         window: int = INCLUSION_WINDOW_EPOCHS,
         cap: int = MAX_INCORPORATIONS_PER_EPOCH
         ) -> list[dict[str, Any]]:
    """R-A14a: any party posts an entry directly to the settlement
    contract. Returns the updated queue.

    The deadline is fixed at posting time from the backlog ahead of
    the entry (M365), and is monotonic in posting order so that the
    oldest open entry always carries the earliest deadline.
    """
    if int(epoch) < 0:
        raise ValueError("epoch must be non-negative")
    backlog = sum(1 for e in queue if e["incorporated_epoch"] is None)
    deadline = deadline_epoch(epoch, backlog, window, cap)
    previous = max((e["deadline_epoch"] for e in queue), default=0)
    queue.append({"entry_id": str(entry_id), "posted_epoch": int(epoch),
                  "deadline_epoch": max(deadline, previous),
                  "incorporated_epoch": None})
    return queue


def due_entries(queue: list[dict[str, Any]], epoch: int
                ) -> list[dict[str, Any]]:
    """R-A14a: entries the librarian must have incorporated by this
    epoch — those whose capped deadline has arrived."""
    return [e for e in queue if e["incorporated_epoch"] is None
            and int(epoch) >= int(e["deadline_epoch"])]


def incorporate(queue: list[dict[str, Any]], entry_id: str,
                epoch: int) -> list[dict[str, Any]]:
    """R-A14a: the librarian incorporates an entry. An incorporation
    that lands after the deadline is a recorded violation (the chain
    is invalid from the deadline until then).

    Incorporation is FIFO (M365): the librarian may not serve a
    friend's entry ahead of a rival's.
    """
    open_entries = [e for e in queue if e["incorporated_epoch"] is None]
    if not open_entries:
        raise KeyError(f"entry {entry_id!r} is not open in the queue")
    head = open_entries[0]
    if head["entry_id"] != str(entry_id):
        if not any(e["entry_id"] == str(entry_id) for e in open_entries):
            raise KeyError(f"entry {entry_id!r} is not open in the queue")
        raise ValueError(
            f"incorporation is FIFO: {head['entry_id']!r} is the head "
            f"of the queue, not {entry_id!r}")
    head["incorporated_epoch"] = int(epoch)
    head["late"] = bool(int(epoch) > int(head["deadline_epoch"]))
    return queue


def chain_valid(queue: list[dict[str, Any]], epoch: int
                ) -> bool:
    """R-A14a: the chain is invalid while any entry sits
    unincorporated past its capped deadline."""
    return not any(
        e["incorporated_epoch"] is None
        and int(epoch) > int(e["deadline_epoch"])
        for e in queue)


def _successor_key(identity: str, epoch: int,
                   anchor_hash: bytes) -> bytes:
    """The ordering key for the deterministic successor order:
    H(identity || epoch || anchor_hash). The fields are separated so
    no concatenation of two inputs can collide with a third.
    """
    h = hashlib.sha256()
    h.update(str(identity).encode("utf-8"))
    h.update(b"\x00" + str(int(epoch)).encode("ascii"))
    h.update(b"\x01" + bytes(anchor_hash))
    return h.digest()


def successor_order(identities: Sequence[str], epoch: int,
                    anchor_hash: bytes = b"",
                    exclude: Collection[str] | None = None
                    ) -> list[str]:
    """M388: the deterministic successor order for the deputy role.

    Candidates are ranked by ``H(identity || epoch || anchor_hash)``
    ascending. The epoch and the anchor keep the order replayable and
    non-grindable: every party can recompute it from the ledger, and
    no party can know it far enough ahead to arrange the outcome.
    ``exclude`` drops the incumbent (and any other identity) from the
    order.
    """
    if int(epoch) < 0:
        raise ValueError("epoch must be non-negative")
    excluded = {str(i) for i in (exclude or ())}
    order = sorted(
        (str(i) for i in identities if str(i) not in excluded),
        key=lambda i: _successor_key(i, int(epoch),
                                     bytes(anchor_hash)))
    return order


def deputy(identities: Sequence[str], epoch: int,
           anchor_hash: bytes = b"",
           exclude: Collection[str] | None = None,
           eligible: Collection[str] | None = None) -> str | None:
    """M388: name the deterministic deputy.

    The deputy is the first identity in ``successor_order`` that is
    eligible (registered and pedigreed, when ``eligible`` is given)
    and not the incumbent. Returns ``None`` when no candidate
    remains — an empty roster names nobody without firing.
    """
    allowed = None if eligible is None else {str(i) for i in eligible}
    for identity in successor_order(identities, epoch, anchor_hash,
                                    exclude):
        if allowed is not None and identity not in allowed:
            continue
        return identity
    return None


def replacement(recorded_reason: str | None,
                endorsing_identities: Sequence[str],
                verified_weights: Mapping[str, float],
                responders: int,
                pool_size: int,
                ratify: RatifyFn,
                pedigreed: Collection[str] | None = None,
                cap_bps: int = VOTING_CAP_BPS,
                epoch: int | None = None,
                anchor_hash: bytes = b"",
                successor_identities: Sequence[str] | None = None,
                exclude_librarian: str | None = None) -> dict[str, Any]:
    """R-A14b as amended by M366: the executable replacement
    procedure, running on the same earned-weight rule as every other
    governance action.

    The TRIGGER stays mechanical — a recorded divergence reason is a
    replay-checkable fact, and no amount of weight substitutes for
    it. Only the ENDORSEMENT is weighted.

    ``verified_weights`` must be the externally-verified balance from
    ``geode.core.voting_weight.weight_snapshot_weights``, not the
    claimable one: cycled revenue buys money, never weight (G13).
    ``pedigreed``, when given, is the set of identities that clear the
    pedigree gate; endorsements from outside it are dropped and
    counted, never silently ignored. ``ratify`` is the shared
    ratification predicate (two-thirds plus the diversity floor).

    M388: when ``successor_identities`` is given (the roster of
    candidates, plus ``epoch`` and optionally ``anchor_hash`` and
    ``exclude_librarian``), the result names the deterministic deputy
    that takes over at the next epoch. ``epoch`` is required whenever
    the roster is supplied — the order is meaningless without it.

    Returns every intermediate quantity, so the verdict is auditable
    rather than a bare boolean.
    """
    if int(responders) <= 0:
        raise ValueError("responders must be positive")
    if int(pool_size) <= 0:
        raise ValueError("pool_size must be positive")
    if successor_identities is not None and epoch is None:
        raise ValueError(
            "epoch is required to name the deputy from a roster")

    endorsers = [str(i) for i in endorsing_identities]
    if len(set(endorsers)) != len(endorsers):
        raise ValueError("an identity may endorse only once")

    if pedigreed is None:
        eligible = list(endorsers)
        dropped: list[str] = []
    else:
        allowed = {str(i) for i in pedigreed}
        eligible = [i for i in endorsers if i in allowed]
        dropped = [i for i in endorsers if i not in allowed]

    # The 20% cap, applied to the same ledger every other governance
    # action uses. The denominator is the RAW total, not the capped
    # one: that is the registered M327 semantic ("three capped
    # identities reach only 60%"), and it is the only reading under
    # which the cap binds. Dividing capped support by a capped total
    # re-inflates a clipped whale back to a majority.
    ledger = EarnedWeightLedger()
    for identity, weight in verified_weights.items():
        ledger.earn(str(identity), float(weight))
    capped = ledger.capped(cap_bps)

    support_weight = sum(capped.get(i, 0.0) for i in eligible)
    total_weight = ledger.total()

    if not recorded_reason:
        fires, reason = False, "no_recorded_reason"
    else:
        fires, reason = ratify(support_weight, total_weight, eligible,
                               int(pool_size), int(responders))

    # M388: name the deputy deterministically. The roster is the set
    # of registered candidates; eligibility narrows it to pedigreed,
    # registered identities; the incumbent is excluded. An empty
    # roster names nobody, and that is reported rather than hidden.
    deputy_name = None
    if successor_identities is not None:
        deputy_name = deputy(successor_identities, int(epoch),
                             anchor_hash,
                             exclude=[exclude_librarian]
                             if exclude_librarian else None,
                             eligible=pedigreed)

    return {"fires": bool(fires),
            "reason": reason,
            "has_recorded_reason": bool(recorded_reason),
            "support_weight": support_weight,
            "total_weight": total_weight,
            "support_fraction": (support_weight / total_weight
                                 if total_weight > 0.0 else 0.0),
            "eligible_endorsers": eligible,
            "unpedigreed_dropped": dropped,
            "distinct_endorsers": len(set(eligible)),
            "cap_bps": int(cap_bps),
            "deputy": deputy_name,
            "deputy_named": deputy_name is not None,
            "deputy_roster_size": (len(successor_identities)
                                   if successor_identities is not None
                                   else 0)}


def liveness_report(anchor_epochs: list[int],
                    inclusion_latencies: list[int]) -> dict[str, Any]:
    """R-A14c: measured, public liveness statistics. ``anchor_epochs``
    are the epochs at which anchors landed; ``inclusion_latencies``
    are per-entry (incorporated - posted) epoch deltas. A stopped
    librarian reads as no anchors and unbounded latency."""
    anchors = [int(a) for a in anchor_epochs]
    latencies = [int(l) for l in inclusion_latencies]
    cadence: list[int] = [b - a for a, b in zip(anchors, anchors[1:])]
    stopped = not anchors
    return {
        "anchor_count": len(anchors),
        "max_anchor_gap": max(cadence) if cadence else math.inf,
        "mean_anchor_gap": (sum(cadence) / len(cadence)
                            if cadence else math.inf),
        "max_inclusion_latency": max(latencies) if latencies else math.inf,
        "mean_inclusion_latency": (sum(latencies) / len(latencies)
                                   if latencies else math.inf),
        "librarian_stopped": stopped,
        "unbounded_latency": not latencies,
    }
