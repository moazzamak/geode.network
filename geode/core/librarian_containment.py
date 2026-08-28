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
"""
from __future__ import annotations

import math
from typing import Any

INCLUSION_WINDOW_EPOCHS = 1      # R-A14a: incorporate within 1 epoch
REPLACEMENT_THRESHOLD = 0.5      # R-A14b: fraction of validators


def post(queue: list[dict[str, Any]], entry_id: str,
         epoch: int) -> list[dict[str, Any]]:
    """R-A14a: any party posts an entry directly to the settlement
    contract. Returns the updated queue."""
    if int(epoch) < 0:
        raise ValueError("epoch must be non-negative")
    queue.append({"entry_id": str(entry_id), "posted_epoch": int(epoch),
                  "incorporated_epoch": None})
    return queue


def due_entries(queue: list[dict[str, Any]], epoch: int,
                window: int = INCLUSION_WINDOW_EPOCHS
                ) -> list[dict[str, Any]]:
    """R-A14a: entries the librarian must have incorporated by this
    epoch (posted at least ``window`` epochs ago)."""
    return [e for e in queue if e["incorporated_epoch"] is None
            and int(epoch) - int(e["posted_epoch"]) >= int(window)]


def incorporate(queue: list[dict[str, Any]], entry_id: str,
                epoch: int) -> list[dict[str, Any]]:
    """R-A14a: the librarian incorporates an entry. An incorporation
    that lands after the window is a recorded violation (the chain is
    invalid from the posting deadline until then)."""
    for e in queue:
        if e["entry_id"] == str(entry_id) and e["incorporated_epoch"] is None:
            e["incorporated_epoch"] = int(epoch)
            e["late"] = bool(
                int(epoch) - int(e["posted_epoch"])
                > INCLUSION_WINDOW_EPOCHS)
            break
    else:
        raise KeyError(f"entry {entry_id!r} is not open in the queue")
    return queue


def chain_valid(queue: list[dict[str, Any]], epoch: int
                ) -> bool:
    """R-A14a: the chain is invalid while any entry sits
    unincorporated past its window."""
    return not any(
        e["incorporated_epoch"] is None
        and int(epoch) - int(e["posted_epoch"])
        > INCLUSION_WINDOW_EPOCHS
        for e in queue)


def replacement(endorsements: int, validators: int,
                recorded_reason: str | None) -> dict[str, Any]:
    """R-A14b: the executable replacement procedure. Fires only with
    a recorded divergence reason and endorsements at/above the
    registered threshold; the deputy operator takes over at the next
    epoch."""
    if int(validators) <= 0:
        raise ValueError("validators must be positive")
    n = int(endorsements)
    if n < 0 or n > int(validators):
        raise ValueError("endorsements must lie in [0, validators]")
    fraction = n / int(validators)
    fires = bool(recorded_reason and fraction >= REPLACEMENT_THRESHOLD)
    return {"fires": fires,
            "endorsement_fraction": fraction,
            "threshold": REPLACEMENT_THRESHOLD,
            "has_recorded_reason": bool(recorded_reason)}


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
