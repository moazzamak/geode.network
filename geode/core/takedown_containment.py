"""M315 — takedown containment for finding A10 (26 Aug 2026).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M315
before any build. A10: with "discretionary power" precisely because
"no replay can settle it", a small colluding set on a thin axis can
irreversibly delete a competitor for one vote round. Four repairs:

- **R-A10a pool-scaled quorum.** Minimum responders scale with the
  pool; never a fixed floor of three.
- **R-A10b appeal path.** An appeal is admissible only if it cites at
  least one registered evidence class, even though the underlying
  judgement is not replayable.
- **R-A10c suspension before permanence.** First ratification
  suspends; permanence only on re-ratification after the suspension.
- **R-A10d revenue-scaled deposit.** The proposer deposit scales
  with the target's trailing revenue, so deleting a valuable
  artifact costs proportionally more than deleting a worthless one.
"""
from __future__ import annotations

import math
from typing import Any

from geode.core.economic_repairs import VERIFIED_SOURCES

# R-A10a: the quorum scales with the pool; floor registered in M314.
RESPONDER_SCALE = 0.1
TAKEDOWN_MIN_RESPONDERS_FLOOR = 3.0

# R-A10b: the registered evidence classes an appeal may cite.
APPEAL_EVIDENCE_CLASSES = frozenset((
    "probe_mismatch_record",
    "session_record",
    "meter_reading",
    "router_trace",
    "reference_run_record",
    "admission_draw",
))

# R-A10c: first ratification suspends for one epoch; permanence only
# on re-ratification after the window.
SUSPENSION_EPOCHS = 1

# R-A10d: the proposer deposit scales with trailing revenue.
DEPOSIT_SCALE = 0.5


def min_responders(pool_size: int) -> int:
    """R-A10a: minimum responders for a quorum, scaled to pool size
    and never below the registered floor."""
    size = int(pool_size)
    if size < 0:
        raise ValueError("pool_size must be non-negative")
    return max(int(TAKEDOWN_MIN_RESPONDERS_FLOOR),
               int(math.ceil(RESPONDER_SCALE * size)))


def appeal_admissible(evidence_classes: list[str] | tuple[str, ...]
                      | set[str]) -> dict[str, Any]:
    """R-A10b: an appeal is admissible only if it cites at least one
    registered evidence class. Returns the adjudication of
    admissibility plus which cited classes are registered."""
    cited = set(str(c) for c in evidence_classes)
    registered = cited & APPEAL_EVIDENCE_CLASSES
    return {"admissible": bool(registered),
            "registered_classes_cited": sorted(registered),
            "unregistered_classes_cited": sorted(
                cited - APPEAL_EVIDENCE_CLASSES)}


def takedown_step(ratification_number: int,
                  previous_suspended: bool) -> dict[str, Any]:
    """R-A10c: one step of the takedown ladder. First ratification
    suspends; a ratification that follows a completed suspension
    window delists permanently. Returns (suspended, delisted)."""
    number = int(ratification_number)
    if number < 1:
        raise ValueError("ratification_number must be >= 1")
    if number == 1:
        return {"suspended": True, "delisted": False,
                "suspension_epochs": SUSPENSION_EPOCHS}
    if previous_suspended:
        return {"suspended": False, "delisted": True,
                "suspension_epochs": 0}
    return {"suspended": True, "delisted": False,
            "suspension_epochs": SUSPENSION_EPOCHS}


def proposer_deposit(trailing_revenue: float) -> float:
    """R-A10d: the proposer deposit, scaled to the target's trailing
    revenue. Deleting a valuable artifact costs proportionally more
    than deleting a worthless one; zero revenue costs zero."""
    revenue = float(trailing_revenue)
    if revenue < 0.0:
        raise ValueError("trailing_revenue must be non-negative")
    return DEPOSIT_SCALE * revenue


def verified_trailing_revenue(session_records: list[dict[str, Any]],
                              value_field: str = "value",
                              source_field: str = "source") -> float:
    """M338 (F10-1): the trailing revenue that feeds the takedown
    deposit, computed from VERIFIED sessions only - the
    R-A20 filter already shipped in economic_repairs. A wash ring
    can self-generate volume at ~5% cost (it pays its own ring
    minus the dev cut), which would inflate a rival's takedown
    deposit; verified-only revenue removes the self-sourced
    inflation channel: ``probe_reference`` means a reference run
    on a probed session initiated by ANOTHER party, so a ring's
    own fake sessions never qualify."""
    total = 0.0
    for record in session_records:
        source = str(record.get(source_field, ""))
        if source not in VERIFIED_SOURCES:
            continue
        value = float(record.get(value_field, 0.0))
        if value < 0.0:
            raise ValueError("session values must be non-negative")
        total += value
    return total


def takedown_deposit_verified(session_records: list[dict[str, Any]],
                              value_field: str = "value") -> float:
    """M338 (F10-1): the proposer deposit on the verified-only
    trailing revenue (the wash-inflation channel closed)."""
    return proposer_deposit(verified_trailing_revenue(
        session_records, value_field=value_field))
