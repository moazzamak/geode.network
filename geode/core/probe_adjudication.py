"""M319 — selective-abort adjudication (A18) and admission resampling
on quorum failure (A19).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M319
(26 Aug 2026, before any build). Two registered decision rules:

- **A18.** A probed session whose host committed its answer and then
  never opened the commitment is adjudicated as a DEVIATION, not as
  downtime. Commit-and-abort must cost at least as much as
  commit-and-mismatch: the cheap abort is the paper's blind spot, and
  the ladder level it earns is the same L1 the mismatch earns.
- **A19.** A failed admission quorum resamples the validator set and
  carries the unspent budget forward — no new fee. Non-response in a
  sampled round accrues a demerit weighted by how close the session
  came to quorum (the closer, the more decisive the silence), applied
  to each non-responder.
"""
from __future__ import annotations

import math
from typing import Any

# the slash ladder levels (mirrors geode.core.economics.SLASH_LADDER)
LEVEL_DOWNTIME = 0   # underperformance/downtime: the market penalizes
LEVEL_DEVIATION = 1  # deviation from the sealed artifact: burn unvested


def adjudicate_probed_session(commit_opened: bool, probed: bool,
                              answers_match: bool) -> dict[str, Any]:
    """The registered adjudication table for a probed session.

    - opened + match: clean.
    - opened + mismatch: deviation (L1) — the pre-existing rule.
    - unopened + probed: DEVIATION (L1) — A18: the selective abort is
      not downtime, it is a refused inspection.
    - unopened + not probed: downtime (L0) — the availability demerit
      remains the only cost outside the probe path.
    """
    if commit_opened and answers_match:
        return {"verdict": "clean", "ladder_level": None}
    if commit_opened:
        return {"verdict": "deviation", "ladder_level": LEVEL_DEVIATION,
                "basis": "opened mismatch"}
    if probed:
        return {"verdict": "deviation", "ladder_level": LEVEL_DEVIATION,
                "basis": "A18: unopened commit on a probed session "
                         "adjudicated as a deviation, not downtime"}
    return {"verdict": "downtime", "ladder_level": LEVEL_DOWNTIME,
            "basis": "unprobed availability demerit"}


def quorum_failure_plan(responders: int, sampled: int,
                        quorum_num: int, quorum_den: int,
                        unspent_budget: int) -> dict[str, Any]:
    """The registered A19 rule on quorum failure: resample the
    validator set, carry the unspent budget forward (no new fee), and
    accrue a per-non-responder demerit weighted by how close the
    session came to quorum. Raises on a nonsensical input, and returns
    a no-op plan when quorum was actually met."""
    if sampled <= 0:
        raise ValueError("sampled must be positive")
    if quorum_num <= 0 or quorum_den <= quorum_num:
        raise ValueError("quorum must be a proper fraction")
    if not 0 <= responders <= sampled:
        raise ValueError("responders must be in [0, sampled]")
    if unspent_budget < 0:
        raise ValueError("unspent_budget must be non-negative")
    needed = math.ceil(sampled * quorum_num / quorum_den)
    if responders >= needed:
        return {"quorum_failed": False, "resample": False,
                "budget_carried_forward": unspent_budget,
                "demerit_per_non_responder": 0.0}
    # proximity weight: a silence is more decisive the closer the
    # session came to quorum; registered as responders/sampled
    weight = responders / sampled
    return {"quorum_failed": True, "resample": True,
            "budget_carried_forward": unspent_budget,
            "new_fee_charged": False,
            "needed": needed, "responders": responders,
            "demerit_per_non_responder": weight,
            "non_responders": sampled - responders}
