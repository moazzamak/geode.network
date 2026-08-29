"""M319 — selective-abort adjudication (A18) and admission resampling
on quorum failure (A19).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M319
(26 Aug 2026, before any build). Two registered decision rules:

- **A18.** A probed session whose host committed its answer and then
  never opened the commitment is charged the full unit price of that
  session, and is escalated to a DEVIATION only when the epoch's
  aborts are *selective* — see M364 below. Commit-and-abort must never
  be a cheap way to refuse inspection.
- **A19.** A failed admission quorum resamples the validator set and
  carries the unspent budget forward — no new fee. Non-response in a
  sampled round accrues a demerit weighted by how close the session
  came to quorum (the closer, the more decisive the silence), applied
  to each non-responder.

M364 (G23, 29 Aug 2026) — the selectivity rule
----------------------------------------------
A18 originally adjudicated *every* unopened commit on a probed session
as a deviation. That made a burn remotely triggerable by a third
party: an attacker who denies service to a rival host immediately
after it commits converts the probe rate into burns, for the cost of
bandwidth. Level 0 says downtime carries no slash, so the two rules
collided, and the collision was the attacker's to fire.

An abort allowance does not close this. It prices the attack; it does
not remove the trigger, and the allowance is itself the attacker's
targeting parameter. What separates the two cases is a statistic:

- the host learns the probe flag only after it commits, and the
  attacker never learns it at all;
- so aborts caused by an attack fall on probed sessions at the probe
  rate, while aborts chosen to dodge inspection fall on probed
  sessions and nowhere else.

Among ``a`` aborts in an epoch, the count landing on probed sessions
is ``Binomial(a, rho)`` under the attack hypothesis and is ``a`` under
the dodging hypothesis. Escalation is therefore a one-sided binomial
test at a registered level, and an honest host under any size of
campaign is escalated only at that level's false-positive rate.

The unit-price charge is what keeps the dodge from being free while
the test is under-powered: a dodging host pays full price for every
session it refuses to have inspected, and earns nothing on it.
"""
from __future__ import annotations

import math
from typing import Any

# the slash ladder levels (mirrors geode.core.economics.SLASH_LADDER)
LEVEL_DOWNTIME = 0   # underperformance/downtime: the market penalizes
LEVEL_DEVIATION = 1  # deviation from the sealed artifact: burn unvested

# M364 registered parameters
ABORT_SELECTIVITY_ALPHA = 1e-3
"""One-sided significance for the selectivity test. An honest host
under attack is falsely escalated in at most this fraction of epochs
in which it aborts at all."""

ABORT_ESCALATION_MIN_ABORTS = 3
"""Small-sample floor: an epoch with fewer aborts than this never
escalates, whatever the probe rate. Without it a very low probe rate
would let a single unlucky abort clear ``alpha`` on its own.

This floor is deliberately a small CONSTANT and not a fraction of
traffic. A fraction was tried first (1% of committed sessions, as
G23's proposed repair registered) and the expected-profit sweep in
:func:`expected_host_profit` measured it as an economic hole: it
scales the dodge budget with the host's own traffic, so a large host
can hide cheating on 1% of its sessions indefinitely, and at the
registered cost parameters that is profitable. A constant floor keeps
the hidden quantity at a few sessions per epoch however large the host
is.
"""


def abort_allowance(committed_sessions: int,
                    minimum: int = ABORT_ESCALATION_MIN_ABORTS) -> int:
    """The small-sample floor, in aborts, for one epoch.

    Does not depend on ``committed_sessions`` — see
    :data:`ABORT_ESCALATION_MIN_ABORTS` for why a traffic-proportional
    floor was rejected. The argument is kept because the floor is a
    property of an epoch and callers hold the epoch, and because a
    later revision may want it back.

    This is a floor on *escalation* only; every abort is still charged
    the unit price from the first one.
    """
    if committed_sessions < 0:
        raise ValueError("committed_sessions must be non-negative")
    if minimum < 1:
        raise ValueError("minimum must be at least 1")
    return minimum


def _binomial_sf(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, p)``, summed in log space.

    Exact term-by-term summation rather than a normal approximation:
    the tail probabilities that decide an escalation run to 1e-9 and
    beyond, where a normal approximation is worthless. The terms are
    formed through ``lgamma`` because ``comb(9000, 450)`` overflows a
    float long before the term it belongs to underflows one.
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    log_p, log_q = math.log(p), math.log1p(-p)
    log_n_fac = math.lgamma(n + 1)

    def _term(i: int) -> float:
        log_t = (log_n_fac - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                 + i * log_p + (n - i) * log_q)
        # a term this small cannot move a sum whose leading term is
        # representable; exp() would raise or denormalise
        return 0.0 if log_t < -745.0 else math.exp(log_t)

    return min(1.0, math.fsum(_term(i) for i in range(k, n + 1)))


def abort_selectivity(aborts_probed: int, aborts_total: int,
                      probe_rate: float,
                      alpha: float = ABORT_SELECTIVITY_ALPHA
                      ) -> dict[str, Any]:
    """Test whether an epoch's aborts are aimed at probed sessions.

    Null hypothesis: the aborts were not chosen with knowledge of the
    probe flag, so each fell on a probed session independently with
    probability ``probe_rate``. Rejecting it is the only thing that
    turns an abort into a deviation.
    """
    if aborts_total < 0 or aborts_probed < 0:
        raise ValueError("abort counts must be non-negative")
    if aborts_probed > aborts_total:
        raise ValueError("aborts_probed cannot exceed aborts_total")
    if not 0.0 <= probe_rate <= 1.0:
        raise ValueError("probe_rate must be in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    p_value = (1.0 if aborts_total == 0
               else _binomial_sf(aborts_probed, aborts_total, probe_rate))
    return {"selective": p_value < alpha, "p_value": p_value,
            "aborts_probed": aborts_probed, "aborts_total": aborts_total,
            "probe_rate": probe_rate, "alpha": alpha}


def adjudicate_epoch_aborts(committed_sessions: int,
                            probed_sessions: int,
                            aborts_probed: int,
                            aborts_unprobed: int,
                            unit_price: float,
                            alpha: float = ABORT_SELECTIVITY_ALPHA,
                            escalation_min_aborts: int
                            = ABORT_ESCALATION_MIN_ABORTS
                            ) -> dict[str, Any]:
    """The M364 epoch-close rule for a host's aborted sessions.

    Returns the total unit-price charge (always levied, users
    refunded) and whether the aborts escalate to Level 1. Escalation
    needs BOTH more aborts than the small-sample floor AND a
    significant probed share; an attacker cannot supply the second
    because it cannot see the probe flag.
    """
    if committed_sessions < 0:
        raise ValueError("committed_sessions must be non-negative")
    if not 0 <= probed_sessions <= committed_sessions:
        raise ValueError("probed_sessions must be in [0, committed]")
    if aborts_probed < 0 or aborts_unprobed < 0:
        raise ValueError("abort counts must be non-negative")
    if aborts_probed + aborts_unprobed > committed_sessions:
        raise ValueError("aborts cannot exceed committed sessions")
    if unit_price < 0:
        raise ValueError("unit_price must be non-negative")

    aborts_total = aborts_probed + aborts_unprobed
    allowance = abort_allowance(committed_sessions,
                                escalation_min_aborts)
    probe_rate = (0.0 if committed_sessions == 0
                  else probed_sessions / committed_sessions)
    test = abort_selectivity(aborts_probed, aborts_total, probe_rate,
                             alpha)
    above_floor = aborts_total > allowance
    escalates = above_floor and test["selective"]
    return {
        "aborts_total": aborts_total,
        "aborts_probed": aborts_probed,
        "aborts_unprobed": aborts_unprobed,
        "allowance": allowance,
        "above_floor": above_floor,
        "selective": test["selective"],
        "p_value": test["p_value"],
        "probe_rate": probe_rate,
        # charged from the first abort, whatever the level
        "unit_price_charge": unit_price * aborts_total,
        "refunded_sessions": aborts_total,
        "escalates": escalates,
        "ladder_level": LEVEL_DEVIATION if escalates else LEVEL_DOWNTIME,
        "basis": ("M364: aborts concentrated on probed sessions beyond "
                  "chance — a refused inspection"
                  if escalates else
                  "M364: aborts are not aimed at probed sessions — "
                  "downtime, charged at the unit price, not slashed"),
    }


def expected_host_profit(cheat_fraction: float, strategy: str, *,
                         sessions: int, unit_price: float,
                         honest_cost: float, cheat_cost_ratio: float,
                         probe_rate: float, burn: float,
                         alpha: float = ABORT_SELECTIVITY_ALPHA,
                         escalation_min_aborts: int
                         = ABORT_ESCALATION_MIN_ABORTS
                         ) -> dict[str, Any]:
    """One epoch's expected profit for a host cheating at
    ``cheat_fraction`` under one of the four strategies M364 has to
    survive. Used to check the thesis "the only profitable behaviour
    is serving the artifact every time" by sweeping, not by assertion.

    - ``honest``: serve everything.
    - ``open_mismatch``: cheat and open anyway; caught at the probe
      rate, which is the pre-existing L1 path.
    - ``dodge``: cheat, and abort exactly the probed cheats. The
      strategy M364 exists to price.
    - ``camouflage``: cheat, and abort every cheated session, probed
      or not, so the aborts do not look aimed.
    """
    if not 0.0 <= cheat_fraction <= 1.0:
        raise ValueError("cheat_fraction must be in [0, 1]")
    if strategy not in ("honest", "open_mismatch", "dodge",
                        "camouflage"):
        raise ValueError(f"unknown strategy {strategy!r}")
    n = sessions
    cheated = round(n * cheat_fraction) if strategy != "honest" else 0
    honest_margin = unit_price - honest_cost
    cheat_cost = honest_cost * cheat_cost_ratio
    probed_cheats = round(cheated * probe_rate)

    if strategy in ("honest", "open_mismatch"):
        aborts_probed = aborts_unprobed = 0
    elif strategy == "dodge":
        aborts_probed, aborts_unprobed = probed_cheats, 0
    else:
        aborts_probed = probed_cheats
        aborts_unprobed = cheated - probed_cheats

    epoch = adjudicate_epoch_aborts(
        committed_sessions=n, probed_sessions=round(n * probe_rate),
        aborts_probed=aborts_probed, aborts_unprobed=aborts_unprobed,
        unit_price=unit_price, alpha=alpha,
        escalation_min_aborts=escalation_min_aborts)
    aborted = aborts_probed + aborts_unprobed

    profit = (n - cheated) * honest_margin
    # cheated sessions that were served and paid for
    profit += (cheated - aborted) * (unit_price - cheat_cost)
    # aborted sessions: compute already spent, no revenue, and the
    # unit-price charge on top
    profit -= aborted * (cheat_cost + unit_price)

    burned = epoch["escalates"]
    if burned:
        profit -= burn
    caught_probability = float(burned)
    if strategy == "open_mismatch" and cheated:
        # every cheated session is independently probed, so a cheater
        # that opens is caught unless every one of them escapes. Taking
        # round(cheated * probe_rate) here instead would let a small
        # cheat rate round its way to zero exposure.
        caught_probability = 1.0 - (1.0 - probe_rate) ** cheated
        profit -= caught_probability * burn

    return {"strategy": strategy, "cheat_fraction": cheat_fraction,
            "profit": profit, "burned": burned,
            "caught_probability": caught_probability,
            "escalates": epoch["escalates"], "aborted": aborted,
            "cheated": cheated, "p_value": epoch["p_value"],
            "baseline_profit": n * honest_margin}


def adjudicate_probed_session(commit_opened: bool, probed: bool,
                              answers_match: bool) -> dict[str, Any]:
    """The registered adjudication table for a single session.

    - opened + match: clean.
    - opened + mismatch: deviation (L1) — the pre-existing rule.
    - unopened: an ABORT. It is charged the unit price immediately,
      and its ladder level is not decidable from one session: whether
      it was a refused inspection or a denied service is a property of
      the epoch's aborts as a whole. See :func:`adjudicate_epoch_aborts`.

    M364 changed the last case. It previously returned L1 whenever the
    session was probed, which let a third party trigger a burn by
    denying service to a host that had already committed.
    """
    if commit_opened and answers_match:
        return {"verdict": "clean", "ladder_level": None,
                "charge_unit_price": False}
    if commit_opened:
        return {"verdict": "deviation", "ladder_level": LEVEL_DEVIATION,
                "charge_unit_price": False, "basis": "opened mismatch"}
    return {"verdict": "abort", "ladder_level": None,
            "charge_unit_price": True, "probed": probed,
            "basis": "M364: an unopened commit is charged the unit "
                     "price; its level is settled at epoch close by "
                     "the selectivity test, not by this session"}



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
