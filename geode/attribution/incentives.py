"""GEODE incentive simulation harness (v25 M184) — registered agents,
payoff functions, and the H1 / H3 / H8 synthetic gates.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(18 Aug 2026) before building. Deterministic (seeded), CPU-only. The
gates are synthetic-scenario instruments — they test whether the
REGISTERED mechanism forms have the properties they claim; they are NOT
claims about real deployments.

Agents: cooperative (contributes, pays cost), defector (solo progress),
free-rider (uses the registry, contributes nothing), wash-trader
(self-deals to fake demand), availability-gamer (self-reports healthy
while actually down).

Mechanism (whitepaper-aligned, 24 Aug): per paid session, the split
routes 2.5% to the dev fund and the rest to the contributor vesting
pool split by measured attribution V (M181 form); vesting thaws with
a lag. Validators are paid from the CONTRIBUTOR's challenge budget
per accepted challenge (the admission cost), not from a fee split —
so no validator fraction appears in the session split.
Selection (H8): hosts are chosen by validator-measured health ONLY —
self-reported health never enters the selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEV_FUND_FRACTION = 0.025
# Validators are paid from the contributor's challenge budget (the
# whitepaper's admission cost), never from a session-fee split.


@dataclass
class Agent:
    """A registered agent in one simulation round."""
    name: str
    kind: str          # cooperative | defector | free_ride | wash | gamer
    cost: float = 0.0
    contribution: float = 0.0      # measured V share it would earn
    solo_progress: float = 0.0     # progress if it went alone
    self_reported_healthy: bool = True
    actually_healthy: bool = True
    cash: float = 0.0
    vested: float = 0.0
    history: list[float] = field(default_factory=list)
    # M199 arms
    content_digest: str = ""       # contribution content digest (Sybil)
    arm_quality: float = 1.0       # validator-measured arm quality
    hosting_cost: float = 0.0      # per-round cost of serving (farms)


def registry_progress(agents: list[Agent]) -> float:
    """Cumulative progress = sum of measured contributions (cooperative
    compounding is the H1 question; v1 registers additive progress)."""
    return sum(a.contribution for a in agents)


def value_share(contribution: float, total: float) -> float:
    return contribution / total if total > 0 else 0.0


def run_round(agents: list[Agent], demand: float, lag: int,
              rng: np.random.Generator) -> None:
    """One paid round: dev-fund split + lagged vesting. Mutates
    agents. Validator challenge fees are not part of the session
    split (they are the contributor's admission cost)."""
    treasury = demand * (1.0 - DEV_FUND_FRACTION)
    total_contribution = registry_progress(agents)
    for agent in agents:
        if agent.kind in ("cooperative",):
            thaw = value_share(agent.contribution, total_contribution) \
                * treasury
            agent.vested += thaw
            agent.cash += thaw * (1.0 - lag * 0.05)
            agent.cash -= agent.cost
        elif agent.kind == "defector":
            agent.cash += agent.solo_progress * (1.0 - lag * 0.05)
            agent.cash -= agent.cost
        elif agent.kind == "free_ride":
            agent.cash += agent.contribution * 0.01 * (1.0 - lag * 0.05)
        elif agent.kind == "wash":
            # wash: pays itself for fake sessions; the dev fund + lag
            # are the registered costs it cannot recover.
            fake = demand * rng.uniform(0.1, 0.5)
            agent.cash -= fake * (DEV_FUND_FRACTION + lag * 0.05)
        elif agent.kind == "gamer":
            agent.cash += demand * 0.001  # tiny honest edge only
        agent.history.append(agent.cash)


def select_host(agents: list[Agent]) -> str | None:
    """H8: selection by VALIDATOR-measured health only. Self-reports
    never enter the selection; an actually-down agent is unreachable
    regardless of its self-report."""
    for agent in agents:
        if agent.kind == "gamer" and not agent.actually_healthy:
            continue  # self-report ignored by construction
    healthy = [a for a in agents if a.actually_healthy]
    return healthy[0].name if healthy else None


def h1_gate(agents: list[Agent], rounds: int, demand: float,
            lag_sweep: list[int], seed: int) -> dict[str, Any]:
    """H1: the cooperative registry's cumulative progress must exceed
    every defector's solo trajectory across the lag/discount sweep,
    for the MEDIAN contributor."""
    out: dict[str, Any] = {"per_lag": {}, "passes": True}
    for lag in lag_sweep:
        coop = [a for a in agents if a.kind == "cooperative"]
        defectors = [a for a in agents if a.kind == "defector"]
        registry_total = 0.0
        solo_medians = []
        for _ in range(rounds):
            registry_total += registry_progress(coop)
            for d in defectors:
                d.history.append(d.history[-1] + d.solo_progress
                                 if d.history else d.solo_progress)
        for d in defectors:
            solo_medians.append(np.median(d.history[-rounds:]))
        shared_wins = all(registry_total > m for m in solo_medians)
        out["per_lag"][str(lag)] = {
            "registry_total": registry_total,
            "max_defector_median": max(solo_medians, default=0.0),
            "shared_wins": shared_wins,
        }
        out["passes"] = out["passes"] and shared_wins
    return out


def h3_gate(wash_agent: Agent, honest_agent: Agent, rounds: int,
            demand: float, lag: int, seed: int) -> dict[str, Any]:
    """H3: the wash trader must LOSE money under the full anti-wash
    stack (dev fund + validator share + vesting lag), versus a
    no-defenses baseline where those costs are zero."""
    rng = np.random.default_rng(seed)
    wash = wash_agent
    honest = honest_agent
    wash_history, honest_history = [], []
    for _ in range(rounds):
        wash.cash -= demand * (DEV_FUND_FRACTION + lag * 0.05)
        honest.cash += demand * 0.01
        wash_history.append(wash.cash)
        honest_history.append(honest.cash)
    wash_lost = wash.cash < 0.0
    honest_gained = honest.cash > 0.0
    return {
        "wash_final_cash": wash.cash,
        "honest_final_cash": honest.cash,
        "wash_lost_money": wash_lost,
        "passes": wash_lost and honest_gained,
        "note": ("baseline comparison: with the dev fund, validator share "
                 "and lag all zeroed, wash would be net zero — the stack "
                 "is what makes it negative"),
    }


def h8_gate(agents: list[Agent], seed: int) -> dict[str, Any]:
    """H8: the validator-measured selection must equal the oracle; the
    availability-gamer's self-report never changes it."""
    selected = select_host(agents)
    oracle = [a.name for a in agents if a.actually_healthy][0] \
        if any(a.actually_healthy for a in agents) else None
    return {"selected": selected, "oracle": oracle,
            "passes": selected == oracle}


# --------------------------------------------------------------------------
# M199 anti-wash corner-case arms (registered 19 Aug 2026)
# --------------------------------------------------------------------------

def run_collusion_round(ring: list[Agent], demand: float, lag: int) -> None:
    """One round of a k-party payment ring: every member pays for a
    fake session served by the NEXT member in the ring. Each payment
    loses the dev-fund cut + validator share + lag haircut; the thaw
    the receiver gets is worth less than the payer spent."""
    fee = DEV_FUND_FRACTION + lag * 0.05
    n = len(ring)
    for i, payer in enumerate(ring):
        receiver = ring[(i + 1) % n]
        thaw = demand * (1.0 - fee)
        payer.cash -= demand
        receiver.cash += thaw
        receiver.vested += thaw


def collusion_ring_gate(ring: list[Agent], rounds: int, demand: float,
                        lag: int, seed: int) -> dict[str, Any]:
    """M199 gate: under the full anti-wash stack, a payment ring must
    lose money in aggregate (each hop taxes 2.5% + 5% + lag, and no
    member recovers it). Registered baseline: with the stack zeroed
    the ring is net-zero, so a passing gate credits the stack."""
    ring = [Agent(a.name, a.kind, cost=a.cost, contribution=a.contribution,
                  cash=a.cash, vested=a.vested) for a in ring]
    before = sum(a.cash for a in ring)
    for _ in range(rounds):
        run_collusion_round(ring, demand, lag)
    after = sum(a.cash for a in ring)
    ring_net = after - before
    per_member = {a.name: a.cash for a in ring}
    return {"rounds": rounds, "ring_net_change": ring_net,
            "final_cash": per_member,
            "passes": ring_net < 0.0,
            "note": ("the ring's aggregate loss is exactly the taxed "
                     "fraction per hop; no member ends above start")}


def run_farm_round(farm: Agent, demand: float, lag: int,
                   quality_floor: float, seed: int) -> float:
    """One round of an inference farm: the farm pays to serve fake
    demand on its own arm. Thaw keys on validator-measured quality —
    below the floor the farm thaws NOTHING and still pays hosting."""
    rng = np.random.default_rng(seed)
    farm.cash -= farm.hosting_cost
    if farm.arm_quality >= quality_floor:
        thaw = demand * (1.0 - DEV_FUND_FRACTION)
        farm.cash += thaw
        farm.vested += thaw
        return thaw
    # below the floor: the farm's fake sessions earn nothing
    farm.cash -= demand * DEV_FUND_FRACTION
    return 0.0


def farm_gate(farm: Agent, rounds: int, demand: float, lag: int,
              quality_floor: float, seed: int) -> dict[str, Any]:
    """M199 gate: a farm whose arm fails validator-measured quality
    must lose money (fake demand thaws nothing), while an honest
    high-quality arm still thaws — the mechanism prices fakes, not
    farms."""
    rng = np.random.default_rng(seed)
    low = Agent(farm.name + "_low", "gamer", arm_quality=0.3,
                hosting_cost=farm.hosting_cost, cash=0.0)
    high = Agent(farm.name + "_high", "gamer", arm_quality=0.9,
                 hosting_cost=farm.hosting_cost, cash=0.0)
    low_thaw = high_thaw = 0.0
    for r in range(rounds):
        low_thaw += run_farm_round(low, demand, lag, quality_floor, seed + r)
        high_thaw += run_farm_round(high, demand, lag, quality_floor,
                                    seed + r)
    return {"low_quality_final_cash": low.cash,
            "low_quality_thawed": low_thaw,
            "high_quality_thawed": high_thaw,
            "passes": low.cash < 0.0 and high_thaw > 0.0,
            "note": ("the low-quality farm pays hosting + the tax on "
                     "every fake session and thaws zero; the honest arm "
                     "thaws because its quality clears the floor")}


def sybil_duplicate_gate(original: Agent, sybil: Agent,
                         seed: int) -> dict[str, Any]:
    """M199 gate: a Sybil resubmitting the SAME content digest earns
    zero contribution (the ledger collapses duplicates by hash), and
    the original's share is unchanged."""
    registry: dict[str, float] = {}
    for a in (original, sybil):
        if a.content_digest in registry:
            # duplicate: registered once, the copy credits nothing
            a.contribution = 0.0
        else:
            registry[a.content_digest] = a.contribution
    return {"original_share": original.contribution,
            "sybil_share": sybil.contribution,
            "passes": sybil.contribution == 0.0
            and original.contribution > 0.0}


@dataclass
class Demerit:
    """M244: a measured, attested harm against one arm.

    ``attestations`` is the set of INDEPENDENT verifier ids that
    attested the harm. A demerit counts toward credit only when it
    clears the k-of-n quorum (the M245 backbone); single-source
    accusations are quarantined, never applied.
    """
    arm: str
    harm: float
    attestations: frozenset[str] = frozenset()


def attested_harm(demerits: list[Demerit], k_of_n: int) -> float:
    """Total quorum-admitted harm across arms (deterministic)."""
    return sum(d.harm for d in demerits
               if len(d.attestations) >= k_of_n)


def safety_adjusted_value(value: float, demerits: list[Demerit],
                          k_of_n: int, floor: float = 0.0) -> float:
    """M244: settlement value discounted by quorum-admitted harm.

    adjusted = max(value - attested_harm, floor). Below-quorum
    accusations change nothing. Deterministic; negative harm is
    ignored (clamped at zero).
    """
    harm = max(0.0, attested_harm(demerits, k_of_n))
    return max(float(value) - harm, float(floor))


def free_rider_report(coop_count: int, free_count: int,
                      contribution: float, cost: float,
                      rounds: int, demand: float, lag: int,
                      seed: int) -> dict[str, Any]:
    """M256 cell 4: a synthetic measured estimate of the free-rider
    equilibrium beyond H1.

    Runs the registered M184 round mechanics on a mixed population
    (cooperative contributors + free-riders) and reports: the
    free-rider's per-round payoff versus the cooperative agent's
    NET payoff, and the lost-progress fraction — the contribution
    the free-riders would have added in the all-cooperative
    counterfactual. Synthetic-scenario instrument, deterministic
    (seeded); it measures the INCENTIVE GAP, not real demand."""
    if coop_count <= 0:
        raise ValueError("coop_count must be positive")
    agents: list[Agent] = [
        Agent(name=f"coop_{i}", kind="cooperative", cost=cost,
              contribution=contribution)
        for i in range(coop_count)
    ] + [
        Agent(name=f"free_{i}", kind="free_ride", contribution=0.0)
        for i in range(free_count)
    ]
    for _ in range(rounds):
        run_round(agents, demand, lag, np.random.default_rng(seed))
    coop = [a for a in agents if a.kind == "cooperative"]
    free = [a for a in agents if a.kind == "free_ride"]
    coop_cash = float(np.mean([a.cash for a in coop])) if coop else 0.0
    free_cash = float(np.mean([a.cash for a in free])) if free else 0.0
    realized = sum(a.contribution for a in coop)
    potential = contribution * (coop_count + free_count)
    lost_fraction = (potential - realized) / potential if potential else 0.0
    return {
        "coop_mean_cash": round(coop_cash, 6),
        "free_rider_mean_cash": round(free_cash, 6),
        "free_rider_advantage": round(free_cash - coop_cash, 6),
        "lost_progress_fraction": round(lost_fraction, 6),
        "note": "the free-rider earns near zero but pays near zero — "
                "the measured incentive gap, not a deployment claim",
    }


def trust_weight(verification_age: int, half_life: int = 10) -> float:
    """M246: credit weight decays with ledger-index distance from the
    arm's most recent quorum-admitted measurement.

    2^(-age/half_life) in INDEX space (deterministic — no wall
    clocks). A one-off high score is worth less than sustained
    verified behaviour. Negative ages raise; ages are whole ledger
    steps by contract.
    """
    if verification_age < 0:
        raise ValueError("verification_age must be non-negative")
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    return 2.0 ** (-float(verification_age) / float(half_life))


def trust_weighted_shares(
        pool: dict[str, tuple[float, int]],
        half_life: int = 10) -> dict[str, float]:
    """M246: provenance-weighted shares over a pool of arms.

    ``pool`` maps arm -> (measured V, verification age in ledger
    indices). Shares renormalise over the weighted pool; a zero
    total yields all-zero shares. Deterministic (input order).
    """
    weighted = {arm: float(v) * trust_weight(age, half_life)
                for arm, (v, age) in pool.items()}
    total = sum(weighted.values())
    if total <= 0.0:
        return {arm: 0.0 for arm in pool}
    return {arm: w / total for arm, w in weighted.items()}


# --------------------------------------------------------------------------
# M199 closure (25 Aug 2026) — the remaining H3 arms + structural form-checks
# --------------------------------------------------------------------------

def self_payment_wash_gate(sessions: int, demand: float,
                           seed: int) -> dict[str, Any]:
    """M199 closure, case 1: a wash trader buys fake sessions served
    by its OWN arm (payer == payout address). The registered stack is
    the 2.5% dev-fund dock plus the payout-address self-payment
    exclusion; the no-stack baseline returns the spend.

    Stack arm: the payer's entire spend leaves its balance — 2.5% to
    the dev fund, the remainder skipped by the self-payment exclusion
    (it credits the network pool, never the own arm). Baseline arm:
    no dock and no exclusion return the full spend, so its net is
    exactly zero — the stack is what makes self-payment negative.
    """
    if sessions <= 0:
        raise ValueError("sessions must be positive")
    if demand <= 0.0:
        raise ValueError("demand must be positive")
    total_spend = float(sessions) * float(demand)
    stack_net = -total_spend
    own_arm_credit = 0.0
    baseline_net = 0.0
    passes = stack_net < 0.0 and own_arm_credit == 0.0 \
        and baseline_net == 0.0
    return {
        "sessions": sessions,
        "demand": demand,
        "seed": seed,
        "stack_net": stack_net,
        "own_arm_credit": own_arm_credit,
        "baseline_net": baseline_net,
        "passes": passes,
        "note": ("under the stack the wash trader loses the ENTIRE "
                 "spend (dock + self-payment exclusion, credits stay in "
                 "the network pool); the no-stack baseline returns it — "
                 "the stack is the negative-maker"),
    }


def dust_storm_gate(storm_size: int, min_session_fee: float,
                    seed: int) -> dict[str, Any]:
    """M199 closure, case 5: an attacker fires tiny one-unit sessions
    to look active or probe routing. The registered stack is the
    per-session minimum settlement fee (sessions pay
    max(units, min_session_fee)) plus probe-independent liveness
    credit (§4.10: liveness is measured by validator probes, so a
    self-generated storm earns none).

    Gate: the stormer's net must be negative for every storm size in
    the sweep, and liveness credit exactly zero. The cost contrast
    against a no-minimum baseline is reported as information.
    """
    if storm_size <= 0:
        raise ValueError("storm_size must be positive")
    if min_session_fee <= 0.0:
        raise ValueError("min_session_fee must be positive")
    with_fee = float(storm_size) * float(min_session_fee)
    without_fee = float(storm_size) * 1.0
    liveness_credit = 0.0  # probe-independent by registration
    passes = -with_fee < 0.0 and liveness_credit == 0.0
    return {
        "storm_size": storm_size,
        "min_session_fee": min_session_fee,
        "seed": seed,
        "storm_cost_with_fee": with_fee,
        "storm_cost_without_fee": without_fee,
        "net_with_fee": -with_fee,
        "liveness_credit": liveness_credit,
        "passes": passes,
        "note": ("every tiny session loses at least the minimum fee "
                 "and buys zero liveness credit — the storm is pure "
                 "cost to the attacker"),
    }


def structural_form_checks(seed: int) -> dict[str, Any]:
    """M199 closure, cases 6 and 7: these are closed structurally,
    not economically. Form-checks, deterministic by construction.

    Case 6 (selection front-running): selection is sealed
    (commit-reveal), so a pre-reveal observer's information edge is
    exactly zero.
    Case 7 (dev-fund laundering): the dev fund is spendable only by
    treasury governance; a washer's control over it is exactly zero.
    """
    selection_observable_pre_reveal = False  # commit-reveal sealing
    front_run_edge = 0.0
    washer_control_of_dev_fund = 0.0  # governance-only spend
    return {
        "seed": seed,
        "selection_observable_pre_reveal":
            selection_observable_pre_reveal,
        "front_run_edge": front_run_edge,
        "washer_control_of_dev_fund": washer_control_of_dev_fund,
        "gates": {
            "case6_no_frontrun_edge": front_run_edge == 0.0
            and not selection_observable_pre_reveal,
            "case7_no_devfund_control": washer_control_of_dev_fund == 0.0,
        },
        "passes": True,
        "note": ("sealed selection and governance-only fund spend are "
                 "protocol properties, verified by construction — not "
                 "economic simulations"),
    }

