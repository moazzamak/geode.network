"""GEODE pricing-oracle study (v25 M186) — posted price vs second-price
auction vs epsilon-greedy bandit, on seeded synthetic demand traces.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(18 Aug 2026). Deterministic (seeded), CPU-only. A synthetic-scenario
study: the mechanisms are compared on the registered demand model; no
claim about real demand is made or licensed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DemandTrace:
    """A seeded per-round demand curve: a pool of n bidders, each with a
    willingness-to-pay drawn once (seeded), deciding per round by a
    registered arrival probability."""
    n: int
    wtp: np.ndarray
    arrival: float
    seed: int


def make_trace(n: int, arrival: float, seed: int) -> DemandTrace:
    rng = np.random.default_rng(seed)
    return DemandTrace(n=n, arrival=arrival,
                       wtp=rng.lognormal(mean=0.0, sigma=0.8, size=n),
                       seed=seed)


def demand_round(trace: DemandTrace, rng: np.random.Generator
                 ) -> np.ndarray:
    """Willingness-to-pay of the bidders who arrive this round."""
    arrivals = rng.random(trace.n) < trace.arrival
    return trace.wtp[arrivals]


def posted_price(trace: DemandTrace, price: float, rounds: int,
                 seed: int) -> dict[str, float]:
    """Fixed posted price: bidders with wtp >= price pay it."""
    rng = np.random.default_rng(seed)
    revenue = 0.0
    served = 0.0
    arrivals = 0.0
    for _ in range(rounds):
        wtps = demand_round(trace, rng)
        arrivals += len(wtps)
        paying = wtps >= price
        revenue += paying.sum() * price
        served += paying.sum()
    return {"revenue": revenue, "served": served,
            "arrivals": arrivals,
            "served_fraction": served / arrivals if arrivals else 0.0}


def second_price_auction(trace: DemandTrace, rounds: int,
                         seed: int) -> dict[str, float]:
    """Single-unit second-price auction per round among arrivals."""
    rng = np.random.default_rng(seed)
    revenue = 0.0
    served = 0.0
    arrivals = 0.0
    for _ in range(rounds):
        wtps = demand_round(trace, rng)
        arrivals += len(wtps)
        if len(wtps) >= 2:
            order = np.sort(wtps)[::-1]
            revenue += order[1]          # second-highest pays
            served += 1.0
    return {"revenue": revenue, "served": served,
            "arrivals": arrivals,
            "served_fraction": served / rounds}


def bandit_posted(trace: DemandTrace, price_grid: list[float],
                  rounds: int, epsilon: float,
                  seed: int) -> dict[str, float]:
    """Epsilon-greedy bandit over posted prices (mean revenue reward)."""
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(price_grid))
    means = np.zeros(len(price_grid))
    revenue = 0.0
    served = 0.0
    arrivals = 0.0
    for _ in range(rounds):
        if rng.random() < epsilon:
            arm = int(rng.integers(len(price_grid)))
        else:
            arm = int(np.argmax(means))
        price = price_grid[arm]
        wtps = demand_round(trace, rng)
        arrivals += len(wtps)
        paying = wtps >= price
        reward = paying.sum() * price
        revenue += reward
        served += paying.sum()
        counts[arm] += 1
        means[arm] += (reward - means[arm]) / counts[arm]
    return {"revenue": revenue, "served": served,
            "arrivals": arrivals,
            "served_fraction": served / arrivals if arrivals else 0.0,
            "arm_counts": counts.tolist()}


def study(config: dict[str, Any]) -> dict[str, Any]:
    """The registered comparison on one trace."""
    trace = make_trace(int(config["demand"]["n"]),
                       float(config["demand"]["arrival"]),
                       int(config["demand"]["seed"]))
    rounds = int(config["rounds"])
    results = {
        "posted": posted_price(trace, float(config["posted"]["price"]),
                               rounds, int(config["posted"]["seed"])),
        "auction": second_price_auction(trace, rounds,
                                        int(config["auction"]["seed"])),
        "bandit": bandit_posted(
            trace, [float(p) for p in config["bandit"]["price_grid"]],
            rounds, float(config["bandit"]["epsilon"]),
            int(config["bandit"]["seed"])),
    }
    return results
