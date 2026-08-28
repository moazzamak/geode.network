"""GEODE attacker-payoff cap (v25 M256 cell 3) — a registry too
small to be worth a nation-state budget is itself a security
property.

Three deterministic pieces: cap the per-session treasury so the
value of a captured decision is bounded; value the capture window
(the sessions a captured committee covers before its freeze window
expires — the M248 bound); and the gate that compares the window
value against an adversary's capture budget. No wall clocks, no
RNG: time is ledger-index space and the budget comparison is pure
arithmetic.
"""
from __future__ import annotations

from typing import Any


def capped_session_value(demand: float, cap: float) -> float:
    """The session treasury after the registered ceiling."""
    if demand < 0.0 or cap < 0.0:
        raise ValueError("demand and cap must be non-negative")
    return min(float(demand), float(cap))


def capture_window_value(capped_session: float,
                         window_sessions: int) -> float:
    """The total value a captured decision can influence before the
    freeze window closes (the M248 time-bound: capture is bounded)."""
    if window_sessions <= 0:
        raise ValueError("window_sessions must be positive")
    return float(capped_session) * int(window_sessions)


def capture_worth_budget(capture_value: float,
                         adversary_budget: float) -> dict[str, Any]:
    """The M256(3) gate: capture is rational only if the window
    value meets the adversary's budget; the design target is
    capture_value < budget (unprofitable capture)."""
    if capture_value < 0.0 or adversary_budget < 0.0:
        raise ValueError("values must be non-negative")
    return {
        "capture_value": float(capture_value),
        "adversary_budget": float(adversary_budget),
        "unprofitable": bool(capture_value < adversary_budget),
    }
