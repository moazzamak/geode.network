"""Coalition-game attribution, incentives, and pricing (the economics
of contributor compensation).

Layering rule: imports only ``geode.*`` and standard libraries.
"""
from geode.attribution.attribution import (
    beta_shapley,
    fingerprint_coverage,
    leave_one_out,
    rank_order,
    ranking_stability,
    shapley,
)
from geode.attribution.incentives import (
    Agent,
    Demerit,
    free_rider_report,
    run_round,
    safety_adjusted_value,
    select_host,
    trust_weight,
    trust_weighted_shares,
)
from geode.attribution.payoff_cap import (
    capped_session_value,
    capture_window_value,
    capture_worth_budget,
)
from geode.attribution.pricing import DemandTrace, make_trace, study
from geode.attribution.stake import (
    MeasurementClass,
    minimum_bond,
    simulate_liar,
    stake_schedule,
)

__all__ = [
    "Agent",
    "DemandTrace",
    "Demerit",
    "MeasurementClass",
    "beta_shapley",
    "capped_session_value",
    "capture_window_value",
    "capture_worth_budget",
    "fingerprint_coverage",
    "free_rider_report",
    "leave_one_out",
    "make_trace",
    "minimum_bond",
    "rank_order",
    "ranking_stability",
    "run_round",
    "safety_adjusted_value",
    "select_host",
    "shapley",
    "simulate_liar",
    "stake_schedule",
    "study",
    "trust_weight",
    "trust_weighted_shares",
]
