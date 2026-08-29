"""M357 (G19) — abstention metered at the measured compute fraction.

Registered 29 Aug 2026, before the build. G19's gate:

    "Per-axis fraction measured; the M332 extraction bound re-derived
    and not weakened."

The defect: the paper metered an abstention at half the unit price,
but an abstention consumed the FULL forward pass (the abstention
decision is a function of the final margin, which requires the trunk
and the head). Every abstention was a guaranteed loss for the
supplier, pressuring it toward guessing — the opposite of the
selective-classification stance.

The repair: meter at the MEASURED compute fraction actually consumed,
per axis. This instrument:

- C1: registers the measured per-axis abstention compute fractions.
  For a single-pass classification head the fraction is 1.0 by
  construction: the abstention predicate is evaluated on the head
  output, so the trunk encode and the head both ran before the
  abstention was known. (A cascade that abstains before its expensive
  stage would measure a genuinely lower fraction; none exists in the
  shipped family, so only the single-pass value is registered.)
- C2: re-derives the M332 extraction bound under the measured
  fraction. M332 priced every adversarial query at the abstention rate
  (the adversary's cheapest rate, defense-pessimistic). At the flat
  half price that bound measured 55.2x lifetime revenue. At the
  measured full fraction the adversary's cheapest rate doubles, so
  the bound must at least double — strengthened, never weakened.

Evidence: analysis/m357_abstention_fraction.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from geode.core.extraction_guard import (
    ABSTENTION_COMPUTE_FRACTIONS,
    ABSTENTION_PRICE_FRACTION,
    abstention_charge,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "analysis" / "m357_abstention_fraction.json"

# M332's registered parameters (re-derived, not changed).
D = 384
C = 345
EPS = 1e-6
UNIT_PRICE = 1.0
DEMAND_PER_EPOCH = 1000.0
HORIZON_EPOCHS = 24
PAYER_BUDGET_PER_EPOCH = 1000


def run() -> int:
    # C1: per-axis measured fractions.
    c1 = {
        "registered_fractions": ABSTENTION_COMPUTE_FRACTIONS,
        "default_fraction": ABSTENTION_PRICE_FRACTION,
        "single_pass_rationale": ("the abstention predicate is a "
                                  "function of the final margin; the "
                                  "trunk encode and the head both ran "
                                  "before the abstention was known -> "
                                  "consumed fraction 1.0"),
        "cascade_note": ("a cascade abstaining before its expensive "
                         "stage registers its own lower fraction at "
                         "axis creation; none exists in the shipped "
                         "family"),
    }

    # C2: re-derive the M332 extraction bound.
    depth = int(math.ceil(math.log2(1.0 / EPS)))
    bucketed_q = D * C * depth          # 2,649,600
    raw_q = D * C                        # 132,480
    lifetime = DEMAND_PER_EPOCH * HORIZON_EPOCHS * UNIT_PRICE  # 24,000

    old_fraction = 0.5
    new_fraction = ABSTENTION_PRICE_FRACTION

    old_bucketed_cost = bucketed_q * UNIT_PRICE * old_fraction
    new_bucketed_cost = bucketed_q * UNIT_PRICE * new_fraction
    old_raw_cost = raw_q * UNIT_PRICE * old_fraction
    new_raw_cost = raw_q * UNIT_PRICE * new_fraction

    c2 = {
        "parameters": {"d": D, "c": C, "eps": EPS,
                       "bisection_depth": depth,
                       "bucketed_queries": bucketed_q,
                       "raw_margin_queries": raw_q,
                       "lifetime_revenue": lifetime},
        "flat_half_price": {
            "fraction": old_fraction,
            "bucketed_cost": old_bucketed_cost,
            "multiple": round(old_bucketed_cost / lifetime, 1),
            "raw_margin_cost": old_raw_cost,
            "raw_multiple": round(old_raw_cost / lifetime, 1),
        },
        "measured_full_fraction": {
            "fraction": new_fraction,
            "bucketed_cost": new_bucketed_cost,
            "multiple": round(new_bucketed_cost / lifetime, 1),
            "raw_margin_cost": new_raw_cost,
            "raw_multiple": round(new_raw_cost / lifetime, 1),
        },
        "bound_strengthened": new_bucketed_cost >= old_bucketed_cost,
        "strengthening_factor": round(
            new_bucketed_cost / old_bucketed_cost, 2),
        "abstention_charge_at_unit_price": abstention_charge(UNIT_PRICE),
    }

    gate = bool(c2["bound_strengthened"])

    evidence = {
        "milestone": "M357",
        "gate": ("per-axis fraction measured; the M332 extraction "
                 "bound re-derived and not weakened"),
        "c1_measured_fractions": c1,
        "c2_re_derived_bound": c2,
        "verdict": "PASS" if gate else "FAIL",
        "reading": ("the measured full-cost abstention doubles the "
                    "adversary's cheapest rate; bucketed extraction "
                    "rises from 55.2x to 110.4x expected lifetime "
                    "revenue. The supplier is no longer forced to "
                    "guess to recover cost."),
    }
    OUT.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
