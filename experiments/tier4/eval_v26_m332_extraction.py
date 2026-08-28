"""M332 gate - the extraction-cost simulation: does recovering W
from bucketed, metered responses cost more than the head's
expected lifetime revenue?

Registered in ``analysis/FEASIBILITY_THREAT_REVIEW_2026-08-28.md``
(F3 gate, R-F3) before the build. The A2 finding: with raw
margin-annotated responses, recovering W in R^{d x C} is a linear
system in d*C unknowns - a low six-figure query count. With
bucketed confidence the adversary is reduced to the labels-only
boundary-search regime: for each class column, d boundary points
found by bisection along d directions, each bisection costing
ceil(log2(1/eps)) queries (eps = the relative boundary
resolution).

Registered parameters (all fixed here, before the run):
- d=384, C=345: the A2 text's vision numbers (CLIP-L-class head).
- eps = 1e-6: bisection depth 20.
- raw-margin extraction: d*C = 132,480 queries (the linear
  system, the pre-repair oracle).
- bucketed extraction: d*C*20 = 2,649,600 queries (the
  labels-only worst case; the buckets are not credited to the
  adversary - conservative for the defense is conservative the
  wrong way, so no bucket benefit is claimed).
- every adversarial query is priced at the ABSTENTION rate (half
  the unit price): the adversary's cheapest rate, the
  defense-pessimistic choice.
- lifetime revenue: the registered M293 scenario demand economics
  - demand 1000 queries/epoch, horizon 24 epochs, unit price 1.0
  (revenue 24,000 units).
- per-payer budget: 1000 queries/epoch on the axis (the axis's
  registered reference demand - a payer may buy at most one
  epoch's demand per epoch), so bucketed extraction needs
  ceil(2,649,600 / 1000) = 2650 epochs.

Gate (registered before the run): the bucketed extraction cost
exceeds the head's expected lifetime revenue on the axis.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from geode.core.extraction_guard import ABSTENTION_PRICE_FRACTION

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m332_extraction")

# the registered gate parameters (the A2 numbers)
D = 384
C = 345
EPS = 1e-6
UNIT_PRICE = 1.0
DEMAND_PER_EPOCH = 1000.0    # the M293 scenario demand
HORIZON_EPOCHS = 24           # the M293 scenario horizon
PAYER_BUDGET_PER_EPOCH = 1000


def raw_margin_extraction_queries(d: int, c: int) -> int:
    """The pre-repair oracle: a linear system in d*c unknowns,
    solved from margin-annotated responses."""
    return d * c


def bucketed_extraction_queries(d: int, c: int, depth: int) -> int:
    """The labels-only boundary-search worst case: per class
    column, d bisections of ``depth`` queries each."""
    return d * c * depth


def run_m332(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    depth = int(math.ceil(math.log2(1.0 / EPS)))
    raw_q = raw_margin_extraction_queries(D, C)
    bucketed_q = bucketed_extraction_queries(D, C, depth)

    # the adversary's cheapest metered rate: the abstention price
    adv_rate = UNIT_PRICE * ABSTENTION_PRICE_FRACTION
    raw_cost = raw_q * adv_rate
    bucketed_cost = bucketed_q * adv_rate
    lifetime_revenue = DEMAND_PER_EPOCH * HORIZON_EPOCHS * UNIT_PRICE
    budget_epochs = int(math.ceil(bucketed_q / PAYER_BUDGET_PER_EPOCH))

    gate = bool(bucketed_cost > lifetime_revenue)
    reading = (
        "PASS: bucketed extraction costs "
        f"{bucketed_cost / lifetime_revenue:.1f}x the head's "
        "expected lifetime revenue - the bucketed, metered, "
        "budgeted oracle is uneconomic to extract against"
        if gate else
        "FAIL: bucketed extraction costs less than the head's "
        "expected lifetime revenue - the repairs do not close A2 "
        "at these parameters")

    evidence: dict[str, Any] = {
        "milestone": "M332",
        "cell": ("extraction-cost gate: raw-margin oracle vs "
                 "bucketed, metered, budgeted responses against "
                 "the head's expected lifetime revenue"),
        "parameters": {
            "d": D, "c": C, "eps": EPS, "bisection_depth": depth,
            "unit_price": UNIT_PRICE,
            "abstention_fraction": ABSTENTION_PRICE_FRACTION,
            "demand_per_epoch": DEMAND_PER_EPOCH,
            "horizon_epochs": HORIZON_EPOCHS,
            "payer_budget_per_epoch": PAYER_BUDGET_PER_EPOCH,
        },
        "extraction": {
            "raw_margin_queries": raw_q,
            "raw_margin_cost": raw_cost,
            "bucketed_queries": bucketed_q,
            "bucketed_cost": bucketed_cost,
            "budget_epochs": budget_epochs,
            "lifetime_revenue": lifetime_revenue,
        },
        "gate": {"ok": gate,
                 "registered": ("bucketed extraction cost exceeds "
                                "the head's expected lifetime "
                                "revenue on the axis")},
        "reading": reading,
        "configuration_hash": payload_hash({
            "d": D, "c": C, "eps": EPS,
            "unit_price": UNIT_PRICE,
            "abstention_fraction": ABSTENTION_PRICE_FRACTION,
            "demand_per_epoch": DEMAND_PER_EPOCH,
            "horizon_epochs": HORIZON_EPOCHS,
            "payer_budget_per_epoch": PAYER_BUDGET_PER_EPOCH}),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gate": gate,
                      "raw_margin_cost": raw_cost,
                      "bucketed_cost": bucketed_cost,
                      "lifetime_revenue": lifetime_revenue,
                      "budget_epochs": budget_epochs,
                      "reading": reading}, indent=1), flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m332(args.output)


if __name__ == "__main__":
    main()
