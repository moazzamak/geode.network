"""M356 (G26) — top-five crowding defense: traffic-share sweep.

Registered 29 Aug 2026, before the build. G26's gate:

    "The traffic-share sweep, extended with a crowding adversary:
    measure delivered accuracy at the user under (a) today's rule,
    (b) reserved slots, (c) the exponent. Publish all three."

Today's rule (M303, `rank_score`): v_a = s_a / (p_a * ubar_a), the
top-k lottery draws with weights = v_a. The crowding adversary (G26):
five barely-passing arms priced at the floor hold all five lottery
slots and evict better arms that price above the floor, because
quality is a gate, not a rank contributor, once through the gate.

The two repairs under test:

- **Reserved slots.** The top five is the UNION of the top three by
  v_a and the top two by s_a (a dominant-quality arm can never be
  evicted by price alone).
- **Quality exponent.** Rank on s_a^gamma / (p_a * ubar_a) with gamma
  registered per axis; gamma = 1 recovers today's rule.

Delivered accuracy at the user = sum over the top-k lottery pool of
(lottery weight) * (arm accuracy) — the expected accuracy of a served
query under the score-weighted lottery.

Registered scenario: floor 1.0; strong arm A (0.95) and good arm B
(0.85) priced at 2.0 (above the floor); a crowd of N arms at accuracy
0.60 (the admission floor) priced at 1.0 (the axis floor). All
expected_units = 1.0. The crowd outranks A and B under v_a because
price dominates.

Evidence: analysis/m356_top_five_crowding.json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "analysis" / "m356_top_five_crowding.json"

FLOOR = 1.0
TOP_K = 5
STRONG = {"accuracy": 0.95, "price": 2.0}
GOOD = {"accuracy": 0.85, "price": 2.0}
CROWD_ACC = 0.60


def v_score(acc: float, price: float, gamma: float = 1.0) -> float:
    """s_a^gamma / (p_a * ubar_a), ubar = 1.0 registered here."""
    return (acc ** gamma) / price


def delivered_accuracy(pool: list[dict]) -> float:
    """Expected accuracy of a served query under the score-weighted
    lottery: sum of (weight * accuracy) over the top-k pool."""
    total = sum(p["v"] for p in pool)
    if total <= 0.0:
        return 0.0
    return sum(p["v"] * p["accuracy"] for p in pool) / total


def pool_under(arms: list[dict], rule: str, gamma: float) -> list[dict]:
    """The top-k pool under one rule. rule: 'today' | 'reserved' |
    'exponent'. Returns pool entries with the v score used for lottery
    weights (v_a for today/reserved, s^gamma/(p ubar) for exponent)."""
    scored = []
    for arm in arms:
        v1 = v_score(arm["accuracy"], arm["price"], 1.0)   # v_a
        if rule == "exponent":
            s = v_score(arm["accuracy"], arm["price"], gamma)
        else:
            s = v1
        scored.append({"arm_id": arm["arm_id"], "accuracy": arm["accuracy"],
                       "price": arm["price"], "v": v1, "s": s,
                       "rank": s})
    if rule == "reserved":
        by_v = sorted(scored, key=lambda r: (-r["v"], r["arm_id"]))
        by_s = sorted(scored, key=lambda r: (-r["accuracy"], r["arm_id"]))
        top_v = by_v[:3]
        top_s = by_s[:2]
        pool_ids = []
        for r in top_v + top_s:
            if r["arm_id"] not in pool_ids:
                pool_ids.append(r["arm_id"])
        pool = [r for r in scored if r["arm_id"] in pool_ids]
        # lottery weights use v_a (today's score), membership is the fix
        for r in pool:
            r["rank"] = r["v"]
    else:
        pool = sorted(scored, key=lambda r: (-r["rank"], r["arm_id"]))[:TOP_K]
    for r in pool:
        r["weight"] = r["rank"]
    return pool


def run() -> int:
    results = {"scenario": {
        "floor": FLOOR, "top_k": TOP_K,
        "strong_A": STRONG, "good_B": GOOD,
        "crowd_accuracy": CROWD_ACC, "crowd_price": FLOOR,
        "expected_units": 1.0,
    }}
    rows = []
    for n_crowd in range(1, 9):
        arms = [{"arm_id": "A", **STRONG},
                {"arm_id": "B", **GOOD}]
        for i in range(n_crowd):
            arms.append({"arm_id": f"crowd{i}",
                         "accuracy": CROWD_ACC, "price": FLOOR})
        row = {"crowd_size": n_crowd}
        for rule, gamma in [("today", 1.0), ("reserved", 1.0),
                            ("exponent_g2", 2.0), ("exponent_g3", 3.0)]:
            pool = pool_under(arms, rule.split("_")[0] if rule.startswith(
                "exponent") else rule, gamma)
            row[rule] = {
                "delivered_accuracy": round(delivered_accuracy(pool), 4),
                "pool": [p["arm_id"] for p in pool],
            }
        rows.append(row)
    results["sweep"] = rows

    # the registered headline scenario (5 crowd arms)
    arms = [{"arm_id": "A", **STRONG}, {"arm_id": "B", **GOOD}]
    for i in range(5):
        arms.append({"arm_id": f"crowd{i}",
                     "accuracy": CROWD_ACC, "price": FLOOR})
    headline = {}
    for rule, gamma in [("today", 1.0), ("reserved", 1.0),
                        ("exponent_g2", 2.0)]:
        pool = pool_under(arms, rule.split("_")[0] if rule.startswith(
            "exponent") else rule, gamma)
        headline[rule] = {
            "delivered_accuracy": round(delivered_accuracy(pool), 4),
            "pool": [p["arm_id"] for p in pool],
            "weights": {p["arm_id"]: round(p["weight"], 4) for p in pool},
        }
    results["headline_5_crowd"] = headline
    results["gate"] = ("delivered accuracy under the crowding "
                       "adversary published for all three rules")

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
