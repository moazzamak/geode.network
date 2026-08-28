"""M303 harness — the registered H26-7 sweep over the repaired router.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M303
(26 Aug 2026, before any build). H26-7 asks: in an M293-style
scenario harness, the repaired router (price floor + anchor-seeded
tie-break + top-k lottery + expected-charge ranking) yields no
price-to-zero equilibrium, no single-winner capture, and no bloat
advantage, on the registered sweep.

Registered cells (written before running):

- **C1 price race.** Two equal-quality arms; the challenger sweeps its
  price down to the floor and below (M303a checks, registered before
  the re-run): below-floor registration is REJECTED; the challenger's
  share rises monotonically as its price falls (price competition is a
  feature of the registered s/(p*u) score); the floor share sits in
  the registered band [0.60, 0.75] (the 2:1 score ratio); no
  above-floor price reaches 90% share (capture is the degenerate
  outcome).
- **C1b equal-price tie.** An equal-priced equal-quality pair at the
  floor splits ~50/50.
- **C2 single-winner capture.** A quality leader over two followers,
  equal prices and units: the leader's share over 2000 anchors must be
  strictly below 100% and below 60%, and the followers win sometimes.
- **C3 bloat.** Equal quality and price; the bloated arm's expected
  units sweep {1, 2, 5, 10}: its share must shrink toward 1/ubar and
  fall below 12% at ubar=10.
- **C4 determinism.** Same anchor twice gives the identical winner and
  seed; distinct anchors give distinct seeds on a tie.
- **C5 floor enforcement.** A price one tick below the floor is
  rejected at registration.

Each cell records which repair closed it (floor / lottery /
expected-charge / anchor tie-break).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from geode.core.router_repair import RepairedRouter

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
                  / "m303_router_repair.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v26" / "m303_router_repair"

FP = [1.0, 0.0]


def _arm(arm_id: str, acc: float, price: float,
         ubar: float | None = None) -> dict:
    spec = {"arm_id": arm_id, "held_out_accuracy": acc, "price": price,
            "availability": {"healthy": True}}
    if ubar is not None:
        spec["expected_units"] = ubar
    return spec


def _share(router: RepairedRouter, n: int) -> dict[str, float]:
    counts: dict[str, int] = {}
    for i in range(n):
        out = router.route(FP, anchor=f"anchor-{i}")
        if not out:
            continue
        winner = out[0]["arm_id"]
        counts[winner] = counts.get(winner, 0) + 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def run_m303(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    floor = float(config["price_floor"])
    n_anchors = int(config["n_anchors"])

    cells: dict[str, Any] = {}

    # ---- C1 price race (M303a checks, registered before re-run) --------
    c1 = {"challenger_shares": {}, "below_floor_rejected": None}
    for price_b in config["c1_price_sweep"]:
        router = RepairedRouter(price_floor=floor)
        router.add_arm(_arm("incumbent", 0.60, 2.0))
        try:
            router.add_arm(_arm("challenger", 0.60, float(price_b)))
            share = _share(router, n_anchors).get("challenger", 0.0)
            c1["challenger_shares"][str(price_b)] = share
        except ValueError:
            c1["below_floor_rejected"] = float(price_b)
    above = {float(p): s for p, s in c1["challenger_shares"].items()
             if float(p) >= floor}
    prices_desc = sorted(above, reverse=True)
    # registered: share RISES as price FALLS -> along descending
    # prices the share must increase, not decrease
    c1["monotone_as_price_falls"] = bool(
        len(prices_desc) >= 2 and all(
            above[prices_desc[i]] < above[prices_desc[i + 1]]
            for i in range(len(prices_desc) - 1)))
    floor_share = above.get(floor, None)
    c1["floor_share_in_band"] = bool(
        floor_share is not None and 0.60 <= floor_share <= 0.75)
    c1["no_capture_above_floor"] = bool(
        all(s < 0.90 for s in above.values()))
    c1["below_floor_rejected_at_registration"] = bool(
        c1["below_floor_rejected"] is not None
        and c1["below_floor_rejected"] < floor)
    c1["closed_by"] = "price floor (R-A3a) + expected-charge ranking " \
                      "(R-A3d)"
    cells["c1_price_race"] = c1

    # ---- C1b equal-price tie at the floor --------------------------------
    router = RepairedRouter(price_floor=floor)
    router.add_arm(_arm("tie_a", 0.60, floor))
    router.add_arm(_arm("tie_b", 0.60, floor))
    tie_share = _share(router, n_anchors).get("tie_a", 0.5)
    c1b = {"share_tie_a": tie_share,
           "tie_splits": abs(tie_share - 0.5) <= 0.05,
           "closed_by": "lottery over the tie (R-A3c) + anchor-seeded "
                        "tie-break (R-A3b)"}
    cells["c1b_equal_price_tie"] = c1b

    # ---- C2 single-winner capture ----------------------------------------
    router = RepairedRouter(price_floor=floor)
    router.add_arm(_arm("leader", 0.70, 1.0))
    router.add_arm(_arm("follower1", 0.69, 1.0))
    router.add_arm(_arm("follower2", 0.60, 1.0))
    shares = _share(router, n_anchors)
    c2 = {"shares": shares,
          "leader_below_100pct": shares.get("leader", 1.0) < 1.0,
          "leader_below_60pct": shares.get("leader", 1.0) < 0.60,
          "followers_win_sometimes":
              shares.get("follower1", 0.0) > 0.0
              and shares.get("follower2", 0.0) > 0.0,
          "closed_by": "top-k lottery (R-A3c)"}
    cells["c2_single_winner_capture"] = c2

    # ---- C3 bloat ---------------------------------------------------------
    c3 = {"bloated_shares": {}}
    for ubar in config["c3_units_sweep"]:
        router = RepairedRouter(price_floor=floor)
        router.add_arm(_arm("lean", 0.60, 1.0, ubar=1.0))
        router.add_arm(_arm("bloated", 0.60, 1.0, ubar=float(ubar)))
        c3["bloated_shares"][str(ubar)] = _share(
            router, n_anchors).get("bloated", 0.0)
    c3["shrinks_with_units"] = bool(
        c3["bloated_shares"]["1"] > c3["bloated_shares"]["2"]
        > c3["bloated_shares"]["5"] > c3["bloated_shares"]["10"])
    c3["below_12pct_at_10"] = c3["bloated_shares"]["10"] < 0.12
    c3["closed_by"] = "expected-charge ranking (R-A3d)"
    cells["c3_bloat"] = c3

    # ---- C4 determinism ---------------------------------------------------
    router = RepairedRouter(price_floor=floor)
    router.add_arm(_arm("a", 0.60, 1.0))
    router.add_arm(_arm("b", 0.60, 1.0))
    first = router.route(FP, anchor="anchor-1")
    second = router.route(FP, anchor="anchor-1")
    other = router.route(FP, anchor="anchor-2")
    c4 = {"same_anchor_same_winner": first[0]["arm_id"] == second[0]["arm_id"],
          "same_anchor_same_seed": first[0]["draw_seed"]
          == second[0]["draw_seed"],
          "distinct_anchor_distinct_seed": first[0]["draw_seed"]
          != other[0]["draw_seed"],
          "closed_by": "anchor-seeded draw (R-A3b/R-A3c)"}
    cells["c4_determinism"] = c4

    # ---- C5 floor enforcement ---------------------------------------------
    router = RepairedRouter(price_floor=floor)
    rejected = None
    try:
        router.add_arm(_arm("sneak", 0.99, floor - 1e-9))
    except ValueError:
        rejected = True
    c5 = {"below_floor_rejected_at_registration": rejected is True,
          "closed_by": "price floor (R-A3a)"}
    cells["c5_floor_enforcement"] = c5

    checks = {
        "c1_monotone_price_competition": c1["monotone_as_price_falls"],
        "c1_floor_share_band": c1["floor_share_in_band"],
        "c1_no_capture_above_floor": c1["no_capture_above_floor"],
        "c1_below_floor_rejected":
            c1["below_floor_rejected_at_registration"],
        "c1b_equal_price_tie_splits": c1b["tie_splits"],
        "c2_no_single_winner": (c2["leader_below_100pct"]
                                and c2["leader_below_60pct"]
                                and c2["followers_win_sometimes"]),
        "c3_no_bloat_advantage": (c3["shrinks_with_units"]
                                  and c3["below_12pct_at_10"]),
        "c4_determinism": (c4["same_anchor_same_winner"]
                           and c4["same_anchor_same_seed"]
                           and c4["distinct_anchor_distinct_seed"]),
        "c5_floor_enforcement":
            c5["below_floor_rejected_at_registration"],
    }
    gates_ok = all(checks.values())

    evidence: dict[str, Any] = {
        "milestone": "M303",
        "cell": "H26-7 sweep over the repaired router",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "cells": cells,
        "checks": checks,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": ("the repaired router shows no price-to-zero "
                        "equilibrium, no single-winner capture, and no "
                        "bloat advantage on the registered sweep "
                        "(M303a checks)"
                        ) if gates_ok else "a check failed — VOID",
        },
        "scope": "synthetic scenario sweep, registered before running",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok, "checks": checks,
                      "cells": {k: v for k, v in cells.items()}},
                     indent=1), flush=True)
    print(f"M303 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m303(args.config, args.output)


if __name__ == "__main__":
    main()
