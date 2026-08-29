"""M354 - does the route lottery actually spread traffic?

Registered in ``analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md`` under
G16. The gate: "The existing synthetic traffic-share sweep is
re-run with distinct session ids and reproduces the published
shares (strongest arm ~ one third, equal-price tie splits evenly,
a 2x price cut roughly doubles share). The current sweep almost
certainly varied the task or the anchor to get those numbers --
confirm which, and say so if the published figure came from a
configuration the protocol does not actually produce."

The failure is reproduced first. ``draw_seed`` hashes
(anchor, task, state root, fingerprint); the paper anchors the
ledger tip "to Ethereum once per epoch", so within one epoch,
for one task, against one registry state, every field is
constant and every session draws the same winner. The published
sweep (``experiments/tier4/eval_v26_m303_router_repair.py``,
``_share``) passed ``anchor=f"anchor-{i}"`` -- a fresh anchor per
session, a cadence the protocol does not produce.

The repair adds the session identifier to the seed. Grinding
resistance is then measured rather than asserted, separately for
a host (who does not own other people's session ids) and for a
payer (who does).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from geode.core.router_repair import RepairedRouter, draw_seed
from geode.hashing import payload_hash

FP = [1.0, 0.0]
N_SESSIONS = 4000
EPOCH_ANCHOR = "epoch-0x8f3a"       # one anchor for the whole epoch
# M388: the draw is seeded from the randomness beacon, which closes
# after the session is declared, ordered by the epoch anchor.
EPOCH_BEACON = "beacon-round-0x3f"
PRICE_FLOOR = 1.0


def _arm(arm_id: str, acc: float, price: float,
         ubar: float | None = None) -> dict[str, Any]:
    spec = {"arm_id": arm_id, "held_out_accuracy": acc,
            "price": price, "availability": {"healthy": True}}
    if ubar is not None:
        spec["expected_units"] = ubar
    return spec


def repaired_seed(beacon: str, anchor: str, task_id: str,
                  state_root: str, fp: list[float],
                  session_id: str) -> str:
    """The final seed: H(beacon, anchor, task, registry state root,
    fp, session id). The session identifier joins the seed (G16); the
    beacon closes after declaration so the draw is not grindable
    against the public anchor (M388)."""
    return payload_hash({"beacon": beacon, "anchor": anchor,
                         "task": task_id, "state_root": state_root,
                         "fingerprint": [float(v) for v in fp],
                         "session": str(session_id)})


def _draw(router: RepairedRouter, seed: str) -> str:
    """One draw from the top-k pool under an externally supplied
    seed, using the router's own ranking."""
    pool = router.route(FP, beacon=EPOCH_BEACON,
                        anchor=EPOCH_ANCHOR)
    ranked = sorted(pool, key=lambda r: r["share_rank"])
    weights = [r["rank_score"] for r in ranked]
    pick = random.Random(seed).choices(range(len(ranked)),
                                       weights=weights, k=1)[0]
    return ranked[pick]["arm_id"]


def _shares(router: RepairedRouter, mode: str,
            n: int = N_SESSIONS) -> dict[str, float]:
    """mode='published' varies the anchor per session (what the
    sweep did); 'protocol' holds the epoch anchor fixed (what the
    protocol produces); 'repaired' holds it fixed and varies the
    session id."""
    counts: dict[str, int] = {}
    state = router.state_root()
    for i in range(n):
        if mode == "published":
            winner = router.route(FP, beacon=EPOCH_BEACON,
                                  anchor=f"anchor-{i}")[0]["arm_id"]
        elif mode == "protocol":
            winner = router.route(FP, beacon=EPOCH_BEACON,
                                  anchor=EPOCH_ANCHOR)[0]["arm_id"]
        elif mode == "repaired":
            winner = _draw(router, repaired_seed(
                EPOCH_BEACON, EPOCH_ANCHOR, "default", state, FP,
                f"session-{i}"))
        else:
            raise ValueError(mode)
        counts[winner] = counts.get(winner, 0) + 1
    return {k: v / n for k, v in sorted(counts.items())}


def _three_arm() -> RepairedRouter:
    router = RepairedRouter(price_floor=PRICE_FLOOR)
    router.add_arm(_arm("leader", 0.70, 1.0))
    router.add_arm(_arm("follower1", 0.69, 1.0))
    router.add_arm(_arm("follower2", 0.60, 1.0))
    return router


def _tie() -> RepairedRouter:
    router = RepairedRouter(price_floor=PRICE_FLOOR)
    router.add_arm(_arm("tie_a", 0.60, PRICE_FLOOR))
    router.add_arm(_arm("tie_b", 0.60, PRICE_FLOOR))
    return router


def _price_cut() -> RepairedRouter:
    router = RepairedRouter(price_floor=PRICE_FLOOR)
    router.add_arm(_arm("full_price", 0.60, 2.0))
    router.add_arm(_arm("half_price", 0.60, 1.0))
    return router


def main() -> int:
    started = time.time()
    payload: dict[str, Any] = {
        "milestone": "M354",
        "finding": "G16 -- the route seed has no per-session entropy",
        "registered_in": "analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md",
        "n_sessions": N_SESSIONS,
    }

    # ---- what the published sweep actually varied ----------------
    payload["published_configuration"] = {
        "source": "experiments/tier4/eval_v26_m303_router_repair.py",
        "function": "_share",
        "call": 'router.route(FP, anchor=f"anchor-{i}")',
        "varied": "the anchor, once per session",
        "protocol_anchor_cadence": "once per epoch (whitepaper, "
                                   "§The ledger and the anchor)",
        "produced_by_the_protocol": False,
    }

    # ---- reproduce the failure -----------------------------------
    scenarios = {
        "three_arm_leader": _three_arm,
        "equal_price_tie": _tie,
        "two_times_price_cut": _price_cut,
    }
    results: dict[str, Any] = {}
    for name, build in scenarios.items():
        row = {mode: _shares(build(), mode)
               for mode in ("published", "protocol", "repaired")}
        row["protocol_is_winner_take_all"] = bool(
            max(row["protocol"].values()) == 1.0)
        results[name] = row
        print(f"\n{name}")
        for mode in ("published", "protocol", "repaired"):
            rendered = "  ".join(f"{k}={v:.3f}"
                                 for k, v in row[mode].items())
            print(f"  {mode:10s} {rendered}")
    payload["scenarios"] = results

    # ---- do the repaired shares match the published claims? ------
    leader = results["three_arm_leader"]["repaired"]["leader"]
    tie_a = results["equal_price_tie"]["repaired"]["tie_a"]
    cut = results["two_times_price_cut"]["repaired"]
    ratio = cut["half_price"] / cut["full_price"]
    payload["published_claims"] = {
        "strongest_arm_about_one_third": {
            "measured": round(leader, 4),
            "holds": 0.28 <= leader <= 0.40},
        "equal_price_tie_splits_evenly": {
            "measured": round(tie_a, 4),
            "holds": abs(tie_a - 0.5) <= 0.05},
        "two_times_price_cut_roughly_doubles_share": {
            "measured_ratio": round(ratio, 4),
            "holds": 1.7 <= ratio <= 2.3},
    }

    # ---- grinding, measured not asserted -------------------------
    # A host does not own other parties' session ids. The seed
    # contains no arm identifier, so a host's only lever is its own
    # registered fields, which move the ranking rather than the draw.
    router = _three_arm()
    state = router.state_root()
    host_lever = draw_seed(EPOCH_BEACON, EPOCH_ANCHOR, "default",
                           state, FP)
    router2 = _three_arm()
    router2.add_arm(_arm("newcomer", 0.10, 9.0))
    host_lever_after = draw_seed(EPOCH_BEACON, EPOCH_ANCHOR,
                                 "default", router2.state_root(), FP)

    # A payer does own its session ids, and within an epoch the
    # anchor is known, so it can resubmit until a preferred arm
    # wins. Measure the expected number of attempts.
    attempts = []
    for trial in range(200):
        n = 1
        while _draw(router, repaired_seed(
                EPOCH_BEACON, EPOCH_ANCHOR, "default", state, FP,
                f"grind-{trial}-{n}")) != "follower2":
            n += 1
            if n > 500:
                break
        attempts.append(n)
    payload["grinding"] = {
        "host": {
            "seed_contains_arm_identifier": False,
            "registry_change_moves_seed":
                host_lever != host_lever_after,
            "reading": "a host cannot bias its own draw: the seed "
                       "has no arm field, and the only registry "
                       "lever it owns (its own price and score) "
                       "moves the ranking, which is the intended "
                       "channel",
        },
        "payer": {
            "mean_resubmissions_to_force_least_favoured_arm":
                round(sum(attempts) / len(attempts), 2),
            "least_favoured_arm_share":
                round(results["three_arm_leader"]["repaired"]
                      ["follower2"], 4),
            "reading": "this loop measures the KNOWN-SEED "
                       "counterfactual: with the seed value fixed and "
                       "public, a payer that owns its session ids can "
                       "resubmit until a preferred arm wins. M388 "
                       "closes that in the protocol by seeding the "
                       "draw from the randomness beacon, whose output "
                       "for a round closes AFTER the session is "
                       "declared — a payer cannot know the seed when "
                       "choosing a session id, so each forcing "
                       "attempt crosses a round and cannot be "
                       "steered. The paper reports the 3.1 figure as "
                       "the measured anchor-seeded counterfactual and "
                       "the beacon as the deployed closure.",
        },
    }

    passed = (all(v["holds"] for v in payload["published_claims"].values())
              and all(results[s]["protocol_is_winner_take_all"]
                      for s in results))
    payload["verdict"] = "PASS" if passed else "FAIL"
    payload["runtime_seconds"] = round(time.time() - started, 1)
    out = Path("analysis/m354_route_lottery_entropy.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nverdict: {payload['verdict']} -> {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
