"""M367 detection-horizon table (28 Aug 2026).

Registered in ``analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md`` under
G27, whose gate reads: "for axis traffic in {10, 100, 1e3, 1e4, 1e5}
sessions/epoch, report median sessions-to-conviction, epochs to
conviction, probe overhead, and honest claim latency, under both the
flat-minimum rule and the traffic-adaptive rule."

The whitepaper's sealed sequential-test measurement is a median of
2383 SESSIONS to convict a 99.5%-agreeing substitute at the default
probe rate rho = 0.05. The test consumes the MISMATCH STREAM, so the
invariant quantity is the number of PROBED sessions it needs:

    probed_needed = 2383 * 0.05 = 119.15  ->  119

Everything below is that constant divided by probed sessions per
epoch. No new measurement is introduced; this is arithmetic over a
sealed number.

The claim freeze at Level 1 lasts ceil(open exposure units / units
per epoch) epochs, and open exposure is exactly the detection window
in sessions, so honest claim latency EQUALS epochs-to-conviction.
That identity is why the horizon is not only a security number: it
is the honest supplier's payout latency.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

PROBED_NEEDED = 119        # sealed 2383 sessions x rho 0.05
SEALED_SESSIONS = 2383
RHO_FLOOR = 0.05           # charter floor, outside ordinary governance
K_E = 2                    # sampled reference executors, default
VESTING_EPOCHS = 4         # charter floor N
FLAT_MIN_PROBED = 1        # today's per-epoch minimum probed sessions
TRAFFIC = [10, 100, 1_000, 10_000, 100_000]


def flat(traffic: int) -> dict[str, float | int | bool | None]:
    """Today's rule: rho fixed at the floor, with a floor of one
    probed session per epoch on quiet axes."""
    probed = max(FLAT_MIN_PROBED, RHO_FLOOR * traffic)
    probed = min(probed, traffic)
    rho_eff = probed / traffic
    return _row(traffic, rho_eff, probed)


def adaptive(traffic: int, probed_floor: int
             ) -> dict[str, float | int | bool | None]:
    """Proposed rule: raise rho on quiet axes to hit a probed-sessions
    floor. rho can only rise, never fall below the charter floor, and
    cannot exceed 1.0 -- which is where the impossibility bites."""
    rho = min(1.0, max(RHO_FLOOR, probed_floor / traffic))
    probed = rho * traffic
    return _row(traffic, rho, probed)


def _row(traffic: int, rho: float, probed: float
         ) -> dict[str, float | int | bool | None]:
    epochs = PROBED_NEEDED / probed
    return {
        "traffic_sessions_per_epoch": traffic,
        "rho_effective": round(rho, 4),
        "probed_sessions_per_epoch": round(probed, 2),
        "sessions_to_conviction_median": round(PROBED_NEEDED / rho),
        "epochs_to_conviction": round(epochs, 3),
        # k_e reference runs on each probed session
        "probe_overhead_fraction_of_serving": round(K_E * rho, 4),
        # the claim freeze is ceil(exposure sessions / traffic),
        # which is exactly the horizon in epochs
        "honest_claim_latency_epochs": math.ceil(epochs),
        "inside_vesting_window": epochs <= VESTING_EPOCHS,
    }


def main() -> int:
    # To land inside the vesting window the test must complete in
    # VESTING_EPOCHS, which fixes the required floor.
    required_floor = math.ceil(PROBED_NEEDED / VESTING_EPOCHS)
    payload = {
        "milestone": "M367",
        "finding": "G27 -- the detection-horizon claim fails on "
                   "quiet axes",
        "registered_in": "analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md",
        "constants": {
            "sealed_median_sessions_at_rho_0_05": SEALED_SESSIONS,
            "probed_sessions_needed": PROBED_NEEDED,
            "rho_floor": RHO_FLOOR,
            "k_e": K_E,
            "vesting_window_epochs": VESTING_EPOCHS,
            "flat_min_probed_per_epoch": FLAT_MIN_PROBED,
        },
        "derived": {
            "required_probed_per_epoch_to_meet_vesting":
                required_floor,
            "review_proposed_floor": 8,
            "review_proposal_sufficient": 8 >= required_floor,
            "min_traffic_for_any_rho_to_meet_vesting":
                required_floor,
        },
        "flat_rule": [flat(t) for t in TRAFFIC],
        "adaptive_rule_review_floor_8":
            [adaptive(t, 8) for t in TRAFFIC],
        "adaptive_rule_corrected_floor":
            [adaptive(t, required_floor) for t in TRAFFIC],
    }
    out = Path("analysis/m367_detection_horizon_table.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"required probed/epoch to meet {VESTING_EPOCHS}-epoch "
          f"vesting: {required_floor}")
    print(f"review's proposed floor of 8 sufficient? "
          f"{8 >= required_floor}")
    hdr = (f"{'traffic':>8} {'rho':>6} {'probed/ep':>10} "
           f"{'epochs':>8} {'overhead':>9} {'claim lat':>10} "
           f"{'in vest':>8}")
    for name, rows in (("FLAT (today)", payload["flat_rule"]),
                       ("ADAPTIVE P=8 (review)",
                        payload["adaptive_rule_review_floor_8"]),
                       (f"ADAPTIVE P={required_floor} (corrected)",
                        payload["adaptive_rule_corrected_floor"])):
        print(f"\n-- {name}")
        print(hdr)
        for r in rows:
            print(f"{r['traffic_sessions_per_epoch']:>8} "
                  f"{r['rho_effective']:>6} "
                  f"{r['probed_sessions_per_epoch']:>10} "
                  f"{r['epochs_to_conviction']:>8} "
                  f"{r['probe_overhead_fraction_of_serving']:>9} "
                  f"{r['honest_claim_latency_epochs']:>10} "
                  f"{str(r['inside_vesting_window']):>8}")
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
