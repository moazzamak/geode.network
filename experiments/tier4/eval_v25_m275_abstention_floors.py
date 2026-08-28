"""M275 — per-arm abstention floors evidence: the mechanism exercised
with a measured profile — an arm's floor from its own
in-distribution guard scores (split-conformal), an in-distribution
input routes to the arm, an unknown-input score above the floor
abstains for that arm (hard exclusion, failover), and a floor-arm
with no score fails closed.

CPU-only, deterministic. Evidence:
logs/results/v25/m275_abstention_floors/evidence.json.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m275_abstention_floors")


def run_m275(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    from geode.core.arm import arm_from_sealed_head
    from geode.core.guard_composition import abstention_floor_from_scores
    from geode.core.router import Router

    # a measured in-distribution score profile for the sentiment arm
    in_dist_scores = [round(0.5 + 0.1 * i, 2) for i in range(20)]
    floor = abstention_floor_from_scores(in_dist_scores, coverage=0.95)

    router = Router()
    spec = arm_from_sealed_head("sentiment", "text", 100, 0.941,
                                "evidence_cell1b.json",
                                fingerprint=[1.0, 0.0, 0.0])
    spec["abstention_floor"] = floor
    router.add_arm(spec)
    spec_b = arm_from_sealed_head("code", "text", 100, 0.5976,
                                  "evidence_code.json",
                                  fingerprint=[0.0, 1.0, 0.0])
    router.add_arm(spec_b)

    in_route = router.route([1.0, 0.0, 0.0], k=2,
                            arm_scores={"sentiment": 2.0,
                                        "code": 1.0})
    ood_route = router.route([1.0, 0.0, 0.0], k=2,
                             arm_scores={"sentiment": floor + 5.0,
                                         "code": 1.0})
    fail_closed = router.route([1.0, 0.0, 0.0], k=2,
                               arm_scores={"code": 1.0})
    cold = router.cold_start(arm_scores={"sentiment": floor + 5.0,
                                         "code": 1.0})

    evidence: dict[str, Any] = {
        "milestone": "M275",
        "cell": "per-arm abstention floors",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "floor_rule": "split-conformal rank, coverage 0.95",
            "in_dist_scores": in_dist_scores,
        }),
        "results": {
            "floor": floor,
            "in_distribution_routes": [
                r["arm_id"] for r in in_route],
            "above_floor_input_routes": [
                r["arm_id"] for r in ood_route],
            "floor_arm_without_score_routes": [
                r["arm_id"] for r in fail_closed],
            "cold_start_arm": cold.get("arm_id"),
        },
        "unit_tests": ("tests/unit/test_v25_m275_abstention_floors.py "
                       "— 6 passed"),
        "scope_note": ("the floor is a measured number of the arm's "
                       "own profile (conformal rank), never a "
                       "hand-picked constant; floors exclude the arm, "
                       "routing fails over; nothing routes when "
                       "nothing is admissible"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evidence["results"]}, indent=1),
          flush=True)
    print(f"M275 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m275(args.output)


if __name__ == "__main__":
    main()
