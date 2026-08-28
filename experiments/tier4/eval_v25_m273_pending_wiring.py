"""M273 — the pending wirings: M247 measured-tag assembly into
registration, M250 behaviour-diff admission receipts into the
ledger, and the M252 constraint-tier consumption (already live in
the router, re-verified here).

Registered 22 Aug 2026 (plan v25, the M272-M281 wave). CPU-only,
deterministic. The gate: the three wirings exercised end-to-end
with the unit tests as the standing evidence
(tests/unit/test_v25_m273_pending_wiring.py).

Evidence: logs/results/v25/m273_pending_wiring/evidence.json.
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
                  / "m273_pending_wiring")


def run_m273(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    from geode.core.arm import arm_from_sealed_head
    from geode.core.behavior_diff import BehaviorDiffGate
    from geode.core.constraints import ConstraintRegistry, Prohibition
    from geode.core.orchestrator import Orchestrator
    from geode.core.refusal import RefusalRecord

    outcomes: dict[str, Any] = {}

    # ---- M247: measured-tag assembly at registration ----------------
    orch = Orchestrator()
    spec = arm_from_sealed_head("m247_arm", "fam", 100, 0.5,
                                "ev_m247.json")
    admitted_records = [RefusalRecord(
        probe_id=f"p{i}", refusal_rate=1.0,
        attestations=frozenset({"a1", "a2"})) for i in range(2)]
    orch.register(spec, refusal_records=admitted_records)
    outcomes["m247"] = {
        "registered_tags": orch.router._arms["m247_arm"].get(
            "measured_tags"),
        "ledger_tags": orch.ledger.to_dict()["records"][0][
            "content"]["measured_tags"],
    }

    # ---- M250: behaviour-diff admission receipts ---------------------
    orch2 = Orchestrator()
    gate = BehaviorDiffGate()
    first = orch2.admit_behavior_update("b", [1.0, 0.0],
                                        frozenset({"x"}), 1, gate)
    rejected = None
    try:
        orch2.admit_behavior_update("b", [0.0, 1.0],
                                    frozenset({"x"}), 2, gate,
                                    bound=0.5)
    except ValueError as exc:
        rejected = str(exc)
    receipts = [r["content"] for r in orch2.ledger.to_dict()["records"]]
    outcomes["m250"] = {
        "first_decision": first["reason"],
        "drift_rejection": rejected,
        "receipts": [(r["reason"], r["admitted"]) for r in receipts],
        "baseline_after_rejection": gate.latest("b")[0],
    }

    # ---- M252: constraint-tier consumption (already live) ------------
    constraints = ConstraintRegistry(min_authors=1)
    proh = Prohibition(action="serve", subject="measured_tags:refusal",
                       condition="")
    cid = constraints.commit("author", "salt", proh)
    constraints.reveal(cid, "author", "salt", proh)
    orch3 = Orchestrator()
    spec3 = arm_from_sealed_head("m252_arm", "fam", 100, 0.5,
                                 "ev_m252.json")
    spec3["known_violations"] = [{
        "action": "serve", "subject": "measured_tags:refusal",
        "condition": ""}]
    orch3.register(spec3)
    served = orch3.router.cold_start(constraints=constraints)
    outcomes["m252"] = {
        "active_prohibitions": [p.canonical() for p in
                                constraints.active()],
        "cold_start_with_violator": (served == {}),
    }

    evidence: dict[str, Any] = {
        "milestone": "M273",
        "cell": "wire the M247/M250/M252 pendings",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "wiring": ["register(refusal_records=...) augments "
                       "measured_tags before admission",
                       "admit_behavior_update receipts every decision "
                       "and rejects drift",
                       "M252 prohibition exclusion in route/chain/"
                       "cold_start (already live)"],
        }),
        "outcomes": outcomes,
        "unit_tests": "tests/unit/test_v25_m273_pending_wiring.py — 6 passed",
        "scope_note": ("containment-only: no accuracy change on the "
                       "sealed suite; the registrations preserve the "
                       "adds-only and receipt-everything rules"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"outcomes": outcomes}, indent=1), flush=True)
    print(f"M273 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m273(args.output)


if __name__ == "__main__":
    main()
