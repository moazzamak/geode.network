"""hello_geode.py — the five-minute tour of the shipped package.

register an arm -> route a task -> guard the input -> contain ->
override -> verify the chain. Every step uses only the public API
and prints what it did. Run with:  python examples/hello_geode.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# runnable from a fresh checkout without installing (the repo root
# goes on the path; an installed package is used when present)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geode.core.freeze import FreezeRegistry
from geode.core.ood import OodGate
from geode.core.orchestrator import Orchestrator
from geode.core.override import OverrideLedger

FINGERPRINT = [0.9, 0.3, 0.2, 0.1]


def main() -> None:
    orch = Orchestrator()

    # 1. Register an arm (append-only; admission is measured-gated).
    orch.router.add_arm({
        "arm_id": "demo_general", "fingerprint": FINGERPRINT,
        "output_contract": {"kind": "class"},
        "held_out_accuracy": 0.5,
        "availability": {"healthy": True},
        "price": 1.0, "general": True, "primitive": False,
    })
    print("registered:", orch.router.list_arms())

    # 2. Route a task.
    recs = orch.router.route(FINGERPRINT)
    print("routed:", [(r["arm_id"], r["ranked_by"]) for r in recs])

    # 3. Guard the input (fail-closed out-of-distribution detection).
    guard = OodGate(threshold=3.0)
    guard.fit_profile([[0.0, 0.0], [1.0, 1.0]])
    in_dist = orch.router.route(FINGERPRINT, ood_guard=guard,
                                input_vec=[0.5, 0.5])
    out_dist = orch.router.route(FINGERPRINT, ood_guard=guard,
                                 input_vec=[50.0, 50.0])
    print("guard: in-distribution ->", len(in_dist), "arms; "
          "out-of-distribution ->", out_dist)

    # 4. Contain the registry with a time-bounded, quorum freeze.
    freeze = FreezeRegistry(k_of_n=2, default_ttl=100)
    freeze.freeze("drill", frozenset({"v1", "v2"}), start_index=0,
                  reason="hello-world drill")
    frozen_route = orch.router.route(FINGERPRINT, freeze=freeze,
                                     as_of_index=10)
    print("while frozen:", frozen_route, "(auto-expires at index",
          freeze.events()[0].expires_index, ")")

    # 5. Record a human override (justification + counterfactual are
    # mandatory — the API rejects blank ones).
    overrides = OverrideLedger()
    idx = overrides.record(
        "operator", "kill_switch", "hello-world drill",
        {"would_have": "routed to demo_general"})
    print("override recorded at index", idx, "| chain tip",
          overrides.tip()[:16] + "...")

    # 6. Verify the decision chain re-hashes clean.
    print("chain verifies:", orch.ledger.verify()["ok"],
          "| override chain verifies:", overrides.verify()["ok"])


if __name__ == "__main__":
    main()
