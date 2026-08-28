"""M315 harness — the registered takedown-containment gate (A10).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M315
(26 Aug 2026, before any build). Gate cells:

- **C1 pool-scaled quorum.** `min_responders` is non-decreasing in
  pool size and never below the registered floor.
- **C2 appeal path.** A crafted appeal citing no registered evidence
  class is inadmissible; citing one is admissible.
- **C3 suspension before permanence.** A first ratification
  suspends and does not delist; a re-ratification after the
  suspension window delists.
- **C4 revenue-scaled deposit.** Zero revenue costs zero; the
  deposit is monotone in trailing revenue.

All four cells must pass.
"""
from __future__ import annotations

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
from geode.core.economics import (
    SECURITY_FLOORS,
    assert_at_or_above_floor,
)
from geode.core.takedown_containment import (
    appeal_admissible,
    min_responders,
    proposer_deposit,
    takedown_step,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
                  / "m315_takedown_containment.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m315_takedown_containment")


def run_m315(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()

    cells: dict[str, Any] = {}

    # ---- C1 pool-scaled quorum ----
    pool_sizes = list(range(0, int(config["max_pool"])))
    responders = [min_responders(s) for s in pool_sizes]
    floor = float(SECURITY_FLOORS["takedown_min_responders"])
    c1 = {
        "floor": floor,
        "min_observed": min(responders),
        "max_observed": max(responders),
        "non_decreasing": responders == sorted(responders),
        "never_below_floor": min(responders) >= floor,
        "floor_guard_rejects_below": _guard_rejects_below(floor),
        "passes": bool(responders == sorted(responders)
                       and min(responders) >= floor),
    }
    cells["c1_pool_scaled_quorum"] = c1

    # ---- C2 appeal path ----
    no_class = appeal_admissible([])
    unregistered_only = appeal_admissible(["vibes", "opinion"])
    registered = appeal_admissible(["probe_mismatch_record"])
    c2 = {
        "no_class_admissible": no_class["admissible"],
        "unregistered_only_admissible": unregistered_only["admissible"],
        "registered_admissible": registered["admissible"],
        "passes": bool(not no_class["admissible"]
                       and not unregistered_only["admissible"]
                       and registered["admissible"]),
    }
    cells["c2_appeal_path"] = c2

    # ---- C3 suspension before permanence ----
    first = takedown_step(1, False)
    rerate = takedown_step(2, True)
    rerate_no_window = takedown_step(2, False)
    c3 = {
        "first_suspends": first["suspended"],
        "first_does_not_delist": not first["delisted"],
        "rerate_after_window_delists": rerate["delisted"],
        "rerate_without_window_suspends": bool(
            rerate_no_window["suspended"]
            and not rerate_no_window["delisted"]),
        "passes": bool(first["suspended"] and not first["delisted"]
                       and rerate["delisted"]
                       and rerate_no_window["suspended"]
                       and not rerate_no_window["delisted"]),
    }
    cells["c3_suspension_before_permanence"] = c3

    # ---- C4 revenue-scaled deposit ----
    revenues = [0.0, 1.0, 10.0, 100.0, 1000.0, 1e6]
    deposits = [proposer_deposit(r) for r in revenues]
    c4 = {
        "zero_revenue_deposit": deposits[0],
        "deposits": deposits,
        "monotone_in_revenue": deposits == sorted(deposits),
        "passes": bool(deposits[0] == 0.0
                       and deposits == sorted(deposits)),
    }
    cells["c4_revenue_scaled_deposit"] = c4

    gates_ok = all(bool(c["passes"]) for c in cells.values())
    elapsed = time.time() - started
    evidence = {
        "milestone": "M315",
        "config_digest": payload_hash(config),
        "gates_ok": gates_ok,
        "cells": cells,
        "registered_checks": ["C1", "C2", "C3", "C4"],
        "runtime_seconds": elapsed,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({
        "gates_ok": gates_ok,
        "c1_floor": floor,
        "c1_max_quorum": max(responders),
        "c4_deposit_at_1e6": deposits[-1],
    }, indent=1))
    return evidence


def _guard_rejects_below(floor: float) -> bool:
    """The registered floor sits behind the M314 guard: a timelocked
    adjustment below it must be rejected."""
    try:
        assert_at_or_above_floor("takedown_min_responders",
                                 floor - 1.0)
        return False
    except ValueError:
        return True


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m315(args.config, args.output)


if __name__ == "__main__":
    main()
