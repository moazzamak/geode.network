"""M312 harness — the registered librarian-containment gate (A14).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M312
(26 Aug 2026, before any build). Gate cells:

- **C1 force-inclusion.** An unincorporated entry past its window
  invalidates the chain; one incorporated within the window does not.
- **C2 executable replacement.** Fires only with a recorded reason
  and at/above the registered endorsement threshold.
- **C3 liveness statistics.** A stopped librarian (no anchors, no
  incorporations) is flagged; a healthy one is bounded.

All three cells must pass.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from geode.core.librarian_containment import (
    chain_valid,
    incorporate,
    liveness_report,
    post,
    replacement,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
                  / "m312_librarian_containment.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m312_librarian_containment")


def run_m312(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()

    cells: dict[str, Any] = {}

    # ---- C1 force-inclusion ----
    queue: list = []
    post(queue, "rival_registration", epoch=5)
    valid_at_deadline = chain_valid(queue, epoch=6)
    invalid_after = chain_valid(queue, epoch=7)
    incorporate(queue, "rival_registration", epoch=6)
    valid_after_incorporation = chain_valid(queue, epoch=7)
    c1 = {
        "valid_at_deadline": valid_at_deadline,
        "invalid_when_withheld": invalid_after,
        "valid_after_incorporation": valid_after_incorporation,
        "passes": bool(valid_at_deadline and not invalid_after
                       and valid_after_incorporation),
    }
    cells["c1_force_inclusion"] = c1

    # ---- C2 executable replacement ----
    below = replacement(4, 10, recorded_reason="divergence")
    at = replacement(5, 10, recorded_reason="divergence")
    no_reason = replacement(10, 10, recorded_reason=None)
    c2 = {
        "below_threshold_fires": below["fires"],
        "at_threshold_fires": at["fires"],
        "no_reason_fires": no_reason["fires"],
        "passes": bool(not below["fires"] and at["fires"]
                       and not no_reason["fires"]),
    }
    cells["c2_executable_replacement"] = c2

    # ---- C3 liveness statistics ----
    stopped = liveness_report([], [])
    healthy = liveness_report([0, 1, 2, 3], [1, 1, 1])
    c3 = {
        "stopped_flagged": stopped["librarian_stopped"],
        "stopped_unbounded": stopped["unbounded_latency"],
        "healthy_not_stopped": not healthy["librarian_stopped"],
        "healthy_bounded": bool(
            healthy["max_anchor_gap"] == 1
            and healthy["max_inclusion_latency"] == 1),
        "passes": bool(stopped["librarian_stopped"]
                       and stopped["unbounded_latency"]
                       and not healthy["librarian_stopped"]
                       and healthy["max_anchor_gap"] == 1
                       and healthy["max_inclusion_latency"] == 1),
    }
    cells["c3_liveness_statistics"] = c3

    gates_ok = all(bool(c["passes"]) for c in cells.values())
    elapsed = time.time() - started
    evidence = {
        "milestone": "M312",
        "config_digest": payload_hash(config),
        "gates_ok": gates_ok,
        "cells": cells,
        "registered_checks": ["C1", "C2", "C3"],
        "runtime_seconds": elapsed,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok,
                      "c1_invalid_when_withheld": invalid_after,
                      "c2_at_threshold_fires": at["fires"],
                      "c3_stopped_flagged": stopped["librarian_stopped"],
                      }, indent=1))
    return evidence


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m312(args.config, args.output)


if __name__ == "__main__":
    main()
