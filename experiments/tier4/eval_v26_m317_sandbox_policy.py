"""M317 harness — the registered standard-library sandbox gate (A24).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M317
(26 Aug 2026, before any build). Gate cells:

- **C1 defect demonstrated.** The pre-repair model connects a
  standard-library primitive to the settlement key, including under
  hash pinning.
- **C2 repair closes it.** The post-repair model has no path from any
  sandboxed primitive to the key.
- **C3 uniform terms enforced.** Any primitive with elevated or
  missing terms is refused by the guard.

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
from geode.core.sandbox_policy import (
    SANDBOX_TERMS,
    assert_uniform_terms,
    post_repair_reachable,
    pre_repair_reachable,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
                  / "m317_sandbox_policy.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m317_sandbox_policy")


def run_m317(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()

    cells: dict[str, Any] = {}
    stdlib = str(config["stdlib_primitive"])
    primitives = [str(p) for p in config["primitives"]]

    # ---- C1 defect demonstrated ----
    pre = pre_repair_reachable(stdlib, str(config["key_process"]))
    pre_pinned = pre_repair_reachable(
        "pinned_" + stdlib, str(config["key_process"]))
    c1 = {
        "pre_repair_path_exists": pre,
        "pre_repair_path_with_hash_pinning": pre_pinned,
        "passes": bool(pre and pre_pinned),
    }
    cells["c1_defect_demonstrated"] = c1

    # ---- C2 repair closes it ----
    post = post_repair_reachable(primitives, str(config["key_process"]))
    c2 = {
        "post_repair_any_path": post,
        "primitives": primitives,
        "passes": bool(not post),
    }
    cells["c2_repair_closes_path"] = c2

    # ---- C3 uniform terms enforced ----
    uniform = dict(SANDBOX_TERMS)
    elevated = dict(SANDBOX_TERMS)
    elevated["settlement_key_access"] = True
    accepted_uniform = True
    rejected_elevated = False
    rejected_missing = False
    try:
        assert_uniform_terms(primitives,
                             {p: dict(uniform) for p in primitives})
    except ValueError:
        accepted_uniform = False
    try:
        assert_uniform_terms([primitives[0]],
                             {primitives[0]: elevated})
    except ValueError:
        rejected_elevated = True
    try:
        assert_uniform_terms(primitives, {})
    except ValueError:
        rejected_missing = True
    c3 = {
        "uniform_terms_accepted": accepted_uniform,
        "elevated_terms_rejected": rejected_elevated,
        "missing_terms_rejected": rejected_missing,
        "passes": bool(accepted_uniform and rejected_elevated
                       and rejected_missing),
    }
    cells["c3_uniform_terms_enforced"] = c3

    gates_ok = all(bool(c["passes"]) for c in cells.values())
    elapsed = time.time() - started
    evidence = {
        "milestone": "M317",
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
                      "pre_repair_path": pre,
                      "post_repair_path": post,
                      }, indent=1))
    return evidence


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m317(args.config, args.output)


if __name__ == "__main__":
    main()
