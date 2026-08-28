"""M178 — capability map v0: the measured task graph + monitoring rule
catalog, sealed as evidence (no new data).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(18 Aug 2026). The map content lives in ``geode/capability.py``; this
cell writes it (with its content hash) plus the rule catalog as sealed
evidence, so the map itself is auditable through M177's L0/L1.
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
from geode.core.capability import (
    CAPABILITY_MAP_V0,
    RULE_CATALOG,
    map_content_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m178_capability.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v25" / "m178_capability"


def run_m178(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible

    evidence: dict[str, Any] = {
        "milestone": "M178",
        "cell": "capability map v0 + monitoring rule catalog",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "capability_map": CAPABILITY_MAP_V0,
        "map_content_hash": map_content_hash(),
        "rule_catalog": RULE_CATALOG,
        "registered_in": "analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md "
                         "section 6 (18 Aug 2026)",
        "note": ("map nodes and edges carry SEALED numbers only; the "
                 "monitoring rules are registration-time instruments, "
                 "not claims — they flag clusters worth monitoring"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"map_content_hash": map_content_hash(),
                      "nodes": len(CAPABILITY_MAP_V0["nodes"]),
                      "edges": len(CAPABILITY_MAP_V0["edges"]),
                      "rules": len(RULE_CATALOG)}, indent=1), flush=True)
    print(f"M178 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m178(args.config, args.output)


if __name__ == "__main__":
    main()
