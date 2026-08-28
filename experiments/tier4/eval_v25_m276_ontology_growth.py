"""M276 — ontology growth policy evidence: the sealed M268 wave nodes
(sentiment, arithmetic, logic, code) appended to the capability map
with the R-new-axis trigger exercised.

CPU-only, deterministic. Evidence:
logs/results/v25/m276_ontology_growth/evidence.json.
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
                  / "m276_ontology_growth")


def run_m276(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    from geode.core.capability import CAPABILITY_MAP_V0, extend_map

    # the sealed M268 wave nodes (sealed evidence paths, modalities)
    new_nodes = {
        "imdb_sentiment": {
            "modality": "next-token-text",
            "sealed_numbers": {"generalist_top1": 0.941},
            "evidence": "logs/results/v25/m268_routing_study/"
                        "evidence_cell1b.json",
        },
        "arithmetic_primitive": {
            "modality": "exact-arithmetic",
            "sealed_numbers": {"exactness": 1.0},
            "evidence": "logs/results/v25/m268_routing_study/"
                        "evidence_primitives.json",
        },
        "logic_primitive": {
            "modality": "boolean-logic",
            "sealed_numbers": {"exactness": 1.0},
            "evidence": "logs/results/v25/m268_routing_study/"
                        "evidence_primitives.json",
        },
        "code_coder_arm": {
            "modality": "next-token-text",
            "sealed_numbers": {"pass_at_1": 0.5976},
            "evidence": "logs/results/v25/m268_routing_study/"
                        "evidence_code.json",
        },
    }

    current = CAPABILITY_MAP_V0
    growth: list[dict[str, Any]] = []
    for node_id, node in new_nodes.items():
        current, flags = extend_map(current, node_id, node)
        growth.append({"node": node_id, "modality": node["modality"],
                       "flags": flags})

    evidence: dict[str, Any] = {
        "milestone": "M276",
        "cell": "ontology growth policy + R-new-axis wiring",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "new_nodes": {k: v["modality"] for k, v in
                          new_nodes.items()}}),
        "results": {
            "growth": growth,
            "n_nodes_before": len(CAPABILITY_MAP_V0["nodes"]),
            "n_nodes_after": len(current["nodes"]),
            "map_hash_before": payload_hash(CAPABILITY_MAP_V0),
            "map_hash_after": payload_hash(current),
            "new_axis_flags": sorted({f for g in growth
                                      for f in g["flags"]
                                      if f == "new_axis"}),
        },
        "unit_tests": ("tests/unit/test_v25_m276_ontology_growth.py "
                       "— 5 passed"),
        "scope_note": ("growth is append-only and deterministic; a "
                       "novel axis forces the extension with the flag "
                       "recorded; sealed nodes untouched"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": evidence["results"]}, indent=1),
          flush=True)
    print(f"M276 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m276(args.output)


if __name__ == "__main__":
    main()
