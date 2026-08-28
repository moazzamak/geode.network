"""GEODE capability map v0 (v25 M178) — measured task graph + monitoring
rule catalog.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(18 Aug 2026) before building. Nodes and edges come from SEALED evidence
only; no new data. Deterministic: no wall clocks, no RNG.

Monitoring rules (registered): R-cap-cluster, R-transfer-spike,
R-new-axis, R-regression — see the docstrings below.
"""
from __future__ import annotations

import json
from typing import Any

from geode.hashing import payload_hash

VISION_KIND = "classification-vision"
TEXT_KIND = "next-token-text"

# The registered map content (sealed numbers only; each value traces to
# its sealed evidence path).
CAPABILITY_MAP_V0: dict[str, Any] = {
    "schema_version": 1,
    "nodes": {
        "domainnet32": {
            "modality": VISION_KIND,
            "sealed_numbers": {
                "sparse_frontier": 0.2786,
                "dense_r70": 0.3118,
                "deep_patch_spm_2048": 0.5899,
            },
            "evidence": "logs/results/v24/m176c_c1/evidence.json",
        },
        "flowers102_bounded": {
            "modality": VISION_KIND,
            "sealed_numbers": {
                "dinov2_cls_baseline": 0.9902,
                "domainnet_spm_arm": 0.1667,
                "info_matched_32x32_cls": 0.8268,
            },
            "evidence": "logs/results/v24/m175_cell_b/evidence.json",
        },
        "wikitext103_next_token": {
            "modality": TEXT_KIND,
            "sealed_numbers": {"uniform_w2_test_ppl": 9.9152},
            "evidence": "logs/results/v24/m175_cell_a0/evidence.json",
        },
        "wikipedia_dump_next_token": {
            "modality": TEXT_KIND,
            "sealed_numbers": {"uniform_w2_held_out_ppl": 9.5142},
            "evidence": "logs/results/v24/m175_cell_d/evidence.json",
        },
    },
    "edges": {
        "domainnet32->flowers102": {
            "kind": "feature_transfer",
            "sealed_verdict": "scoped negative (0.167 vs 0.990)",
            "note": "blocker measured: the construction, not the 32x32 input",
        },
        "wikitext103->wikipedia_dump": {
            "kind": "text_to_text_transfer",
            "sealed_verdict": "HOLDS (gap factor 1.041)",
        },
        "cross_modality_guard": {
            "kind": "routing_safety",
            "sealed_verdict": "PASS (contract guard, wrong-kind arms unreachable)",
        },
    },
}

COS_THRESHOLD = 0.9
SPIKE_FACTOR = 1.1


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def rule_r_cap_cluster(task: dict[str, Any],
                       map_v0: dict[str, Any]) -> list[str]:
    """R-cap-cluster: a new task whose registered fingerprint cosine >=
    COS_THRESHOLD to an existing task in a DIFFERENT modality flags a
    capability cluster (unexpected transfer across modalities)."""
    flags: list[str] = []
    fp = task.get("fingerprint") or []
    if not fp:
        return flags
    for node_id, node in map_v0["nodes"].items():
        node_fp = node.get("fingerprint") or []
        if not node_fp or node["modality"] == task.get("modality"):
            continue
        if _cos(fp, node_fp) >= COS_THRESHOLD:
            flags.append(f"cap_cluster:{node_id}")
    return flags


def rule_r_transfer_spike(transfer: dict[str, Any]) -> list[str]:
    """R-transfer-spike: a cross-family transfer with gap factor <
    SPIKE_FACTOR flags a verification demand (same-family transfers
    holding is measured; cross-family spikes need checking)."""
    if transfer.get("same_family", True):
        return []
    if float(transfer.get("gap_factor", 0.0)) < SPIKE_FACTOR:
        return ["transfer_spike"]
    return []


def rule_r_new_axis(task: dict[str, Any], map_v0: dict[str, Any]) -> list[str]:
    """R-new-axis: a novel output_contract kind requires a map extension."""
    kinds = {n["modality"] for n in map_v0["nodes"].values()}
    if task.get("modality") not in kinds:
        return ["new_axis"]
    return []


def rule_r_regression(node_id: str, re_measured: dict[str, Any],
                      map_v0: dict[str, Any]) -> list[str]:
    """R-regression: a re-measurement of a sealed node outside its
    registered tolerance flags. The tolerance travels WITH the node
    (per-number), never fitted after the fact."""
    node = map_v0["nodes"].get(node_id)
    if not node:
        return ["unknown_node"]
    flags = []
    for key, sealed in node["sealed_numbers"].items():
        if key in re_measured:
            tol = node.get("tolerances", {}).get(key, 0.002)
            if abs(float(re_measured[key]) - float(sealed)) > tol:
                flags.append(f"regression:{key}")
    return flags


RULE_CATALOG: dict[str, Any] = {
    "R-cap-cluster": {
        "fn": "rule_r_cap_cluster",
        "trigger": f"fingerprint cosine >= {COS_THRESHOLD} across modalities",
    },
    "R-transfer-spike": {
        "fn": "rule_r_transfer_spike",
        "trigger": f"cross-family transfer gap factor < {SPIKE_FACTOR}",
    },
    "R-new-axis": {
        "fn": "rule_r_new_axis",
        "trigger": "novel output_contract kind",
    },
    "R-regression": {
        "fn": "rule_r_regression",
        "trigger": "sealed node re-measured outside its registered tolerance",
    },
}


def map_content_hash() -> str:
    return payload_hash(json.dumps(CAPABILITY_MAP_V0, sort_keys=True,
                                   ensure_ascii=True,
                                   separators=(",", ":")))


def extend_map(map_v0: dict[str, Any], node_id: str,
               node: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """M276 — the ontology growth policy. Every new measured task
    node appends its descriptor (modality + sealed numbers +
    fingerprint + evidence path) to the capability map; a novel
    output-contract kind fires R-new-axis, and the extension is
    FORCED with the flag recorded (the map must grow, the flag must
    travel with it). Duplicate node ids are rejected (append-only);
    the map stays deterministic (content-hashed)."""
    if node_id in map_v0["nodes"]:
        raise ValueError(f"node {node_id!r} already in the map "
                         "(append-only)")
    if "modality" not in node or "sealed_numbers" not in node:
        raise ValueError("node must carry modality + sealed_numbers")
    flags = rule_r_new_axis(node, map_v0)
    flags.extend(rule_r_cap_cluster(node, map_v0))
    updated = json.loads(json.dumps(map_v0))
    updated["nodes"][node_id] = {
        "modality": node["modality"],
        "sealed_numbers": node["sealed_numbers"],
        "fingerprint": node.get("fingerprint") or [],
        "evidence": node.get("evidence", ""),
    }
    return updated, flags
