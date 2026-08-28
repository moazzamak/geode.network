"""M223 — API persistence: the service's registry and decision chain
survive a restart.

A snapshot stores the registered arm specs plus the route REQUESTS
(which carry the fingerprints — the ledger's route records do not).
Loading re-registers and re-serves in order through the public
orchestrator API, so the restored chain verifies and is
deterministically identical to the original.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from geode.core.orchestrator import Orchestrator
from geode.hashing import canonical_json, payload_hash


def save_snapshot(orch: Orchestrator,
                  route_requests: list[dict[str, Any]],
                  path: str | Path) -> str:
    """Write the registry snapshot; returns its payload hash."""
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "arms": [dict(orch.router._arms[arm_id])
                 for arm_id in orch.router.list_arms()],
        "routes": [dict(req) for req in route_requests],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(snapshot) + "\n", encoding="utf-8",
                      newline="\n")
    return payload_hash(snapshot)


def load_snapshot(orch: Orchestrator, path: str | Path) -> None:
    """Replay a snapshot into a FRESH orchestrator through the public
    API: register arms, re-serve the routes in order."""
    snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != 1:
        raise ValueError("unsupported snapshot schema")
    for spec in snapshot["arms"]:
        orch.register(dict(spec))
    for req in snapshot["routes"]:
        orch.serve(str(req["query_id"]), list(req.get("fingerprint", [])),
                   k=int(req.get("k", 1)), task_id=req.get("task_id"),
                   contract_kind=req.get("contract_kind"))
