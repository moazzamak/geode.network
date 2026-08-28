"""M317 — standard-library sandboxing for finding A24 (26 Aug 2026).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M317
before any build. A24: the standard library runs directly inside the
signing process while only third-party code is sandboxed; the
settlement key lives in a host process; trusted-by-hash is not
memory-safe. Repair (R-A24): sandbox the standard library on the same
terms as third-party primitives.

This module is an executable capability model of the reachability
question — the policy it encodes is the deliverable. It demonstrates
the pre-repair key-reachability path (including under hash pinning),
the post-repair absence of any path, and enforces uniform sandbox
terms.
"""
from __future__ import annotations

from typing import Any

# The registered uniform sandbox terms (R-A24): every primitive runs
# under these, standard-library and third-party alike.
SANDBOX_TERMS: dict[str, Any] = {
    "memory_isolation": True,        # no host-memory access
    "network": "settlement_only",    # no arbitrary egress
    "filesystem": "readonly_catalog",
    "settlement_key_access": False,
    "hash_pinned": True,             # trusted-by-hash, as before
}


def _reachable(edges: dict[str, list[str]], start: str,
               target: str,
               required: dict[str, set[str]] | None = None,
               capabilities: dict[str, set[str]] | None = None
               ) -> bool:
    """Capability-weighted reachability: to cross an edge the
    PRINCIPAL (the process the path starts from) must hold the edge's
    required capabilities — intermediate nodes relay, they do not
    lend capabilities. This is what "sandboxed" means in the model:
    a primitive that lacks ``settlement_key_access`` cannot cross the
    host-to-key edge, however many brokers it may legitimately call."""
    required = required or {}
    capabilities = capabilities or {}
    principal_caps = capabilities.get(start, set())
    seen: set[str] = set()
    frontier = [start]
    while frontier:
        node = frontier.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        for nxt in edges.get(node, []):
            need = required.get(node, {}).get(nxt, set())
            if need <= principal_caps:
                frontier.append(nxt)
    return False


def pre_repair_reachable(stdlib_primitive: str,
                         key_process: str) -> bool:
    """The registered A24 defect: the standard-library primitive runs
    DIRECTLY in the host, so it holds the host's capabilities — key
    access included — and the call graph connects it to the
    settlement-key process. Hash pinning does not remove the
    capability: pinning guarantees the intended code runs, including
    its intended bugs, on attacker-chosen input."""
    edges = {
        stdlib_primitive: ["host"],          # runs directly (fig:isolation)
        "host": [key_process, "ledger"],
        "ledger": [],
        key_process: [],
    }
    required = {"host": {key_process: {"settlement_key_access"}}}
    capabilities = {
        stdlib_primitive: {"settlement_key_access"},   # in-host
        "host": {"settlement_key_access"},
    }
    return _reachable(edges, stdlib_primitive, key_process,
                      required, capabilities)


def post_repair_reachable(primitives: list[str],
                          key_process: str) -> bool:
    """R-A24: every primitive runs sandboxed with identical
    capability sets — no ``settlement_key_access`` — so no path from
    any primitive runtime to the key crosses the host-to-key edge."""
    edges: dict[str, list[str]] = {
        "host": [key_process, "ledger"],
        "ledger": [],
        key_process: [],
        "sandbox_broker": ["host"],          # mediated, no key lending
    }
    for primitive in primitives:
        edges[primitive] = ["sandbox_broker"]
    required = {"host": {key_process: {"settlement_key_access"}}}
    capabilities = {primitive: set() for primitive in primitives}
    capabilities["sandbox_broker"] = set()
    capabilities["host"] = {"settlement_key_access"}
    return any(_reachable(edges, primitive, key_process,
                          required, capabilities)
               for primitive in primitives)


def assert_uniform_terms(primitives: list[str],
                         terms_by_primitive: dict[str, dict[str, Any]]
                         ) -> None:
    """The registered policy: every primitive's terms must equal the
    registered uniform terms. Raises ValueError on any deviation."""
    for primitive in primitives:
        terms = terms_by_primitive.get(primitive)
        if terms is None:
            raise ValueError(f"primitive {primitive!r} has no "
                             "registered sandbox terms")
        if terms != SANDBOX_TERMS:
            raise ValueError(
                f"primitive {primitive!r} deviates from the registered "
                f"uniform terms: {terms}")
