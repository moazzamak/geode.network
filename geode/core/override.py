"""GEODE override ledger (v25 M243) — the human-alignment surface.

Every human intervention (manual re-rank, admission exception,
kill-switch, constraint waiver) is appended to an append-only,
hash-chained ledger WITH a justification and the counterfactual
("what the system would have done"). An override with an empty
justification or a missing counterfactual is REJECTED (raises) —
interventions without recorded reasons cannot happen silently.

Deterministic: no wall clocks in content (the standing
reproducibility rule); the chain reuses AppendOnlyLedger. Lives in
``core`` (not ``audit``) because it depends on the ledger, and the
registered layer table lets ``audit`` import only ``hashing``.
"""
from __future__ import annotations

from typing import Any

from geode.core.ledger import AppendOnlyLedger

# The intervention kinds this ledger records (open set; new kinds are
# appended as records, never enumerated exhaustively here).
KNOWN_ACTIONS = frozenset({
    "manual_rerank",
    "admission_exception",
    "kill_switch",
    "constraint_waiver",
})


class OverrideLedger:
    """Append-only human-intervention ledger with mandatory
    justification + counterfactual."""

    def __init__(self) -> None:
        self._ledger = AppendOnlyLedger()

    def record(self, actor: str, action: str, justification: str,
               counterfactual: dict[str, Any],
               subject: str | None = None,
               key: str | None = None) -> int:
        """Append one intervention; returns the record index.

        Raises ValueError when the justification is blank or the
        counterfactual is missing/empty — a human override without a
        recorded reason or a recorded what-the-system-would-have-done
        is refused by construction.
        """
        if not str(actor).strip():
            raise ValueError("an override must name its actor")
        if action not in KNOWN_ACTIONS:
            raise ValueError(f"unknown override action {action!r} "
                             f"(known: {sorted(KNOWN_ACTIONS)})")
        if not str(justification).strip():
            raise ValueError("an override requires a non-empty "
                             "justification (M243)")
        if not counterfactual:
            raise ValueError("an override requires the counterfactual "
                             "of what the system would have done (M243)")
        content: dict[str, Any] = {
            "kind": "override",
            "actor": str(actor),
            "action": action,
            "justification": str(justification),
            "counterfactual": dict(counterfactual),
        }
        if subject is not None:
            content["subject"] = str(subject)
        return self._ledger.append(
            {**content, "key": key} if key is not None else content)

    def verify(self) -> dict[str, Any]:
        """Re-hash the whole chain; report tampering."""
        return self._ledger.verify()

    def tip(self) -> str:
        return self._ledger.tip()

    def to_dict(self) -> dict[str, Any]:
        return self._ledger.to_dict()
