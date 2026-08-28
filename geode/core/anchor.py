"""GEODE public-chain anchoring (v25 M254, structure) — the
trust-minimized publication layer.

The anchor is the registry's global tamper-evidence: the ledger tip,
the record count, and the last record hash, published to a public
chain so no state actor can quietly rewrite local history anywhere.
This cell ships the deterministic spec, the offline verifier, and the
submission interface; the actual endpoint is the M194 decision
(``AnchorClient.submit`` raises with that note until then).

Deterministic: hashes only, no wall clocks, no RNG.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from geode.hashing import payload_hash


@dataclass(frozen=True)
class AnchorSpec:
    """The registered M185 fields a public chain will carry."""
    tip: str
    record_count: int
    last_record_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {"fields": ["tip", "record_count", "last_record_hash"],
                "values": {"tip": self.tip,
                           "record_count": self.record_count,
                           "last_record_hash": self.last_record_hash}}

    def digest(self) -> str:
        """The canonical anchor digest (what a chain would carry)."""
        return payload_hash(str(sorted(self.to_dict()["values"]
                                       .items())))


def anchor_from_ledger(ledger: Any) -> AnchorSpec:
    """Build the spec from an AppendOnlyLedger's tip state."""
    d = ledger.to_dict()
    records = d["records"]
    return AnchorSpec(
        tip=d["tip"],
        record_count=d["record_count"],
        last_record_hash=records[-1]["hash"] if records else "")


def verify_anchor_entry(entry: dict[str, Any],
                        expected: AnchorSpec) -> dict[str, Any]:
    """Verify a published anchor entry against the local chain:
    tamper-evidence across copies (the anti-rewrite check)."""
    ok = bool(entry.get("values") == expected.to_dict()["values"])
    return {"ok": ok, "expected": expected.to_dict()["values"],
            "got": entry.get("values")}


class AnchorClient:
    """The publication interface; gated on M194."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def submit(self, spec: AnchorSpec) -> str:
        """The registered M194 gate: the endpoint decision is the
        only missing piece."""
        raise NotImplementedError(
            "public-chain anchoring is GATED on the M194 endpoint + "
            "funded-key decision (M254, 20 Aug 2026)")

    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)
