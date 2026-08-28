"""GEODE ledger prototype (v25 M185, LOCAL part) — append-only registry
with content hashes chained to the previous record.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(18 Aug 2026). Tamper-evident by construction: each record's hash covers
its content plus the previous record's hash, so any change to any record
invalidates every later hash. No wall clocks in any content hash (the
standing reproducibility rule). Public-testnet anchoring is the M194
anchor-audit gate; ``anchor_spec`` records the exact fields M194 will
submit.
"""
from __future__ import annotations

import hmac
import json
from typing import Any

from geode.hashing import payload_hash
from geode.audit import TIMING_FIELDS


def record_hash(content: dict[str, Any], previous: str) -> str:
    """Chain hash: covers the canonical content (timing fields excluded
    by the standing rule) and the previous hash."""
    stripped = {k: v for k, v in content.items()
                if k not in TIMING_FIELDS}
    return payload_hash(json.dumps(
        {"content": stripped, "previous": previous},
        sort_keys=True, ensure_ascii=True, separators=(",", ":")))


def answer_commitment(answer: str, nonce: str) -> str:
    """H(answer, nonce): the only form an answer takes on the ledger
    (v26 M310). The opening (answer, nonce) stays inside the sealed
    replay environment; it never becomes a ledger record."""
    return payload_hash({"answer": str(answer), "nonce": str(nonce)})


def opens_commitment(commitment: str, answer: str, nonce: str) -> bool:
    """True iff (answer, nonce) opens the recorded commitment.
    Constant-time comparison."""
    return hmac.compare_digest(
        str(commitment), answer_commitment(answer, nonce))


def registry_state_root(state: dict[str, Any]) -> str:
    """Merkle-style commitment over the registry state a route decided
    against (scores, prices, qualification). Every route entry carries
    this root, so route replay is a local check against a committed
    root instead of a whole-chain reconstruction (v26 M310)."""
    return payload_hash(state)


class AppendOnlyLedger:
    """Append-only, hash-chained registry. Deterministic; read-only
    except for append."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._chain: list[str] = [record_hash({"genesis": True}, "")]
        self._index: dict[str, int] = {}

    def append(self, record: dict[str, Any]) -> int:
        """Append a record; returns its index. Records are immutable
        once appended (append-only)."""
        content = dict(record)
        if "index" in content or "hash" in content:
            raise ValueError("record may not set 'index' or 'hash'")
        index = len(self._records)
        h = record_hash(content, self._chain[-1])
        self._records.append({"index": index, "content": content,
                              "hash": h})
        self._chain.append(h)
        key = content.get("key")
        if key is not None:
            if key in self._index:
                raise ValueError(f"duplicate key {key!r} (append-only)")
            self._index[str(key)] = index
        return index

    def tip(self) -> str:
        return self._chain[-1]

    def verify(self) -> dict[str, Any]:
        """Re-hash every record from genesis; report tampering."""
        tip = record_hash({"genesis": True}, "")
        tampered: list[int] = []
        for rec in self._records:
            expected = record_hash(rec["content"], tip)
            if expected != rec["hash"]:
                tampered.append(rec["index"])
            tip = expected
        return {"ok": not tampered, "tampered_records": tampered,
                "tip": tip, "record_count": len(self._records)}

    def get(self, key: str) -> dict[str, Any] | None:
        idx = self._index.get(key)
        return self._records[idx] if idx is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {"records": self._records, "tip": self.tip(),
                "record_count": len(self._records)}

    def anchor_spec(self) -> dict[str, Any]:
        """The exact fields M194 will submit to a public testnet."""
        return {
            "fields": ["tip", "record_count", "last_record_hash"],
            "values": {"tip": self.tip(),
                       "record_count": len(self._records),
                       "last_record_hash":
                           self._records[-1]["hash"] if self._records
                           else self._chain[0]},
            "note": ("anchoring is the M194 anchor-audit gate; this cell "
                     "seals the local chain and the spec fields only"),
        }
