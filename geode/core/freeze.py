"""GEODE emergency freeze (v25 M248) — containment with no permanent
censorship.

A freeze is a quorum-attested, TIME-BOUNDED event: it takes effect
only with k-of-n distinct attestations and AUTO-EXPIRES at a ledger
index. A captured quorum can therefore delay the registry, but never
silence it forever (the registered anti-censorship amendment). An
early unfreeze attests to a SPECIFIC freeze event and requires the
same threshold — pre-signed unfreezes cannot exist, because the
attestation names the event it lifts.

Deterministic: no RNG, no wall clocks — time is ledger-index space.
Consumed by the router (a frozen registry returns empty routes) and
by admission (add_arm rejects while frozen).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class FreezeError(RuntimeError):
    """Raised when a frozen registry is asked to admit or route."""


@dataclass
class FreezeEvent:
    """One freeze occurrence."""
    event_id: str
    reason: str
    attestations: frozenset[str] = field(default_factory=frozenset)
    start_index: int = 0
    expires_index: int = 0
    lifted: bool = False
    lifted_attestations: frozenset[str] = field(default_factory=frozenset)


class FreezeRegistry:
    """Quorum-attested, auto-expiring freeze state."""

    def __init__(self, k_of_n: int = 2, default_ttl: int = 1000):
        if k_of_n < 1:
            raise ValueError("k_of_n must be >= 1")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be positive")
        self.k_of_n = int(k_of_n)
        self.default_ttl = int(default_ttl)
        self._events: list[FreezeEvent] = []

    def freeze(self, event_id: str, attestations: frozenset[str],
               start_index: int, reason: str = "",
               ttl: int | None = None) -> FreezeEvent:
        """Register a freeze; it is EFFECTIVE only with quorum
        attestations. Auto-expires at start_index + ttl."""
        ttl = self.default_ttl if ttl is None else int(ttl)
        if ttl <= 0:
            raise ValueError("ttl must be positive (a freeze must end)")
        event = FreezeEvent(
            event_id=str(event_id),
            reason=str(reason),
            attestations=frozenset(attestations),
            start_index=int(start_index),
            expires_index=int(start_index) + ttl)
        self._events.append(event)
        return event

    def _effective(self, event: FreezeEvent) -> bool:
        return (len(event.attestations) >= self.k_of_n
                and not event.lifted)

    def is_frozen(self, as_of_index: int) -> bool:
        """Frozen iff any quorum freeze covers this index and is not
        lifted. Expired freezes never count (auto-release)."""
        for event in self._events:
            if not self._effective(event):
                continue
            if event.start_index <= as_of_index < event.expires_index:
                return True
        return False

    def unfreeze(self, event_id: str, attestations: frozenset[str],
                 as_of_index: int) -> FreezeEvent:
        """Lift a SPECIFIC freeze event before its expiry; requires
        the same threshold and the event must currently cover
        ``as_of_index`` (attestation names the event it lifts — no
        pre-signed unfreezes)."""
        for event in self._events:
            if event.event_id != event_id:
                continue
            if not self._effective(event):
                raise FreezeError(
                    f"freeze {event_id!r} is not effective (quorum "
                    f"not met or already lifted)")
            if not (event.start_index <= as_of_index
                    < event.expires_index):
                raise FreezeError(
                    f"freeze {event_id!r} does not cover index "
                    f"{as_of_index}")
            if len(attestations) < self.k_of_n:
                raise FreezeError("unfreeze requires the same quorum "
                                  f"(got {len(attestations)} "
                                  f"attestations, need {self.k_of_n})")
            event.lifted = True
            event.lifted_attestations = frozenset(attestations)
            return event
        raise FreezeError(f"unknown freeze event {event_id!r}")

    def events(self) -> list[FreezeEvent]:
        return list(self._events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_of_n": self.k_of_n,
            "default_ttl": self.default_ttl,
            "events": [
                {"event_id": e.event_id, "reason": e.reason,
                 "attestations": sorted(e.attestations),
                 "start_index": e.start_index,
                 "expires_index": e.expires_index,
                 "lifted": e.lifted,
                 "lifted_attestations": sorted(e.lifted_attestations)}
                for e in self._events],
        }
