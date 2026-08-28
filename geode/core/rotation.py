"""GEODE verifier rotation (v25 M253, rotation half) — no standing
committee.

Quorum verifier sets rotate on a deterministic ledger-index
schedule: the active committee for a given index is a pure function
of the ordered verifier list and the index, so no committee can
persist beyond its window and no authority selects committees (the
trustless-world amendment: capture of the CURRENT committee buys an
attacker only the remainder of its window).

Deterministic: no RNG, no wall clocks. The staking half (slashing
of false attestations) is the M194-gated cell; this module records
the interface it will feed (``stake_weights`` is reserved and
ignored until then).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class VerifierRotation:
    """Deterministic rotating committees over a fixed verifier set."""

    def __init__(self, verifiers: Sequence[str],
                 committee_size: int | None = None,
                 epoch_length: int = 100):
        if not verifiers:
            raise ValueError("the verifier set must be non-empty")
        ordered = sorted(set(verifiers))
        size = len(ordered) if committee_size is None else int(
            committee_size)
        if not 1 <= size <= len(ordered):
            raise ValueError(f"committee_size {size} out of range "
                             f"1..{len(ordered)}")
        if epoch_length <= 0:
            raise ValueError("epoch_length must be positive")
        self._verifiers = ordered
        self.committee_size = size
        self.epoch_length = int(epoch_length)

    @property
    def verifiers(self) -> list[str]:
        return list(self._verifiers)

    def epoch(self, ledger_index: int) -> int:
        """The epoch a ledger index falls in."""
        if ledger_index < 0:
            raise ValueError("ledger_index must be non-negative")
        return int(ledger_index) // self.epoch_length

    def committee_for(self, ledger_index: int) -> list[str]:
        """The active committee at an index: a windowed rotation
        through the ordered verifier list (epoch e starts at offset
        e % n; wraps around). Deterministic."""
        n = len(self._verifiers)
        start = self.epoch(ledger_index) % n
        idx = [(start + i) % n for i in range(self.committee_size)]
        return [self._verifiers[i] for i in idx]

    def committee_span(self, verifier: str) -> int:
        """The maximum number of CONSECUTIVE epochs a verifier can
        sit on the committee (ceil(k/n) windows per full cycle):
        the anti-capture bound — capture cannot persist forever."""
        n = len(self._verifiers)
        k = self.committee_size
        return (k + n - 1) // n if verifier in self._verifiers else 0

    def quorum_met(self, attestations: Sequence[str],
                   ledger_index: int, k_of_n: int) -> bool:
        """Whether the attestations clear k-of-n WITHIN the active
        committee at this index (attestations from outside the
        committee do not count)."""
        committee = set(self.committee_for(ledger_index))
        inside = [a for a in attestations if a in committee]
        return len(inside) >= k_of_n

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifiers": self._verifiers,
            "committee_size": self.committee_size,
            "epoch_length": self.epoch_length,
        }
