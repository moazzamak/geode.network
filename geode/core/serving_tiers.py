"""Serving-tier auditability - every session records which tier
served it, and a plaintext session sold as private is a contract
violation.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md``
§8.34 (27 Aug 2026, before any build). Launch gate 1 of the
privacy launch gates (``TESTNET_LAUNCH_PLAN_v26.md`` §3): the
serving-tier record and the public tier mix. Core module - no
privacy import (the M216 direction table).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ServingTier(str, Enum):
    ON_DEVICE = "on_device"        # phone-class encoder runs locally
    FHE_PRIVATE = "fhe_private"    # ciphertext-only head evaluation
    PLAINTEXT = "plaintext"        # disclosed as such


class TierViolation(RuntimeError):
    """A session's tier record contradicts what the session was
    sold as."""


@dataclass(frozen=True)
class TierSession:
    session_id: str
    tier: ServingTier
    disclosed_as_plaintext: bool = False
    ciphertext_only: bool = False
    note: str = ""


@dataclass
class TierAuditLedger:
    """The public record of which tier served every session. The
    mix is a public statistic; nobody can quietly downgrade a
    session's privacy without the record showing it."""
    sessions: list[TierSession] = field(default_factory=list)

    def record(self, session: TierSession) -> None:
        self.assert_tier_integrity(session)
        self.sessions.append(session)

    def tier_mix(self) -> dict[str, int]:
        mix: dict[str, int] = {tier.value: 0 for tier in ServingTier}
        for session in self.sessions:
            mix[session.tier.value] += 1
        return mix

    def assert_tier_integrity(self, session: TierSession) -> None:
        """W-G2: a plaintext session must be disclosed as
        plaintext; an FHE session must be ciphertext-only. Anything
        else is a ledger-visible contract violation."""
        if session.tier == ServingTier.PLAINTEXT and \
                not session.disclosed_as_plaintext:
            raise TierViolation(
                f"session {session.session_id} served on the "
                "plaintext tier without disclosure")
        if session.tier == ServingTier.FHE_PRIVATE and \
                not session.ciphertext_only:
            raise TierViolation(
                f"session {session.session_id} is recorded as FHE "
                "private but is not ciphertext-only")

    def plaintext_sessions_sold_as_private(self) -> list[str]:
        """The violation scan over the record: every plaintext
        session must carry its disclosure."""
        bad = []
        for session in self.sessions:
            try:
                self.assert_tier_integrity(session)
            except TierViolation:
                bad.append(session.session_id)
        return bad
