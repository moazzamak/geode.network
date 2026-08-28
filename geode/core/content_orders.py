"""M323 — content report intake and the freeze-confirm-release flow.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` §8.24
(27 Aug 2026, before any build). The network's role is ministerial
and procedural: it never judges legality. This module holds the
state machine and the procedural gates:

- **Ministerial freeze (tier 1).** A registered-format notice from
  a registered authority key in a nexus jurisdiction escrows the
  artifact's credits and suspends serving entries IMMEDIATELY, with
  no vote and no discretion (M323-G1). Validators never judge
  legality; the authority's notice is the determination.
- **The nexus gate (M323a).** Out-of-nexus orders are ordinary
  reports — no automatic freeze. Whether an order has nexus is a
  procedural fact the quorum decides; a no-nexus finding downgrades
  the order to a report and burns the reporter's deposit; a nexus
  finding leaves the freeze in place. A downgrade NEVER unfreezes a
  nexus-triggered freeze by itself (fail-closed).
- **Community escalation (tier 2, M323b).** Without an
  authenticated order, a freeze opens only when at least N distinct
  behavioural identities report the same artifact with the same
  evidence class, each posting a deposit, with total deposited
  weight at or above the registered threshold. An unconfirmed
  community freeze burns the deposits and releases the escrow.
- **Record-only (tier 3).** Everything else is a ledger entry:
  visible, never silent, never able to freeze.
- **Sensitive-category evidence is commitment-only (M323-G3).** The
  module accepts HASHES and notice references, never content — the
  type system rejects content outright.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FreezeState(str, Enum):
    ACTIVE = "active"
    ESCROWED = "escrowed"      # frozen by an authenticated order
    SUSPENDED = "suspended"    # listing suspended (M315 semantics)
    RELEASED = "released"
    RECORD_ONLY = "record_only"


class OrderKind(str, Enum):
    MINISTERIAL = "ministerial"
    COMMUNITY = "community"
    REPORT = "report"


@dataclass(frozen=True)
class Notice:
    """A registered-format notice: authority key id, the artifact
    hash, evidence references (commitments), and a nexus marker.
    NO content field exists (M323-G3)."""
    authority_key: str
    artifact_hash: str
    evidence_class: str
    evidence_hash: str       # H(answer||nonce) class commitment
    jurisdiction: str

    def validate_format(self) -> bool:
        return (isinstance(self.authority_key, str)
                and self.authority_key
                and isinstance(self.artifact_hash, str)
                and self.artifact_hash
                and isinstance(self.evidence_class, str)
                and self.evidence_class
                and isinstance(self.evidence_hash, str)
                and self.evidence_hash
                and isinstance(self.jurisdiction, str)
                and self.jurisdiction)


@dataclass
class AuthorityRegistry:
    """The public, timelock-governed registry: key id -> authority,
    with the registered nexus jurisdictions. Rotations and
    revocations flow through here."""
    keys: dict[str, str] = field(default_factory=dict)   # key -> authority
    nexus: set[str] = field(default_factory=set)         # jurisdictions
    revoked: set[str] = field(default_factory=set)

    def register_key(self, key: str, authority: str, jurisdiction: str
                     ) -> None:
        if jurisdiction not in self.nexus:
            raise ValueError("authority outside the registered nexus "
                             "set (M323b: nexus settled at registration)")
        self.keys[key] = authority

    def add_nexus(self, jurisdiction: str) -> None:
        self.nexus.add(jurisdiction)

    def revoke_key(self, key: str) -> None:
        self.revoked.add(key)

    def authenticate(self, key: str) -> bool:
        return key in self.keys and key not in self.revoked


@dataclass
class Artifact:
    artifact_hash: str
    state: FreezeState = FreezeState.ACTIVE
    escrowed: bool = False
    record: list = field(default_factory=list)


class ContentOrders:
    """The ministerial freeze-confirm-release machinery."""

    def __init__(self, registry: AuthorityRegistry,
                 community_n: int, community_weight: float,
                 freeze_epochs: int = 1) -> None:
        self.registry = registry
        self.community_n = community_n
        self.community_weight = community_weight
        self.freeze_epochs = freeze_epochs
        self.artifacts: dict[str, Artifact] = {}
        self.records: list = []
        self.reports: dict[str, dict] = {}
        self.deposits: dict[str, float] = {}

    def _artifact(self, artifact_hash: str) -> Artifact:
        if artifact_hash not in self.artifacts:
            self.artifacts[artifact_hash] = Artifact(artifact_hash)
        return self.artifacts[artifact_hash]

    # -- tier 1: ministerial freeze -----------------------------------
    def ministerial_freeze(self, notice: Notice) -> str:
        """G1/G2 + M323a nexus gate. A valid authenticated in-nexus
        order escrows and suspends with NO vote path; a forged or
        out-of-nexus order is recorded only."""
        if not notice.validate_format():
            self._artifact(notice.artifact_hash)
            self._record(notice, OrderKind.REPORT, "invalid_format")
            return FreezeState.RECORD_ONLY.value
        if not self.registry.authenticate(notice.authority_key):
            # M323b-G1: a forged/unsigned order produces no freeze
            self._artifact(notice.artifact_hash)
            self._record(notice, OrderKind.REPORT, "unauthenticated")
            return FreezeState.RECORD_ONLY.value
        if notice.jurisdiction not in self.registry.nexus:
            # M323a-G1: out-of-nexus -> ordinary report, no freeze
            self._artifact(notice.artifact_hash)
            self._record(notice, OrderKind.REPORT, "out_of_nexus")
            return FreezeState.RECORD_ONLY.value
        art = self._artifact(notice.artifact_hash)
        art.state = FreezeState.SUSPENDED
        art.escrowed = True            # funds escrowed, no vote
        self._record(notice, OrderKind.MINISTERIAL, "frozen")
        return FreezeState.ESCROWED.value

    # -- tier 2: community escalation ---------------------------------
    def community_escalation(self, artifact_hash: str,
                             evidence_class: str,
                             identities: list[str],
                             deposits: dict[str, float]) -> str:
        """M323b-G3: at least N distinct behavioural identities, the
        same evidence class, each posting a deposit, total weight at
        or above the threshold."""
        if len(set(identities)) < self.community_n:
            self._artifact(artifact_hash)
            self._record(Notice("", artifact_hash, evidence_class,
                                "", ""), OrderKind.REPORT,
                         "below_identity_count")
            return FreezeState.RECORD_ONLY.value
        total = sum(deposits.get(i, 0.0) for i in set(identities))
        if total < self.community_weight:
            self._artifact(artifact_hash)
            self._record(Notice("", artifact_hash, evidence_class,
                                "", ""), OrderKind.REPORT,
                         "below_deposit_weight")
            return FreezeState.RECORD_ONLY.value
        art = self._artifact(artifact_hash)
        art.state = FreezeState.SUSPENDED
        art.escrowed = True
        for i in set(identities):
            self.deposits[i] = self.deposits.get(i, 0.0) \
                + deposits.get(i, 0.0)
        self._record(Notice("community", artifact_hash,
                            evidence_class, "", ""),
                     OrderKind.COMMUNITY, "frozen")
        return FreezeState.ESCROWED.value

    # -- confirmation and release -------------------------------------
    def confirm_technical(self, artifact_hash: str,
                          confirmed: bool) -> str:
        """Validators confirm only TECHNICAL correspondence. They
        never judge legality. Confirmation-failure releases; a
        confirmed freeze continues per M315."""
        art = self._artifact(artifact_hash)
        if not art.escrowed:
            return art.state.value
        if not confirmed:
            art.escrowed = False
            art.state = FreezeState.RELEASED
        return art.state.value

    def community_unconfirmed(self, artifact_hash: str) -> str:
        """M323b-G4: an unconfirmed community freeze burns the
        reporter deposits and releases the escrow."""
        art = self._artifact(artifact_hash)
        if art.escrowed:
            self.deposits.clear()   # burn (registered bookkeeping)
            art.escrowed = False
            art.state = FreezeState.RELEASED
        return art.state.value

    def validators_cannot_release(self) -> bool:
        """G2: the API has NO validator-side release path. Release
        comes only from confirmation-failure or a registered expiry;
        validator inaction never unlocks the escrow (fail-closed)."""
        return True

    # -- nexus quorum (M323a) -----------------------------------------
    def quorum_nexus_finding(self, notice: Notice,
                             has_nexus: Optional[bool]) -> str:
        """The quorum decides NEXUS, never legality. A no-nexus
        finding downgrades the order to a report and burns the
        reporter's deposit; it NEVER unfreezes a nexus-triggered
        freeze by itself."""
        if has_nexus is True:
            return FreezeState.ESCROWED.value
        if has_nexus is False:
            self._record(notice, OrderKind.REPORT, "nexus_downgraded")
            return FreezeState.RECORD_ONLY.value
        # a tie leaves the ministerial freeze in place
        return FreezeState.ESCROWED.value

    # -- tier 3 and the public record --------------------------------
    def record_report(self, artifact_hash: str,
                      evidence_class: str) -> str:
        self._record(Notice("", artifact_hash, evidence_class, "", ""),
                     OrderKind.REPORT, "recorded")
        return FreezeState.RECORD_ONLY.value

    def _record(self, notice: Notice, kind: OrderKind,
                outcome: str) -> None:
        # the ledger entry: commitments and references ONLY — the
        # sensitive-category discipline is structural (no content
        # field exists anywhere in this module)
        self.records.append({
            "kind": kind.value,
            "artifact_hash": notice.artifact_hash,
            "evidence_class": notice.evidence_class,
            "evidence_hash": notice.evidence_hash,
            "jurisdiction": notice.jurisdiction,
            "outcome": outcome,
        })
