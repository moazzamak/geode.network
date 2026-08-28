"""M324 - control-escalation resistance: schema/capability audits.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md``
§8.25 (27 Aug 2026, before any build). The protocol-level defense
against state escalation is INEXPRESSIBILITY: a demand for a
capability the system does not contain is a category error, not a
refusal. This module turns the registered invariants into
executable audits:

- **M324-G1** no code path or ledger entry can express user-level
  exclusion (schema audit over the entry-type whitelist);
- **M324-G2** the developer's key set holds no capability matching
  an escalation demand (M317-style capability model);
- **M324-G3** every compliance action is artifact-scoped and
  publicly recorded;
- **M324-G4** compliance-policy changes require the
  cross-jurisdiction quorum and the standard timelock;
- **M324a** the gateway-operator role is separable from the
  developer role in the registry schema (G1/G2/G3);
- **M324b** a released frontend is content-addressed, has no
  developer-held pointer, is pinned by the registered number of
  independent parties, and works against any gateway set (G1-G4).

The module is a spec module in the M311/M317/M319 style: the
policy it encodes is the deliverable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ----------------------------------------------------------------------
# G1: the ledger schema cannot express user-level exclusion
# ----------------------------------------------------------------------

REGISTERED_ENTRY_TYPES = frozenset({
    "route", "answer", "abstention", "payment", "price_table",
    "registry", "challenge", "probe", "order", "freeze",
    "policy_change", "anchor", "disposal",
})

# fields that would express a user-level selection surface; their
# presence in any ledger entry is an audit failure by construction
BANNED_FIELDS = frozenset({
    "user_id", "user", "ip", "ip_address", "region", "jurisdiction",
    "device", "address_blacklist", "blocklist", "identity_attribute",
})


class SchemaViolation(RuntimeError):
    """A ledger entry or code path expresses something the protocol
    does not contain."""


class LedgerSchemaAudit:
    """The ledger entry whitelist plus the banned-field scan."""

    def __init__(self, entry_types: frozenset[str] | None = None):
        self.entry_types = entry_types or REGISTERED_ENTRY_TYPES

    def assert_known_type(self, entry: dict[str, Any]) -> None:
        if entry.get("type") not in self.entry_types:
            raise SchemaViolation(
                f"unregistered ledger entry type {entry.get('type')!r}")

    def assert_no_user_selection_fields(self, entry: dict[str, Any]
                                        ) -> None:
        for key in entry:
            if key.lower() in BANNED_FIELDS:
                raise SchemaViolation(
                    f"ledger field {key!r} would express a user-level "
                    "selection surface that the protocol does not contain")
        # nested scan: the audit is over the whole payload
        for value in entry.values():
            if isinstance(value, dict):
                self.assert_no_user_selection_fields(value)

    def audit(self, entry: dict[str, Any]) -> None:
        self.assert_known_type(entry)
        self.assert_no_user_selection_fields(entry)


# ----------------------------------------------------------------------
# G2: the developer key set holds no escalation capability
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Capability:
    name: str
    targets: frozenset[str]   # the object classes it can act on


@dataclass
class Key:
    key_id: str
    roles: frozenset[str] = frozenset()

    def capabilities(self, registry: "CapabilityRegistry"
                     ) -> set[Capability]:
        caps: set[Capability] = set()
        for role in self.roles:
            caps |= registry.roles.get(role, frozenset())
        return caps


class CapabilityRegistry:
    def __init__(self, roles: dict[str, frozenset[Capability]]):
        self.roles = roles


# the registered role/capability table: every capability acts on
# ARTIFACTS and LEDGER entries - never on users, regions, or traffic
REGISTERED_ROLES: dict[str, frozenset[Capability]] = {
    "librarian": frozenset({
        Capability("append_entry", frozenset({"ledger"})),
        Capability("file_freeze", frozenset({"artifact"})),
        Capability("file_lift", frozenset({"artifact"})),
        Capability("file_policy_change", frozenset({"policy"})),
    }),
    "developer": frozenset({
        Capability("operate_gateway", frozenset({"frontend"})),
        Capability("operate_registry", frozenset({"registry"})),
    }),
    "gateway_operator": frozenset({
        Capability("operate_gateway", frozenset({"frontend"})),
    }),
    "validator": frozenset({
        Capability("submit_challenge", frozenset({"artifact"})),
        Capability("vote", frozenset({"artifact", "policy"})),
    }),
}

# escalation demands, in the capability language. Every one of them
# names a target class the registered table does not contain.
ESCALATION_DEMANDS: dict[str, Capability] = {
    "block users from a jurisdiction": Capability(
        "exclude_users", frozenset({"user"})),
    "block IP ranges": Capability("exclude_ips", frozenset({"ip"})),
    "throttle a region": Capability("throttle", frozenset({"region"})),
    "price by identity": Capability("price_discriminate",
                                    frozenset({"identity"})),
}


class CapabilityAudit:
    def __init__(self, roles: dict[str, frozenset[Capability]]
                 | None = None):
        self.registry = CapabilityRegistry(roles or REGISTERED_ROLES)

    def key_capabilities(self, key: Key) -> set[Capability]:
        return key.capabilities(self.registry)

    def assert_no_escalation_capability(self, key: Key) -> None:
        """M324-G2: no developer (or any) key holds a capability
        matching an escalation demand."""
        held = key.capabilities(self.registry)
        for name, demand in ESCALATION_DEMANDS.items():
            if demand in held:
                raise SchemaViolation(
                    f"key {key.key_id} holds capability "
                    f"{demand.name!r} matching the escalation demand "
                    f"{name!r}; the protocol contains no such surface")
        # also: no held capability may target user/ip/region classes
        for cap in held:
            if cap.targets & frozenset({"user", "ip", "region",
                                        "identity"}):
                raise SchemaViolation(
                    f"capability {cap.name!r} targets "
                    f"{sorted(cap.targets)}: outside the registered "
                    "artifact/policy surface")


# ----------------------------------------------------------------------
# G3: compliance actions are artifact-scoped and publicly recorded
# ----------------------------------------------------------------------

class ComplianceRecordAudit:
    """Every compliance action must name an artifact and produce a
    public record. Actions without an artifact hash, or with a
    user-level field, fail."""

    def audit(self, record: dict[str, Any]) -> None:
        if "artifact_hash" not in record:
            raise SchemaViolation(
                "a compliance action must be artifact-scoped: "
                "no artifact_hash in the record")
        LedgerSchemaAudit().assert_no_user_selection_fields(record)
        if record.get("kind") not in ("order", "freeze", "policy_change",
                                      "report", "takedown"):
            raise SchemaViolation(
                f"unregistered compliance record kind "
                f"{record.get('kind')!r}")


# ----------------------------------------------------------------------
# G4: policy changes need the quorum and the timelock
# ----------------------------------------------------------------------

@dataclass
class PolicyChangeGate:
    """Compliance-policy changes (authority-key admissions,
    thresholds, notice formats) require cross-jurisdiction quorum
    agreement under the standard timelock. A single jurisdiction
    cannot dictate policy."""
    quorum_jurisdictions: frozenset[str]
    timelock_epochs: int = 1
    endorsements: dict[str, set[str]] = field(default_factory=dict)
    applied: set[str] = field(default_factory=set)
    applied_at: dict[str, int] = field(default_factory=dict)
    _endorsed_at: dict[str, int] = field(default_factory=dict)

    def propose(self, change: str, jurisdiction: str, epoch: int
                ) -> str:
        if jurisdiction not in self.quorum_jurisdictions:
            raise SchemaViolation(
                f"jurisdiction {jurisdiction!r} is outside the "
                "registered nexus quorum")
        self.endorsements.setdefault(change, set()).add(jurisdiction)
        # the timelock clock starts when the FULL quorum is reached
        if self.can_apply(change):
            self._endorsed_at.setdefault(change, epoch)
        return "proposed"

    def endorsers(self, change: str) -> set[str]:
        return self.endorsements.get(change, set())

    def can_apply(self, change: str) -> bool:
        return (len(self.endorsements.get(change, set()))
                >= len(self.quorum_jurisdictions))

    def apply(self, change: str, epoch: int) -> str:
        """Application requires the FULL quorum AND the registered
        timelock since the change was fully endorsed."""
        if not self.can_apply(change):
            raise SchemaViolation(
                f"change {change!r} lacks the cross-jurisdiction quorum")
        endorsed_at = self._endorsed_at.get(change, epoch)
        if epoch - endorsed_at < self.timelock_epochs:
            raise SchemaViolation(
                f"change {change!r} is inside the timelock")
        self.applied.add(change)
        self.applied_at[change] = epoch
        return "applied"


# ----------------------------------------------------------------------
# M324a: the gateway role is separable from the developer
# ----------------------------------------------------------------------

class GatewaySeparabilityAudit:
    """The registry schema names a gateway-operator role that any
    third party can hold; the developer's own services are not
    frontends; the bootstrap-gateway sunset is a registered
    governance path with a timelock, not an aspiration."""

    def __init__(self, sunset_timelock_epochs: int = 1):
        self.sunset_timelock_epochs = sunset_timelock_epochs
        self.operators: dict[str, str] = {}      # gateway -> operator
        self.sunset_registered: bool = False

    def register_operator(self, gateway: str, operator: str) -> None:
        self.operators[gateway] = operator

    def separable(self, developer: str) -> bool:
        """M324a-G2: at least one gateway is operated by a party
        other than the developer."""
        return any(op != developer for op in self.operators.values())

    def register_sunset(self) -> None:
        self.sunset_registered = True

    def developer_services_are_not_frontends(self, services: set[str],
                                             frontends: set[str]) -> bool:
        return services.isdisjoint(frontends)


# ----------------------------------------------------------------------
# M324b: the immutable frontend release model
# ----------------------------------------------------------------------

class ContentAddressedReleaseAudit:
    """A released frontend is content-addressed, has no
    developer-held pointer, is pinned by enough independent
    parties, and works against any gateway set."""

    def __init__(self, required_pinners: int = 3):
        self.required_pinners = required_pinners
        self.releases: dict[str, str] = {}        # cid -> bytes digest
        self.pinners: dict[str, set[str]] = {}    # cid -> pinners
        self.pointer: str | None = None           # the banned lever

    def release(self, cid: str, digest: str) -> None:
        if self.pointer is not None:
            raise SchemaViolation("releases are ledger-side; the "
                                  "pointer lever does not exist")
        self.releases[cid] = digest

    def assert_immutable(self, cid: str, digest: str) -> None:
        """M324b-G1: the CID's content cannot change; any digest
        disagreement is a content-address violation."""
        if self.releases.get(cid) != digest:
            raise SchemaViolation(
                f"CID {cid} is content-addressed: digest mismatch")

    def assert_no_developer_pointer(self) -> None:
        """M324b-G2: no developer-held pointer may resolve a
        frontend; discovery is ledger-side."""
        if self.pointer is not None:
            raise SchemaViolation("a developer-held canonical pointer "
                                  "is a de facto control lever")

    def pin(self, cid: str, pinner: str) -> None:
        self.pinners.setdefault(cid, set()).add(pinner)

    def availability_gate(self, cid: str) -> bool:
        """M324b-G3: at least the registered number of INDEPENDENT
        parties pin the release."""
        return len(self.pinners.get(cid, set())) >= self.required_pinners

    def endpoint_agnostic(self, frontend: dict[str, Any],
                          gateway_set: set[str]) -> bool:
        """M324b-G4: the frontend discovers gateways from the
        ledger and functions against ANY of them - none is
        hardcoded as a dependency."""
        discovered = frontend.get("gateways", set())
        return bool(discovered) and gateway_set.issubset(discovered)
