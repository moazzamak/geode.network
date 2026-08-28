"""M323b - the authority-key registry with multi-channel pinning.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md``
§8.35 (27 Aug 2026, before any build). A government key is
accepted only when at least three registered independent channels
agree on the same key-to-authority binding:

- a binding under the channel minimum does not settle;
- three agreeing channels settle it;
- a conflicting channel blocks settlement and the disagreement
  stays visible (the registry never silently chooses);
- rotations and revocations flow through their OWN channel
  minimum (the same three reports a revocation must be three
  revocation reports, never the three binding reports re-read);
- a revoked key never authenticates again;
- nexus admission is a compliance-policy change: it routes through
  the registered cross-jurisdiction quorum (M324-G4), never a
  single channel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

MIN_CHANNELS = 3               # the registered settlement minimum


class ChannelConflict(RuntimeError):
    """Channels disagree on a binding; the registry records the
    disagreement instead of settling."""


class BelowChannelMinimum(RuntimeError):
    """A binding has not reached the registered channel minimum."""


class AnnouncementKind(str, Enum):
    BIND = "bind"
    REVOKE = "revoke"


@dataclass(frozen=True)
class ChannelAnnouncement:
    channel: str
    key: str
    authority: str
    jurisdiction: str
    kind: AnnouncementKind = AnnouncementKind.BIND


@dataclass
class MultiChannelPinner:
    """The network's view of which key belongs to which authority,
    settled once by multi-channel agreement. Bindings and
    revocations accumulate on their own channel sets."""
    channels: set[str] = field(default_factory=set)
    bind_views: dict[str, dict[str, ChannelAnnouncement]] = field(
        default_factory=dict)     # key -> channel -> announcement
    revoke_views: dict[str, dict[str, ChannelAnnouncement]] = field(
        default_factory=dict)     # key -> channel -> announcement
    bindings: dict[str, dict[str, str]] = field(
        default_factory=dict)     # key -> {authority, jurisdiction}
    conflicts: dict[str, set[str]] = field(default_factory=dict)
    revoked: set[str] = field(default_factory=set)
    min_channels: int = MIN_CHANNELS

    def register_channel(self, channel: str) -> None:
        self.channels.add(channel)

    # -- the multi-channel rule -------------------------------------

    def _binding_state(self, key: str) -> str:
        per_channel = self.bind_views.get(key, {})
        views = {(a.authority, a.jurisdiction)
                 for a in per_channel.values()}
        if len(views) > 1:
            self.conflicts[key] = {a.channel
                                   for a in per_channel.values()}
            return "conflicted"
        if len(per_channel) < self.min_channels:
            return "pending"
        if key in self.revoked:
            return "revoked"
        first = next(iter(per_channel.values()))
        self.bindings[key] = {"authority": first.authority,
                              "jurisdiction": first.jurisdiction}
        return "settled"

    def _revocation_state(self, key: str) -> str:
        per_channel = self.revoke_views.get(key, {})
        if len(per_channel) < self.min_channels:
            return "pending"
        self.revoked.add(key)
        self.bindings.pop(key, None)
        self.conflicts.pop(key, None)
        return "revoked"

    # -- the public surface -----------------------------------------

    def announce(self, announcement: ChannelAnnouncement) -> str:
        """One channel's binding report. The same channel
        reporting twice counts once (R-G5)."""
        self.register_channel(announcement.channel)
        if announcement.kind == AnnouncementKind.REVOKE:
            self.revoke_views.setdefault(
                announcement.key, {})[announcement.channel] = announcement
            return self._revocation_state(announcement.key)
        self.bind_views.setdefault(
            announcement.key, {})[announcement.channel] = announcement
        return self._binding_state(announcement.key)

    def authenticate(self, key: str) -> bool:
        return (key in self.bindings and key not in self.revoked)

    def rotate(self, old_key: str, new_key: str,
               announcement: ChannelAnnouncement) -> str:
        """A rotation settles through the new key's own channel
        minimum; the old key revokes when the new one settles."""
        if not self.authenticate(old_key):
            return "pending"
        state = self.announce(ChannelAnnouncement(
            channel=announcement.channel, key=new_key,
            authority=announcement.authority,
            jurisdiction=announcement.jurisdiction))
        if state == "settled":
            self.revoked.add(old_key)
            self.bindings.pop(old_key, None)
            self.conflicts.pop(old_key, None)
        return state

    def revoke(self, key: str, announcement: ChannelAnnouncement
               ) -> str:
        """A revocation settles at its own channel minimum; the
        revoked key never authenticates again (R-G4)."""
        return self.announce(ChannelAnnouncement(
            channel=announcement.channel, key=key,
            authority=announcement.authority,
            jurisdiction=announcement.jurisdiction,
            kind=AnnouncementKind.REVOKE))

    def settlement(self, key: str) -> str:
        """The public settlement state of a key."""
        if key in self.revoked:
            return "revoked"
        if key in self.bindings:
            return "settled"
        if key in self.conflicts:
            return "conflicted"
        if key in self.bind_views or key in self.revoke_views:
            return "pending"
        return "unknown"
