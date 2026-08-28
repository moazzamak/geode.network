"""M323b unit tests - multi-channel authority-key pinning."""
from __future__ import annotations

import unittest

from geode.core.authority_key_registry import (
    AnnouncementKind,
    ChannelAnnouncement,
    MultiChannelPinner,
)


def _bind(pinner: MultiChannelPinner, channel: str, key: str,
          authority: str = "Court A", jurisdiction: str = "GB"
          ) -> str:
    return pinner.announce(ChannelAnnouncement(
        channel=channel, key=key, authority=authority,
        jurisdiction=jurisdiction))


def _revoke(pinner: MultiChannelPinner, channel: str, key: str
            ) -> str:
    return pinner.revoke(key, ChannelAnnouncement(
        channel=channel, key=key, authority="Court A",
        jurisdiction="GB", kind=AnnouncementKind.REVOKE))


class TestMultiChannelPinning(unittest.TestCase):
    def test_under_minimum_does_not_settle(self):
        pinner = MultiChannelPinner()
        self.assertEqual(_bind(pinner, "ch-1", "k-1"), "pending")
        self.assertEqual(_bind(pinner, "ch-2", "k-1"), "pending")
        self.assertEqual(pinner.settlement("k-1"), "pending")
        self.assertFalse(pinner.authenticate("k-1"))

    def test_three_agreeing_channels_settle(self):
        pinner = MultiChannelPinner()
        _bind(pinner, "ch-1", "k-1")
        _bind(pinner, "ch-2", "k-1")
        self.assertEqual(_bind(pinner, "ch-3", "k-1"), "settled")
        self.assertTrue(pinner.authenticate("k-1"))
        self.assertEqual(pinner.bindings["k-1"]["authority"],
                         "Court A")

    def test_same_channel_counts_once(self):
        pinner = MultiChannelPinner()
        _bind(pinner, "ch-1", "k-1")
        _bind(pinner, "ch-1", "k-1")
        _bind(pinner, "ch-1", "k-1")
        self.assertEqual(pinner.settlement("k-1"), "pending")

    def test_conflicting_channels_block_and_record(self):
        pinner = MultiChannelPinner()
        _bind(pinner, "ch-1", "k-1", authority="Court A")
        _bind(pinner, "ch-2", "k-1", authority="Court A")
        _bind(pinner, "ch-3", "k-1", authority="Court B")
        self.assertEqual(pinner.settlement("k-1"), "conflicted")
        self.assertFalse(pinner.authenticate("k-1"))
        self.assertEqual(pinner.conflicts["k-1"],
                         {"ch-1", "ch-2", "ch-3"})

    def test_revocation_needs_its_own_channels(self):
        pinner = MultiChannelPinner()
        for ch in ("ch-1", "ch-2", "ch-3"):
            _bind(pinner, ch, "k-1")
        self.assertTrue(pinner.authenticate("k-1"))
        # one revocation report: the key still authenticates
        self.assertEqual(_revoke(pinner, "ch-1", "k-1"), "pending")
        self.assertTrue(pinner.authenticate("k-1"))
        # three revocation reports: revoked forever
        _revoke(pinner, "ch-2", "k-1")
        self.assertEqual(_revoke(pinner, "ch-3", "k-1"), "revoked")
        self.assertFalse(pinner.authenticate("k-1"))
        self.assertEqual(pinner.settlement("k-1"), "revoked")

    def test_rotation_revokes_the_old_key(self):
        pinner = MultiChannelPinner()
        for ch in ("ch-1", "ch-2", "ch-3"):
            _bind(pinner, ch, "k-old")
        states = []
        for ch in ("ch-1", "ch-2", "ch-3"):
            states.append(pinner.rotate("k-old", "k-new",
                                        ChannelAnnouncement(
                                            channel=ch, key="k-new",
                                            authority="Court A",
                                            jurisdiction="GB")))
        self.assertEqual(states[-1], "settled")
        self.assertFalse(pinner.authenticate("k-old"))
        self.assertTrue(pinner.authenticate("k-new"))

    def test_rotation_requires_the_old_key_authenticated(self):
        pinner = MultiChannelPinner()
        state = pinner.rotate("unknown", "k-new",
                              ChannelAnnouncement(
                                  channel="ch-1", key="k-new",
                                  authority="Court A",
                                  jurisdiction="GB"))
        self.assertEqual(state, "pending")


if __name__ == "__main__":
    unittest.main()
