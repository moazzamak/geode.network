"""Unit tests for the M323 content-order machinery.

Pins the registered gates: the ministerial freeze (G1), the
no-validator-release property (G2), commitment-only evidence (G3),
the nexus gate and its quorum (M323a-G1..G4), and the tier-2/3
triggers (M323b-G1..G4).
"""
from __future__ import annotations

import unittest

from geode.core.content_orders import (
    AuthorityRegistry,
    ContentOrders,
    FreezeState,
    Notice,
)


def _setup() -> tuple[ContentOrders, AuthorityRegistry]:
    reg = AuthorityRegistry()
    reg.add_nexus("Freedonia")
    reg.add_nexus("Elbonia")
    reg.register_key("court-key-1", "Freedonia Court", "Freedonia")
    co = ContentOrders(reg, community_n=3, community_weight=5.0)
    return co, reg


class TestMinisterialFreeze(unittest.TestCase):
    """M323-G1."""

    def test_valid_in_nexus_order_freezes_with_no_vote(self):
        co, _ = _setup()
        notice = Notice("court-key-1", "artifact-A", "session_record",
                        "h(123)", "Freedonia")
        self.assertEqual(co.ministerial_freeze(notice),
                         FreezeState.ESCROWED.value)
        art = co.artifacts["artifact-A"]
        self.assertTrue(art.escrowed)
        self.assertEqual(art.state, FreezeState.SUSPENDED)

    def test_forged_order_produces_no_freeze(self):
        """M323b-G1."""
        co, _ = _setup()
        notice = Notice("fake-key", "artifact-A", "session_record",
                        "h(1)", "Freedonia")
        self.assertEqual(co.ministerial_freeze(notice),
                         FreezeState.RECORD_ONLY.value)
        self.assertFalse(co.artifacts["artifact-A"].escrowed)

    def test_out_of_nexus_order_is_a_report(self):
        """M323a-G1."""
        co, _ = _setup()
        notice = Notice("court-key-1", "artifact-A", "session_record",
                        "h(1)", "Nowhere")
        self.assertEqual(co.ministerial_freeze(notice),
                         FreezeState.RECORD_ONLY.value)
        self.assertFalse(co.artifacts["artifact-A"].escrowed)

    def test_invalid_format_is_a_report(self):
        co, _ = _setup()
        notice = Notice("", "artifact-A", "", "", "Freedonia")
        self.assertEqual(co.ministerial_freeze(notice),
                         FreezeState.RECORD_ONLY.value)


class TestNoValidatorRelease(unittest.TestCase):
    """M323-G2."""

    def test_validators_have_no_release_path(self):
        co, _ = _setup()
        self.assertTrue(co.validators_cannot_release())
        notice = Notice("court-key-1", "artifact-A", "session_record",
                        "h(1)", "Freedonia")
        co.ministerial_freeze(notice)
        # the only exits are confirmation-failure and expiry; nothing
        # in the API lets a validator move funds during the freeze
        self.assertTrue(co.artifacts["artifact-A"].escrowed)
        co.confirm_technical("artifact-A", confirmed=False)
        self.assertFalse(co.artifacts["artifact-A"].escrowed)


class TestNexusQuorum(unittest.TestCase):
    """M323a-G3/G4."""

    def test_no_nexus_finding_downgrades_and_burns(self):
        co, _ = _setup()
        notice = Notice("court-key-1", "artifact-A", "session_record",
                        "h(1)", "Freedonia")
        co.ministerial_freeze(notice)
        self.assertEqual(co.quorum_nexus_finding(notice, False),
                         FreezeState.RECORD_ONLY.value)
        # the downgrade does NOT release the escrow by itself
        self.assertTrue(co.artifacts["artifact-A"].escrowed)

    def test_tie_leaves_the_freeze_in_place(self):
        co, _ = _setup()
        notice = Notice("court-key-1", "artifact-A", "session_record",
                        "h(1)", "Freedonia")
        co.ministerial_freeze(notice)
        self.assertEqual(co.quorum_nexus_finding(notice, None),
                         FreezeState.ESCROWED.value)


class TestCommunityEscalation(unittest.TestCase):
    """M323b-G3/G4."""

    def test_below_n_identities_no_freeze(self):
        co, _ = _setup()
        out = co.community_escalation("artifact-A", "session_record",
                                      ["a", "b"],
                                      {"a": 10.0, "b": 10.0})
        self.assertEqual(out, FreezeState.RECORD_ONLY.value)
        self.assertFalse(co.artifacts["artifact-A"].escrowed)

    def test_below_deposit_weight_no_freeze(self):
        co, _ = _setup()
        out = co.community_escalation("artifact-A", "session_record",
                                      ["a", "b", "c"],
                                      {"a": 1.0, "b": 1.0, "c": 1.0})
        self.assertEqual(out, FreezeState.RECORD_ONLY.value)

    def test_n_identities_and_weight_freeze(self):
        co, _ = _setup()
        out = co.community_escalation("artifact-A", "session_record",
                                      ["a", "b", "c"],
                                      {"a": 2.0, "b": 2.0, "c": 2.0})
        self.assertEqual(out, FreezeState.ESCROWED.value)
        self.assertTrue(co.artifacts["artifact-A"].escrowed)

    def test_unconfirmed_community_freeze_burns_and_releases(self):
        """M323b-G4."""
        co, _ = _setup()
        co.community_escalation("artifact-A", "session_record",
                                ["a", "b", "c"],
                                {"a": 2.0, "b": 2.0, "c": 2.0})
        self.assertEqual(co.community_unconfirmed("artifact-A"),
                         FreezeState.RELEASED.value)
        self.assertEqual(co.deposits, {})
        self.assertFalse(co.artifacts["artifact-A"].escrowed)


class TestCommitmentOnly(unittest.TestCase):
    """M323-G3: no content field exists anywhere."""

    def test_notice_has_no_content_field(self):
        fields = set(Notice.__dataclass_fields__)
        self.assertNotIn("content", fields)
        self.assertNotIn("image", fields)
        self.assertNotIn("payload", fields)

    def test_record_entries_are_commitments_only(self):
        co, _ = _setup()
        notice = Notice("court-key-1", "artifact-A", "session_record",
                        "h(1)", "Freedonia")
        co.ministerial_freeze(notice)
        entry = co.records[-1]
        self.assertEqual(set(entry), {"kind", "artifact_hash",
                                      "evidence_class", "evidence_hash",
                                      "jurisdiction", "outcome"})


if __name__ == "__main__":
    unittest.main()
