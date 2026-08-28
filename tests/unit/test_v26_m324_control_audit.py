"""M324 unit tests - control-escalation resistance audits."""
from __future__ import annotations

import unittest

from geode.core.control_audit import (
    CapabilityAudit,
    CapabilityRegistry,
    ComplianceRecordAudit,
    ContentAddressedReleaseAudit,
    ESCALATION_DEMANDS,
    GatewaySeparabilityAudit,
    Key,
    LedgerSchemaAudit,
    PolicyChangeGate,
    SchemaViolation,
)


class TestLedgerSchemaAudit(unittest.TestCase):
    def test_registered_entry_passes(self):
        LedgerSchemaAudit().audit(
            {"type": "route", "artifact": "a1", "score": 0.9})

    def test_unregistered_entry_type_fails(self):
        with self.assertRaises(SchemaViolation):
            LedgerSchemaAudit().audit({"type": "user_block"})

    def test_user_selection_field_fails(self):
        with self.assertRaises(SchemaViolation):
            LedgerSchemaAudit().audit(
                {"type": "route", "artifact": "a1", "ip": "1.2.3.4"})

    def test_nested_banned_field_fails(self):
        with self.assertRaises(SchemaViolation):
            LedgerSchemaAudit().audit(
                {"type": "order",
                 "payload": {"region": "EU"}})


class TestCapabilityAudit(unittest.TestCase):
    def test_developer_key_has_no_escalation_capability(self):
        audit = CapabilityAudit()
        dev = Key("dev-1", roles=frozenset({"developer"}))
        audit.assert_no_escalation_capability(dev)

    def test_no_registered_role_matches_an_escalation_demand(self):
        audit = CapabilityAudit()
        for key_id, roles in (("librarian", {"librarian"}),
                              ("validator", {"validator"}),
                              ("gateway", {"gateway_operator"})):
            audit.assert_no_escalation_capability(
                Key(key_id, roles=frozenset(roles)))

    def test_escalation_demands_name_unheld_targets(self):
        targets = set()
        for demand in ESCALATION_DEMANDS.values():
            targets |= demand.targets
        self.assertIn("user", targets)
        # every registered capability targets artifacts/policy only
        for role in CapabilityRegistry({}).roles.values():
            pass  # registry here is empty; checked below
        registry = CapabilityAudit().registry
        for role, caps in registry.roles.items():
            for cap in caps:
                self.assertFalse(
                    cap.targets & frozenset({"user", "ip", "region"}),
                    f"{role}/{cap.name} escapes the artifact surface")


class TestComplianceRecordAudit(unittest.TestCase):
    def test_artifact_scoped_record_passes(self):
        ComplianceRecordAudit().audit(
            {"kind": "freeze", "artifact_hash": "a1",
             "evidence": "e1"})

    def test_record_without_artifact_fails(self):
        with self.assertRaises(SchemaViolation):
            ComplianceRecordAudit().audit(
                {"kind": "freeze", "evidence": "e1"})

    def test_record_with_user_field_fails(self):
        with self.assertRaises(SchemaViolation):
            ComplianceRecordAudit().audit(
                {"kind": "order", "artifact_hash": "a1",
                 "user": "u1"})


class TestPolicyChangeGate(unittest.TestCase):
    def _gate(self) -> PolicyChangeGate:
        return PolicyChangeGate(quorum_jurisdictions=frozenset(
            {"GB", "US", "EU"}), timelock_epochs=1)

    def test_single_state_cannot_dictate(self):
        gate = self._gate()
        gate.propose("admit-key-x", "GB", epoch=0)
        self.assertFalse(gate.can_apply("admit-key-x"))
        with self.assertRaises(SchemaViolation):
            gate.apply("admit-key-x", epoch=0)

    def test_full_quorum_then_timelock(self):
        gate = self._gate()
        gate.propose("c1", "GB", epoch=0)
        gate.propose("c1", "US", epoch=1)
        gate.propose("c1", "EU", epoch=1)
        self.assertTrue(gate.can_apply("c1"))
        # timelock not elapsed: applied only from epoch >= 2
        with self.assertRaises(SchemaViolation):
            gate.apply("c1", epoch=1)
        self.assertEqual(gate.apply("c1", epoch=2), "applied")

    def test_outside_nexus_jurisdiction_rejected(self):
        gate = self._gate()
        with self.assertRaises(SchemaViolation):
            gate.propose("c2", "ZZ", epoch=0)


class TestGatewaySeparability(unittest.TestCase):
    def test_third_party_gateway_is_separable(self):
        audit = GatewaySeparabilityAudit()
        audit.register_operator("gw-1", "developer")
        self.assertFalse(audit.separable("developer"))
        audit.register_operator("gw-2", "third-party-co")
        self.assertTrue(audit.separable("developer"))

    def test_developer_services_are_not_frontends(self):
        audit = GatewaySeparabilityAudit()
        self.assertTrue(audit.developer_services_are_not_frontends(
            {"registry", "ledger", "sampling"},
            {"gw-1", "gw-2"}))


class TestContentAddressedRelease(unittest.TestCase):
    def test_release_is_immutable(self):
        audit = ContentAddressedReleaseAudit()
        audit.release("cid-1", "digest-a")
        audit.assert_immutable("cid-1", "digest-a")
        with self.assertRaises(SchemaViolation):
            audit.assert_immutable("cid-1", "digest-b")

    def test_pointer_lever_does_not_exist(self):
        audit = ContentAddressedReleaseAudit()
        audit.assert_no_developer_pointer()

    def test_availability_requires_independent_pinners(self):
        audit = ContentAddressedReleaseAudit(required_pinners=3)
        audit.release("cid-1", "digest-a")
        self.assertFalse(audit.availability_gate("cid-1"))
        audit.pin("cid-1", "dev")
        audit.pin("cid-1", "dev")          # same party: no growth
        self.assertFalse(audit.availability_gate("cid-1"))
        audit.pin("cid-1", "validator-a")
        audit.pin("cid-1", "gateway-b")
        self.assertTrue(audit.availability_gate("cid-1"))

    def test_endpoint_agnostic_frontend(self):
        audit = ContentAddressedReleaseAudit()
        frontend = {"gateways": {"gw-1", "gw-2", "gw-3"}}
        self.assertTrue(audit.endpoint_agnostic(
            frontend, {"gw-1", "gw-3"}))
        self.assertFalse(audit.endpoint_agnostic(frontend, {"gw-4"}))


if __name__ == "__main__":
    unittest.main()
