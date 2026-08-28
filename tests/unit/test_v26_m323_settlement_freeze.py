"""M323 settlement-wiring tests: the ministerial freeze bridge."""
from __future__ import annotations

import unittest

from geode.core.content_orders import (
    AuthorityRegistry,
    ContentOrders,
    FreezeState,
    Notice,
)
from geode.core.settlement_freeze import (
    EPOCH_SECONDS,
    FREEZE_SIGNATURE,
    LIFT_SIGNATURE,
    NoFreezeOrderError,
    SettlementFreezeBridge,
    _selector,
    keccak256,
)

# the empty-string Keccak-256 vector (the standard test vector)
KECCAK_EMPTY = "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
# selectors cross-checked against ethers.id() in the EVM workspace
FREEZE_SELECTOR = "57093b23"
LIFT_SELECTOR = "0d7d2752"


def _orders() -> ContentOrders:
    registry = AuthorityRegistry()
    registry.add_nexus("GB")
    registry.register_key("gov-1", "Court A", "GB")
    return ContentOrders(registry, community_n=3,
                         community_weight=10.0)


def _notice(artifact_hash: str = "art-1") -> Notice:
    return Notice(authority_key="gov-1", artifact_hash=artifact_hash,
                  evidence_class="session_record",
                  evidence_hash="e1e1e1", jurisdiction="GB")


class TestKeccakAndSelectors(unittest.TestCase):
    def test_keccak_empty_vector(self):
        self.assertEqual(keccak256(b"").hex(), KECCAK_EMPTY)

    def test_selectors_match_ethers(self):
        self.assertEqual(_selector(FREEZE_SIGNATURE).hex(),
                         FREEZE_SELECTOR)
        self.assertEqual(_selector(LIFT_SIGNATURE).hex(),
                         LIFT_SELECTOR)


class TestFreezeBridge(unittest.TestCase):
    def test_escrowed_order_produces_a_deterministic_filing(self):
        orders = _orders()
        state = orders.ministerial_freeze(_notice())
        self.assertEqual(state, FreezeState.ESCROWED.value)
        bridge = SettlementFreezeBridge()
        filing = bridge.file_freeze(orders, "art-1", "e1e1e1",
                                    epochs=1)
        self.assertEqual(filing.selector.hex(), FREEZE_SELECTOR)
        self.assertEqual(len(filing.calldata), 4 + 32 * 3)
        filing2 = bridge.file_freeze(orders, "art-1", "e1e1e1",
                                     epochs=1)
        self.assertEqual(filing.calldata, filing2.calldata)
        self.assertEqual(filing.digest, filing2.digest)

    def test_record_only_order_never_reaches_the_contract(self):
        orders = _orders()
        bad = _notice(artifact_hash="art-2")
        bad = Notice(authority_key="forged-key",
                     artifact_hash="art-2", evidence_class="x",
                     evidence_hash="y", jurisdiction="GB")
        state = orders.ministerial_freeze(bad)
        self.assertEqual(state, FreezeState.RECORD_ONLY.value)
        with self.assertRaises(NoFreezeOrderError):
            SettlementFreezeBridge().file_freeze(
                orders, "art-2", "y", epochs=1)

    def test_out_of_nexus_order_never_reaches_the_contract(self):
        orders = _orders()
        notice = Notice(authority_key="gov-1", artifact_hash="art-3",
                        evidence_class="x", evidence_hash="y",
                        jurisdiction="FR")
        self.assertEqual(orders.ministerial_freeze(notice),
                         FreezeState.RECORD_ONLY.value)
        with self.assertRaises(NoFreezeOrderError):
            SettlementFreezeBridge().file_freeze(
                orders, "art-3", "y", epochs=1)

    def test_zero_window_raises(self):
        orders = _orders()
        orders.ministerial_freeze(_notice())
        with self.assertRaises(NoFreezeOrderError):
            SettlementFreezeBridge().file_freeze(
                orders, "art-1", "e1e1e1", epochs=0)

    def test_lift_only_from_released_state(self):
        orders = _orders()
        orders.ministerial_freeze(_notice())
        # still escrowed: no lift can be filed
        with self.assertRaises(NoFreezeOrderError):
            SettlementFreezeBridge().file_lift(orders, "art-1")
        # confirmation failure releases
        self.assertEqual(orders.confirm_technical("art-1", False),
                         FreezeState.RELEASED.value)
        lift = SettlementFreezeBridge().file_lift(orders, "art-1")
        self.assertEqual(lift.selector.hex(), LIFT_SELECTOR)
        self.assertEqual(len(lift.calldata), 4 + 32)

    def test_validators_have_no_move_path(self):
        self.assertTrue(
            SettlementFreezeBridge.validators_have_no_move_path())

    def test_epoch_window_registered(self):
        self.assertEqual(EPOCH_SECONDS, 7 * 24 * 3600)


if __name__ == "__main__":
    unittest.main()
