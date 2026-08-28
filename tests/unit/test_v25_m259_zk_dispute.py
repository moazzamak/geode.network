"""Unit tests for M259: zk measurement disputes through the
SlashLedger (the M256 cell 2 structure half).
"""
from __future__ import annotations

import unittest

from geode.privacy.zk_dispute import (
    build_dispute_payload,
    verify_dispute_payload,
)
from geode.settlement.slashing import SlashLedger

# a 2-dim linear relation y = W x + b over small integers
W = [[1, 2], [3, 4]]
B = [5, 6]
X = [7, 8]          # y = Wx + b = [1*7+2*8+5, 3*7+4*8+6] = [28, 59]
Y = [28, 59]
R = 12345


class TestM259ZkDispute(unittest.TestCase):

    def test_honest_payload_verifies(self):
        payload = build_dispute_payload(X, R, W, B, Y)
        self.assertTrue(verify_dispute_payload(payload))

    def test_lying_claim_fails_verification(self):
        # a liar proves the WRONG y; their own proof cannot verify
        liar = build_dispute_payload(X, R, W, B, [999, 999])
        self.assertFalse(verify_dispute_payload(liar))

    def test_slash_accused_when_their_claim_fails(self):
        honest = build_dispute_payload(X, R, W, B, Y)
        lying = build_dispute_payload(X, R, W, B, [999, 999])
        ledger = SlashLedger()
        ledger.deposit("attester", 100.0)
        ledger.deposit("challenger", 50.0)
        out = ledger.dispute(
            "d1", "attester", "challenger", "m1",
            accused_proof=lying, challenger_proof=honest,
            verify_fn=verify_dispute_payload)
        self.assertEqual(out["verdict"], "slash_accused")
        self.assertEqual(ledger.stake_of("attester"), 0.0)

    def test_false_accusation_slashes_challenger(self):
        honest_a = build_dispute_payload(X, R, W, B, Y)
        honest_c = build_dispute_payload(X, R, W, B, Y)
        ledger = SlashLedger()
        ledger.deposit("attester", 100.0)
        ledger.deposit("challenger", 50.0)
        out = ledger.dispute(
            "d2", "attester", "challenger", "m1",
            accused_proof=honest_a, challenger_proof=honest_c,
            verify_fn=verify_dispute_payload)
        self.assertEqual(out["verdict"], "slash_challenger")
        self.assertEqual(ledger.stake_of("challenger"), 0.0)

    def test_malformed_payload_fails_closed(self):
        self.assertFalse(verify_dispute_payload({}))


if __name__ == "__main__":
    unittest.main()
