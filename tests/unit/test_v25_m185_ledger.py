"""Unit tests for M185: the append-only hash-chained ledger."""
from __future__ import annotations

import unittest

from geode.core.ledger import AppendOnlyLedger, record_hash


class TestLedger(unittest.TestCase):

    def test_chain_integrity(self):
        ledger = AppendOnlyLedger()
        ledger.append({"key": "a", "value": 1})
        ledger.append({"key": "b", "value": 2})
        self.assertTrue(ledger.verify()["ok"])
        self.assertEqual(ledger.verify()["record_count"], 2)

    def test_tamper_detected(self):
        ledger = AppendOnlyLedger()
        ledger.append({"key": "a", "value": 1})
        ledger.append({"key": "b", "value": 2})
        ledger._records[0]["content"]["value"] = 999
        result = ledger.verify()
        self.assertFalse(result["ok"])
        self.assertIn(0, result["tampered_records"])

    def test_append_only_duplicate_key_rejected(self):
        ledger = AppendOnlyLedger()
        ledger.append({"key": "a"})
        with self.assertRaises(ValueError):
            ledger.append({"key": "a"})

    def test_index_and_hash_reserved(self):
        ledger = AppendOnlyLedger()
        with self.assertRaises(ValueError):
            ledger.append({"index": 1})
        with self.assertRaises(ValueError):
            ledger.append({"hash": "x"})

    def test_deterministic(self):
        a = AppendOnlyLedger()
        b = AppendOnlyLedger()
        for ledger in (a, b):
            ledger.append({"key": "x", "value": [1, 2, 3]})
        self.assertEqual(a.tip(), b.tip())
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_no_wall_clock_in_hash(self):
        self.assertEqual(
            record_hash({"value": 1}, "prev"),
            record_hash({"value": 1, "runtime_seconds": 999.0}, "prev"))


if __name__ == "__main__":
    unittest.main()
