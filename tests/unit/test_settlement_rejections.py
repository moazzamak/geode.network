"""Unit tests for the settlement rule-checker's rejection branches
(whitepaper batch schema: artifactId-based entries)."""
import unittest

from geode.settlement.settlement import (
    MAX_BATCH,
    address_of,
    artifact_id_of,
    verify_batch_rules,
)


def _batch(payers, amounts):
    return [{"payers": payers,
             "entries": [{"artifactId": artifact_id_of(f"arm{i}"),
                          "who": address_of(f"arm{i}"),
                          "amount": a,
                          "proofHash": "0x" + "ab" * 32}
                         for i, a in enumerate(amounts)]}]


class TestRuleRejections(unittest.TestCase):
    def test_empty_report(self):
        self.assertIn("no batches built", verify_batch_rules({}))

    def test_shape_mismatch(self):
        report = {"batches": [{"payers": ["0x1", "0x2"],
                               "entries": [{"artifactId": "0x0",
                                            "who": "0x1",
                                            "amount": 5,
                                            "proofHash":
                                                "0x" + "ab" * 32}]}]}
        self.assertTrue(any("shape mismatch" in v
                            for v in verify_batch_rules(report)))

    def test_missing_artifact_id(self):
        report = {"batches": [{"payers": ["0x1"],
                               "entries": [{"who": "0x1",
                                            "amount": 5,
                                            "proofHash":
                                                "0x" + "ab" * 32}]}]}
        self.assertTrue(any("missing artifactId" in v
                            for v in verify_batch_rules(report)))

    def test_missing_proof_hash(self):
        report = {"batches": [{"payers": ["0x1"],
                               "entries": [{"artifactId": "0x0",
                                            "who": "0x1",
                                            "amount": 5}]}]}
        self.assertTrue(any("proofHash" in v
                            for v in verify_batch_rules(report)))

    def test_oversized_batch(self):
        report = {"batches": _batch([f"0x{i:02x}" for i in
                                     range(MAX_BATCH + 1)],
                                    [1] * (MAX_BATCH + 1))}
        self.assertTrue(any("> MAX_BATCH" in v
                            for v in verify_batch_rules(report)))

    def test_nonpositive_amount(self):
        report = {"batches": _batch(["0x1"], [0])}
        self.assertTrue(any("positive integer" in v
                            for v in verify_batch_rules(report)))

    def test_pool_exceeded_in_order(self):
        report = {"batches": _batch(["0x1", "0x2"], [5, 6]),
                  "pool_expected": 10}
        self.assertTrue(any("exceeds remaining pool" in v
                            for v in verify_batch_rules(report,
                                                        pool=10)))

    def test_total_credited_exceeds_pool_expected(self):
        report = {"batches": _batch(["0x1", "0x2"], [5, 6]),
                  "pool_expected": 10}
        self.assertTrue(any("exceeds pool_expected" in v
                            for v in verify_batch_rules(report)))


if __name__ == "__main__":
    unittest.main()
