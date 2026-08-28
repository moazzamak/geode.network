"""M282 — commit-reveal arm submissions: the honest path commits
before probes and admits after a hashing reveal; every after-the-
fact re-description path fails by construction."""
import unittest

from geode.core.submission_commit import (
    AdmissionRule,
    SubmissionLedger,
    submission_commit,
)


class TestSubmissionCommit(unittest.TestCase):
    def setUp(self):
        self.ledger = SubmissionLedger()
        self.claim = {"per_task": {"d0": 0.85, "d3": 0.90}}
        self.salt = "s4lt-20260823"
        self.weight_digest = "ab12cd34"

    def test_honest_path_commits_reveals_admits(self):
        cid = self.ledger.commit("alice", self.salt, "arm_x",
                                 "sentiment", self.claim,
                                 self.weight_digest)
        self.assertTrue(
            self.ledger.reveal(cid, "alice", self.salt, "arm_x",
                               "sentiment", self.claim,
                               self.weight_digest))
        receipt = self.ledger.admit(cid, {"d0": 0.86, "d3": 0.91})
        self.assertTrue(receipt["admitted"])
        self.assertEqual(receipt["stage"], "admit")
        self.assertEqual(self.ledger.admitted_ids(), [cid])

    def test_measured_below_committed_rejected(self):
        cid = self.ledger.commit("bob", self.salt, "arm_y",
                                 "logic", self.claim,
                                 self.weight_digest)
        self.ledger.reveal(cid, "bob", self.salt, "arm_y", "logic",
                           self.claim, self.weight_digest)
        receipt = self.ledger.admit(cid, {"d0": 0.80, "d3": 0.90})
        self.assertFalse(receipt["admitted"])
        self.assertEqual(receipt["stage"], "reject")
        self.assertEqual(receipt["committed_per_task"],
                         {"d0": 0.85, "d3": 0.90})

    def test_reveal_without_commit_raises(self):
        cid = submission_commit("carol", self.salt, "arm_z",
                                "arithmetic", self.claim,
                                self.weight_digest)
        with self.assertRaises(ValueError):
            self.ledger.reveal(cid, "carol", self.salt, "arm_z",
                               "arithmetic", self.claim,
                               self.weight_digest)

    def test_admit_without_reveal_raises(self):
        cid = self.ledger.commit("dave", self.salt, "arm_w",
                                 "logic", self.claim,
                                 self.weight_digest)
        with self.assertRaises(ValueError):
            self.ledger.admit(cid, {"d0": 0.99, "d3": 0.99})

    def test_redescribed_claim_after_results_fails(self):
        # the after-the-fact cheat: commit a modest claim, see the
        # probes, then re-describe a higher claim at reveal time
        cid = self.ledger.commit("eve", self.salt, "arm_v",
                                 "sentiment", self.claim,
                                 self.weight_digest)
        higher = {"per_task": {"d0": 0.99, "d3": 0.99}}
        with self.assertRaises(ValueError):
            self.ledger.reveal(cid, "eve", self.salt, "arm_v",
                               "sentiment", higher, self.weight_digest)
        # and no admission is possible afterwards
        with self.assertRaises(ValueError):
            self.ledger.admit(cid, {"d0": 0.99, "d3": 0.99})

    def test_tampered_weight_digest_fails_reveal(self):
        cid = self.ledger.commit("frank", self.salt, "arm_u",
                                 "arithmetic", self.claim,
                                 self.weight_digest)
        with self.assertRaises(ValueError):
            self.ledger.reveal(cid, "frank", self.salt, "arm_u",
                               "arithmetic", self.claim,
                               "00000000")

    def test_double_admit_raises(self):
        cid = self.ledger.commit("grace", self.salt, "arm_t",
                                 "logic", self.claim,
                                 self.weight_digest)
        self.ledger.reveal(cid, "grace", self.salt, "arm_t", "logic",
                           self.claim, self.weight_digest)
        self.ledger.admit(cid, {"d0": 0.85, "d3": 0.90})
        with self.assertRaises(ValueError):
            self.ledger.admit(cid, {"d0": 0.85, "d3": 0.90})

    def test_receipts_record_every_stage(self):
        cid = self.ledger.commit("heidi", self.salt, "arm_s",
                                 "sentiment", self.claim,
                                 self.weight_digest)
        self.ledger.reveal(cid, "heidi", self.salt, "arm_s",
                           "sentiment", self.claim, self.weight_digest)
        self.ledger.admit(cid, {"d0": 0.85, "d3": 0.90})
        stages = [r["stage"] for r in self.ledger.receipts]
        self.assertEqual(stages, ["commit", "reveal", "admit"])

    def test_rule_tolerance(self):
        rule = AdmissionRule(tolerance=0.01)
        self.assertTrue(rule.passes({"a": 0.80}, {"a": 0.795}))
        self.assertFalse(rule.passes({"a": 0.80}, {"a": 0.78}))


if __name__ == "__main__":
    unittest.main()
