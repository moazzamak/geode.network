"""Serving-tier and FHE-gateway wiring tests (§8.34)."""
from __future__ import annotations

import unittest

import numpy as np

from geode.core.serving_tiers import (
    ServingTier,
    TierAuditLedger,
    TierSession,
    TierViolation,
)
from geode.privacy.fhe_gateway import FheServingSession, FheTranscript


class TestTierAuditLedger(unittest.TestCase):
    def test_tier_mix_is_a_public_statistic(self):
        ledger = TierAuditLedger()
        ledger.record(TierSession("s1", ServingTier.ON_DEVICE))
        ledger.record(TierSession("s2", ServingTier.FHE_PRIVATE,
                                  ciphertext_only=True))
        ledger.record(TierSession("s3", ServingTier.PLAINTEXT,
                                  disclosed_as_plaintext=True))
        mix = ledger.tier_mix()
        self.assertEqual(mix["on_device"], 1)
        self.assertEqual(mix["fhe_private"], 1)
        self.assertEqual(mix["plaintext"], 1)

    def test_plaintext_must_be_disclosed(self):
        with self.assertRaises(TierViolation):
            TierAuditLedger().record(
                TierSession("s1", ServingTier.PLAINTEXT))

    def test_fhe_must_be_ciphertext_only(self):
        with self.assertRaises(TierViolation):
            TierAuditLedger().record(
                TierSession("s2", ServingTier.FHE_PRIVATE,
                            ciphertext_only=False))

    def test_violation_scan(self):
        ledger = TierAuditLedger()
        ledger.sessions.append(TierSession(
            "s1", ServingTier.PLAINTEXT))  # bypasses record() on purpose
        self.assertEqual(
            ledger.plaintext_sessions_sold_as_private(), ["s1"])


class TestFheGateway(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.d, cls.C = 64, 8
        cls.W = rng.uniform(-0.05, 0.05, size=(cls.d, cls.C))
        cls.b = rng.uniform(-0.5, 0.5, size=(cls.C,))
        cls.session = FheServingSession(cls.W, cls.b)

    def test_transcript_type_has_no_plaintext_field(self):
        t = FheTranscript(session_id="s1")
        for banned in ("z", "q_z", "scores", "plaintext", "answer"):
            self.assertFalse(hasattr(t, banned))

    def test_full_ciphertext_round_trip(self):
        z = np.random.default_rng(3).uniform(
            -3.0, 3.0, size=(self.d,))
        cts, transcript = self.session.device_encrypt(z, "s-1")
        # the host side sees ciphertexts only
        score_ct = self.session.host_evaluate(cts)
        self.assertTrue(transcript.ciphertext_only)
        self.assertTrue(len(transcript.input_ciphertexts) >= 1)
        answer, fhe_scores = self.session.device_decrypt(score_ct)
        plain = self.session.plaintext_scores(z)
        self.assertEqual(answer, int(np.argmax(plain)))
        gate = self.session.agreement_gate(fhe_scores, plain)
        self.assertTrue(gate["ok"], gate)

    def test_tier_session_is_fhe_private(self):
        z = np.random.default_rng(4).uniform(
            -3.0, 3.0, size=(self.d,))
        _cts, transcript = self.session.device_encrypt(z, "s-2")
        tier = self.session.tier_session("s-2", transcript)
        self.assertEqual(tier.tier, ServingTier.FHE_PRIVATE)
        self.assertTrue(tier.ciphertext_only)
        TierAuditLedger().record(tier)   # must not raise

    def test_gateway_never_holds_device_plaintext(self):
        # the session object exposes no attribute holding z or the
        # plaintext scores (W-G5: the gateway process is the host)
        session = self.session
        for banned in ("z", "q_z", "plaintext_scores_cache"):
            self.assertFalse(hasattr(session, banned))


if __name__ == "__main__":
    unittest.main()
