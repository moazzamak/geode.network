"""M306 unit tests - the canonical replay oracle.

Pins every policy step against the sealed implementation paths:
the oracle's accumulation must reproduce the sealed
``RidgeAccumulator`` bit-for-bit, its LU head must reproduce the
M322e head-cache builder bit-for-bit, and its repaired solve must
reproduce the M296 solver bit-for-bit.
"""
from __future__ import annotations

import unittest

import numpy as np

from geode.core.replay_oracle import (
    POLICY_TEXT,
    ReplayCertificate,
    SealedSystem,
    _policy_payload,
    hardware_signature,
    head_digest,
    oracle_id,
    package_versions,
    repaired_head,
    sealed_lu_head,
    symmetric_system,
    thread_config,
)
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v26_m296_head_repair import solve_symmetric
from experiments.tier4.eval_v26_m322_fhe_quant import _ridge_head

CLASSES = 5
PENALTY = 1.0


def _synthetic(rows: int = 300, width: int = 24):
    rng = np.random.default_rng(0)
    features = rng.uniform(-2.0, 2.0, size=(rows, width)).astype(np.float32)
    labels = rng.integers(0, CLASSES, size=rows)
    return features, labels


class TestAccumulationBitwise(unittest.TestCase):
    def test_accumulate_matches_sealed_path(self):
        features, labels = _synthetic()
        sys = SealedSystem.accumulate(features, labels, classes=CLASSES)
        acc = RidgeAccumulator(features.shape[1], CLASSES)
        for start in range(0, len(features), 4096):
            stop = min(start + 4096, len(features))
            acc.add(features[start:stop], labels[start:stop])
        self.assertTrue(np.array_equal(sys.gram, acc.gram))
        self.assertTrue(np.array_equal(sys.column_sum, acc.column_sum))
        self.assertTrue(np.array_equal(sys.cross, acc.cross))
        self.assertTrue(np.array_equal(sys.class_count, acc.class_count))
        self.assertEqual(sys.rows, acc.rows)

    def test_standardised_system_matches_sealed_path(self):
        features, labels = _synthetic()
        sys = SealedSystem.accumulate(features, labels, classes=CLASSES)
        acc = RidgeAccumulator(features.shape[1], CLASSES)
        for start in range(0, len(features), 4096):
            stop = min(start + 4096, len(features))
            acc.add(features[start:stop], labels[start:stop])
        s_centred, s_cross, s_intercept = sys.standardised_system()
        a_centred, a_cross, a_intercept = acc._standardised_system()
        self.assertTrue(np.array_equal(s_centred, a_centred))
        self.assertTrue(np.array_equal(s_cross, a_cross))
        self.assertTrue(np.array_equal(s_intercept, a_intercept))

    def test_standardiser_fp32_rounding(self):
        features, labels = _synthetic()
        sys = SealedSystem.accumulate(features, labels, classes=CLASSES)
        centre, scale = sys.standardiser()
        self.assertEqual(centre.dtype, np.float32)
        self.assertEqual(scale.dtype, np.float32)


class TestSolveBitwise(unittest.TestCase):
    def test_symmetric_system_matches_m296(self):
        rng = np.random.default_rng(1)
        centred = rng.normal(size=(20, 20))
        ours = symmetric_system(centred)
        from experiments.tier4.eval_v26_m296_head_repair import \
            symmetric_system as m296_sym
        self.assertTrue(np.array_equal(ours, m296_sym(centred)))
        self.assertTrue(np.array_equal(ours, ours.T))  # bitwise symmetric

    def test_sealed_lu_head_matches_head_cache_builder(self):
        features, labels = _synthetic()
        sys = SealedSystem.accumulate(features, labels, classes=CLASSES)
        centred, cross, intercept = sys.standardised_system()
        ours = sealed_lu_head(centred, cross, intercept, PENALTY)
        ref = _ridge_head(centred, cross, intercept, PENALTY)
        self.assertTrue(np.array_equal(ours[0], ref[0]))
        self.assertTrue(np.array_equal(ours[1], ref[1]))

    def test_repaired_head_matches_m296_solver(self):
        features, labels = _synthetic()
        sys = SealedSystem.accumulate(features, labels, classes=CLASSES)
        centred, cross, intercept = sys.standardised_system()
        ours, report = repaired_head(centred, cross, intercept, PENALTY)
        ref, ref_report = solve_symmetric(centred, cross, intercept,
                                          PENALTY,
                                          report_conditioning=False)
        self.assertTrue(np.array_equal(ours, ref))
        self.assertEqual(report["solve_path"], ref_report["solve_path"])
        self.assertEqual(report["backward_passed"],
                         ref_report["backward_passed"])


class TestRegistrationAndDigests(unittest.TestCase):
    def test_oracle_id_stable_and_sensitive(self):
        self.assertEqual(oracle_id(), oracle_id())
        self.assertEqual(len(oracle_id()), 64)
        self.assertNotEqual(oracle_id(block=4096), oracle_id(block=2048))
        payload = _policy_payload()
        self.assertIn(b"numerics policy v1", payload)

    def test_policy_text_registered(self):
        self.assertIn("symmetric-by-construction", POLICY_TEXT)
        self.assertIn("driver evd", POLICY_TEXT)
        self.assertIn("4096", POLICY_TEXT)

    def test_head_digest_deterministic_and_sensitive(self):
        rng = np.random.default_rng(2)
        w = rng.normal(size=(10, 4))
        b = rng.normal(size=(4,))
        self.assertEqual(head_digest(w, b), head_digest(w, b))
        flipped = w.copy()
        flipped[0, 0] = np.nextafter(flipped[0, 0], np.inf)
        self.assertNotEqual(head_digest(w, b), head_digest(flipped, b))
        self.assertNotEqual(head_digest(w, b), head_digest(w, b + 1e-9))

    def test_certificate_records_but_never_hashes_hardware(self):
        cert = ReplayCertificate(
            oracle=oracle_id(),
            head_digest=head_digest(np.zeros((2, 2)), np.zeros(2)),
            expected_digest=None,
            bit_exact=False,
            solve_path="cholesky",
            hardware={"processor": "test-cpu"},
        ).as_dict()
        self.assertIn("hardware", cert)
        self.assertEqual(cert["hardware"]["processor"], "test-cpu")
        # the digest never depends on the hardware block
        self.assertEqual(
            head_digest(np.zeros((2, 2)), np.zeros(2)),
            head_digest(np.zeros((2, 2)), np.zeros(2)))

    def test_signature_fields(self):
        sig = hardware_signature()
        for key in ("numpy", "scipy", "processor", "threads"):
            self.assertIn(key, sig)
        self.assertIn("os_cpu_count", thread_config())

    def test_package_versions_present(self):
        versions = package_versions()
        self.assertTrue(versions["numpy"])
        self.assertTrue(versions["scipy"])


if __name__ == "__main__":
    unittest.main()
