"""Unit tests for M168: the fingerprint embedder build (G1 determinism,
dimension, unit norm, additive structure with the MLP off, OOV path).
"""
from __future__ import annotations

import unittest

import torch

from geode.core.descriptor import normalise
from geode.core.fingerprint import FingerprintEncoder

DESC_A = normalise({"input.modality": "image",
                    "output.kind": "class",
                    "latent.label_cardinality": 345})
DESC_B = normalise({"input.modality": "numeric-series",
                    "output.kind": "regression",
                    "latent.label_cardinality": 1000})
DESC_OOV = normalise({"input.modality": "hologram"})


class TestFingerprintBuild(unittest.TestCase):

    def test_g1_determinism_same_encoder(self):
        enc = FingerprintEncoder(seed=11)
        f1 = enc.fingerprint(DESC_A)
        f2 = enc.fingerprint(DESC_A)
        self.assertTrue(torch.equal(f1, f2))

    def test_g1_determinism_fresh_encoder_same_seed(self):
        f1 = FingerprintEncoder(seed=11).fingerprint(DESC_A)
        f2 = FingerprintEncoder(seed=11).fingerprint(DESC_A)
        self.assertTrue(torch.equal(f1, f2))

    def test_dimension_and_unit_norm(self):
        enc = FingerprintEncoder(f_dim=16, seed=11)
        f = enc.fingerprint(DESC_A)
        self.assertEqual(f.shape, (16,))
        self.assertAlmostEqual(float(f.norm()), 1.0, places=6)

    def test_distinct_descriptors_differ(self):
        enc = FingerprintEncoder(seed=11)
        self.assertFalse(torch.equal(enc.fingerprint(DESC_A),
                                     enc.fingerprint(DESC_B)))

    def test_oov_does_not_crash(self):
        enc = FingerprintEncoder(seed=11)
        f = enc.fingerprint(DESC_OOV)
        self.assertTrue(torch.isfinite(f).all())

    def test_additive_structure_with_mlp_off(self):
        enc = FingerprintEncoder(mlp_on=False, seed=11)
        with torch.no_grad():
            enc.eval()
            pre_norm = enc.forward(DESC_A)
        idx = enc._indices(DESC_A)
        expected = torch.nn.functional.normalize(
            enc.token_emb(idx).sum(dim=0), dim=-1)
        self.assertTrue(torch.allclose(pre_norm, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
