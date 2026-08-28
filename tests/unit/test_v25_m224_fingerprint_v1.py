"""Unit tests for M224: the shipped v1 fingerprint weights
(strict load, determinism, pinned descriptor vector, unit norm,
differs from the random-init constructor).
"""
from __future__ import annotations

import unittest

import torch

from geode.core.descriptor import normalise
from geode.core.fingerprint import FingerprintEncoder

DESC_A = normalise({"input.modality": "image",
                    "output.kind": "class",
                    "latent.label_cardinality": 345})

# The v1 fingerprint of DESC_A, pinned at ship time (2026-08-20, the
# M224 sealed run - see logs/results/v25/m224_fingerprint_v1_train).
PINNED_A_V1 = [
    0.4869911373, 0.2311680615, -0.1959175169, -0.1343872696,
    -0.1884384900, 0.3611462712, -0.1713467985, -0.0221466254,
    -0.4305131435, 0.4674756229, -0.0277779438, -0.0509828478,
    -0.0366070010, -0.1771251559, 0.0107689044, -0.1307523847,
]


class TestFingerprintV1Ship(unittest.TestCase):

    def test_v1_loads_and_is_deterministic(self):
        f1 = FingerprintEncoder.pretrained_v1().fingerprint(DESC_A)
        f2 = FingerprintEncoder.pretrained_v1().fingerprint(DESC_A)
        self.assertTrue(torch.equal(f1, f2))

    def test_v1_pinned_descriptor_vector(self):
        f = FingerprintEncoder.pretrained_v1().fingerprint(DESC_A)
        self.assertEqual(f.shape, (16,))
        self.assertTrue(torch.allclose(
            f, torch.tensor(PINNED_A_V1), atol=1e-6))

    def test_v1_unit_norm(self):
        f = FingerprintEncoder.pretrained_v1().fingerprint(DESC_A)
        self.assertAlmostEqual(float(f.norm()), 1.0, places=6)

    def test_v1_differs_from_random_init(self):
        f_v1 = FingerprintEncoder.pretrained_v1().fingerprint(DESC_A)
        f_rand = FingerprintEncoder(seed=11).fingerprint(DESC_A)
        self.assertFalse(torch.allclose(f_v1, f_rand, atol=1e-4))

    def test_v1_mlp_weights_shipped(self):
        enc = FingerprintEncoder.pretrained_v1()
        self.assertTrue(enc.mlp_on)
        self.assertIsNotNone(enc.mlp)
        self.assertEqual(enc.f_dim, 16)


if __name__ == "__main__":
    unittest.main()
