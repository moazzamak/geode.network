"""Unit tests for the M192 secret-sharing primitives."""
import itertools
import unittest

import numpy as np

from geode.privacy.secret_sharing import (
    PRIME,
    recombine_additive,
    replicated_gram_shares,
    shamir_reconstruct,
    shamir_split,
    signed_from_field,
    split_additive,
    to_field,
)


class TestReplicatedGram(unittest.TestCase):
    def test_shares_sum_to_gram(self):
        rng = np.random.default_rng(3)
        for _ in range(5):
            block = rng.normal(size=(64, 24))
            shares = replicated_gram_shares(block, 3, rng)
            total = np.zeros_like(shares[0])
            for share in shares:
                total += share
            ref = block.T @ block
            rel = float(np.abs(total - ref).max()
                        / max(np.abs(ref).max(), 1e-300))
            self.assertLess(rel, 1e-10)

    def test_rejects_non_three_parties(self):
        rng = np.random.default_rng(3)
        block = rng.normal(size=(8, 8))
        with self.assertRaises(ValueError):
            replicated_gram_shares(block, 5, rng)


class TestAdditiveSplit(unittest.TestCase):
    def test_recombine(self):
        rng = np.random.default_rng(7)
        row = np.linspace(-40.0, 40.0, 1428)
        shares = split_additive(row, 3, rng)
        total = recombine_additive(shares)
        self.assertLess(float(np.abs(total - row).max()), 1e-9)

    def test_single_share_noise(self):
        rng = np.random.default_rng(7)
        row = np.linspace(-40.0, 40.0, 512)
        shares = split_additive(row, 3, rng)
        r = float(np.corrcoef(shares[1], row)[0, 1])
        self.assertLess(abs(r), 0.05)


class TestShamir(unittest.TestCase):
    def test_all_subsets_reconstruct(self):
        rng = np.random.default_rng(11)
        secret = 123456789
        k, n = 3, 5
        shares = shamir_split(to_field(secret), k, n, rng)
        for subset in itertools.combinations(range(1, n + 1), k):
            pairs = [(x, shares[x - 1]) for x in subset]
            self.assertEqual(shamir_reconstruct(pairs), secret % PRIME)

    def test_corrupted_share_detected(self):
        rng = np.random.default_rng(11)
        k, n = 3, 5
        shares = shamir_split(to_field(42), k, n, rng)
        shares[3] = (shares[3] + int(rng.integers(1, PRIME))) % PRIME
        values = set()
        for subset in itertools.combinations(range(1, n + 1), k):
            pairs = [(x, shares[x - 1]) for x in subset]
            values.add(shamir_reconstruct(pairs))
        self.assertGreater(len(values), 1)

    def test_signed_roundtrip(self):
        for v in (-12345, 0, 12345):
            self.assertEqual(signed_from_field(to_field(v)), v)


if __name__ == "__main__":
    unittest.main()
