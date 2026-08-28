"""Unit tests for the M192b field Gram protocol pieces."""
import random
import unittest

import numpy as np

from experiments.tier4.eval_v25_m192b_all_random_gram import (
    FIELD,
    _lagrange_coeffs,
    _mod_outer,
    _party_local,
    _shamir_vector_share,
)
from geode.privacy.secret_sharing import shamir_reconstruct


class TestModOuter(unittest.TestCase):
    def test_matches_python_reference(self):
        rnd = random.Random(5)
        a = np.asarray([rnd.randrange(FIELD) for _ in range(17)],
                       dtype=np.int64)
        b = np.asarray([rnd.randrange(FIELD) for _ in range(17)],
                       dtype=np.int64)
        got = _mod_outer(a, b)
        ref = np.asarray(
            [[(int(x) * int(y)) % FIELD for y in b] for x in a],
            dtype=object)
        self.assertTrue(np.all(np.asarray(got, dtype=object) == ref))


class TestFieldGram(unittest.TestCase):
    def test_party_locals_sum_to_gram(self):
        rng = np.random.default_rng(6)
        cols, rows = 16, 16
        k, n = 2, 3
        block = rng.integers(-1000, 1000, size=(rows, cols)).astype(np.int64)
        gram_ref = np.zeros((cols, cols), dtype=object)
        for r in range(rows):
            row = block[r] % FIELD
            gram_ref = (gram_ref + _mod_outer(row, row)) % FIELD
        lam = _lagrange_coeffs(list(range(1, n + 1)), 0)
        party = [np.zeros((cols, cols), dtype=object) for _ in range(n)]
        for r in range(rows):
            shares = _shamir_vector_share(block[r], k, n, rng)
            for p in range(n):
                q = (p + 1) % n
                party[p] = (party[p] + _party_local(
                    shares[p], shares[q], lam[p], lam[q])) % FIELD
        total = np.zeros((cols, cols), dtype=object)
        for p in range(n):
            total = (total + party[p]) % FIELD
        self.assertTrue(np.all(np.asarray(total, dtype=object)
                               == np.asarray(gram_ref, dtype=object)))

    def test_any_pair_reconstructs(self):
        rng = np.random.default_rng(6)
        k, n = 2, 3
        values = rng.integers(-10 ** 6, 10 ** 6, size=16).astype(np.int64)
        shares = _shamir_vector_share(values, k, n, rng)
        for pair in ((1, 2), (2, 3), (1, 3)):
            rec = np.zeros(16, dtype=np.int64)
            for i in range(16):
                vals = [(x, int(shares[x - 1][i])) for x in pair]
                rec[i] = shamir_reconstruct(vals, modulus=FIELD)
            self.assertTrue(np.all(rec == values % FIELD))


if __name__ == "__main__":
    unittest.main()
