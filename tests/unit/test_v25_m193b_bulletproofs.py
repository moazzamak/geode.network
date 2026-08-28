"""Unit tests for the M193b log-sized argument."""
import random
import unittest

from geode.privacy.zk_bulletproofs import (
    Q_ORDER,
    commit_vec,
    proof_size_bytes,
    prove,
    verify,
)


def _case(n: int = 64, seed: int = 9):
    rnd = random.Random(seed)
    x = [rnd.randrange(Q_ORDER) for _ in range(n)]
    w = [rnd.randrange(Q_ORDER) for _ in range(n)]
    claim = sum((xi * wi) % Q_ORDER for xi, wi in zip(x, w)) % Q_ORDER
    r = rnd.randrange(Q_ORDER)
    return x, r, w, claim


class TestBulletproofsStyle(unittest.TestCase):
    def test_honest_proof_verifies(self):
        for n in (1, 3, 64):
            x, r, w, claim = _case(n)
            proof = prove(x, r, w, claim)
            self.assertTrue(verify(proof, w))

    def test_tampered_claim_rejected(self):
        x, r, w, claim = _case(64)
        proof = prove(x, r, w, claim)
        proof["claim"] = (claim + 1) % Q_ORDER
        self.assertFalse(verify(proof, w))

    def test_tampered_weights_rejected(self):
        x, r, w, claim = _case(64)
        proof = prove(x, r, w, claim)
        w_bad = list(w)
        w_bad[0] = (w_bad[0] + 1) % Q_ORDER
        self.assertFalse(verify(proof, w_bad))

    def test_deterministic(self):
        x, r, w, claim = _case(64)
        self.assertEqual(prove(x, r, w, claim), prove(x, r, w, claim))

    def test_proof_size_logarithmic(self):
        x, r, w, claim = _case(1024)
        size = proof_size_bytes(prove(x, r, w, claim))
        self.assertLess(size, 2048)  # ~10 rounds of 32-byte elements


if __name__ == "__main__":
    unittest.main()
