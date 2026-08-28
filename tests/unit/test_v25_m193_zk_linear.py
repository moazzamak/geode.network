"""Unit tests for the M193 zk linear-relation argument."""
import random
import unittest

from geode.privacy.zk_linear import (
    Q,
    commit,
    proof_size_bytes,
    prove,
    verify,
)


def _case(n: int = 16, outputs: int = 5, seed: int = 1):
    rnd = random.Random(seed)
    x = [rnd.randrange(Q) for _ in range(n)]
    w = [[rnd.randrange(Q) for _ in range(n)] for _ in range(outputs)]
    b = [rnd.randrange(Q) for _ in range(outputs)]
    y = [(sum(wi * xi for wi, xi in zip(row, x)) + bi) % Q
         for row, bi in zip(w, b)]
    r = rnd.randrange(Q)
    return x, r, w, b, y


class TestZkLinear(unittest.TestCase):
    def test_honest_proof_verifies(self):
        x, r, w, b, y = _case()
        proof = prove(x, r, w, b, y)
        self.assertTrue(verify(proof, commit(x, r), w, b, y))

    def test_tampered_score_rejected(self):
        x, r, w, b, y = _case()
        proof = prove(x, r, w, b, y)
        y_bad = list(y)
        y_bad[0] = (y_bad[0] + 1) % Q
        self.assertFalse(verify(proof, commit(x, r), w, b, y_bad))

    def test_tampered_weights_rejected(self):
        x, r, w, b, y = _case()
        proof = prove(x, r, w, b, y)
        w_bad = [list(row) for row in w]
        w_bad[0][0] = (w_bad[0][0] + 1) % Q
        self.assertFalse(verify(proof, commit(x, r), w_bad, b, y))

    def test_wrong_input_commitment_rejected(self):
        x, r, w, b, y = _case()
        proof = prove(x, r, w, b, y)
        x_bad = list(x)
        x_bad[0] = (x_bad[0] + 1) % Q
        self.assertFalse(verify(proof, commit(x_bad, r), w, b, y))

    def test_deterministic(self):
        x, r, w, b, y = _case()
        self.assertEqual(prove(x, r, w, b, y), prove(x, r, w, b, y))

    def test_proof_size_reported(self):
        x, r, w, b, y = _case(16)
        size = proof_size_bytes(prove(x, r, w, b, y))
        self.assertGreater(size, 0)


if __name__ == "__main__":
    unittest.main()
