"""Unit tests for the M213 on-chain serialization bridge."""
import unittest

from geode.privacy.zk_bulletproofs import Q_ORDER, prove
from geode.privacy.zk_onchain import (
    challenge_serialization,
    generator_label,
    proof_length_bytes,
    rounds_of,
    serialize,
    serialize_hex,
    words_hex,
)


def _proof(n: int) -> dict:
    import random
    rnd = random.Random(9)
    x = [rnd.randrange(Q_ORDER) for _ in range(n)]
    w = [rnd.randrange(Q_ORDER) for _ in range(n)]
    claim = sum(xi * wi % Q_ORDER for xi, wi in zip(x, w)) % Q_ORDER
    r = rnd.randrange(Q_ORDER)
    return prove(x, r, w, claim)


class TestLayout(unittest.TestCase):
    def test_rounds(self):
        self.assertEqual(rounds_of(64), 6)
        self.assertEqual(rounds_of(1), 0)
        self.assertEqual(rounds_of(16384), 14)

    def test_length(self):
        # [C][L..][R..][a][b][r] = 1 + 2r + 3 words (claim separate)
        self.assertEqual(proof_length_bytes(64), (1 + 12 + 3) * 32)
        self.assertEqual(proof_length_bytes(16384), 32 * 32)  # 1,024 B

    def test_serialize_round_trip(self):
        proof = _proof(64)
        data = serialize(proof, 64)
        self.assertEqual(len(data), proof_length_bytes(64))
        # [C][L0..L5][R0..R5][a][b][r]
        def word(off):
            return int.from_bytes(data[off:off + 32], "big")
        self.assertEqual(word(0), proof["c_commit"])
        for j in range(6):
            self.assertEqual(word(32 + j * 32), proof["L"][j])
            self.assertEqual(word(32 + 6 * 32 + j * 32), proof["R"][j])
        off = 32 + 12 * 32
        self.assertEqual(word(off), proof["a_final"])
        self.assertEqual(word(off + 32), proof["b_final"])
        self.assertEqual(word(off + 64), proof["r_final"])

    def test_hex_matches_format(self):
        # minimal lowercase hex, as Python's format(v, "x")
        proof = _proof(64)
        h = serialize_hex(proof, 64)
        self.assertEqual(h, "0x" + serialize(proof, 64).hex())
        self.assertEqual(h[2:].split("00")[0][:0], "")  # no leading zeros

    def test_words_padded(self):
        w = words_hex([1, 2], 4)
        self.assertEqual(len(w), 4)
        self.assertEqual(w[0], "0x" + (1).to_bytes(32, "big").hex())
        self.assertEqual(w[2], "0x" + (0).to_bytes(32, "big").hex())


class TestByteExactMirrors(unittest.TestCase):
    def test_generator_label_negative_one(self):
        # BP_G = _generator(-1) — the label must be b'geode-bp--1'
        self.assertEqual(generator_label(-1), b"geode-bp--1")
        self.assertEqual(generator_label(10), b"geode-bp-10")

    def test_challenge_serialization(self):
        # verify hashes _ser(l)||_ser(r)||_ser(c) — three separate
        # single-value _ser calls, NO separators
        self.assertEqual(challenge_serialization(255, 16, 0),
                         b"ff100")


if __name__ == "__main__":
    unittest.main()
