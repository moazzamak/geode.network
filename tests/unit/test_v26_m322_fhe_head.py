"""Unit tests for the M322e-B FHE-head quantization arithmetic.

Pins the registered scheme: uniform fixed-point scales, exact int64
MACs, the dequantization identity, the gating headroom guard, and
the simulated path's self-consistency.
"""
from __future__ import annotations

import unittest

import numpy as np

from geode.privacy.fhe_head import (
    SCALE_BITS_8,
    SCALE_BITS_16,
    class_exponents,
    dequantize,
    dequantize_perclass,
    fhe_simulated_scores,
    fhe_simulated_scores_perclass,
    quantize_head,
    quantize_head_perclass,
    quantize_input,
    quantized_scores,
)


class TestQuantizeHead(unittest.TestCase):

    def test_scales(self):
        W = np.array([[1.0, -0.5], [0.25, 2.0]])
        b = np.array([0.5, -1.0])
        head = quantize_head(W, b, SCALE_BITS_16)
        self.assertEqual(head["scale"], 1 << 32)
        self.assertTrue(np.array_equal(head["W_q"],
                                       np.rint(W * (1 << 16)).astype(np.int64)))
        self.assertTrue(np.array_equal(head["b_q"],
                                       np.rint(b * (1 << 32)).astype(np.int64)))

    def test_eight_bit_scales(self):
        head = quantize_head(np.ones((3, 2)), np.ones(2), SCALE_BITS_8)
        self.assertEqual(head["scale"], 1 << 16)

    def test_headroom_guard_gates(self):
        W = np.array([[1e15]])
        b = np.zeros(1)
        with self.assertRaises(ValueError):
            quantize_head(W, b, SCALE_BITS_16)

    def test_input_headroom_guard_gates(self):
        with self.assertRaises(ValueError):
            quantize_input(np.array([1e15]), SCALE_BITS_16)


class TestIntegerMACs(unittest.TestCase):

    def test_exact_integer_identity(self):
        W = np.array([[3.0, 1.0], [2.0, -4.0]])
        b = np.array([0.5, -0.25])
        z = np.array([0.75, -1.5])
        head = quantize_head(W, b, SCALE_BITS_16)
        q_z = quantize_input(z, SCALE_BITS_16)
        scores_q = quantized_scores(q_z, head["W_q"], head["b_q"])
        # integer exactness: the quantized quantities reproduce the
        # integer MACs by construction
        expected = (head["W_q"].T @ q_z + head["b_q"])
        self.assertTrue(np.array_equal(scores_q, expected))
        self.assertEqual(scores_q.dtype, np.int64)


class TestSimulatedPath(unittest.TestCase):

    def test_matches_direct_quantization(self):
        rng = np.random.default_rng(0)
        z = rng.uniform(-3.0, 3.0, size=(64,))
        W = rng.uniform(-0.05, 0.05, size=(64, 12))
        b = rng.uniform(-0.5, 0.5, size=(12,))
        via_pipeline = fhe_simulated_scores(z, W, b, SCALE_BITS_16)
        head = quantize_head(W, b, SCALE_BITS_16)
        q_z = quantize_input(z, SCALE_BITS_16)
        direct = dequantize(quantized_scores(q_z, head["W_q"],
                                             head["b_q"]), SCALE_BITS_16)
        self.assertTrue(np.allclose(via_pipeline, direct, rtol=0.0,
                                    atol=0.0))

    def test_quantization_error_bounded(self):
        rng = np.random.default_rng(1)
        z = rng.uniform(-3.0, 3.0, size=(128,))
        W = rng.uniform(-0.05, 0.05, size=(128, 16))
        b = rng.uniform(-0.5, 0.5, size=(16,))
        s = W.T @ z + b
        s_q = fhe_simulated_scores(z, W, b, SCALE_BITS_16)
        rel = float(np.max(np.abs(s_q - s)) / max(float(np.max(np.abs(s))),
                                                  1e-12))
        self.assertLessEqual(rel, 2 ** -9)


class TestPerClassEncoding(unittest.TestCase):

    def test_exponents_normalize_columns(self):
        rng = np.random.default_rng(2)
        W = rng.lognormal(0.0, 3.0, size=(64, 16))  # wide dynamic range
        k = class_exponents(W, SCALE_BITS_16)
        scaled_max = np.max(np.abs(W * (2.0 ** k)[None, :]), axis=0)
        # each column normalizes into a band around 2^15
        self.assertTrue(np.all(scaled_max > 2 ** 13))
        self.assertTrue(np.all(scaled_max <= 2 ** 16))

    def test_perclass_matches_direct_roundtrip(self):
        rng = np.random.default_rng(3)
        W = rng.lognormal(0.0, 2.0, size=(64, 12))
        b = rng.uniform(-0.5, 0.5, size=(12,))
        z = rng.uniform(-3.0, 3.0, size=(64,))
        s = W.T @ z + b
        head = quantize_head_perclass(W, b, SCALE_BITS_16)
        q_z = quantize_input(z, SCALE_BITS_16)
        q = quantized_scores(q_z, head["W_q"], head["b_q"])
        s_q = dequantize_perclass(q, head["exponents"], SCALE_BITS_16)
        self.assertTrue(np.allclose(s_q, s, rtol=2 ** -9))

    def test_perclass_handles_outlier_columns(self):
        rng = np.random.default_rng(4)
        W = rng.uniform(-0.05, 0.05, size=(128, 16))
        W[:, 3] *= 1000.0  # one outlier column
        b = rng.uniform(-0.5, 0.5, size=(16,))
        z = rng.uniform(-3.0, 3.0, size=(128,))
        s = W.T @ z + b
        s_q = fhe_simulated_scores_perclass(z, W, b, SCALE_BITS_16)
        rel = float(np.max(np.abs(s_q - s)) / max(float(np.max(np.abs(s))),
                                                  1e-12))
        self.assertLessEqual(rel, 2 ** -9)


if __name__ == "__main__":
    unittest.main()
