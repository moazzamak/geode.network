"""M322e-B — quantization and integer-MAC simulation for the FHE head.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` §8.22
M322e-B (27 Aug 2026, before any FHE code). BFV arithmetic is exact
over integers: the decoded score vector equals the registered
integer multiply-accumulate exactly. This module therefore contains
the ARITHMETIC the FHE backend will reproduce bit-for-bit:

- ``quantize_head`` / ``quantize_input`` — fixed-point, uniform
  per-vector scales: 16-bit cell scales W and z by 2**16, b by
  2**32; the 8-bit cell scales by 2**8 / 2**16.
- ``quantized_scores`` — the integer MACs (int64 accumulation).
- ``dequantize`` — divide by 2**(2*bits).

The BFV backend (TenSEAL) is a separate stage that must agree with
this simulation EXACTLY (M322-QG3); no approximation tolerance is
registered between the two.
"""
from __future__ import annotations

import numpy as np

# Registered scale bits per cell.
SCALE_BITS_16 = 16
SCALE_BITS_8 = 8

# Registered int64 headroom guard: the largest intermediate of the
# 16-bit cell is ~768 * 2**32 * max|Wz|; it must stay far below
# 2**62 for exact signed accumulation.
INT64_GUARD = 2 ** 62


def _scale(bits: int) -> int:
    return 1 << bits


def quantize_head(W: np.ndarray, b: np.ndarray,
                  bits: int = SCALE_BITS_16) -> dict:
    """Quantize the frozen head: ``q_W = round(W*2^bits)``,
    ``q_b = round(b*2^(2*bits))``. Premise checks are GATING:
    out-of-range values raise instead of silently wrapping.
    """
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    s_w = _scale(bits)
    s_b = _scale(2 * bits)
    q_W_float = np.rint(W * s_w)
    q_b_float = np.rint(b * s_b)
    if np.abs(q_W_float).max() >= INT64_GUARD:
        raise ValueError("quantized W exceeds the registered int64 "
                         "headroom guard (M322e-B)")
    if np.abs(q_b_float).max() >= INT64_GUARD:
        raise ValueError("quantized b exceeds the registered int64 "
                         "headroom guard (M322e-B)")
    q_W = q_W_float.astype(np.int64)
    q_b = q_b_float.astype(np.int64)
    return {"W_q": q_W, "b_q": q_b, "bits": bits,
            "scale": s_b}


def quantize_input(z: np.ndarray, bits: int = SCALE_BITS_16) -> np.ndarray:
    """Quantize the device-side input: ``q_z = round(z*2^bits)``."""
    z = np.asarray(z, dtype=np.float64)
    q_z_float = np.rint(z * _scale(bits))
    if np.abs(q_z_float).max() >= INT64_GUARD:
        raise ValueError("quantized input exceeds the registered "
                         "int64 headroom guard (M322e-B)")
    return q_z_float.astype(np.int64)


def quantized_scores(q_z: np.ndarray, W_q: np.ndarray,
                     b_q: np.ndarray) -> np.ndarray:
    """The integer MACs: ``W_q^T q_z + b_q`` in exact int64. This is
    the arithmetic BFV decodes to, bit for bit.
    """
    return W_q.T @ q_z + b_q


def dequantize(scores_q: np.ndarray, bits: int = SCALE_BITS_16) -> np.ndarray:
    """Divide the decoded integers back to score scale."""
    return np.asarray(scores_q, dtype=np.float64) / float(_scale(2 * bits))


def fhe_simulated_scores(z: np.ndarray, W: np.ndarray, b: np.ndarray,
                         bits: int = SCALE_BITS_16) -> np.ndarray:
    """One pass through the registered arithmetic: quantize the
    input, integer MACs, dequantize. The reference for the BFV
    backend's exact-agreement gate (M322-QG3).
    """
    head = quantize_head(W, b, bits)
    q_z = quantize_input(z, bits)
    return dequantize(quantized_scores(q_z, head["W_q"], head["b_q"]),
                      bits)


# ---------------------------------------------------------------------------
# M322e-C: per-class block exponents (registered after the M322e-B negative
# finding — the real head's dynamic range dominates a uniform scale)
# ---------------------------------------------------------------------------
EXPONENT_BOUND = 40  # |k_c| is clamped into [-EXPONENT_BOUND, EXPONENT_BOUND]


def class_exponents(W: np.ndarray, bits: int = SCALE_BITS_16
                    ) -> np.ndarray:
    """Per-class block exponents: k_c = round(log2(2^(bits-1) /
    max_j |W[j,c]|)), clamped. The exponents normalize each class's
    weight range to ~2^(bits-1) before the uniform quantization.
    """
    W = np.asarray(W, dtype=np.float64)
    col_max = np.max(np.abs(W), axis=0)
    col_max = np.maximum(col_max, 1e-300)
    k = np.rint(np.log2(2.0 ** (bits - 1) / col_max))
    return np.clip(k, -EXPONENT_BOUND, EXPONENT_BOUND).astype(np.int64)


def quantize_head_perclass(W: np.ndarray, b: np.ndarray,
                           bits: int = SCALE_BITS_16) -> dict:
    """M322e-C: per-class rescale, then the uniform quantization.
    ``W_q = round(W * 2^k_c * 2^bits)`` per column;
    ``b_q = round(b * 2^k_c * 2^(2*bits))``. Gating guards as in
    the uniform scheme.
    """
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    k = class_exponents(W, bits)
    W_scaled = W * (2.0 ** k)[None, :]
    b_scaled = b * (2.0 ** k)
    head = quantize_head(W_scaled, b_scaled, bits)
    head["exponents"] = k
    return head


def dequantize_perclass(scores_q: np.ndarray, exponents: np.ndarray,
                        bits: int = SCALE_BITS_16) -> np.ndarray:
    """Device-side per-class dequantization: divide class c by
    2^(2*bits + k_c). The exponents are public registry constants.
    """
    scores_q = np.asarray(scores_q, dtype=np.float64)
    exponents = np.asarray(exponents, dtype=np.int64)
    factors = 2.0 ** (2 * bits + exponents)
    return scores_q / factors


def fhe_simulated_scores_perclass(z: np.ndarray, W: np.ndarray,
                                  b: np.ndarray,
                                  bits: int = SCALE_BITS_16
                                  ) -> np.ndarray:
    """One pass through the M322e-C arithmetic: per-class rescale,
    quantize, integer MACs, per-class dequantize.
    """
    head = quantize_head_perclass(W, b, bits)
    q_z = quantize_input(z, bits)
    q = quantized_scores(q_z, head["W_q"], head["b_q"])
    return dequantize_perclass(q, head["exponents"], bits)
