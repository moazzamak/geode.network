"""M322e-D CKKS backend — TenSEAL stage for the FHE head.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` §8.22
M322e-D (27 Aug 2026, before the backend build). TenSEAL 0.3.17 on
this machine exposes NO rotation on its Python surface (mechanical
evidence in the plan); the only rotation-free matvec primitive is
``dot`` against a plain vector, combined with the static
``pack_vectors`` for the result. The registered construction:

- the device encrypts ``z_enc = q_z / 2^32`` padded to the CKKS
  slot capacity;
- the host computes ``s_c = ct.dot(W_enc[:, c])`` for each class
  ``c`` (a size-1 ciphertext each) with ``W_enc = q_W' / 2^32``
  (per-class block exponents folded in), adds the scaled bias
  ``b_enc = q_b' / 2^64``, and packs the C scalars into ONE
  ciphertext via ``pack_vectors``;
- the device multiplies the decoded vector by ``2^64`` and
  compares against the integer MACs (CKKS-QG3a/QG3b).

CKKS is APPROXIMATE; the gates are the registered noise bounds,
never a bit-exact claim.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import tenseal as ts

# Registered CKKS parameters (poly degree 8192 was the only working
# TenSEAL 0.3.17 CKKS context on this machine — probe recorded).
# CKKS vectors hold N/2 slots.
POLY_DEGREE = 8192
SLOTS = POLY_DEGREE // 2
COEFF_BITS = [60, 40, 40, 60]
GLOBAL_SCALE = 2 ** 40
ENC_Z_SCALE = 2.0 ** 16  # q_z (~2^16) enters at O(1)
ENC_W_SCALE = 2.0 ** 32  # q_W' (~2^31) enters at O(0.5)
ENC_B_SCALE = 2.0 ** 48  # q_b' (~2^47) enters at O(0.5)
DEC_SCALE = 2.0 ** 48   # device-side recovery of the integer MACs


def build_context() -> ts.Context:
    ctx = ts.context(ts.SCHEME_TYPE.CKKS,
                     poly_modulus_degree=POLY_DEGREE,
                     coeff_mod_bit_sizes=COEFF_BITS)
    ctx.global_scale = GLOBAL_SCALE
    ctx.generate_galois_keys()
    return ctx


def encrypt_input(context: ts.Context, q_z: np.ndarray) -> Any:
    """Device step: q_z / 2^16, chunked across ceil(d / SLOTS)
    ciphertexts (the real head dimension, 13244, exceeds one
    CKKS ciphertext's 4096 slots). O(1) magnitudes keep the
    CKKS representation noise small."""
    q_z = np.asarray(q_z, dtype=np.float64)
    chunks = []
    for start in range(0, len(q_z), SLOTS):
        vec = np.zeros(SLOTS, dtype=np.float64)
        block = q_z[start:start + SLOTS]
        vec[: len(block)] = block / ENC_Z_SCALE
        chunks.append(ts.ckks_vector(context, vec))
    return chunks


def evaluate_head(cts: Any, W_q: np.ndarray, b_q: np.ndarray,
                  d: int, c: int) -> Any:
    """Host step: one dot-plain per (class, chunk), chunk partials
    summed per class, the C scalar ciphertexts packed into a single
    C-slot ciphertext, then ONE plain bias add on the packed vector.
    (Per-class biases added BEFORE packing lose every bias past the
    first slot — the TenSEAL 0.3.17 pack interaction, probed and
    worked around; the bias-after-pack order is the registered
    form.)
    """
    scalars = []
    for j in range(c):
        partial = None
        for k, ct in enumerate(cts):
            start = k * SLOTS
            col = np.zeros(SLOTS, dtype=np.float64)
            block = W_q[start:start + SLOTS, j] / ENC_W_SCALE
            col[: len(block)] = block
            p_k = ct.dot(ts.plain_tensor(col.tolist(), dtype="float"))
            partial = p_k if partial is None else partial + p_k
        scalars.append(partial)
    packed = ts.CKKSVector.pack_vectors(scalars)
    bias = (np.asarray(b_q, dtype=np.float64) / ENC_B_SCALE)
    return packed + ts.plain_tensor(bias.tolist(), dtype="float")


def decrypt_scaled(ct: Any, c: int) -> np.ndarray:
    """Device step: decoded vector, scaled back by 2^48, slots
    0..C-1 — comparable with the integer MACs."""
    out = np.asarray(ct.decrypt(), dtype=np.float64).ravel()[: c]
    return out * DEC_SCALE
