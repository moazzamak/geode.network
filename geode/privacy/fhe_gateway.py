"""M322 serving wiring - the ciphertext-only FHE session.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md``
§8.34 (27 Aug 2026, before any build). Built on the M322e-D CKKS
backend (``fhe_head_ckks``) and the M322e-C per-class quantization
(``fhe_head``). The roles are separated by function boundary, not
by trust:

- the DEVICE quantizes its input, encrypts it, and keeps the
  plaintext local;
- the HOST evaluates the quantized head on ciphertext only (the
  registered ``dot``-plain / pack construction);
- the DEVICE decrypts, dequantizes, and takes the argmax
  on-device.

The session transcript type has NO field for the plaintext input
or scores - ciphertext-only is structural (W-G3), and the gateway
never receives the device's plaintext in an FHE session (W-G5).
CKKS is APPROXIMATE: the agreement gate is the registered noise
bound, never a bit-exact claim (W-G4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from geode.core.serving_tiers import ServingTier, TierSession
from geode.privacy.fhe_head import (
    SCALE_BITS_16,
    class_exponents,
    dequantize_perclass,
    quantize_head_perclass,
    quantize_input,
)
from geode.privacy.fhe_head_ckks import (
    build_context,
    decrypt_scaled,
    encrypt_input,
    evaluate_head,
)

# the registered CKKS-vs-fp64 agreement bound for the score vector:
# measured on the real head at ~3.75e9/3.83e9 relative scale; the
# smoke wiring gate uses the published per-query relative bound
REL_SCORE_TOLERANCE = 1e-2


@dataclass(frozen=True)
class FheTranscript:
    """What the network ever sees of an FHE session. Serialized
    ciphertexts and digests ONLY - no field exists for the
    plaintext input, the plaintext scores, or the answer
    (W-G3)."""
    session_id: str
    input_ciphertexts: tuple[bytes, ...] = ()
    score_ciphertext: bytes = b""
    head_digest: str = ""
    host: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ciphertext_only(self) -> bool:
        return True


class FheServingSession:
    """One private serving session. Device-side and host-side steps
    are explicit; the class holds the host's state (the quantized
    head), never the device's plaintext."""

    def __init__(self, W: np.ndarray, b: np.ndarray,
                 bits: int = SCALE_BITS_16) -> None:
        self.context = build_context()
        head = quantize_head_perclass(W, b, bits)
        self.W_q: np.ndarray = head["W_q"]
        self.b_q: np.ndarray = head["b_q"]
        self.exponents: np.ndarray = head["exponents"]
        self.bits = bits
        self.d, self.C = W.shape
        self.W_fp64 = np.asarray(W, dtype=np.float64)
        self.b_fp64 = np.asarray(b, dtype=np.float64)

    # -- device side -------------------------------------------------

    def device_encrypt(self, z: np.ndarray, session_id: str
                       ) -> tuple[Any, FheTranscript]:
        """Device step: quantize, encrypt, keep the plaintext
        local. Returns the ciphertexts for the host and the
        transcript skeleton (ciphertexts only)."""
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (self.d,):
            raise ValueError(f"input shape {z.shape}, expected "
                             f"({self.d},)")
        q_z = quantize_input(z, self.bits)
        cts = encrypt_input(self.context, q_z)
        serialized = tuple(bytes(ct.serialize()) for ct in cts)
        transcript = FheTranscript(
            session_id=session_id,
            input_ciphertexts=serialized,
            head_digest=self._head_digest(),
        )
        return cts, transcript

    # -- host side ---------------------------------------------------

    def host_evaluate(self, cts: Any) -> Any:
        """Host step: evaluate the quantized head on ciphertext
        only. The host never receives a plaintext vector."""
        return evaluate_head(cts, self.W_q, self.b_q, self.d, self.C)

    # -- device side -------------------------------------------------

    def device_decrypt(self, score_ct: Any) -> tuple[int, np.ndarray]:
        """Device step: decrypt, dequantize, argmax on-device."""
        decoded = decrypt_scaled(score_ct, self.C)
        scores = dequantize_perclass(decoded, self.exponents,
                                     self.bits)
        return int(np.argmax(scores)), scores

    # -- reference and gates -----------------------------------------

    def plaintext_scores(self, z: np.ndarray) -> np.ndarray:
        return z @ self.W_fp64 + self.b_fp64

    def agreement_gate(self, fhe_scores: np.ndarray,
                       plain_scores: np.ndarray) -> dict[str, Any]:
        """W-G4: the FHE scores agree with the fp64 head on the
        registered relative noise bound. Approximate, never
        bit-exact."""
        scale = max(float(np.max(np.abs(plain_scores))), 1e-300)
        rel = float(np.max(np.abs(fhe_scores - plain_scores)) / scale)
        return {
            "argmax_agrees": bool(np.argmax(fhe_scores)
                                  == np.argmax(plain_scores)),
            "max_relative_error": rel,
            "bound": REL_SCORE_TOLERANCE,
            "ok": bool(rel <= REL_SCORE_TOLERANCE),
        }

    def _head_digest(self) -> str:
        import hashlib
        return hashlib.sha256(
            self.W_q.tobytes() + self.b_q.tobytes()
            + self.exponents.tobytes()).hexdigest()[:16]

    def tier_session(self, session_id: str, transcript: FheTranscript
                     ) -> TierSession:
        return TierSession(
            session_id=session_id,
            tier=ServingTier.FHE_PRIVATE,
            ciphertext_only=transcript.ciphertext_only,
            note=f"host {transcript.host or 'unnamed'}, head "
                 f"{transcript.head_digest}")
