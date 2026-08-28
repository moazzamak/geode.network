"""M270 C4 — signed-request authentication.

Ed25519-class request signing and verification. Identity is used for
quotas and ledger provenance ONLY; no routing function receives it
(the hard rule: identity never routes).

Prior art, cited not invented: Ed25519 (Bernstein, Duif, Lange,
Schwabe & Yang 2012, J. Cryptogr. Eng. 2(2)); PyNaCl is the bindings
layer. Nonces and window timestamps belong to the signature input,
never to a content hash (the standing reproducibility rule).
"""
from __future__ import annotations

import base64
import time
from typing import Any

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey

_SEP = b"|"


def canonical_signed_bytes(method: str, path: str, payload_hash: str,
                           nonce: str, not_before: int,
                           not_after: int) -> bytes:
    """The exact bytes the requester signs and the verifier checks.
    Any change to field order or separator is a protocol change and
    must be versioned."""
    return _SEP.join([
        method.encode("utf-8"),
        path.encode("utf-8"),
        payload_hash.encode("utf-8"),
        nonce.encode("utf-8"),
        str(not_before).encode("utf-8"),
        str(not_after).encode("utf-8"),
    ])


def generate_keypair() -> dict[str, str]:
    """A fresh Ed25519 keypair (base64). Test/dev identity issuance;
    production issuance is out of scope for this cell."""
    signing = SigningKey.generate()
    return {
        "private_key": base64.b64encode(signing.encode()).decode("ascii"),
        "public_key": base64.b64encode(
            signing.verify_key.encode()).decode("ascii"),
    }


def sign_request(private_key_b64: str, method: str, path: str,
                 payload_hash: str, nonce: str, not_before: int,
                 not_after: int) -> str:
    """Sign the canonical request bytes; returns the base64 signature."""
    signing = SigningKey(base64.b64decode(private_key_b64))
    signed = signing.sign(canonical_signed_bytes(
        method, path, payload_hash, nonce, not_before, not_after))
    return base64.b64encode(signed.signature).decode("ascii")


def verify_request(public_key_b64: str, method: str, path: str,
                   payload_hash: str, nonce: str, not_before: int,
                   not_after: int, signature_b64: str,
                   nonce_store: dict[str, Any] | None = None,
                   now: int | None = None) -> dict[str, Any]:
    """Verify one signed request. Outcomes: ok | bad_signature |
    replayed_nonce | expired. On 'ok' the nonce is consumed in
    `nonce_store` (callers pass one store to share replay protection
    across requests). `now` is injectable for determinism in tests."""
    store = nonce_store if nonce_store is not None else {}
    now = int(time.time()) if now is None else int(now)
    if now < not_before or now > not_after:
        return {"ok": False, "outcome": "expired"}
    if nonce in store:
        return {"ok": False, "outcome": "replayed_nonce"}
    try:
        verify_key = VerifyKey(base64.b64decode(public_key_b64))
        verify_key.verify(
            canonical_signed_bytes(method, path, payload_hash, nonce,
                                   not_before, not_after),
            base64.b64decode(signature_b64))
    except (BadSignatureError, ValueError):
        return {"ok": False, "outcome": "bad_signature"}
    store[nonce] = now
    return {"ok": True, "outcome": "ok"}
