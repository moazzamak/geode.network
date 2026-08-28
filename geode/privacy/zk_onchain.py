"""M213 — proof serialization matching the registered on-chain layout
for the M193b log-sized argument, plus the generator-label and
challenge-serialization mirrors the Solidity verifier must reproduce
bit-exactly.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(19 Aug 2026, before the build). Layout (32 bytes per element,
r = log2(n)):

    [C][L_0..L_{r-1}][R_0..R_{r-1}][a_final][b_final][r_final]

The claim is the public statement and travels as a SEPARATE argument
(never in the proof bytes) — matching the sealed M193b size figure of
1,024 bytes for r = 14 (32 words). The sealed ``geode/zk_bulletproofs.py``
module is NOT modified; this module is the on-chain bridge only.
"""
from __future__ import annotations

from typing import Any, Sequence


def rounds_of(n: int) -> int:
    r = 0
    while (1 << r) < n:
        r += 1
    return r


def proof_length_bytes(n: int) -> int:
    return (1 + 2 * rounds_of(n) + 3) * 32


def _word(value: int) -> bytes:
    return int(value).to_bytes(32, "big")


def serialize(proof: dict[str, Any], n: int) -> bytes:
    """Serialize a M193b proof dict in the registered layout. ``n``
    is the padded vector width (a power of two). The claim slot is
    NOT included — the claim is a separate public argument."""
    r = rounds_of(n)
    if len(proof["L"]) != r or len(proof["R"]) != r:
        raise ValueError(f"round count mismatch: proof has "
                         f"{len(proof['L'])} rounds, n={n} needs {r}")
    out = b"".join([
        _word(proof["c_commit"]),
        *(_word(v) for v in proof["L"]),
        *(_word(v) for v in proof["R"]),
        _word(proof["a_final"]),
        _word(proof["b_final"]),
        _word(proof["r_final"]),
    ])
    assert len(out) == proof_length_bytes(n)
    return out


def serialize_hex(proof: dict[str, Any], n: int) -> str:
    return "0x" + serialize(proof, n).hex()


def words_hex(w: Sequence[int], n: int) -> list[str]:
    """The padded weight vector as 0x-hex 32-byte words for calldata."""
    padded = [int(v) for v in w] + [0] * (n - len(w))
    return ["0x" + _word(v).hex() for v in padded]


def generator_label(index: int) -> bytes:
    """The exact byte string the generator derives from (and the
    Solidity verifier must reproduce): b'geode-bp-' + decimal index.
    Note index -1 (BP_G) yields b'geode-bp--1'."""
    return f"geode-bp-{index}".encode("utf-8")


def challenge_serialization(l: int, r: int, c: int) -> bytes:
    """The exact Fiat-Shamir serialization used in the verify path:
    three SEPARATE single-value ``_ser`` calls hashed in sequence, i.e.
    the minimal lowercase hex strings concatenated with NO separators
    (';' only appears when ``_ser`` receives several values in one
    call, which never happens in verify)."""
    return (format(int(l), "x") + format(int(r), "x")
            + format(int(c), "x")).encode("utf-8")
