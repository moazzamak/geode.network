"""M193 — zero-knowledge proof of linear computation for the ridge
head (single-move Fiat-Shamir linear-relation argument).

Registered in ``RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(19 Aug 2026, before the build). Proves, in zero knowledge, that the
claimed score vector y satisfies y = W x + b for a committed input x,
with public weights W and bias b.

System: Pedersen vector commitment over the QR subgroup of the
RFC 3526 2048-bit MODP group; a single-move Schnorr-style proof of
the aggregated linear relation (Fiat-Shamir non-interactive). Proof
size is O(n) — the log-sized (Bulletproofs) cell is the registered
next step. Special honest-verifier zero knowledge; soundness from the
discrete-log assumption in the QR subgroup.
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

# RFC 3526 group 14 (2048-bit MODP): p safe prime, q = (p-1)/2 prime,
# g = 2 generates the prime-order subgroup.
P = int("FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
        "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
        "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
        "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
        "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
        "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
        "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
        "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
        "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
        "DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
        "15728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)
Q = (P - 1) // 2
G = 2


def generator_for(index: int) -> int:
    """Deterministic non-trivial subgroup generator h_i (no trusted
    setup beyond the public group)."""
    digest = hashlib.sha256(f"geode-zk-{index}".encode("utf-8")).digest()
    h = int.from_bytes(digest, "big") % P
    h = pow(h, 2, P)  # force QR subgroup
    if h == 1:
        h = G
    return h


def _challenge(*parts: bytes) -> int:
    state = hashlib.sha256()
    for part in parts:
        state.update(part)
    return int.from_bytes(state.digest(), "big") % Q


def commit(x: Sequence[int], r: int, start: int = 0) -> int:
    """Pedersen commitment Com(x; r) = g^r * prod h_i^x_i (mod p)."""
    acc = pow(G, int(r), P)
    for i, xi in enumerate(x):
        acc = (acc * pow(generator_for(start + i), int(xi) % Q, P)) % P
    return acc


def _ser(*values: int) -> bytes:
    """Deterministic canonical serialization of field elements."""
    return ";".join(format(int(v), "x") for v in values).encode("utf-8")


def aggregate(w: Sequence[Sequence[int]], b: Sequence[int],
              y: Sequence[int]) -> tuple[list[int], int, int]:
    """Linear-combination aggregate: w' = sum t^j w_j, b' = sum t^j b_j,
    y' = sum t^j y_j with t = H(w || b || y). The aggregate relation
    <w', x> = y' - b' holds iff every row holds (up to 345/q soundness)."""
    payload = hashlib.sha256()
    for row in w:
        payload.update(_ser(*row))
    payload.update(_ser(*b))
    payload.update(_ser(*y))
    t = int.from_bytes(payload.digest(), "big") % Q
    w_agg = [0] * len(w[0])
    b_agg, y_agg = 0, 0
    power = 1
    for row, bi, yi in zip(w, b, y):
        for i, v in enumerate(row):
            w_agg[i] = (w_agg[i] + power * (int(v) % Q)) % Q
        b_agg = (b_agg + power * (int(bi) % Q)) % Q
        y_agg = (y_agg + power * (int(yi) % Q)) % Q
        power = (power * t) % Q
    return w_agg, b_agg, y_agg


def prove(x: Sequence[int], r: int, w: Sequence[Sequence[int]],
          b: Sequence[int], y: Sequence[int]) -> dict[str, Any]:
    """Produce the non-interactive proof for y = W x + b with x
    committed under randomness r. Deterministic for fixed x, r."""
    n = len(x)
    w_agg, b_agg, y_agg = aggregate(w, b, y)
    # random masking (seeded deterministically from the transcript so
    # the proof is reproducible: registered g3 determinism)
    seed = _challenge(b"geode-zk-mask", _ser(int(r)))
    import random as _random
    rng = _random.Random(seed)
    s = [rng.randrange(Q) for _ in range(n)]
    tr = rng.randrange(Q)
    a_commit = commit(s, tr)
    u = sum((si * wi) % Q for si, wi in zip(s, w_agg)) % Q
    c = _challenge(_ser(a_commit), _ser(u), _ser(*w_agg), _ser(b_agg),
                   _ser(y_agg))
    z = [(si + c * (int(xi) % Q)) % Q for si, xi in zip(s, x)]
    zr = (tr + c * (int(r) % Q)) % Q
    return {"A": a_commit, "u": u, "z": z, "zr": zr,
            "w_agg": w_agg, "b_agg": b_agg, "y_agg": y_agg}


def verify(proof: dict[str, Any], c_commit: int,
           w: Sequence[Sequence[int]], b: Sequence[int],
           y: Sequence[int]) -> bool:
    """Verify against the published commitment C = Com(x; r)."""
    w_agg, b_agg, y_agg = aggregate(w, b, y)
    if (list(proof["w_agg"]) != w_agg
            or proof["b_agg"] != b_agg or proof["y_agg"] != y_agg):
        return False
    a_commit, u, z, zr = proof["A"], proof["u"], proof["z"], proof["zr"]
    c = _challenge(_ser(a_commit), _ser(u), _ser(*w_agg), _ser(b_agg),
                   _ser(y_agg))
    # commitment check: g^zr prod h_i^z_i == A * C^c
    lhs = pow(G, int(zr), P)
    for i, zi in enumerate(z):
        lhs = (lhs * pow(generator_for(i), int(zi) % Q, P)) % P
    rhs = (a_commit * pow(c_commit, c, P)) % P
    if lhs != rhs:
        return False
    # linear relation: <w', z> == u + c * (y' - b')
    inner = sum((zi * wi) % Q for zi, wi in zip(z, w_agg)) % Q
    return inner == (u + c * ((y_agg - b_agg) % Q)) % Q


def proof_size_bytes(proof: dict[str, Any]) -> int:
    """Serialized size of the proof (group elements 256 B, scalars 32 B)."""
    n = len(proof["z"])
    return 256 + 32 + n * 32 + 32 + n * 32 + 32 + 32
