"""M318 — Pedersen commitment of the quantized head as the registry
binding key (R-A15a).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M318
(A15/A16). The registry key is a content hash, but the proof layer's
inner-product statement needs a HOMOMORPHIC binding. This module
commits the QUANTIZED head — the artifact the private serving path
actually evaluates (M322e-C) — column by column with multi-generator
Pedersen commitments over the registered group:

    C_j = prod_i g_i^{q_W[i,j]} * h^{r_j}   (mod P)

The commitment is the binding key alongside the content hash. Any
proof or verification statement about ``W`` binds to this
commitment; the A16 statement restates over ledger-registered
reference CODES (never raw inputs): ``y_i = decode(W^T z_i)``.

Group: the seed-derived safe-prime group of ``zk_bulletproofs``
(prototype security parameter, registered). The generators are
hash-derived (unknown discrete logs between them).
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from geode.privacy.zk_bulletproofs import P, Q_ORDER, BP_G

G_BASE = BP_G
H_BASE = pow(int.from_bytes(hashlib.sha256(
    b"geode-m318-head-h").digest(), "big"), 2, P) % P


def _generator(index: int) -> int:
    digest = hashlib.sha256(f"geode-m318-g-{index}".encode("utf-8")
                            ).digest()
    h = int.from_bytes(digest, "big") % P
    h = pow(h, 2, P)
    return h if h != 1 else pow(4, 2, P)


def commit_head(q_W: np.ndarray, r: Sequence[int]) -> tuple[np.ndarray]:
    """Per-column multi-generator Pedersen commitments to the
    quantized head ``q_W`` (d, C). Returns the C commitment values.
    """
    q_W = np.asarray(q_W, dtype=np.int64)
    r = [int(x) % Q_ORDER for x in r]
    if len(r) != q_W.shape[1]:
        raise ValueError("one opening per column")
    gens = [_generator(i) for i in range(q_W.shape[0])]
    C = []
    for j in range(q_W.shape[1]):
        acc = pow(H_BASE, r[j], P)
        for i in range(q_W.shape[0]):
            if q_W[i, j] == 0:
                continue
            acc = (acc * pow(gens[i], int(q_W[i, j]), P)) % P
        C.append(acc)
    return tuple(C)


def verify_head(C: Sequence[int], q_W: np.ndarray,
                r: Sequence[int]) -> bool:
    """Verify the commitments open to the given quantized head."""
    try:
        C2 = commit_head(q_W, r)
    except ValueError:
        return False
    return tuple(int(x) for x in C) == tuple(int(x) for x in C2)


def bind_artifact(q_W: np.ndarray, r: Sequence[int]) -> dict:
    """The registry binding record: the commitment vector plus the
    artifact's content hash (the existing registry key stays)."""
    payload = {"commitments": [int(c) for c in commit_head(q_W, r)]}
    content_hash = hashlib.sha256(repr(payload).encode("utf-8")
                                  ).hexdigest()
    return {"commitments": payload["commitments"],
            "content_hash": content_hash}
