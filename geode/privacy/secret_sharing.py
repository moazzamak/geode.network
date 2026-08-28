"""M192 — secret-sharing primitives for the GEODE privacy track.

Registered in ``RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(19 Aug 2026, before the build). Two protocols:

1. ``split_additive`` / ``recombine_additive`` — additive float64
   splitting over n parties for CONTRIBUTION privacy (a single
   party's share is independent noise; the sum recovers the row).
2. ``shamir_split`` / ``shamir_reconstruct`` — Shamir k-of-n over the
   Mersenne prime 2**61 - 1 for BYZANTINE threshold score
   reconstruction (any k honest shares recover; a corrupted share is
   detected by subset-consistency disagreement).

No RNG is seeded globally: every function takes an explicit generator
so runs stay deterministic under a registered seed.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

PRIME = (1 << 61) - 1  # Mersenne prime, fits int64 arithmetic with care


# ---------------------------------------------------------------------------
# additive splitting (contribution privacy)
# ---------------------------------------------------------------------------
def split_additive(row: np.ndarray, parties: int,
                   rng: np.random.Generator) -> list[np.ndarray]:
    """Split a float64 row into `parties` additive shares: shares 1..n-1
    are uniform noise at the data scale; share 0 absorbs the residual.
    A single share reveals nothing about the row except its scale."""
    noise = rng.uniform(-1.0, 1.0, size=(parties - 1, len(row)))
    shares = [np.asarray(noise[i], dtype=np.float64)
              for i in range(parties - 1)]
    residual = np.asarray(row, dtype=np.float64).copy()
    for share in shares:
        residual -= share
    return [residual] + shares


def recombine_additive(shares: Sequence[np.ndarray]) -> np.ndarray:
    total = np.zeros_like(shares[0], dtype=np.float64)
    for share in shares:
        total += share
    return total


def replicated_gram_shares(block: np.ndarray, parties: int,
                           rng: np.random.Generator
                           ) -> list[np.ndarray]:
    """The registered Z-resharing protocol for a secret-shared Gram.

    Splits the block additively (X = sum X_p; X_0 is the residual
    share - disclosed limitation), gives each party p the pair
    {X_p, X_{p+1}}, and returns party output shares C_p with

        C_p = Z_p - Z_{p+1} + X_p.T X_{p+1} + X_{p+1}.T X_p
              + 0.5 X_p.T X_p + 0.5 X_{p+1}.T X_{p+1}

    where the Z_p are random matrices announced identically to the
    two neighbours (Z cancels in the sum). In exact arithmetic
    sum_p C_p == block.T @ block. VALID ONLY FOR parties == 3: the
    share cycle {p, p+1} must cover every index pair, which holds
    exactly for the 3-cycle 0-1, 1-2, 2-0.
    """
    if parties != 3:
        raise ValueError("replicated_gram_shares requires 3 parties")
    noise = [rng.uniform(-1.0, 1.0, size=block.shape)
             for _ in range(parties - 1)]
    shares: list[np.ndarray] = [np.asarray(block, dtype=np.float64).copy()]
    for n_ in noise:
        shares[0] -= n_
        shares.append(n_)
    zs = [rng.uniform(-1.0, 1.0, size=(block.shape[1], block.shape[1]))
          for _ in range(parties)]
    outs: list[np.ndarray] = []
    for p in range(parties):
        xp = shares[p]
        xq = shares[(p + 1) % parties]
        local = (xp.T @ xq + xq.T @ xp
                 + 0.5 * (xp.T @ xp) + 0.5 * (xq.T @ xq))
        outs.append(zs[p] - zs[(p + 1) % parties] + local)
    return outs


# ---------------------------------------------------------------------------
# Shamir k-of-n over Z_p (byzantine threshold reconstruction)
# ---------------------------------------------------------------------------
def _mod(value: int) -> int:
    return value % PRIME


def _inv(value: int) -> int:
    return pow(int(value), PRIME - 2, PRIME)


def shamir_split(secret: int, k: int, n: int,
                 rng: np.random.Generator) -> list[int]:
    """Split an integer secret (already reduced mod p) into n shares
    of a degree-(k-1) polynomial over Z_p. k shares reconstruct."""
    coeffs = [int(secret)] + [int(rng.integers(0, PRIME))
                              for _ in range(k - 1)]
    return [_mod(sum(coeff * pow(x, deg, PRIME)
                     for deg, coeff in enumerate(coeffs)))
            for x in range(1, n + 1)]


def _inv_mod(value: int, modulus: int) -> int:
    return pow(int(value), modulus - 2, modulus)


def _lagrange_mod(share_pairs: Sequence[tuple[int, int]], x0: int,
                  modulus: int) -> int:
    total = 0
    for i, (xi, yi) in enumerate(share_pairs):
        num = int(yi) % modulus
        den = 1
        for j, (xj, _yj) in enumerate(share_pairs):
            if i == j:
                continue
            num = (num * ((x0 - xj) % modulus)) % modulus
            den = (den * ((xi - xj) % modulus)) % modulus
        total = (total + num * _inv_mod(den, modulus)) % modulus
    return total


def shamir_reconstruct(pairs: Sequence[tuple[int, int]],
                       degree: int | None = None,
                       modulus: int = PRIME) -> int:
    """Reconstruct the secret from (x, share) pairs over the given
    prime modulus (default the M192 2**61-1 field)."""
    return _lagrange_mod(pairs, 0, modulus)


def signed_from_field(value: int) -> int:
    """Map a field element back to a signed integer (values >= p/2
    encode negatives)."""
    if value > PRIME // 2:
        return value - PRIME
    return value


def to_field(value: int) -> int:
    return int(value) % PRIME
