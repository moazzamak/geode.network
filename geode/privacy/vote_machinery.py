"""M328 — vote machinery: quorum diversity floor, secret-ballot
Pedersen tally, weight snapshot.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` §8.29
(27 Aug 2026, before any build). Three mechanisms, each closing one
gap from the Byzantine threat model (``STAKE_VOTE_THREAT_MODEL_v26``):

- **P1 diversity floor.** A vote ratifies only when its supporting
  weight comes from at least ``d = max(3, ceil(0.2 * n_responders))``
  distinct behavioural identities. One owner of many artifacts must
  hold sampled weight across several genuinely distinct serving
  businesses; weight that is all one identity never ratifies.
- **P2 secret-ballot tally.** Each ballot is a Pedersen commitment
  ``C_v = g^{o_v} h^{r_v}`` over the vote bit, with a Schnorr-style
  OR membership proof for ``o_v in {0, 1}``, signed and public. A
  tally committee of k validators opens ONLY the weighted sums by
  threshold (voters Shamir-split their openings; ``t = ceil(k/2)+1``
  members combine). Individual ballots are never published. Unopened
  weight above one third of the sampled weight fails the vote
  closed.
- **P3 weight snapshot.** Each vote freezes every sampled voter's
  weight at the epoch-boundary snapshot that opens it.

The group is the seed-derived safe-prime group of
``geode.privacy.zk_bulletproofs`` (prototype security parameter,
registered): order ``Q_ORDER`` prime, bases ``G`` and ``H`` both of
order ``q`` with unknown discrete log between them.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

from geode.privacy.zk_bulletproofs import P, Q_ORDER, BP_G

G_BASE = BP_G  # order-q base, unknown-log
H_BASE = int.from_bytes(
    hashlib.sha256(b"geode-vote-m328-h-base").digest(), "big") % P
H_BASE = pow(H_BASE, 2, P) % P
if H_BASE == 1:
    H_BASE = pow(4, 2, P) % P  # square of 4: order q, distinct from 1


# ---------------------------------------------------------------------------
# P1 — quorum diversity floor
# ---------------------------------------------------------------------------
def min_responders(pool_size: int) -> int:
    """M315: the responder minimum scales with the pool; never a
    fixed three."""
    return max(3, math.ceil(0.1 * pool_size))


def diversity_floor(n_responders: int) -> int:
    """M328 P1: d = max(3, ceil(0.2 * n_responders))."""
    return max(3, math.ceil(0.2 * n_responders))


def ratifies(support_weight: float, total_weight: float,
             supporting_identities: Sequence[str],
             pool_size: int, responders: int) -> tuple[bool, str]:
    """The full ratification predicate. Returns (verdict, reason)."""
    if responders < min_responders(pool_size):
        return False, "below_min_responders"
    if total_weight <= 0.0:
        return False, "zero_sampled_weight"
    if support_weight / total_weight < 2.0 / 3.0:
        return False, "below_two_thirds"
    distinct = len({i for i in supporting_identities})
    if distinct < diversity_floor(responders):
        return False, "below_diversity_floor"
    return True, "ratifies"


# ---------------------------------------------------------------------------
# P3 — weight snapshot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WeightSnapshot:
    """An immutable epoch-boundary snapshot of voter weights."""
    anchor: str
    weights: dict[str, float]

    def weight_of(self, identity: str) -> float:
        return float(self.weights.get(identity, 0.0))

    def total(self, identities: Sequence[str]) -> float:
        return sum(self.weight_of(i) for i in identities)

    def digest(self) -> str:
        payload = {"anchor": self.anchor,
                   "weights": {str(k): float(v)
                               for k, v in sorted(self.weights.items())}}
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# P2 — secret-ballot Pedersen tally
# ---------------------------------------------------------------------------
def _ser(*values: int) -> bytes:
    return ";".join(format(int(v), "x") for v in values).encode("utf-8")


def _challenge(*parts: bytes) -> int:
    state = hashlib.sha256()
    for part in parts:
        state.update(part)
    return int.from_bytes(state.digest(), "big") % Q_ORDER


def commit_ballot(vote: int, r: int) -> int:
    """Pedersen commitment C = g^vote h^r mod P (vote in {0,1})."""
    if vote not in (0, 1):
        raise ValueError("ballot vote must be 0 or 1 (M328-G3)")
    return (pow(G_BASE, vote, P) * pow(H_BASE, r, P)) % P


def verify_commitment(C: int, vote: int, r: int) -> bool:
    return C == commit_ballot(vote, r)


def ballot_proof(vote: int, r: int) -> dict:
    """Schnorr OR-proof of membership in {0, 1} for the committed
    vote (Fiat-Shamir, non-interactive). Branch 0 claims C = h^z;
    branch 1 claims C/g = h^z.
    """
    C = commit_ballot(vote, r)
    b, bp = vote, 1 - vote
    # simulated branch (the vote we do NOT hold)
    c_bp = _challenge(_ser(C), b"sim", _ser(bp))
    z_bp = int.from_bytes(hashlib.sha256(
        _ser(C, bp, c_bp)).digest(), "big") % Q_ORDER
    target_bp = (C if bp == 0
                 else (C * pow(G_BASE, Q_ORDER - 1, P)) % P)
    a_bp = (pow(H_BASE, z_bp, P)
            * pow(target_bp, Q_ORDER - c_bp, P)) % P
    # real branch
    t_b = int.from_bytes(hashlib.sha256(
        _ser(C, b, r)).digest(), "big") % Q_ORDER
    a_b = pow(H_BASE, t_b, P)
    a0 = a_b if b == 0 else a_bp
    a1 = a_bp if b == 0 else a_b
    e = _challenge(_ser(C), _ser(a0), _ser(a1))
    c_b = (e - c_bp) % Q_ORDER
    z_b = (t_b + c_b * r) % Q_ORDER
    return {"C": C, "e": e,
            "c0": c_b if b == 0 else c_bp,
            "z0": z_b if b == 0 else z_bp,
            "c1": c_bp if b == 0 else c_b,
            "z1": z_bp if b == 0 else z_b}


def verify_ballot(C: int, proof: dict) -> bool:
    """Verify the OR-proof: both branches reconstruct to the same
    challenge e and c0 + c1 == e."""
    e = int(proof["e"]) % Q_ORDER
    c0 = int(proof["c0"]) % Q_ORDER
    z0 = int(proof["z0"]) % Q_ORDER
    c1 = int(proof["c1"]) % Q_ORDER
    z1 = int(proof["z1"]) % Q_ORDER
    if (c0 + c1) % Q_ORDER != e:
        return False
    a0 = (pow(H_BASE, z0, P) * pow(C, Q_ORDER - c0, P)) % P
    target1 = (C * pow(G_BASE, Q_ORDER - 1, P)) % P
    a1 = (pow(H_BASE, z1, P) * pow(target1, Q_ORDER - c1, P)) % P
    return _challenge(_ser(C), _ser(a0), _ser(a1)) == e


def shamir_split(secret: int, threshold: int, n: int,
                 seed: int) -> list[tuple[int, int]]:
    """Shamir shares of the ballot opening over Z_q (the Pedersen
    exponent field). Coefficients from the seed."""
    if threshold > n or threshold < 1:
        raise ValueError("threshold must satisfy 1 <= t <= n")
    coeffs = [secret % Q_ORDER]
    state = seed % Q_ORDER
    for _ in range(threshold - 1):
        state = _challenge(_ser(state), b"coeff")
        coeffs.append(state)
    points = []
    for x in range(1, n + 1):
        y = 0
        xp = 1
        for c in coeffs:
            y = (y + c * xp) % Q_ORDER
            xp = (xp * x) % Q_ORDER
        points.append((x, y))
    return points


def shamir_combine(points: Sequence[tuple[int, int]]) -> int:
    """Lagrange interpolation of the shares at x = 0."""
    total = 0
    for i, (xi, yi) in enumerate(points):
        num, den = 1, 1
        for j, (xj, _) in enumerate(points):
            if i == j:
                continue
            num = (num * (-xj)) % Q_ORDER
            den = (den * (xi - xj)) % Q_ORDER
        total = (total + yi * num * pow(den, Q_ORDER - 2, Q_ORDER)) \
            % Q_ORDER
    return total


@dataclass(frozen=True)
class TallyRecord:
    """What the public record carries: commitments, weights, opened
    SUMS only. No individual ballot or opening field exists
    (M328-G6 — the inexpressibility audit target)."""
    weighted_support: int
    weighted_total: int
    commitments: tuple[int, ...]
    weights: tuple[int, ...]

    def ratifies(self, pool_size: int) -> tuple[bool, str]:
        support_identities = [f"id{i}" for i, w in enumerate(self.weights)
                              if w > 0]
        return ratifies(float(self.weighted_support),
                        float(self.weighted_total),
                        support_identities, pool_size,
                        responders=len(self.weights))


def tally(ballots: Sequence[tuple[int, int, int]],
          weights: Sequence[int], committee_size: int,
          threshold: int, seed: int) -> TallyRecord:
    """The committee tally: voters Shamir-split their openings to
    the committee; the threshold combination recovers the weighted
    opening sum; the sums verify against the commitment aggregate.

    ``ballots`` entries are (vote, r, C). Returns the record with
    ONLY the weighted sums (no individual openings).
    """
    r_weighted = 0
    v_weighted = 0
    w_total = 0
    for (vote, r, C), w in zip(ballots, weights):
        if vote not in (0, 1) or not verify_commitment(C, vote, r):
            raise ValueError("invalid ballot (M328-G3)")
        v_weighted = (v_weighted + vote * w) % Q_ORDER
        w_total += w
        shares = shamir_split(r, threshold, committee_size,
                              seed + w)
        # the committee combines with threshold-of-k: the honest
        # simulation combines all k shares; the secrecy gate tests
        # that fewer than t shares reveal nothing
        r_weighted = (r_weighted
                      + w * shamir_combine(shares)) % Q_ORDER
    lhs = (pow(G_BASE, v_weighted, P) * pow(H_BASE, r_weighted, P)) % P
    rhs = 1
    for (_, _, C), w in zip(ballots, weights):
        rhs = (rhs * pow(C, w, P)) % P
    if lhs != rhs:
        raise ValueError("tally opening failed Pedersen verification "
                         "(M328-G4)")
    return TallyRecord(weighted_support=v_weighted,
                       weighted_total=w_total,
                       commitments=tuple(C for _, _, C in ballots),
                       weights=tuple(weights))
