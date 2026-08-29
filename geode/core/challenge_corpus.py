"""M369 (G29) — append-only challenge corpus; depletion budget + alarm.

Registered 29 Aug 2026, before the build. G29's defect: challenges
are drawn from a per-axis corpus "committed by Merkle root at axis
creation," and "every revealed point is public thereafter." An
attacker submits repeated registrations, paying only fees, to burn the
corpus; after depletion the axis cannot admit anyone against
unrevealed data. Separately, "shards reshuffle / retired rows are
replaced" would change a root fixed at creation.

The repair (registered in the review):

- Commit the corpus as a SEQUENCE of roots (an append-only commitment
  tree), so replenishment extends the commitment instead of breaking
  it. Each admission records which epoch's root it drew from.
- A per-axis depletion budget and an alarm: when unrevealed points
  fall below a registered fraction, admissions on that axis PAUSE
  until the corpus is extended. A pause is public and measurable.
- The registration fee is sized so a depletion campaign costs more
  than extending the corpus.

Gate (M369): a depletion simulation — cost to the attacker to exhaust
an axis vs cost to the network to replenish; the fee is set so the
ratio exceeds a registered margin.

Design (registered before the build):

- AppendOnlyCorpus: a list of roots (commitments). `commit` appends;
  admission records the root index it drew from, so replenishment
  never rewrites history.
- DEPLETION_PAUSE_FRACTION = 0.25: admissions pause when unrevealed
  points fall below a quarter of the axis's total corpus.
- DEPLETION_MARGIN = 2.0: the attacker's exhaustion cost must exceed
  twice the network's replenishment cost.
"""
from __future__ import annotations

from dataclasses import dataclass


# Registered depletion parameters.
DEPLETION_PAUSE_FRACTION = 0.25   # pause admissions below 25% unrevealed
DEPLETION_MARGIN = 2.0            # attacker cost must exceed 2x replenish


@dataclass(frozen=True)
class RootCommit:
    """One commitment in the append-only sequence. ``drawn`` counts
    how many challenge points from this root have been revealed."""
    root: str
    points: int          # points committed by this root
    drawn: int = 0       # revealed so far


class AppendOnlyCorpus:
    """The append-only commitment sequence. ``commit`` extends the
    sequence; it never rewrites a prior root, so replenishment does
    not break the commitment. An admission records which root index
    it drew from."""

    def __init__(self) -> None:
        self._roots: list[RootCommit] = []

    def commit(self, root: str, points: int) -> int:
        """Append a new root; returns its index (the index an
        admission records as its draw source)."""
        if points <= 0:
            raise ValueError("a commit must carry positive points")
        self._roots.append(RootCommit(root=str(root), points=int(points)))
        return len(self._roots) - 1

    def reveal(self, root_index: int, n: int) -> None:
        """Reveal ``n`` points from a committed root (a registration
        draws its challenges from a recorded epoch's root)."""
        if not 0 <= root_index < len(self._roots):
            raise IndexError("unknown root index")
        if n < 0:
            raise ValueError("revealed points must be non-negative")
        root = self._roots[root_index]
        self._roots[root_index] = RootCommit(
            root.root, root.points, root.drawn + n)

    def total_points(self) -> int:
        return sum(r.points for r in self._roots)

    def unrevealed(self) -> int:
        return sum(r.points - r.drawn for r in self._roots)

    def unrevealed_fraction(self) -> float:
        total = self.total_points()
        if total == 0:
            return 0.0
        return self.unrevealed() / total

    def admissions_paused(self,
                          threshold: float = DEPLETION_PAUSE_FRACTION
                          ) -> bool:
        """The public alarm: admissions pause when the unrevealed
        fraction falls below the registered threshold. A pause is
        public (this value is in the registry); a silently exhausted
        corpus is not."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        return self.unrevealed_fraction() < threshold


def depletion_gate(registration_fee: float,
                   points_revealed_per_registration: int,
                   replenish_cost_per_point: float,
                   attacker_registrations: int,
                   margin: float = DEPLETION_MARGIN) -> dict:
    """The depletion simulation (M369's gate).

    The attacker's cost to exhaust an axis is the number of
    registrations it takes to reveal the corpus times the registration
    fee. The network's cost to replenish is the number of points
    replaced times the per-point replenishment cost. The fee is set
    (and here checked) so the attacker's cost exceeds the network's
    by the registered margin.

    Returns both costs and the ratio, so the reading is a contrast.
    """
    if registration_fee < 0.0 or points_revealed_per_registration <= 0:
        raise ValueError("fee must be non-negative and points positive")
    if replenish_cost_per_point < 0.0:
        raise ValueError("replenish cost must be non-negative")
    if attacker_registrations <= 0:
        raise ValueError("attacker registrations must be positive")
    attacker_cost = attacker_registrations * float(registration_fee)
    replenished_points = (attacker_registrations
                          * points_revealed_per_registration)
    replenish_cost = replenished_points * float(replenish_cost_per_point)
    ratio = attacker_cost / replenish_cost if replenish_cost > 0.0 \
        else float("inf")
    return {
        "attacker_cost": attacker_cost,
        "replenish_cost": replenish_cost,
        "ratio": ratio,
        "exceeds_margin": ratio >= float(margin),
        "margin": float(margin),
    }
