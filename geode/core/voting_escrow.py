"""M359 (G14) — voluntary voting escrow: the burn base decouples
from the weight base.

Registered 29 Aug 2026, before the build. G14's defect: the ladder
burns "vested-but-unclaimed credits" at level 3, and voting weight IS
unclaimed credits, so holding weight strictly increases slashable
exposure. The rational actor claims every epoch and abstains from
governance — the ladder prices participation. The at-risk-capital
argument and the participation tax are the same number.

The repair (registered in the review): a VOLUNTARY escrow. A
participant may lock claimable (externally-verified, M358) credits
into weight-bearing escrow for a fixed term (8 epochs). Escrowed
credits carry weight and are burnable at level 3; unescrowed vested
credits are claimable and carry NO weight. The choice becomes explicit
and priced rather than a hidden penalty: a voter who wants weight
accepts the escrow term and the L3 exposure; a participant who claims
every epoch simply has no weight.

Design decisions (registered before the build):

- ESCROW_TERM_EPOCHS = 8 (the review's proposal).
- Escrow is per-identity, a stack of slots with maturity epochs.
- lock(): consumes from the identity's verified claimable balance
  (the M358 weight base) and creates a slot maturing term epochs out.
- weight(): the sum of not-yet-matured slots — the ONLY weight base.
- burn(): L3 reduces the escrow (and therefore weight), newest-matured
  first; it can never touch a claimable balance it does not cover.
- unlock(): matured slots return to the claimable balance (pull).
- Unescrowed credits carry no weight by construction: weight reads
  only escrow slots.

Gate (M359): escrow is voluntary, term-bounded, burnable at L3;
unescrowed vested credits carry no weight.
"""
from __future__ import annotations

from dataclasses import dataclass


# The registered escrow term (G14's proposal: 8 epochs).
ESCROW_TERM_EPOCHS = 8


@dataclass(frozen=True)
class EscrowSlot:
    """One locked tranche: ``amount`` credits maturing at
    ``matures_at`` (an epoch number)."""
    amount: float
    matures_at: int


class InsufficientBalance(RuntimeError):
    """The identity does not have enough claimable credits to lock."""


class EscrowNotFound(RuntimeError):
    """No burnable escrow covers the requested burn."""


class VotingEscrow:
    """Voluntary, term-bounded weight escrow.

    Claimable = externally-verified credits (M358) that the
    participant has not locked. Escrowed = locked credits that carry
    voting weight. Claimable credits carry no weight; weight reads
    only the escrow.
    """

    def __init__(self) -> None:
        self._claimable: dict[str, float] = {}
        self._escrow: dict[str, list[EscrowSlot]] = {}
        self._burnt: dict[str, float] = {}

    def grant_claimable(self, identity: str, amount: float) -> None:
        """Credit an identity's verified claimable balance (the M358
        weight base enters here)."""
        if amount < 0.0:
            raise ValueError("amount must be non-negative")
        self._claimable[str(identity)] = \
            self._claimable.get(str(identity), 0.0) + float(amount)

    def claimable(self, identity: str) -> float:
        return self._claimable.get(str(identity), 0.0)

    def lock(self, identity: str, amount: float,
             current_epoch: int) -> None:
        """Lock ``amount`` claimable credits into escrow maturing
        ``ESCROW_TERM_EPOCHS`` epochs from now. Voluntary: only what
        the participant chooses to lock carries weight."""
        identity = str(identity)
        if amount <= 0.0:
            raise ValueError("lock amount must be positive")
        if self.claimable(identity) < amount:
            raise InsufficientBalance(
                f"{identity} has {self.claimable(identity)} claimable, "
                f"cannot lock {amount}")
        self._claimable[identity] -= amount
        self._escrow.setdefault(identity, []).append(EscrowSlot(
            amount=float(amount),
            matures_at=int(current_epoch) + ESCROW_TERM_EPOCHS))

    def weight(self, identity: str, current_epoch: int) -> float:
        """Voting weight: the sum of escrow slots not yet matured at
        ``current_epoch``. Unescrowed credits carry no weight by
        construction — weight never reads the claimable balance."""
        current = int(current_epoch)
        return sum(slot.amount for slot in self._escrow.get(
            str(identity), []) if slot.matures_at > current)

    def unlock_matured(self, identity: str, current_epoch: int) -> float:
        """Return matured slots to the claimable balance (pull)."""
        identity = str(identity)
        current = int(current_epoch)
        slots = self._escrow.get(identity, [])
        matured = [s for s in slots if s.matures_at <= current]
        if not matured:
            return 0.0
        self._escrow[identity] = [s for s in slots
                                  if s.matures_at > current]
        amount = sum(s.amount for s in matured)
        self._claimable[identity] = \
            self._claimable.get(identity, 0.0) + amount
        return amount

    def burn(self, identity: str, amount: float,
             current_epoch: int) -> float:
        """Level-3 burn: consume escrowed credits (and therefore
        weight), newest-matured first. Never touches the claimable
        balance beyond what the escrow covers. Returns what was
        actually burned."""
        identity = str(identity)
        if amount <= 0.0:
            raise ValueError("burn amount must be positive")
        slots = [s for s in self._escrow.get(identity, [])
                 if s.matures_at > int(current_epoch)]
        total = sum(s.amount for s in slots)
        if total < amount:
            raise EscrowNotFound(
                f"{identity} has {total} unexpired escrow, cannot "
                f"burn {amount}")
        remaining = float(amount)
        kept: list[EscrowSlot] = []
        for slot in sorted(slots, key=lambda s: s.matures_at):
            if remaining <= 0.0:
                kept.append(slot)
                continue
            if slot.amount <= remaining:
                remaining -= slot.amount
            else:
                kept.append(EscrowSlot(amount=slot.amount - remaining,
                                       matures_at=slot.matures_at))
                remaining = 0.0
        self._escrow[identity] = kept
        self._burnt[identity] = self._burnt.get(identity, 0.0) \
            + float(amount) - remaining
        return float(amount) - remaining

    def total_burnt(self, identity: str) -> float:
        return self._burnt.get(str(identity), 0.0)
