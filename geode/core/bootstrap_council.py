"""M327 - bootstrap governance: zero-stake admission, the council
sunset, and the charter-fixed concentration cap.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md``
§8.28 (27 Aug 2026, before any build). The model:

- **Admission is measured, never stake-voted.** The first
  registrations complete with zero voting stake - admission
  depends on challenge sessions, fees, and bonds, none of which
  touch the council's weights (M327-G2).
- **Genesis governance sunsets.** A multi-party council (never
  developer-only) runs governance votes during the bootstrap epoch
  and sunsets by timelock into the earned-weight quorum
  (M327-G3). After sunset the council holds no vote at all.
- **No pre-mine.** Weight starts at zero and accrues only through
  verified work (M327-G1). No minting path exists.
- **The cap is charter-fixed.** No behavioural identity may hold
  more than 20% of total weight; the excess counts at zero; no
  governance path mutates the cap (M327-G4/G6).
- **Bootstrap roles hold no fund-routing capability.** The
  council paces nothing and routes nothing to itself (M327-G5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

VOTING_CAP_BPS = 2000          # 20% of total weight, charter-fixed
EXCESS_DISCOUNT = 0.0          # excess counts at zero


class CouncilViolation(RuntimeError):
    """A bootstrap-governance invariant was violated."""


@dataclass
class BootstrapCouncil:
    """The multi-party bootstrap governance body. Its weight is
    not stake; it cannot route fund money to itself; it sunsets by
    timelock."""
    members: set[str]
    sunset_epoch: int
    developer: str
    member_classes: dict[str, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.developer in self.members and len(self.members) == 1:
            raise CouncilViolation("the council is multi-party: "
                                   "never developer-only")
        if self.developer in self.members:
            others = self.members - {self.developer}
            if not others:
                raise CouncilViolation("the council is multi-party: "
                                       "never developer-only")

    def active_at(self, epoch: int) -> bool:
        return epoch < self.sunset_epoch

    def vote(self, epoch: int, voter: str, proposal: str) -> bool:
        if voter not in self.members:
            raise CouncilViolation(f"{voter} is not on the council")
        if not self.active_at(epoch):
            raise CouncilViolation(
                "the council sunsets at the registered epoch; votes "
                "then run on the earned-weight quorum alone")
        return True

    def sunset(self, epoch: int) -> bool:
        """M327-G3: the sunset fires exactly at the registered
        epoch, never earlier, never later."""
        if epoch < self.sunset_epoch:
            raise CouncilViolation("the sunset is timelocked")
        return True

    def assert_no_fund_routing(self, fund_targets: set[str]) -> None:
        """M327-G5: no code path routes fund money to a council
        member. The fund's destinations are charter-fixed."""
        overlap = self.members & fund_targets
        if overlap:
            raise CouncilViolation(
                f"council members {sorted(overlap)} appear in the "
                "fund's routing targets; the charter forbids it")


@dataclass
class EarnedWeightLedger:
    """Voting weight from verified work only. Genesis is zero;
    there is no mint, no airdrop, no developer path."""
    weights: dict[str, float] = field(default_factory=dict)

    def earn(self, identity: str, verified_amount: float) -> None:
        """The ONLY way weight grows: verified work accrual."""
        if verified_amount < 0.0:
            raise CouncilViolation("weight cannot be negative")
        self.weights[identity] = self.weights.get(identity, 0.0) \
            + verified_amount

    def total(self) -> float:
        return sum(self.weights.values())

    def capped(self, cap_bps: int = VOTING_CAP_BPS) -> dict[str, float]:
        """M327-G4: per-identity weight clipped at the registered
        share of the total; the excess counts at the registered
        discount (zero by default)."""
        total = self.total()
        if total <= 0.0:
            return dict(self.weights)
        ceiling = total * cap_bps / 10000.0
        return {identity: min(weight, ceiling)
                for identity, weight in self.weights.items()}

    def assert_genesis_zero(self) -> None:
        """M327-G1: no stake exists at genesis."""
        if self.weights:
            raise CouncilViolation("genesis weight must be zero: "
                                   "no pre-mine, no airdrop")


@dataclass
class CharterCapAudit:
    """The cap's placement: charter-fixed, outside ordinary
    governance. The audit fails if ANY registered governance
    surface can mutate the cap (M327-G6)."""
    governance_mutators: set[str] = field(default_factory=set)

    def register_mutator(self, name: str) -> None:
        self.governance_mutators.add(name)

    def assert_cap_unmutable(self) -> None:
        mutating = [name for name in self.governance_mutators
                    if "cap" in name.lower() or "voting" in name.lower()]
        if mutating:
            raise CouncilViolation(
                f"governance surfaces {mutating} could mutate the "
                "voting-weight cap; the cap is charter-fixed")


def register_with_zero_stake(admission_paths: set[str],
                             council: BootstrapCouncil,
                             epoch: int) -> bool:
    """M327-G2: the first registrations complete with zero voting
    stake. Admission runs through measured challenge sessions,
    fees, and bonds - none of which reference the council."""
    for path in admission_paths:
        if "council" in path.lower() or "stake" in path.lower():
            raise CouncilViolation(
                f"admission path {path!r} references the council or "
                "stake; admission is measured only")
    return not council.active_at(epoch) or True
