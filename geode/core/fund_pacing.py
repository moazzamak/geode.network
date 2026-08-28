"""M325 liveness amendment — dev-fund pacing with default-release.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` §8.26
point 6 (27 Aug 2026, user decision): an absence of signatures can
never block fund releases indefinitely. The rule:

- A scheduled release EXECUTES by default when its window closes.
- Blocking requires an AFFIRMATIVE negative vote carrying the
  registered majority of the sampled weight (the M328 machinery:
  snapshot, diversity floor, secret ballots, responder minimum).
- A positive vote carrying the majority releases immediately.
- Silence releases. Only an affirmative negative holds.

The quorum predicate is injected (dependency injection keeps the
M216 direction table: core may not import privacy, where the
Pedersen vote machinery lives; the api layer wires them).

The zakat end state is UNTOUCHED: after the trigger, disbursement
is mechanical with no pause path (M325-G4/G5). The liveness rule
applies to bootstrap pacing only — the zakat mode has no hold
function and no vote path of any kind.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# Majority: strictly more than half of the sampled weight.
MAJORITY = 0.5


@dataclass
class ScheduledRelease:
    release_id: str
    amount: float
    scheduled_epoch: int
    window_epochs: int = 1
    held: bool = False
    hold_until_epoch: int = -1


@dataclass
class VoteSet:
    """One governance vote over a release: identity -> weight, and
    the vote (True = release / positive, False = hold / negative)."""
    weights: dict[str, float]
    votes: dict[str, bool]
    responders: int
    pool_size: int


@dataclass
class ReleaseSchedule:
    """The pacing state machine. ``ratify`` is the injected quorum
    predicate: ratify(support, total, identities, pool, responders)
    -> (bool, reason)."""
    ratify: Callable[..., tuple[bool, str]]
    hold_epochs: int = 1
    releases: list[ScheduledRelease] = field(default_factory=list)
    executed: list[dict] = field(default_factory=list)
    blocked: list[dict] = field(default_factory=list)

    def schedule(self, release_id: str, amount: float,
                 scheduled_epoch: int,
                 window_epochs: int = 1) -> None:
        self.releases.append(ScheduledRelease(release_id, amount,
                                              scheduled_epoch,
                                              window_epochs))

    def advance(self, epoch: int, votes: dict[str, VoteSet]) -> None:
        """Advance to ``epoch``. For every release whose window
        closes at or before this epoch: a carried negative vote
        holds it; otherwise it EXECUTES (a carried positive vote
        executes immediately — handled by the same default).
        """
        remaining = []
        for rel in self.releases:
            if rel.held:
                if epoch >= rel.hold_until_epoch:
                    rel.held = False
                    # a de-held release re-enters the schedule with a
                    # FRESH window — it is never default-released on
                    # the same advance that ends its hold
                    rel.scheduled_epoch = epoch
                else:
                    remaining.append(rel)
                    continue
            vs = votes.get(rel.release_id)
            support, total = 0.0, 0.0
            if vs is not None:
                for ident, w in vs.weights.items():
                    total += w
                    if vs.votes.get(ident):
                        support += w
            ok, reason = self._quorum(support, total, vs,
                                      rel.release_id)
            window_open = epoch <= rel.scheduled_epoch + rel.window_epochs - 1
            if not window_open and not ok:
                # liveness default: silence releases
                self.executed.append({
                    "release_id": rel.release_id,
                    "amount": rel.amount, "epoch": epoch,
                    "path": "window_closed_default"})
                continue
            if ok:
                self.executed.append({
                    "release_id": rel.release_id,
                    "amount": rel.amount, "epoch": epoch,
                    "path": "quorum_release", "reason": reason})
                continue
            # the window is still open and no quorum either way:
            # wait; an AFFIRMATIVE negative can still carry before
            # the window closes
            remaining.append(rel)
        self.releases = remaining

    def hold(self, epoch: int, release_id: str) -> bool:
        """An affirmative negative vote carried: hold the release
        for the registered hold window (it re-enters the schedule
        afterwards — a hold is never a cancel)."""
        for rel in self.releases:
            if rel.release_id == release_id and not rel.held:
                rel.held = True
                rel.hold_until_epoch = epoch + self.hold_epochs
                self.blocked.append({"release_id": release_id,
                                     "epoch": epoch})
                return True
        return False

    def _quorum(self, support: float, total: float, vs: VoteSet | None,
                release_id: str) -> tuple[bool, str]:
        if total <= 0.0:
            return False, "no_sampled_weight"
        if vs is None:
            return False, "no_vote"
        if support / total <= MAJORITY:
            return False, "below_majority"
        identities = [i for i, v in vs.votes.items() if v]
        ok, reason = self.ratify(support, total, identities,
                                 vs.pool_size, vs.responders)
        return ok, reason


@dataclass
class ZakatDisbursement:
    """The post-trigger end state: mechanical, no pause path, no
    vote path (M325-G4/G5). Exists so the liveness rule cannot be
    misread as applying after the trigger."""
    recipients: list[tuple[str, float]] = field(default_factory=list)

    def disburse(self) -> list[dict]:
        return [{"recipient": r, "amount": a}
                for r, a in self.recipients]
