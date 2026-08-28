"""GEODE quorum takedown (v25 M294) — network-majority delisting of
socially destructive artifacts.

Registered in ``analysis/v25_m294_quorum_takedown_spec.md`` and the
plan (25 Aug 2026, M294 ACTIVE) BEFORE building. The slash ladder
delists for REPLAY-GATED offenses (the math decides); takedown is a
JUDGMENT, so it is the system's one discretionary power and is
contained accordingly: quorum-gated, recorded, permanent, distinct
from slashing, and identity-free (C1 — an artifact is judged, never
a person).

Deterministic: no RNG, no wall clocks; sampling orders the validator
pool by a content hash, verdicts are counted votes. The librarian
FILES a verdict; it never decides one.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


class TakedownError(RuntimeError):
    """Raised on malformed takedown operations."""


SUPPORT = "support"
OPPOSE = "oppose"


@dataclass
class TakedownProposal:
    proposal_id: str
    artifact_id: str
    evidence_refs: tuple[str, ...]
    proposer: str
    deposit: float
    votes: dict[str, str] = field(default_factory=dict)
    verdict: str | None = None  # "ratified" | "rejected" | None


class QuorumTakedown:
    """Quorum-gated, permanent artifact delisting.

    Voter eligibility (M294 amendments, 25 Aug): a validator is
    sampleable only after the activation window (A epochs from
    registration), while above the activity floor (responded in at
    least half of its sampled rounds), and only with a responded
    round within the trailing recency window (W epochs) — tenure
    alone is not enough; voting power is earned by RECENT work.
    Its vote carries tenure weight min(1, tenure/T). A flood of
    fresh registrations therefore contributes ~zero quorum weight.
    """

    def __init__(self, k: int = 9, quorum_num: int = 2,
                 quorum_den: int = 3, min_responders: int = 3,
                 librarian: str = "librarian",
                 activation_epochs: int = 2,
                 tenure_full_epochs: int = 4,
                 recency_epochs: int = 2):
        if k < 1:
            raise ValueError("k must be positive")
        if quorum_num <= 0 or quorum_den <= quorum_num:
            raise ValueError("quorum must be a proper fraction")
        if min_responders < 1:
            raise ValueError("min_responders must be positive")
        if activation_epochs < 0:
            raise ValueError("activation_epochs must be >= 0")
        if tenure_full_epochs <= 0:
            raise ValueError("tenure_full_epochs must be positive")
        if recency_epochs <= 0:
            raise ValueError("recency_epochs must be positive")
        self.k = int(k)
        self.quorum_num = int(quorum_num)
        self.quorum_den = int(quorum_den)
        self.min_responders = int(min_responders)
        self.librarian = str(librarian)
        self.activation_epochs = int(activation_epochs)
        self.tenure_full_epochs = int(tenure_full_epochs)
        self.recency_epochs = int(recency_epochs)
        self._proposals: dict[str, TakedownProposal] = {}
        self._delisted: set[str] = set()
        self._activation: dict[str, int] = {}   # validator -> epoch
        self._sampled_rounds: dict[str, int] = {}
        self._responded_rounds: dict[str, int] = {}
        self._last_responded: dict[str, int] = {}

    # --------------------------------------------------------- eligibility
    def note_registration(self, validator: str, epoch: int) -> int:
        """A validator becomes sampleable at epoch + A (the
        activation window). Returns the activation epoch."""
        activation = int(epoch) + self.activation_epochs
        self._activation[str(validator)] = activation
        self._sampled_rounds.setdefault(str(validator), 0)
        self._responded_rounds.setdefault(str(validator), 0)
        return activation

    def record_round(self, validator: str, epoch: int,
                     responded: bool) -> None:
        """One sampled round at ``epoch``: the validator responded or
        stayed silent. The activity floor is half of sampled rounds;
        the recency gate needs a responded round inside the trailing
        window."""
        v = str(validator)
        if v not in self._activation:
            raise TakedownError(f"unregistered validator {v!r}")
        self._sampled_rounds[v] += 1
        if responded:
            self._responded_rounds[v] += 1
            self._last_responded[v] = int(epoch)

    def is_active(self, validator: str) -> bool:
        """Above the activity floor: responded in at least half of
        sampled rounds; zero rounds counts as active."""
        v = str(validator)
        if v not in self._activation:
            return False
        return 2 * self._responded_rounds.get(v, 0) >= \
            self._sampled_rounds.get(v, 0)

    def is_recent(self, validator: str, epoch: int) -> bool:
        """Performed the role within the trailing recency window:
        a responded round exists and is strictly inside W epochs.
        Tenure alone never qualifies."""
        last = self._last_responded.get(str(validator))
        return last is not None \
            and int(epoch) - last < self.recency_epochs

    def is_eligible(self, validator: str, epoch: int) -> bool:
        activation = self._activation.get(str(validator))
        return activation is not None and int(epoch) >= activation \
            and self.is_active(validator) \
            and self.is_recent(validator, epoch)

    def voter_weight(self, validator: str, epoch: int) -> float:
        """Tenure weight min(1, tenure/T): zero at activation, full
        at T epochs of tenure. Ineligible validators weigh zero."""
        if not self.is_eligible(validator, epoch):
            return 0.0
        tenure = int(epoch) - self._activation[str(validator)]
        return min(1.0, tenure / self.tenure_full_epochs)

    # ------------------------------------------------------------ sampling
    def sampled_set(self, epoch: int, artifact_id: str,
                    pool: list[str]) -> list[str]:
        """The first k ELIGIBLE entries of the pool ordered by
        hash(epoch, artifactId, validator) — no one chooses their
        judges, and fresh/dormant registrations are never sampled."""
        eligible = [v for v in pool if self.is_eligible(v, epoch)]
        keyed = sorted(
            ((self._sample_hash(epoch, artifact_id, v), v)
             for v in eligible),
            key=lambda pair: (pair[0], pair[1]))
        return [v for _, v in keyed[:self.k]]

    @staticmethod
    def _sample_hash(epoch: int, artifact_id: str,
                     validator: str) -> str:
        raw = f"{epoch}:{artifact_id}:{validator}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    # ------------------------------------------------------------ proposal
    def propose(self, artifact_id: str, evidence_refs: list[str],
                proposer: str, deposit: float,
                proposal_id: str | None = None) -> TakedownProposal:
        if not evidence_refs:
            raise TakedownError("a proposal must cite evidence")
        if deposit <= 0.0:
            raise TakedownError("a proposal requires a positive deposit")
        pid = str(proposal_id or
                  self._sample_hash(-1, artifact_id,
                                    ":".join(evidence_refs))[:16])
        if pid in self._proposals:
            raise TakedownError(f"duplicate proposal {pid!r}")
        proposal = TakedownProposal(
            proposal_id=pid,
            artifact_id=str(artifact_id),
            evidence_refs=tuple(str(r) for r in evidence_refs),
            proposer=str(proposer),
            deposit=float(deposit))
        self._proposals[pid] = proposal
        return proposal

    # ---------------------------------------------------------------- vote
    def vote(self, proposal_id: str, validator: str, choice: str,
             epoch: int, artifact_id: str,
             pool: list[str]) -> None:
        """One vote per validator per proposal; the first counts
        (G2). Votes from outside the sampled set are ignored."""
        if choice not in (SUPPORT, OPPOSE):
            raise TakedownError(f"bad choice {choice!r}")
        if validator not in self.sampled_set(epoch, artifact_id, pool):
            return
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise TakedownError(f"unknown proposal {proposal_id!r}")
        if validator in proposal.votes:
            return  # duplicates count once: the first vote stands
        proposal.votes[validator] = choice

    # -------------------------------------------------------------- verdict
    def verdict(self, proposal_id: str, epoch: int, artifact_id: str,
                pool: list[str]) -> dict[str, Any]:
        """G1 (amended): ratified iff support_weight >= 2/3 of the
        total sampled weight AND responders >= min_responders AND the
        total weight is at least 1.0 (fail-closed on cold starts).
        Anything else fails closed."""
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise TakedownError(f"unknown proposal {proposal_id!r}")
        sampled = self.sampled_set(epoch, artifact_id, pool)
        weights = {v: self.voter_weight(v, epoch) for v in sampled}
        total_weight = sum(weights.values())
        responders = [v for v in sampled if v in proposal.votes]
        support_weight = sum(weights.get(v, 0.0) for v in responders
                             if proposal.votes[v] == SUPPORT)
        ratified = (total_weight >= 1.0
                    and support_weight >= self.quorum_num * total_weight
                    / self.quorum_den
                    and len(responders) >= self.min_responders)
        proposal.verdict = "ratified" if ratified else "rejected"
        return {
            "proposal_id": proposal_id,
            "sampled": sampled,
            "total_weight": total_weight,
            "support_weight": support_weight,
            "need_weight": self.quorum_num * total_weight
            / self.quorum_den,
            "responders": len(responders),
            "support": sum(1 for v in responders
                           if proposal.votes[v] == SUPPORT),
            "ratified": ratified,
            "verdict": proposal.verdict,
            "deposit_returned": ratified,
        }

    # -------------------------------------------------------------- filing
    def file_delist(self, artifact_id: str, quorum_record_hash: str,
                    by: str) -> None:
        """The librarian FILES a ratified verdict (G5 enforcement on
        the contract side carries the same record hash). Permanent —
        no un-delist path exists (G3)."""
        if by != self.librarian:
            raise TakedownError("only the librarian files delists")
        self._delisted.add(str(artifact_id))

    def is_delisted(self, artifact_id: str) -> bool:
        return str(artifact_id) in self._delisted

    def delisted(self) -> frozenset[str]:
        return frozenset(self._delisted)
