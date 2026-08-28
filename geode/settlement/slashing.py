"""GEODE slashing ledger (whitepaper-aligned, 24 Aug 2026) — disputes
are OPEN, DEPOSITED, and decided by deterministic replay; penalties
are GRADED BURNS (L0-L3), never recovered by anyone.

There is no adjudicator role. Any party may file a dispute with a
deposit and an evidence reference; the computation (here the injected
``verify_fn``, standing in for the replay of sealed data) decides
guilt. Slashed amounts move to the BURNED bucket — unreachable by any
claim path; nobody gains from a slash.

Rules (registered):

- both sides submit evidence for the same measurement;
- the challenger's evidence verifies and the accused's does not ->
  BURN the accused's deposited stake (level 1: the unvested promise
  in the contract's terms; higher levels burn more, up to level 3);
- both verify -> the challenger's deposit BURNS (false accusation);
- neither verifies -> UNRESOLVED (quarantined, never applied).

Deterministic; hash-chained on AppendOnlyLedger; no wall clocks.
"""
from __future__ import annotations

from typing import Any, Callable

from geode.core.ledger import AppendOnlyLedger


class SlashLedger:
    """Open deposited disputes and graded burns, append-only."""

    def __init__(self) -> None:
        self._ledger = AppendOnlyLedger()
        self._deposit: dict[str, float] = {}
        self._burned: dict[str, float] = {}

    def deposit(self, verifier: str, amount: float) -> float:
        """A dispute deposit (replaces the retired stake; the
        whitepaper's slash collateral is the unvested promise, the
        deposit here prices frivolous filings)."""
        if amount < 0.0:
            raise ValueError("deposit must be non-negative")
        self._deposit[str(verifier)] = self._deposit.get(str(verifier),
                                                         0.0) + amount
        self._ledger.append({
            "kind": "dispute_deposit", "key": f"deposit:{verifier}:"
            f"{len(self._ledger.to_dict()['records'])}",
            "verifier": str(verifier), "amount": amount})
        return self._deposit[str(verifier)]

    def stake_of(self, verifier: str) -> float:
        """The party's current deposited amount (kept as the
        registered accessor name)."""
        return self._deposit.get(str(verifier), 0.0)

    def burned_of(self, verifier: str) -> float:
        return self._burned.get(str(verifier), 0.0)

    @property
    def burned_total(self) -> float:
        return sum(self._burned.values())

    def dispute(self, dispute_id: str, accused: str, challenger: str,
                measurement_ref: str, accused_proof: Any,
                challenger_proof: Any,
                verify_fn: Callable[[Any, Any], bool],
                level: int = 1,
                evidence_hash: str = "",
                ) -> dict[str, Any]:
        """One deposited dispute; ``verify_fn(proof, reference)`` is
        the injected verifier (zk proof check or chain adapter),
        standing in for the replay of sealed data. ``level`` is the
        graded-ladder level (1-3; the amount burned grows with the
        level in the contract)."""
        if level < 1 or level > 3:
            raise ValueError("level must be in 1..3 (L0 is no slash)")
        accused_ok = bool(verify_fn(accused_proof, measurement_ref))
        challenger_ok = bool(verify_fn(challenger_proof,
                                       measurement_ref))
        outcome: dict[str, Any] = {"dispute_id": str(dispute_id),
                                   "accused": str(accused),
                                   "challenger": str(challenger),
                                   "level": int(level),
                                   "evidence_hash": str(evidence_hash)}
        if challenger_ok and not accused_ok:
            # the accused's claim failed verification: burn
            outcome.update({"verdict": "slash_accused",
                            "slashed": self.stake_of(accused)})
            self._burned[str(accused)] = \
                self._burned.get(str(accused), 0.0) + self.stake_of(accused)
            self._deposit[str(accused)] = 0.0
        elif accused_ok and challenger_ok:
            # false accusation: the challenger pays
            outcome.update({"verdict": "slash_challenger",
                            "slashed": self.stake_of(challenger)})
            self._burned[str(challenger)] = \
                self._burned.get(str(challenger), 0.0) \
                + self.stake_of(challenger)
            self._deposit[str(challenger)] = 0.0
        else:
            outcome.update({"verdict": "unresolved", "slashed": 0.0})
        self._ledger.append({
            "kind": "dispute", "key": f"dispute:{dispute_id}",
            **outcome})
        return outcome

    def verify(self) -> dict[str, Any]:
        return self._ledger.verify()

    def tip(self) -> str:
        return self._ledger.tip()

    def to_dict(self) -> dict[str, Any]:
        return {"deposits": dict(self._deposit),
                "burned": dict(self._burned),
                "ledger": self._ledger.to_dict()}
