"""GEODE behavioural diffing gate (v25 M250) — goal-content
integrity.

Arm OUTPUT behaviour is snapshotted as vectors; only quorum-admitted
snapshots (k-of-n, the M245 backbone) become baselines. An append-
only arm update whose behaviour diverges from the latest admitted
baseline beyond the drift bound is GATED (rejected); admitted
updates become the new baseline. A first update with no baseline is
admitted (it establishes the baseline — gating starts from the
second update).

Deterministic: no RNG, no wall clocks; drift is 1 - cosine (the
M242 metric). Structure-only: behaviour-vector extraction from real
arm outputs is a future data artifact.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from geode.core.fingerprint import DriftGate


class BehaviorDiffGate:
    """Quorum-admitted behaviour baselines + drift gating."""

    def __init__(self, drift_bound: float = 0.2, k_of_n: int = 2):
        self.drift_bound = float(drift_bound)
        self.k_of_n = int(k_of_n)
        self._baselines: dict[str, tuple[list[float], int]] = {}
        self._drift = DriftGate(drift_bound=self.drift_bound,
                                staleness_window=0)

    def record_snapshot(self, arm_id: str, vector: Sequence[float],
                        attestations: frozenset[str],
                        ledger_index: int) -> bool:
        """Record a snapshot; only a quorum-admitted one becomes (or
        replaces) the baseline. Returns True when it did."""
        if len(attestations) < self.k_of_n:
            return False  # below quorum: quarantined, never a baseline
        self._baselines[str(arm_id)] = (list(vector),
                                        int(ledger_index))
        return True

    def latest(self, arm_id: str) -> tuple[list[float], int] | None:
        return self._baselines.get(str(arm_id))

    def admits_update(self, arm_id: str, new_vector: Sequence[float],
                      ledger_index: int,
                      bound: float | None = None) -> dict[str, Any]:
        """The append-only update decision (deterministic)."""
        base = self.latest(arm_id)
        if base is None:
            # first update establishes the baseline
            self._baselines[str(arm_id)] = (list(new_vector),
                                            int(ledger_index))
            return {"admitted": True, "reason": "baseline_established",
                    "drift": None}
        baseline, _ = base
        b = self.drift_bound if bound is None else float(bound)
        d = self._drift.drift(new_vector, baseline)
        if d > b:
            return {"admitted": False,
                    "reason": f"behavior_drift ({d:.6f} > {b:.6f})",
                    "drift": d}
        # admitted: the update becomes the new baseline
        self._baselines[str(arm_id)] = (list(new_vector),
                                        int(ledger_index))
        return {"admitted": True, "reason": "admitted", "drift": d}
