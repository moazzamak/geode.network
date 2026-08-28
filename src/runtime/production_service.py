"""Replicated bundle serving and crash-recoverable promotion coordination."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.runtime.model_bundle import LocalModelBundleStore


Predictor = Callable[[np.ndarray], np.ndarray]
BundleLoader = Callable[[str], Predictor]


@dataclass(frozen=True)
class ShadowServiceObservation:
    production_bundle_id: str
    candidate_bundle_id: str
    agreement: float
    production_latency_seconds: float
    candidate_latency_seconds: float
    candidate_controls_outputs: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "production_bundle_id": self.production_bundle_id,
            "candidate_bundle_id": self.candidate_bundle_id,
            "agreement": self.agreement,
            "production_latency_seconds": self.production_latency_seconds,
            "candidate_latency_seconds": self.candidate_latency_seconds,
            "candidate_controls_outputs": self.candidate_controls_outputs,
        }


class ReplicatedBundleService:
    def __init__(
        self,
        store: LocalModelBundleStore,
        loader: BundleLoader,
        *,
        replica_count: int,
    ) -> None:
        if replica_count < 2:
            raise ValueError("production rehearsal requires at least two replicas")
        self.store = store
        self.loader = loader
        self.replica_count = replica_count
        self._replicas: list[tuple[str, Predictor]] = []
        self._next_replica = 0
        self._latencies: list[float] = []
        self.reload_current()

    def reload_current(self) -> None:
        current = self.store.current()
        if current is None:
            raise ValueError("replicated service requires an active bundle")
        self._replicas = [
            (current.bundle_id, self.loader(current.bundle_id))
            for _ in range(self.replica_count)
        ]
        self._next_replica = 0

    @property
    def serving_bundle_ids(self) -> tuple[str, ...]:
        return tuple(bundle_id for bundle_id, _ in self._replicas)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        _, predictor = self._replicas[self._next_replica]
        self._next_replica = (self._next_replica + 1) % self.replica_count
        started = time.perf_counter()
        result = np.asarray(predictor(np.asarray(inputs)))
        self._latencies.append(time.perf_counter() - started)
        return result

    def shadow(self, candidate_bundle_id: str, inputs: np.ndarray) -> ShadowServiceObservation:
        production_bundle_id, production = self._replicas[0]
        candidate = self.loader(candidate_bundle_id)
        values = np.asarray(inputs)
        started = time.perf_counter()
        authoritative = np.asarray(production(values))
        production_seconds = time.perf_counter() - started
        started = time.perf_counter()
        shadow = np.asarray(candidate(values))
        candidate_seconds = time.perf_counter() - started
        if authoritative.shape != shadow.shape:
            agreement = 0.0
        else:
            agreement = float(np.mean(authoritative == shadow))
        return ShadowServiceObservation(
            production_bundle_id,
            candidate_bundle_id,
            agreement,
            production_seconds,
            candidate_seconds,
        )

    def telemetry(self) -> dict[str, Any]:
        latencies = np.asarray(self._latencies, dtype=np.float64)
        return {
            "replicas": self.replica_count,
            "requests": len(latencies),
            "serving_bundle_ids": list(self.serving_bundle_ids),
            "latency_p95_seconds": (
                float(np.percentile(latencies, 95)) if len(latencies) else 0.0
            ),
        }


class ProductionPromotionCoordinator:
    def __init__(self, store: LocalModelBundleStore) -> None:
        self.store = store
        self.journal_path = store.root / "promotion_journal.json"

    def _write(self, payload: dict[str, Any]) -> None:
        self.store.root.mkdir(parents=True, exist_ok=True)
        temporary = self.journal_path.with_suffix(".partial")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, self.journal_path)

    def state(self) -> dict[str, Any]:
        if not self.journal_path.exists():
            current = self.store.current()
            return {
                "phase": "stable",
                "production_bundle_id": None if current is None else current.bundle_id,
            }
        return json.loads(self.journal_path.read_text(encoding="utf-8"))

    def begin_promotion(self, candidate_bundle_id: str) -> dict[str, Any]:
        current = self.store.current()
        if current is None:
            raise ValueError("promotion requires an active production bundle")
        candidate = self.store.load(candidate_bundle_id)
        if candidate.parent_bundle_id != current.bundle_id:
            raise ValueError("canary must be a direct child of production")
        state = {
            "phase": "promotion_in_progress",
            "previous_bundle_id": current.bundle_id,
            "candidate_bundle_id": candidate.bundle_id,
        }
        self._write(state)
        self.store.activate(candidate.bundle_id)
        return state

    def complete_promotion(self) -> dict[str, Any]:
        state = self.state()
        if state.get("phase") != "promotion_in_progress":
            raise ValueError("no promotion is in progress")
        current = self.store.current()
        if current is None or current.bundle_id != state["candidate_bundle_id"]:
            raise ValueError("active bundle does not match promotion candidate")
        stable = {"phase": "stable", "production_bundle_id": current.bundle_id}
        self._write(stable)
        return stable

    def recover(self) -> dict[str, Any]:
        state = self.state()
        if state.get("phase") != "promotion_in_progress":
            return state
        restored = self.store.activate(state["previous_bundle_id"])
        stable = {
            "phase": "stable",
            "production_bundle_id": restored.bundle_id,
            "recovered_candidate_bundle_id": state["candidate_bundle_id"],
        }
        self._write(stable)
        return stable

    def promote_or_rollback(
        self,
        candidate_bundle_id: str,
        *,
        canary_gate_passed: bool,
    ) -> dict[str, Any]:
        self.begin_promotion(candidate_bundle_id)
        return self.complete_promotion() if canary_gate_passed else self.recover()