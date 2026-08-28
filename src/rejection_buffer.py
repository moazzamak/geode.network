from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from src.open_set import OpenSetPrediction


@dataclass(frozen=True)
class RejectionRecord:
    record_id: int
    embedding: tuple[float, ...]
    timestamp: float
    window_id: int
    source_model_signature: str
    support_profile_version: str
    novelty_score: float
    decision_margin: float
    nearest_candidates: tuple[Any, ...]
    source_sample_id: Any | None = None
    class_order_version: str = ""
    threshold_lineage_hash: str = ""
    acceptance_policy_hash: str = ""

    def __post_init__(self) -> None:
        if self.record_id < 0 or self.window_id < 0:
            raise ValueError("record_id and window_id must be non-negative.")
        if not self.embedding or not all(map(math.isfinite, self.embedding)):
            raise ValueError("embedding must be non-empty and finite.")
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite.")
        if not self.source_model_signature or not self.support_profile_version:
            raise ValueError("source model and profile versions must be non-empty.")
        if not math.isfinite(self.novelty_score) or self.decision_margin < 0.0:
            raise ValueError("rejection novelty evidence must be finite and non-negative.")
        lineage = (
            self.class_order_version,
            self.threshold_lineage_hash,
            self.acceptance_policy_hash,
        )
        if any(lineage) and not all(lineage):
            raise ValueError("rejection lineage must be supplied as a complete set.")
        for name in ("threshold_lineage_hash", "acceptance_policy_hash"):
            value = getattr(self, name)
            if value and (
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest.")


class RejectionBuffer:
    """Bounded FIFO store for rejected embeddings; it never inspects labels."""

    def __init__(self, max_records: int, max_embedding_dimensions: int) -> None:
        if max_records <= 0 or max_embedding_dimensions <= 0:
            raise ValueError("buffer limits must be positive.")
        self.max_records = max_records
        self.max_embedding_dimensions = max_embedding_dimensions
        self._records: deque[RejectionRecord] = deque()
        self._next_record_id = 0
        self.evicted_records = 0

    def __len__(self) -> int:
        return len(self._records)

    def append_rejection(
        self,
        embedding: np.ndarray,
        *,
        timestamp: float,
        window_id: int,
        prediction: OpenSetPrediction,
        nearest_candidates: Iterable[Any] = (),
        source_sample_id: Any | None = None,
    ) -> RejectionRecord:
        if prediction.accepted:
            raise ValueError("Accepted predictions cannot enter the rejection buffer.")
        vector = np.asarray(embedding, dtype=np.float64)
        if vector.ndim != 1 or not len(vector):
            raise ValueError("embedding must be a non-empty vector.")
        if len(vector) > self.max_embedding_dimensions:
            raise ValueError("embedding exceeds the configured dimension cap.")
        candidates = tuple(nearest_candidates)
        if len(candidates) != len(set(candidates)):
            raise ValueError("nearest_candidates must be unique and ordered.")
        record = RejectionRecord(
            record_id=self._next_record_id,
            embedding=tuple(float(value) for value in vector),
            timestamp=float(timestamp),
            window_id=window_id,
            source_model_signature=prediction.candidate_model_signature,
            support_profile_version=prediction.support_profile_version,
            novelty_score=prediction.calibrated_novelty_score,
            decision_margin=prediction.decision_margin,
            nearest_candidates=candidates,
            source_sample_id=source_sample_id,
        )
        self._next_record_id += 1
        if len(self._records) == self.max_records:
            self._records.popleft()
            self.evicted_records += 1
        self._records.append(record)
        return record

    def snapshot(self) -> tuple[RejectionRecord, ...]:
        return tuple(self._records)

    def records_in_windows(self, window_ids: Iterable[int]) -> tuple[RejectionRecord, ...]:
        selected = set(window_ids)
        return tuple(record for record in self._records if record.window_id in selected)

    @property
    def windows_present(self) -> tuple[int, ...]:
        return tuple(sorted({record.window_id for record in self._records}))