"""Append-only local metric ledger for resumable GEODE runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from src.runtime.schemas import MetricEvent


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _logical_key(event: MetricEvent) -> tuple[Any, ...]:
    return (
        event.attempt_id,
        event.stage_name,
        event.epoch,
        event.global_step,
        event.metric_name,
        event.split,
        event.namespace,
    )


class MetricLedger:
    """Write validated metric events without overwriting prior history."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def read_events(self) -> list[MetricEvent]:
        if not self.path.exists():
            return []
        events: list[MetricEvent] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    events.append(MetricEvent.from_dict(payload))
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid metric event at {self.path}:{line_number}"
                    ) from error
        return events

    def append(self, event: MetricEvent) -> bool:
        """Append an event, returning False only for an identical event ID retry."""
        if not isinstance(event, MetricEvent):
            raise TypeError("event must be a MetricEvent")
        with self._lock:
            existing = self.read_events()
            for prior in existing:
                if prior.event_id == event.event_id:
                    if prior == event:
                        return False
                    raise ValueError(f"event_id {event.event_id!r} has conflicting content")
                if _logical_key(prior) == _logical_key(event):
                    raise ValueError("metric logical key already exists with another event_id")

            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(_canonical_json(event.to_dict()) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        return True