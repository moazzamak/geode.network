"""GEODE API metrics (v25 M258 cell 3) — the live-operations view.

A deterministic collector over RECORDED durations: request counts
and p50/p99 percentiles (linear interpolation over the sorted
window). No RNG; the output depends only on the recorded data. The
API service exposes it at /metrics; the M240 one-off latency
measurement becomes a live quantity through this collector.
"""
from __future__ import annotations

from typing import Any


class MetricsCollector:
    """Counts + percentile summaries from recorded durations (ms)."""

    def __init__(self, window: int = 4096):
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = int(window)
        self._durations: list[float] = []
        self._count: int = 0

    def record(self, duration_ms: float) -> None:
        if duration_ms < 0.0:
            raise ValueError("duration must be non-negative")
        self._durations.append(float(duration_ms))
        self._count += 1
        if len(self._durations) > self.window:
            self._durations = self._durations[-self.window:]

    def _percentile(self, p: float) -> float | None:
        if not self._durations:
            return None
        xs = sorted(self._durations)
        rank = (len(xs) - 1) * p
        lo = int(rank)
        hi = min(lo + 1, len(xs) - 1)
        frac = rank - lo
        return xs[lo] * (1.0 - frac) + xs[hi] * frac

    def summary(self) -> dict[str, Any]:
        return {
            "count": self._count,
            "windowed": len(self._durations),
            "p50_ms": self._percentile(0.5),
            "p99_ms": self._percentile(0.99),
        }
