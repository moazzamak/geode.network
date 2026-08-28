"""R-A7a / R-A7c - the coverage-adjusted axis metric.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md``
§8.36 (27 Aug 2026, before any build). The A7 exploit: a head that
refuses most classes reports a high raw accuracy on the subset it
keeps, and abstention inflates the axis score instead of trading
against it. The repair:

- the axis metric is coverage-adjusted: ``accuracy x coverage``
  (the registered simple form; risk-coverage AUC is the named
  alternative);
- coverage is published next to every score, and a score without
  a coverage figure is not routable (R-A7c);
- the temperature/ECE calibration half (R-A7b) remains a measured
  M302 item - the scale forgery it closes is named here, not
  silently absorbed.
"""
from __future__ import annotations

from dataclasses import dataclass


class MissingCoverage(RuntimeError):
    """A score without a coverage figure is not interpretable and
    must not be routable (R-A7c)."""


@dataclass(frozen=True)
class AxisScore:
    accuracy: float
    coverage: float          # share of the axis's rows the artifact
                             # answers (never abstains)
    axis: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError("accuracy must be in [0, 1]")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be in [0, 1]")

    @property
    def coverage_adjusted(self) -> float:
        return self.accuracy * self.coverage


def coverage_adjusted_score(accuracy: float, coverage: float) -> float:
    """The registered simple form of R-A7a: accuracy times
    coverage. Abstention trades against the score instead of
    inflating it."""
    return AxisScore(accuracy=accuracy, coverage=coverage
                     ).coverage_adjusted


def refuse_missing_coverage(raw: dict) -> AxisScore:
    """R-A7c enforcement point: a score without a coverage figure
    is not interpretable and must not be routable. Legacy dict
    scores pass through here and raise."""
    if "coverage" not in raw or raw["coverage"] is None:
        raise MissingCoverage("a score without a coverage figure "
                              "is not routable (R-A7c)")
    return AxisScore(accuracy=float(raw["accuracy"]),
                     coverage=float(raw["coverage"]),
                     axis=str(raw.get("axis", "")))


def compare(score_a: AxisScore, score_b: AxisScore) -> int:
    """The routing comparison on the coverage-adjusted metric:
    -1 if a ranks below b, +1 if above, 0 if tied."""
    a = score_a.coverage_adjusted
    b = score_b.coverage_adjusted
    return (a > b) - (a < b)
