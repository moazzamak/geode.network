"""M205 DNN-component admission validator (registered 19 Aug 2026).

Implements the admission contract of
``analysis/v25_m205_dnn_component_spec.md`` (v25 plan §4.13): the
registry VERIFIES a submitted DNN artifact, never trains it. Ranking
comes later from the M151/M180 coalition machinery on cached codes —
admission only rejects invalid or implausible submissions.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

MIN_TEST = 500          # registered minimum held-out size
CHANCE_FLOOR = 1.0 / 345  # + 1e-6 slack: a chance-level artifact is
# implausible and carries no signal (registered 19 Aug 2026)
HEX_RE = re.compile(r"^[0-9a-fA-F]{40,128}$")


@dataclass
class DNNSubmission:
    architecture_hash: str
    seed_hash: str
    data_digest: str
    software_hash: str
    weights_hash: str
    training_log_digest: str
    eval_report: dict[str, Any]

    def replay_hash(self) -> str:
        """The deterministic inference-replay anchor H6 will use on
        every paid session: H(architecture || weights)."""
        return hashlib.sha256(
            (self.architecture_hash + "||" + self.weights_hash)
            .encode("utf-8")).hexdigest()


@dataclass
class AdmissionResult:
    admitted: bool
    replay_hash: str = ""
    reasons: list[str] = field(default_factory=list)
    duplicate: bool = False


def _check_hex(name: str, value: str, reasons: list[str]) -> None:
    if not isinstance(value, str) or not HEX_RE.fullmatch(value):
        reasons.append(f"{name} must be a 40-128 char hex hash")


def validate_submission(sub: DNNSubmission) -> AdmissionResult:
    """Admission per the registered contract. Returns admitted=True
    only for a complete, plausible, non-duplicate submission."""
    reasons: list[str] = []
    for name, value in (
            ("architecture_hash", sub.architecture_hash),
            ("seed_hash", sub.seed_hash),
            ("data_digest", sub.data_digest),
            ("software_hash", sub.software_hash),
            ("weights_hash", sub.weights_hash),
            ("training_log_digest", sub.training_log_digest)):
        _check_hex(name, value, reasons)
    report = sub.eval_report
    if not isinstance(report, dict):
        reasons.append("eval_report must be an object")
        return AdmissionResult(False, reasons=reasons)
    if report.get("split") not in ("test", "heldout"):
        reasons.append("eval_report.split must declare a held-out split")
    n_test = report.get("n_test")
    if not isinstance(n_test, int) or n_test < MIN_TEST:
        reasons.append(f"eval_report.n_test must be >= {MIN_TEST}")
    accuracy = report.get("accuracy")
    if not isinstance(accuracy, (int, float)) \
            or not 0.0 <= float(accuracy) <= 1.0:
        reasons.append("eval_report.accuracy must be in [0, 1]")
    elif float(accuracy) <= CHANCE_FLOOR + 1e-6:
        reasons.append("eval_report.accuracy is at the chance floor — "
                       "the artifact carries no signal")
    if reasons:
        return AdmissionResult(False, reasons=reasons)
    return AdmissionResult(True, replay_hash=sub.replay_hash())


class AdmissionRegistry:
    """A registry keyed by (architecture, data) digests; duplicates
    collapse (the M199 Sybil rule) and earn no attribution."""

    def __init__(self) -> None:
        self._admitted: dict[tuple[str, str], str] = {}

    def admit(self, sub: DNNSubmission) -> AdmissionResult:
        result = validate_submission(sub)
        if not result.admitted:
            return result
        key = (sub.architecture_hash, sub.data_digest)
        if key in self._admitted:
            result.duplicate = True
            result.admitted = False
            result.reasons.append(
                "duplicate: this architecture+data digest is already "
                "admitted and earns no attribution")
            return result
        self._admitted[key] = sub.replay_hash()
        return result

    def admitted_count(self) -> int:
        return len(self._admitted)

    def replay_hash_of(self, sub: DNNSubmission) -> str | None:
        return self._admitted.get(
            (sub.architecture_hash, sub.data_digest))
