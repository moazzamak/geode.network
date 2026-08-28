"""M302 H26-9 measurement - the coverage-adjusted ranking inversion
on the sealed M286 scoped-arm numbers.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md``
§8.39 (27 Aug 2026, before the run). No re-fit: the sealed M286
evidence supplies both arms (the scoped reading and the
full-coverage reading of the same head). The R-A7a metric module
(`geode/core/coverage_adjusted.py`) does the comparison; the
coverage figure is published with every score (R-A7c).

Gates: the coverage-adjusted ranking inverts; both scores carry
coverage; a coverage-less score is refused.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from geode.core.coverage_adjusted import (
    AxisScore,
    MissingCoverage,
    compare,
    refuse_missing_coverage,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
M286_EVIDENCE = (REPO_ROOT / "logs" / "results" / "v25"
                 / "m286_served_subset" / "evidence.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m302_coverage_metric")


def main() -> int:
    started = time.time()
    sealed = json.loads(M286_EVIDENCE.read_text(encoding="utf-8"))
    results = sealed["results"]

    # the two arms, from the sealed record
    scoped = AxisScore(
        accuracy=float(results["served_subset_overall_top1"]),
        coverage=float(results["served_test_row_coverage"]),
        axis="oid-vision")
    full = AxisScore(
        accuracy=float(results["overall_top1_all_classes"]),
        coverage=1.0,
        axis="oid-vision")

    raw_ranking = compare_unadjusted = (
        1 if scoped.accuracy > full.accuracy else -1)
    adjusted_ranking = compare(scoped, full)
    inverted = bool(adjusted_ranking < 0)

    # R-A7c: a coverage-less score is refused
    refused = False
    try:
        refuse_missing_coverage({"accuracy": 0.901})
    except MissingCoverage:
        refused = True

    gates_ok = inverted and refused
    evidence: dict[str, Any] = {
        "milestone": "M302",
        "cell": "H26-9: coverage-adjusted ranking inversion on the "
                "sealed M286 scoped-arm numbers",
        "sealed_source": str(M286_EVIDENCE),
        "arms": {
            "scoped": {"accuracy": scoped.accuracy,
                       "coverage": scoped.coverage,
                       "coverage_adjusted":
                           scoped.coverage_adjusted,
                       "note": "129 served classes, 472 refused "
                               "(M286 rule)"},
            "full_coverage": {"accuracy": full.accuracy,
                              "coverage": full.coverage,
                              "coverage_adjusted":
                                  full.coverage_adjusted,
                              "note": "the same head without the "
                                      "refusal rule"},
        },
        "raw_ranking": "scoped outranks full"
        if raw_ranking > 0 else "full outranks scoped",
        "coverage_adjusted_ranking": "full outranks scoped"
        if adjusted_ranking < 0 else "scoped outranks full",
        "gates": {
            "h26_9_ranking_inverts": {
                "ok": inverted,
                "scoped_adjusted": scoped.coverage_adjusted,
                "full_adjusted": full.coverage_adjusted},
            "ra7c_coverage_published": {
                "ok": True,
                "note": "both AxisScore records carry their "
                        "coverage figure"},
            "ra7c_coverage_less_refused": {"ok": refused},
        },
        "gates_ok": bool(gates_ok),
        "ra7b_pending": ("the temperature/ECE calibration half is a "
                         "registered pending measurement; it needs "
                         "per-row score distributions, not this "
                         "evidence table"),
        "configuration_hash": payload_hash({
            "metric": "accuracy x coverage (the registered simple "
                      "form of R-A7a)",
            "source": "sealed M286 evidence",
        }),
        "runtime_seconds": round(time.time() - started, 3),
    }
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    write_canonical_json(DEFAULT_OUTPUT / "evidence.json", evidence)
    build_artifact_index(DEFAULT_OUTPUT)
    print(json.dumps({"gates_ok": bool(gates_ok),
                      "scoped_adjusted": scoped.coverage_adjusted,
                      "full_adjusted": full.coverage_adjusted,
                      "inverted": inverted}))
    return 0 if gates_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
