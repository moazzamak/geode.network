"""M282 — commit-reveal arm submissions: scripted evidence run.

Exercises the SubmissionLedger end-to-end: the honest path
(commit -> reveal -> admit) admits; every after-the-fact path —
re-described claim at reveal, tampered weight digest,
reveal-without-commit, admit-before-reveal, measured-below-commit —
fails by construction. Receipts are recorded at every stage.

Deterministic; no wall clocks inside hashes. Evidence:
logs/results/v25/m282_submission_commit/evidence.json.
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
from geode.core.submission_commit import (
    SubmissionLedger,
    submission_commit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m282_submission_commit")

SALT = "m282-20260823"
DIGEST_A = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
DIGEST_B = "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"


def run_m282(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    outcomes: list[dict[str, Any]] = []
    ledger = SubmissionLedger()

    def record(name: str, ok: bool, detail: Any = None,
               expected: str = "") -> None:
        outcomes.append({"scenario": name, "passed": bool(ok),
                         "expected": expected, "detail": detail})

    claim_modest = {"per_task": {"d0": 0.85, "d3": 0.90}}

    # 1. the honest path admits
    cid_honest = ledger.commit("alice", SALT, "arm_honest",
                               "sentiment", claim_modest, DIGEST_A)
    ledger.reveal(cid_honest, "alice", SALT, "arm_honest",
                  "sentiment", claim_modest, DIGEST_A)
    receipt = ledger.admit(cid_honest, {"d0": 0.86, "d3": 0.91})
    record("honest_commit_reveal_admit", receipt["admitted"] is True,
           receipt, "admitted=True")
    record("honest_admitted_ids", ledger.admitted_ids() == [cid_honest],
           ledger.admitted_ids(), "[cid_honest]")

    # 2. measured below the committed claim rejects with the
    #    COMMITTED claim recorded
    cid_low = ledger.commit("bob", SALT, "arm_low", "logic",
                            claim_modest, DIGEST_B)
    ledger.reveal(cid_low, "bob", SALT, "arm_low", "logic",
                  claim_modest, DIGEST_B)
    receipt = ledger.admit(cid_low, {"d0": 0.80, "d3": 0.90})
    record("measured_below_commit_rejects",
           receipt["admitted"] is False
           and receipt["committed_per_task"] == {"d0": 0.85, "d3": 0.90},
           receipt, "admitted=False, committed claim recorded")

    # 3. the after-the-fact re-description fails at reveal
    cid_cheat = ledger.commit("eve", SALT, "arm_cheat", "sentiment",
                              claim_modest, DIGEST_A)
    try:
        ledger.reveal(cid_cheat, "eve", SALT, "arm_cheat", "sentiment",
                      {"per_task": {"d0": 0.99, "d3": 0.99}}, DIGEST_A)
        record("redescribed_claim_fails_reveal", False, "no raise",
               "ValueError")
    except ValueError as exc:
        record("redescribed_claim_fails_reveal", True, str(exc),
               "ValueError")

    # 4. tampered weight digest fails at reveal
    cid_tamper = ledger.commit("frank", SALT, "arm_tamper",
                               "arithmetic", claim_modest, DIGEST_A)
    try:
        ledger.reveal(cid_tamper, "frank", SALT, "arm_tamper",
                      "arithmetic", claim_modest, "deadbeef")
        record("tampered_digest_fails_reveal", False, "no raise",
               "ValueError")
    except ValueError as exc:
        record("tampered_digest_fails_reveal", True, str(exc),
               "ValueError")

    # 5. reveal without a prior commit fails
    orphan = submission_commit("carol", SALT, "arm_orphan",
                               "arithmetic", claim_modest, DIGEST_A)
    try:
        ledger.reveal(orphan, "carol", SALT, "arm_orphan",
                      "arithmetic", claim_modest, DIGEST_A)
        record("reveal_without_commit_fails", False, "no raise",
               "ValueError")
    except ValueError as exc:
        record("reveal_without_commit_fails", True, str(exc),
               "ValueError")

    # 6. admit before reveal fails
    cid_norev = ledger.commit("dave", SALT, "arm_norev", "logic",
                              claim_modest, DIGEST_B)
    try:
        ledger.admit(cid_norev, {"d0": 0.99, "d3": 0.99})
        record("admit_before_reveal_fails", False, "no raise",
               "ValueError")
    except ValueError as exc:
        record("admit_before_reveal_fails", True, str(exc),
               "ValueError")

    all_passed = all(o["passed"] for o in outcomes)
    results = {
        "n_scenarios": len(outcomes),
        "all_passed": all_passed,
        "outcomes": outcomes,
        "receipts": ledger.receipts,
        "admitted_ids": ledger.admitted_ids(),
        "verdict": ("M282 PASS — the honest path admits; every "
                    "after-the-fact path fails by construction"
                    if all_passed else
                    "M282 FAIL — a scenario did not behave as "
                    "registered"),
    }
    evidence: dict[str, Any] = {
        "milestone": "M282",
        "cell": "commit-reveal arm submissions",
        "admissible_as_evidence": True,
        "smoke": False,
        "configuration_hash": payload_hash({
            "module": "geode/core/submission_commit.py",
            "rule": {"tolerance": ledger.rule.tolerance},
            "scenarios": ["honest_commit_reveal_admit",
                          "measured_below_commit_rejects",
                          "redescribed_claim_fails_reveal",
                          "tampered_digest_fails_reveal",
                          "reveal_without_commit_fails",
                          "admit_before_reveal_fails"],
        }),
        "results": results,
        "scope_note": ("the contributor commits (capability claim + "
                       "weight digest, salted) BEFORE probes run; the "
                       "reveal must hash to the commit at admission; "
                       "selection metrics compare the measured values "
                       "against the claim AS COMMITTED — re-described "
                       "claims never enter the comparison"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"results": results}, indent=1), flush=True)
    print(f"M282 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


if __name__ == "__main__":
    run_m282(DEFAULT_OUTPUT)
