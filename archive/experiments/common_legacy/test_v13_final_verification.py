"""Tests for M86's finalization verifier.

The verifier is the thing that licenses the v13 ledger, so the tests here are
mostly about making it **fail**. A gate that has only ever been observed to pass
has not been shown to be a gate.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.common.v5_artifacts import payload_hash
from experiments.tier4.verify_v13_final import (
    _conclusion_operands,
    _recompute_hash,
    _verify_v12_ledger,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = REPO_ROOT / "logs" / "results" / "v13" / "m86_final_verification" / "evidence.json"
CONFIG = REPO_ROOT / "experiments" / "configs" / "v13" / "m86_final_verification.json"

RULES = [
    {"name": "whole_payload", "exclude": ["evidence_hash"]},
    {"name": "without_generated_at", "exclude": ["evidence_hash", "generated_at"]},
]


def _sealed() -> dict:
    if not EVIDENCE.exists():
        pytest.skip("M86 has not been run in this checkout")
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_recompute_hash_reports_the_rule_that_worked() -> None:
    evidence = {"generated_at": "now", "value": 1}
    evidence["evidence_hash"] = payload_hash({"value": 1})
    report = _recompute_hash(evidence, RULES)
    assert report["status"] == "recomputed"
    assert report["rule"] == "without_generated_at"


def test_recompute_hash_catches_a_tampered_payload() -> None:
    evidence = {"value": 1}
    evidence["evidence_hash"] = payload_hash({"value": 1})
    evidence["value"] = 2
    assert _recompute_hash(evidence, RULES)["status"] == "not_reproducible"


def test_recompute_hash_reports_an_absent_hash_as_absent() -> None:
    # Four v13 milestones store no evidence_hash. They must not be counted as
    # verified, and they must not be counted as failing either.
    report = _recompute_hash({"value": 1}, RULES)
    assert report["status"] == "absent"
    assert report["stored"] is None


def test_verify_v12_ledger_finds_both_amendments() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = _verify_v12_ledger(config)
    assert report["all_present"] is True
    assert len(report["amendments"]) == 2
    assert all(item["cited_evidence_exists"] for item in report["amendments"])


def test_verify_v12_ledger_fails_on_a_missing_amendment() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["v12_ledger"]["required_amendments"].append(
        {
            "heading": "## Amendment A9 — never written",
            "evidence": "logs/results/v13/m77_probe_degeneracy/evidence.json",
            "source_milestone": "m77_probe_degeneracy",
        }
    )
    assert _verify_v12_ledger(config)["all_present"] is False


def _operand_inputs() -> tuple[dict, dict, dict, dict]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    evidence = {}
    for name, specification in config["milestones"].items():
        path = REPO_ROOT / specification["directory"] / "evidence.json"
        if not path.exists():
            pytest.skip("v13 sealed evidence is not present in this checkout")
        evidence[name] = json.loads(path.read_text(encoding="utf-8"))
    ledger = _verify_v12_ledger(config)
    replay = {"identical": True}
    milestones = {
        name: {
            "evidence_hash": _recompute_hash(block, config["hash_exclusion_rules"]),
            "final_labels_opened": block.get("final_labels_opened", "field_absent"),
        }
        for name, block in evidence.items()
    }
    return evidence, ledger, replay, milestones


def test_conclusion_operands_all_pass_on_the_sealed_evidence() -> None:
    operands = _conclusion_operands(*_operand_inputs())
    assert all(operands.values()), sorted(k for k, v in operands.items() if not v)
    assert len(operands) == 19


def test_a_flipped_verdict_fails_its_operand() -> None:
    evidence, ledger, replay, milestones = _operand_inputs()
    evidence = copy.deepcopy(evidence)
    evidence["m84_exposure_ladder"]["gate"]["verdict"] = "ladder_rises"
    operands = _conclusion_operands(evidence, ledger, replay, milestones)
    assert operands["m84_exposure_ladder_flat"] is False


def test_a_broken_cross_milestone_reproduction_fails_its_operand() -> None:
    # M85a's whole claim to be ranking M84's object rests on this one number.
    evidence, ledger, replay, milestones = _operand_inputs()
    evidence = copy.deepcopy(evidence)
    evidence["m84_exposure_ladder"]["gate"]["baseline"] = 0.5
    operands = _conclusion_operands(evidence, ledger, replay, milestones)
    assert operands["m85a_reproduces_m84_zero_rung"] is False


def test_an_opened_final_label_fails_its_operand() -> None:
    evidence, ledger, replay, milestones = _operand_inputs()
    milestones = copy.deepcopy(milestones)
    milestones["m81_sparse_head"]["final_labels_opened"] = True
    operands = _conclusion_operands(evidence, ledger, replay, milestones)
    assert operands["final_labels_never_opened"] is False


def test_a_failed_replay_fails_its_operand() -> None:
    evidence, ledger, _, milestones = _operand_inputs()
    operands = _conclusion_operands(evidence, ledger, {"identical": False}, milestones)
    assert operands["designated_milestone_replays_identically"] is False


def test_a_frontier_citing_a_stale_hash_fails_its_operand() -> None:
    evidence, ledger, replay, milestones = _operand_inputs()
    evidence = copy.deepcopy(evidence)
    evidence["m85_frontier"]["source_hashes"]["m84_exposure_ladder"] = "0" * 64
    operands = _conclusion_operands(evidence, ledger, replay, milestones)
    assert operands["m85c_frontier_cites_the_sealed_hashes"] is False


def test_the_sealed_run_passed_and_records_no_final_label_access() -> None:
    sealed = _sealed()
    assert sealed["verification_passed"] is True
    assert sealed["failed_operands"] == []
    assert sealed["final_labels_opened"] is False
    assert sealed["features_loaded"] is False
    assert sealed["outcome"] == "C"


def test_the_sealed_run_replayed_a_milestone_rather_than_trusting_a_flag() -> None:
    replay = _sealed()["replay"]
    assert replay["identical"] is True
    assert replay["differing_fields"] == []
    assert (
        replay["stable_payload_hash_sealed"] == replay["stable_payload_hash_replayed"]
    )
    # The stored hashes differ because runtime_seconds sits inside them. That is
    # the defect N86.3 registered, and it is asserted here so a later change to
    # the hashing rule does not quietly erase the record of it.
    assert replay["sealed_evidence_hash"] != replay["replayed_evidence_hash"]


def test_the_void_runs_were_retained_not_deleted() -> None:
    void = _sealed()["retained_void_runs"]
    assert set(void) == {"m83_boundary", "m78_sample_adequacy_void_r1"}
    assert all(item["retained"] for item in void.values())
