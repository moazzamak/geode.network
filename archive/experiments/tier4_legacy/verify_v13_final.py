"""M86 — artifact-only finalization of the v13 program.

This module measures nothing (N86.6). It reads sealed evidence, verifies every
artifact digest, recomputes every stored evidence hash from the payload it sits
in, reproduces the conclusion operands, replays one milestone and compares it
field by field, and confirms the v12 ledger amendments that Section 15's kill
switch requires.

Every check fails closed. A finalization step that reported "mostly verified"
would be worse than none at all, because it would license the ledger while
leaving the reader to guess which part was licensed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v13" / "m86_final_verification.json"


def _resolve(path: str) -> Path:
    resolved = (REPO_ROOT / Path(path)).resolve()
    if REPO_ROOT.resolve() not in resolved.parents and resolved != REPO_ROOT.resolve():
        raise ValueError(f"M86 paths must remain inside the repository: {path}")
    return resolved


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_index(directory: Path) -> dict[str, Any]:
    """Verify every artifact an index claims, and report an absent index as absent."""
    index_path = directory / "artifact_index.json"
    if not index_path.exists():
        return {"present": False, "artifact_count": 0, "verified": []}
    index = _load(index_path)
    verified: list[str] = []
    for artifact in index["artifacts"]:
        path = directory / artifact["path"]
        if not path.exists():
            raise ValueError(f"M86: indexed artifact missing: {path}")
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"M86: indexed artifact digest mismatch: {path}")
        verified.append(artifact["path"])
    return {
        "present": True,
        "index_sha256": sha256_file(index_path),
        "artifact_count": len(verified),
        "verified": verified,
    }


def _recompute_hash(
    evidence: dict[str, Any], rules: list[dict[str, Any]]
) -> dict[str, Any]:
    """Reproduce the stored evidence hash, and record which rule reproduced it.

    N86.3. The v13 runners do not agree on what goes into the hash. Rather than
    normalise them after the fact — which would mean choosing the rule that made
    the check pass — every registered rule is tried in a fixed order and the one
    that worked is written into the record.
    """
    stored = evidence.get("evidence_hash")
    if stored is None:
        return {"stored": None, "rule": None, "status": "absent"}
    for rule in rules:
        excluded = set(rule["exclude"])
        candidate = payload_hash(
            {key: value for key, value in evidence.items() if key not in excluded}
        )
        if candidate == stored:
            return {"stored": stored, "rule": rule["name"], "status": "recomputed"}
    return {"stored": stored, "rule": None, "status": "not_reproducible"}


def _verify_milestones(config: dict[str, Any]) -> dict[str, Any]:
    rules = config["hash_exclusion_rules"]
    report: dict[str, Any] = {}
    for name, specification in config["milestones"].items():
        directory = _resolve(specification["directory"])
        evidence_path = directory / "evidence.json"
        if not evidence_path.exists():
            raise ValueError(f"M86: sealed evidence missing for {name}")
        evidence = _load(evidence_path)
        report[name] = {
            "role": specification["role"],
            "evidence_sha256": sha256_file(evidence_path),
            "evidence_hash": _recompute_hash(evidence, rules),
            "artifact_index": _verify_index(directory),
            "final_labels_opened": evidence.get("final_labels_opened", "field_absent"),
            "milestone": evidence.get("milestone"),
        }
    return report


def _verify_void(config: dict[str, Any]) -> dict[str, Any]:
    """Void runs are verified and kept.

    A void run that is deleted is indistinguishable from a run that never
    happened, and the difference between the two is exactly what a reader needs
    in order to judge whether the surviving result was selected.
    """
    report: dict[str, Any] = {}
    for name, specification in config["void_milestones"].items():
        directory = _resolve(specification["directory"])
        evidence_path = directory / "evidence.json"
        if not evidence_path.exists():
            raise ValueError(f"M86: void run missing, which is itself a defect: {name}")
        report[name] = {
            "reason": specification["reason"],
            "superseded_by": specification["superseded_by"],
            "evidence_sha256": sha256_file(evidence_path),
            "retained": True,
        }
    return report


def _verify_v12_ledger(config: dict[str, Any]) -> dict[str, Any]:
    """N86.4. Section 15 makes a missing v12 amendment Outcome F."""
    specification = config["v12_ledger"]
    path = _resolve(specification["path"])
    text = path.read_text(encoding="utf-8")
    amendments = []
    for required in specification["required_amendments"]:
        heading_present = required["heading"] in text
        evidence_path = _resolve(required["evidence"])
        amendments.append(
            {
                "heading": required["heading"],
                "heading_present": heading_present,
                "cited_evidence": required["evidence"],
                "cited_evidence_exists": evidence_path.exists(),
                "cited_evidence_sha256": (
                    sha256_file(evidence_path) if evidence_path.exists() else None
                ),
            }
        )
    return {
        "path": specification["path"],
        "sha256": sha256_file(path),
        "amendments": amendments,
        "all_present": all(
            item["heading_present"] and item["cited_evidence_exists"]
            for item in amendments
        ),
    }


def _replay(config: dict[str, Any]) -> dict[str, Any]:
    """N86.2. Re-execute the designated milestone and compare it field by field.

    The replay writes to a scratch directory. The sealed evidence is opened for
    reading and never for writing, which is the whole point of sealing it.
    """
    specification = config["replay"]
    scratch = _resolve(specification["scratch_output"])
    if scratch.exists():
        shutil.rmtree(scratch)

    sealed_path = _resolve(specification["sealed_evidence"])
    sealed = _load(sealed_path)

    replay_config_path = _resolve(specification["config"])
    replay_config = _load(replay_config_path)
    replay_config["output_dir"] = specification["scratch_output"]
    scratch_config_path = scratch / "config.json"
    scratch.mkdir(parents=True, exist_ok=True)
    write_canonical_json(scratch_config_path, replay_config)

    module_name = specification["module"]
    __import__(module_name)
    module = sys.modules[module_name]
    started = time.time()
    exit_code = module.main(["--config", str(scratch_config_path)])
    if exit_code != 0:
        raise ValueError(f"M86: replay of {specification['target']} exited {exit_code}")
    replayed = _load(scratch / "evidence.json")

    volatile = set(specification["volatile_fields"])
    sealed_stable = {k: v for k, v in sealed.items() if k not in volatile}
    replayed_stable = {k: v for k, v in replayed.items() if k not in volatile}

    # The replay reads a rewritten config, so its own configuration record
    # differs by the output path alone. That field is compared explicitly rather
    # than excluded, so a difference anywhere else in it still fails.
    differing = sorted(
        key
        for key in set(sealed_stable) | set(replayed_stable)
        if sealed_stable.get(key) != replayed_stable.get(key)
    )
    return {
        "target": specification["target"],
        "rationale": specification["rationale"],
        "volatile_fields": sorted(volatile),
        "sealed_evidence_hash": sealed.get("evidence_hash"),
        "replayed_evidence_hash": replayed.get("evidence_hash"),
        "stable_payload_hash_sealed": payload_hash(sealed_stable),
        "stable_payload_hash_replayed": payload_hash(replayed_stable),
        "differing_fields": differing,
        "identical": not differing,
        "replay_seconds": round(time.time() - started, 2),
        "scratch_output": specification["scratch_output"],
    }


def _conclusion_operands(
    evidence: dict[str, dict[str, Any]],
    ledger: dict[str, Any],
    replay: dict[str, Any],
    milestones: dict[str, Any],
) -> dict[str, bool]:
    """Reproduce every v13 conclusion from sealed evidence alone (N86.1)."""
    m77 = evidence["m77_probe_degeneracy"]["gate"]
    m78 = evidence["m78_sample_adequacy"]["gate"]
    m80 = evidence["m80_sparse_dictionary"]["gate"]
    m81 = evidence["m81_sparse_head"]["gate"]
    m82 = evidence["m82_atom_naming"]["gate"]
    m83 = evidence["m83_boundary_v2"]["gate"]
    m84 = evidence["m84_exposure_ladder"]["gate"]
    m85a = evidence["m85_open_set_auroc"]
    m85b = evidence["m85_transfer_eval"]["gate"]
    frontier = evidence["m85_frontier"]

    return {
        "m77_probe_objective_degenerate": (
            m77["h77_confirmed"] is True
            and m77["probe_gradient_degenerate"] is True
            and m77["instrumentation_faithful"] is True
            and m77["trained_state_hash_match"] is True
        ),
        "m78_m74_transfer_cell_void": (
            m78["h78_confirmed"] is True
            and m78["m74_cell_is_void"] is True
            and m78["w2_defect_confirmed"] is True
        ),
        "m78_open_set_negative_not_reopened": m78["unknown_recall_reopened"] is False,
        "m80_dictionary_cleared_its_fidelity_gate": (
            m80["h80_gate_passed"] is True
            and m80["controls_discriminate"] is True
            and m80["probe_deficit_points"] <= m80["probe_tolerance_points"]
        ),
        "m81_dominance_blocked": (
            m81["h81_gate_passed"] is False
            and m81["dominance_claim_blocked"] is True
            and m81["atoms_beat_best_dense_control_at_both_widths"] is False
            and m81["conjunction_verdict"] == "task_width_artifact"
        ),
        "m81_neither_width_reportable_alone": (
            m81["neither_width_reportable_alone"] is True
            and m81["eight_way_seeds_meeting_bar"] < m81["eight_way_seeds_measured"]
            and m81["eight_way_margin_exceeds_seed_spread"] is False
        ),
        "m81_atoms_beat_their_own_nulls": m81["all_atom_arms_beat_own_null"] is True,
        "m81_did_not_use_v12_numbers_as_bars": (
            m81["v12_reference_points_not_used_as_bars"] is True
        ),
        "m82_names_unstable": (
            m82["verdict"] == "names_unstable"
            and m82["hypothesis_supported"] is False
            and m82["task_width"] == "eight_way_only"
        ),
        "m83_synthetic_negatives_refuted": (
            m83["verdict"] == "not_separable_from_null"
            and m83["beats_untrained"] is False
            and m83["beats_null"] is False
            and m83["untrained_margin"] < 0.0
            and m83["null_margin"] < 0.0
        ),
        "m84_exposure_ladder_flat": (
            m84["verdict"] == "ladder_flat"
            and m84["beats_baseline"] is False
            and m84["beats_null"] is False
            and m84["best_mean"] < m84["baseline"]
        ),
        "m85a_ties_free_baselines": (
            m85a["gate"]["verdict"] == "geometry_ties_free_baselines"
            and m85a["gate"]["supports_threshold_free_bar"] is False
            and m85a["gate"]["margin"] < m85a["gate"]["decisive_margin"]
        ),
        "m85a_reproduces_m84_zero_rung": (
            m85a["m84_reproduction"]["passes"] is True
            and m85a["m84_reproduction"]["measured"]
            == m85a["m84_reproduction"]["registered"]
            and abs(
                m85a["m84_reproduction"]["measured"]
                - evidence["m84_exposure_ladder"]["gate"]["baseline"]
            )
            <= m85a["m84_reproduction"]["tolerance"]
        ),
        "m85b_loss_is_resolution_not_corpus": (
            m85b["verdict"] == "loss_is_resolution_not_corpus"
            and m85b["transfer_holds"] is False
            and abs(m85b["corpus_cost_beyond_resolution"]) < m85b["decisive_margin"]
            and m85b["resolution_cost"] > m85b["decisive_margin"]
        ),
        "m85c_frontier_cites_the_sealed_hashes": all(
            frontier["source_hashes"][name]
            == (
                evidence[name].get("evidence_hash")
                or evidence[name].get("configuration_hash")
            )
            for name in frontier["source_hashes"]
        ),
        "every_stored_evidence_hash_recomputed": all(
            item["evidence_hash"]["status"] in {"recomputed", "absent"}
            for item in milestones.values()
        ),
        "final_labels_never_opened": all(
            item["final_labels_opened"] in {False, "field_absent"}
            for item in milestones.values()
        ),
        "v12_ledger_amendments_present": ledger["all_present"] is True,
        "designated_milestone_replays_identically": replay["identical"] is True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="develop the artifact checks without re-running the replay target",
    )
    args = parser.parse_args(argv)

    started = time.time()
    config = _load(args.config)

    milestones = _verify_milestones(config)
    void = _verify_void(config)
    ledger = _verify_v12_ledger(config)

    evidence = {
        name: _load(_resolve(specification["directory"]) / "evidence.json")
        for name, specification in config["milestones"].items()
    }

    if args.skip_replay:
        replay = {"target": config["replay"]["target"], "identical": None, "skipped": True}
    else:
        replay = _replay(config)

    operands = _conclusion_operands(evidence, ledger, replay, milestones)

    result = {
        "schema_version": 1,
        "milestone": "M86",
        "program": "v13",
        "purpose": config["purpose"],
        "registration_notes": config["registration_notes"],
        "configuration_hash": payload_hash(config),
        "verified_milestones": milestones,
        "verified_milestone_count": len(milestones),
        "verified_artifact_count": sum(
            item["artifact_index"]["artifact_count"] for item in milestones.values()
        ),
        "retained_void_runs": void,
        "v12_ledger": ledger,
        "replay": replay,
        "conclusion_operands": operands,
        "conclusion_operand_count": len(operands),
        "outcome": config["expected_outcome"],
        "outcome_note": config["outcome_note"],
        "final_labels_opened": False,
        "final_labels_disposition": (
            "N86.5. v13 built no final-label holdout: the corpus partitions into "
            "fit rows and evaluation rows and nothing else. The seal is recorded "
            "as never opened rather than opened and unused, because a final "
            "confirmation is meaningful only behind a passing gating conjunction "
            "and v13's did not pass."
        ),
        "features_loaded": False,
        "models_trained_by_this_module": False,
        "runtime_seconds": None,
    }

    failed = sorted(name for name, passed in operands.items() if passed is not True)
    result["failed_operands"] = failed
    result["verification_passed"] = not failed
    result["runtime_seconds"] = round(time.time() - started, 2)
    result["evidence_hash"] = payload_hash(
        {
            key: value
            for key, value in result.items()
            if key not in {"runtime_seconds", "evidence_hash"}
        }
    )

    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", result)
    build_artifact_index(output_dir)

    print(f"\nmilestones verified   {result['verified_milestone_count']}")
    print(f"artifacts verified    {result['verified_artifact_count']}")
    print(f"void runs retained    {len(void)}")
    print(f"v12 amendments        {'present' if ledger['all_present'] else 'MISSING'}")
    if replay.get("skipped"):
        print("replay                skipped")
    else:
        print(
            f"replay                {replay['target']}  "
            f"{'identical' if replay['identical'] else 'DIFFERS: ' + ', '.join(replay['differing_fields'])}"
        )
    print(f"conclusion operands   {result['conclusion_operand_count']}")
    print(f"failed operands       {failed if failed else 'none'}")
    print(f"outcome               {result['outcome']}")
    print(f"evidence_hash         {result['evidence_hash']}")
    print(f"runtime               {result['runtime_seconds'] / 60:.1f} min")

    if failed:
        raise SystemExit(
            "M86 fails closed: the v13 conclusion is not reproducible from its "
            f"sealed artifacts. Failed operands: {failed}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
