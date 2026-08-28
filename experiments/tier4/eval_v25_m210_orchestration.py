"""M210 — model-agnostic orchestration closure demonstration.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(19 Aug 2026, before the build). Five sealed arms of three code
families and two architectures (spm/ms/pool closed-form heads, the
f6144 ridge head, and the M206 MLP DNN arm) are registered into one
orchestrator, routed per domain task, replayed, attributed (LOO over
arms), and every decision is hash-chained on the M185 ledger.

Gates:
- G1 ledger chain verifies; record count = registrations + routes.
- G2 determinism: a second identical orchestrator produces the same
  content hash and the same ledger tip.
- G3 size-agnosticism: a synthetic 1B-param arm validates with zero
  reasons (no size bound anywhere); a malformed spec is rejected.
- G4 replay: every routed decision carries a non-null replay handle;
  the MLP arm's handle equals its sealed M206 replay hash.
- G5 heterogeneous routing: top-1 across the six domain tasks spans at
  least two arm families, per the sealed per-domain accuracies.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from geode.core.arm import arm_from_admission, arm_from_sealed_head, \
    validate_arm_spec
from geode.core.dnn_admission import AdmissionRegistry, DNNSubmission
from geode.core.orchestrator import Orchestrator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m210_orchestration.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m210_orchestration")


def _mlp_arm(arm_cfg: dict[str, Any],
             m206_evidence: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Reconstruct the M206 submission from its sealed evidence and
    re-admit it through the M205 contract; the arm carries the same
    replay hash. Returns (arm_spec, sealed_replay_hash)."""
    hashes = m206_evidence["hashes"]
    arch = m206_evidence["architecture"]
    sub = DNNSubmission(
        architecture_hash=hashes["architecture"],
        seed_hash=hashes["seed"],
        data_digest=hashes["data"],
        software_hash=hashes["software"],
        weights_hash=hashes["weights"],
        training_log_digest=hashes["training_log"],
        eval_report={
            "split": "test",
            "n_test": int(m206_evidence["evaluation"]["n_test"]),
            "accuracy": float(m206_evidence["evaluation"]
                              ["test_accuracy"]),
        },
    )
    result = AdmissionRegistry().admit(sub)
    params = (arch["input"] * arch["hidden"] + arch["hidden"]
              + arch["hidden"] * arch["output"] + arch["output"])
    arm = arm_from_admission(
        sub, result, "mlp_dnn",
        held_out_accuracy=float(arm_cfg["held_out_accuracy"]),
        param_count=int(params),
        size_bytes=int(params * 4))
    return arm, m206_evidence["admission"]["replay_hash"]


def _build_orchestrator(config: dict[str, Any],
                        m206_evidence: dict[str, Any]
                        ) -> tuple[Orchestrator, dict[str, Any]]:
    orch = Orchestrator()
    meta: dict[str, Any] = {}
    for arm_id, arm_cfg in config["arms"].items():
        if arm_cfg["kind"] == "sealed_head":
            spec = arm_from_sealed_head(
                arm_id, arm_cfg["family"], int(arm_cfg["width"]),
                float(arm_cfg["held_out_accuracy"]),
                arm_cfg["sealed_source"],
                per_task=arm_cfg.get("per_task"))
        else:
            spec, sealed_replay = _mlp_arm(arm_cfg, m206_evidence)
            meta["mlp_sealed_replay"] = sealed_replay
        reasons = validate_arm_spec(spec)
        if reasons:
            raise SystemExit(f"arm {arm_id} failed validation: {reasons}")
        orch.register(spec)
    return orch, meta


def run_m210(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible

    m206_path = REPO_ROOT / config["m206_evidence"]
    m206_evidence = json.loads(m206_path.read_text(encoding="utf-8"))
    if m206_evidence.get("admissible_as_evidence") is not True:
        raise SystemExit("M210 premise failure: M206 evidence inadmissible")

    orch, meta = _build_orchestrator(config, m206_evidence)

    # ---- serve the six domain tasks ---------------------------------------
    routes: dict[str, Any] = {}
    for query in config["queries"]:
        routed = orch.serve(query["query_id"], [],
                            task_id=query["task_id"])
        routes[query["task_id"]] = {
            "top_k": [{"arm_id": r["arm_id"],
                       "held_out": r["held_out_accuracy"],
                       "route_cos": r.get("route_cos"),
                       "replay_handle": r.get("replay_handle")
                       if "replay_handle" in r else
                       orch.replay_handle(r["arm_id"])}
                      for r in routed],
        }

    # ---- gates ------------------------------------------------------------
    chain = orch.chain_verify()
    top1_families = set()
    handles_ok = True
    for task, recs in routes.items():
        arm_id = recs["top_k"][0]["arm_id"]
        arm = orch.router._arms[arm_id]
        top1_families.add(arm.get("family") or arm.get("kind"))
        handles_ok = handles_ok and all(
            r.get("replay_handle") is not None for r in recs["top_k"])
    mlp_anchor_ok = (orch.replay_handle("mlp_dnn")
                     == meta["mlp_sealed_replay"])
    # G2 determinism: a second orchestrator, identical operations
    orch2, _meta2 = _build_orchestrator(config, m206_evidence)
    for query in config["queries"]:
        orch2.serve(query["query_id"], [], task_id=query["task_id"])
    deterministic = (orch2.content_hash() == orch.content_hash()
                     and orch2.chain_verify()["tip"] == chain["tip"])
    # G3 size-agnosticism: no bound anywhere in the contract
    huge = arm_from_sealed_head("huge_dummy", "synthetic", 100_000,
                                0.5, "synthetic")
    huge["kind"] = "dnn"
    huge["replay_hash"] = "0" * 64
    huge["param_count"] = 1_000_000_000
    huge["size_bytes"] = 4_000_000_000
    size_free = validate_arm_spec(huge) == []
    malformed = {"arm_id": "bad"}  # missing everything
    rejects = len(validate_arm_spec(malformed)) > 0

    gates = {
        "g1_chain": chain,
        "g2_deterministic": deterministic,
        "g3_size_agnostic": {"synthetic_1b_params_validates": size_free,
                             "malformed_rejected": rejects},
        "g4_replay": {"handles_non_null": handles_ok,
                      "mlp_replay_anchor": mlp_anchor_ok},
        "g5_heterogeneous_top1": {
            "top1_families": sorted(top1_families),
            "ok": len(top1_families) >= 2},
    }
    gates_ok = all([chain["ok"], deterministic, size_free, rejects,
                    handles_ok, mlp_anchor_ok,
                    gates["g5_heterogeneous_top1"]["ok"]])

    attribution = orch.attribute()

    evidence: dict[str, Any] = {
        "milestone": "M210",
        "cell": "model-agnostic orchestration closure (register -> "
                "route -> replay -> attribute -> record)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "routes": routes,
        "size_table": orch.size_table(),
        "attribution": attribution,
        "gates": gates,
        "void": not gates_ok,
        "void_reason": "" if gates_ok else
        "one or more orchestration gates failed",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok,
                      "top1_families": sorted(top1_families),
                      "routes": {t: [r["arm_id"] for r in
                                     routes[t]["top_k"]]
                                 for t in routes},
                      "loo": attribution["loo_marginals"]}, indent=1),
          flush=True)
    print(f"M210 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m210(args.config, args.output)


if __name__ == "__main__":
    main()
