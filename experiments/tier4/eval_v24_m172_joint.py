"""M172 — multi-task differentiation cell: A, B, A+B, B+A (the gated
north-star test).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` (section 7
Phase C M172; section 6 capability 8): "tasks A, B fed separately and
jointly: fingerprints distinct for A and B, the joint task routed
distinguishably".

Cell (registered): A = mackey_glass (numeric-series regression),
B = dyck (token-text next-token), control C = tabular (unrelated), and
the two joint descriptors A+B / B+A (the dominant task's axes with
coupling = mixture — there is NO fitted mixture arm in the MVP, by
registration). Gates:

- G5a cos(A, B) < 0.9 (pure fingerprints distinct);
- G5b cos(A+B, B+A) < 0.99 (the two mixtures are distinguishable);
- G5c chain head of A+B != chain head of B+A (distinguishable routing);
- G5d chain head of A+B == chain head of A and chain head of B+A ==
  chain head of B (the mixtures land on the dominant side);
- G5e chain head of A+B not in {chain head of C} (the mixture does not
  route to an unrelated arm).

Premise gate (void on failure): the M171 sealed evidence must be
admissible. M172b: the run trains ONE encoder in-process and uses it
for every fingerprint (arms and queries share one fingerprint space);
cross-run reuse is not gated until the encoder weights are saved as a
frozen artifact (registered future item) — GPU training is measured
non-deterministic across processes. The runner also records a
second in-process training to measure training determinism directly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.data_cache import configure_external_cache_environment
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v24_m169_fingerprint_train import TASK_DESCRIPTORS
from experiments.tier4.eval_v24_m171_router import _build_encoder
from geode.core.descriptor import normalise
from geode.core.router import Router

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m172_joint.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v24" / "m172_joint")
M171_EVIDENCE = (REPO_ROOT / "logs" / "results" / "v24" / "m171_router"
                 / "evidence.json")

TASK_A, TASK_B, TASK_C = "mackey_glass", "dyck", "tabular"


def _cos(a: list[float], b: list[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(aa @ bb / (np.linalg.norm(aa) * np.linalg.norm(bb)))


def run_m172(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible

    configure_external_cache_environment()

    # ---- premise: the sealed M171 evidence --------------------------------
    m171 = json.loads(M171_EVIDENCE.read_text(encoding="utf-8"))
    premise = {
        "m171_admissible": bool(m171.get("admissible_as_evidence")),
        "m171_milestone": m171.get("milestone"),
    }

    # ---- rebuild the encoder (self-consistent, ONE training) ------------
    # M172b: the M172 run trained its encoder and used it for every
    # fingerprint. Cross-run fingerprint reuse is not gated until the
    # encoder weights are saved as a frozen artifact (registered future
    # item); GPU training is measured to be non-deterministic across
    # processes, so no cross-run equivalence premise is asserted here.
    enc, _ = _build_encoder(config)
    fp = {}
    for name in TASK_DESCRIPTORS:
        fp[name] = [float(v) for v in
                    enc.fingerprint(normalise(TASK_DESCRIPTORS[name]))
                    .detach().cpu().numpy()]

    # measured fact: training determinism across two in-process rebuilds
    enc2, _ = _build_encoder(config)
    fp2 = {n: [float(v) for v in
               enc2.fingerprint(normalise(TASK_DESCRIPTORS[n]))
               .detach().cpu().numpy()] for n in TASK_DESCRIPTORS}
    nondet = {n: _cos(fp[n], fp2[n]) for n in
              ["mackey_glass", "dyck", "tabular", "lorenz"]}
    premise["training_nondeterminism_min_cos"] = min(nondet.values())

    # joint descriptors (registered: dominant task axes + coupling mixture)
    fp["A+B"] = [float(v) for v in enc.fingerprint(normalise(
        {**TASK_DESCRIPTORS[TASK_A], "coupling": "mixture"}))
        .detach().cpu().numpy()]
    fp["B+A"] = [float(v) for v in enc.fingerprint(normalise(
        {**TASK_DESCRIPTORS[TASK_B], "coupling": "mixture"}))
        .detach().cpu().numpy()]

    # ---- rebuild the router from the sealed M171 evidence arms ------------
    # (arm fingerprints from THIS run's encoder, so arms and queries share
    # one fingerprint space; accuracy records from the sealed M171 matrix)
    router = Router()
    arm_id = {"mackey_glass": "mackey_glass-ridge",
              "lorenz": "lorenz-ridge", "tabular": "tabular-ridge",
              "dyck": "dyck-ridge"}
    kinds = {"mackey_glass": "regression", "lorenz": "regression",
             "tabular": "regression", "dyck": "classification"}
    for task, aid in arm_id.items():
        router.add_arm({
            "arm_id": aid, "task_id": task, "fingerprint": fp[task],
            "output_contract": {"kind": kinds[task],
                                "dim": 4 if task == "dyck" else 1},
            "held_out_accuracy": {task: m171["r1"][task]["best_accuracy"]},
            "selection_accuracy": float(m171["r1"][task]
                                        ["best_accuracy"]),
            "availability": {"contract_hash": "sealed-m171",
                             "payload_hash": "sealed-m171",
                             "healthy": True},
            "price": 0.0, "general": False, "primitive": False,
        })

    def head(task_name: str) -> str:
        chain = router.chain(fp[task_name], task_id=(
            task_name if task_name in arm_id else None))
        return chain[0]["arm_id"]

    # ---- the measurements -------------------------------------------------
    cos_ab = _cos(fp[TASK_A], fp[TASK_B])
    cos_joint = _cos(fp["A+B"], fp["B+A"])
    head_a, head_b = head(TASK_A), head(TASK_B)
    head_ab, head_ba = head("A+B"), head("B+A")
    head_c = head(TASK_C)

    gates = {
        "g5a_distinct_pure": {"passed": cos_ab < 0.9, "cos_ab": cos_ab},
        "g5b_distinct_joints": {"passed": cos_joint < 0.99,
                                "cos_ab_ba": cos_joint},
        "g5c_distinct_routing": {"passed": head_ab != head_ba,
                                 "head_ab": head_ab, "head_ba": head_ba},
        "g5d_dominant_side": {
            "passed": bool(head_ab == head_a and head_ba == head_b),
            "head_a": head_a, "head_ab": head_ab,
            "head_b": head_b, "head_ba": head_ba},
        "g5e_unrelated_control": {"passed": bool(head_ab != head_c),
                                  "head_c": head_c},
    }

    void = not premise["m171_admissible"]

    evidence: dict[str, Any] = {
        "milestone": "M172",
        "cell": "multi-task differentiation A/B/A+B/B+A (v0)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
        "interpretation_registered_before_running":
            config["interpretation_registered_before_running"],
        "premise": premise,
        "training_nondeterminism": nondet,
        "void": void,
        "void_reason": "M171 evidence inadmissible" if void else "",
        "gates": gates,
        "gates_all_passed": bool(all(g["passed"] for g in gates.values())),
        "cos_matrix": {"cos_ab": cos_ab, "cos_ab_ba": cos_joint,
                       "cos_ab_with_a": _cos(fp["A+B"], fp[TASK_A]),
                       "cos_ab_with_b": _cos(fp["A+B"], fp[TASK_B]),
                       "cos_ba_with_a": _cos(fp["B+A"], fp[TASK_A]),
                       "cos_ba_with_b": _cos(fp["B+A"], fp[TASK_B])},
        "routing_table": {"A": head_a, "B": head_b, "A+B": head_ab,
                          "B+A": head_ba, "C": head_c},
        "fingerprints": {n: fp[n] for n in
                         [TASK_A, TASK_B, "A+B", "B+A"]},
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"M172 complete -> {output_dir / 'evidence.json'}", flush=True)
    print(json.dumps({"void": void, "gates": gates,
                      "gates_all_passed": evidence["gates_all_passed"]},
                     indent=1)[:1600], flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m172(args.config, args.output)


if __name__ == "__main__":
    main()
