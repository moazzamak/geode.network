"""M173 — MVP acceptance run: capabilities 1-10 with every gate and
anchor recorded.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` (section 7
Phase C M173; section 6 capability list). This is the milestone the user
revisits: it does NOT fit anything new. It assembles the sealed M165-M172
evidence into one pass/fail report per MVP capability, re-verifies the
registries' transactional hashes, re-runs the registered unit tests, and
re-hashes every sealed evidence file against its artifact index (I5).

Fresh checks only where a capability has no sealed number yet:
capability 1 (a no-crash normalisation loop over the registered
descriptors plus a genuine OOV descriptor) and capability 10 (file-level
sha256 re-verification). No GPU, no training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v24_m169_fingerprint_train import TASK_DESCRIPTORS
from geode.core.descriptor import normalise

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m173_acceptance.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v24"
                  / "m173_acceptance")

SEALED = {
    "m169": REPO_ROOT / "logs" / "results" / "v24"
            / "m169_fingerprint_gates" / "evidence.json",
    "m171": REPO_ROOT / "logs" / "results" / "v24" / "m171_router"
            / "evidence.json",
    "m172": REPO_ROOT / "logs" / "results" / "v24" / "m172_joint"
            / "evidence.json",
    "m142_c4": REPO_ROOT / "logs" / "results" / "v16" / "m142_c4"
               / "evidence.json",
}
M170_ARTIFACT = (REPO_ROOT / "analysis" / "traversability_set_v0.md")
RIDGE_ANCHOR = 0.2273623188405797
ANCHOR_TOL = 1e-6
UNIT_TESTS = [
    "experiments.common.test_v24_m165_registry",
    "experiments.common.test_v24_m168_fingerprint",
    "experiments.common.test_v24_m171_router",
]
OOV_DESCRIPTOR = {
    "input.modality": "audio", "input.submodality": "spectrogram",
    "input.value_kind": "continuous", "input.temporal_structure": "sequential",
    "output.kind": "regression", "output.ordinality": "cardinal",
    "latent.recurrence": "markov", "latent.stationarity": "stationary",
    "latent.noise_regime": "medium", "latent.label_cardinality": 1,
    "latent.sample_regime": "small", "coupling": "single-task",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str) -> dict[str, Any]:
    return json.loads(SEALED[name].read_text(encoding="utf-8"))


def run_m173(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()

    m169, m171, m172, m142 = (_load(k) for k in
                              ("m169", "m171", "m172", "m142_c4"))
    caps: dict[int, dict[str, Any]] = {}

    # ---- capability 1: task ingestion (fresh, no crash, I4) --------------
    try:
        descs = {n: normalise(d) for n, d in TASK_DESCRIPTORS.items()}
        oov = normalise(OOV_DESCRIPTOR)
        hashes = {n: d.hash() for n, d in descs.items()}
        distinct = len(set(hashes.values()))
        oov_logged = any(e["kind"] == "oov" for e in oov.events)
        caps[1] = {
            "passed": bool(distinct == 5 and oov_logged),
            "distinct_hashes_of_6": distinct,
            "expected": 5,
            "oov_event_logged": oov_logged,
            "note": "mg and lorenz share a descriptor under ontology v0 "
                    "(the registered identical pair), so 6 tasks -> 5 "
                    "distinct hashes",
        }
    except Exception as exc:  # noqa: BLE001 - acceptance must not crash
        caps[1] = {"passed": False, "error": repr(exc)}

    # ---- capabilities 2-4: the M169 gates --------------------------------
    caps[2] = {"passed": bool(m169["gates"]["g1_deterministic"]),
               "g1": m169["gates"]["g1_deterministic"]}
    caps[3] = {"passed": bool(m169["gates"]["g2_similarity_ordering"]
                              ["passed"]),
               "margin": m169["gates"]["g2_similarity_ordering"]["margin"]}
    caps[4] = {"passed": bool(m169["gates"]["g3_traversality"]["passed"]
                              and M170_ARTIFACT.exists()),
               "min_cos": m169["gates"]["g3_traversality"]["min_cos"],
               "artifact": M170_ARTIFACT.as_posix()}

    # ---- capability 5: routing (R1 + the registered eps-advance rule) ----
    caps[5] = {
        "passed": bool(m171["r1_eps_advance_all_passed"]),
        "r1_raw": m171["r1"],
        "r1_eps_advance": m171["r1_eps_advance"],
        "eps_registered": m171["eps_registered"],
        "note": "raw nearest-arm R1 is 3/4 (dyck negative: bigram primitive "
                "beats the specialist); the registered section-5 "
                "eps-advance rule recovers 4/4",
    }

    # ---- capability 6: fit-and-report + anchor reproduction ---------------
    anchor = m142["cells_138k"]["p0.5"]
    caps[6] = {
        "passed": bool(m171["admissible_as_evidence"]
                       and len(m171["competence_matrix"]) >= 4
                       and abs(anchor - RIDGE_ANCHOR) <= ANCHOR_TOL),
        "sealed_anchor": anchor,
        "registered_anchor": RIDGE_ANCHOR,
        "abs_delta": abs(anchor - RIDGE_ANCHOR),
        "fitted_tasks": len(m171["competence_matrix"]),
    }

    # ---- capability 7: registry operations (transactional, I1) -----------
    tx = m171["registry_transactional_hashes"]
    keys = sorted(tx)
    stable = True
    for i in range(1, len(keys)):
        prev, cur = tx[keys[i - 1]], tx[keys[i]]
        for tid in prev:
            if tid in cur and prev[tid] != cur[tid]:
                stable = False
    suite = subprocess.run(
        [sys.executable, "-m", "unittest", *UNIT_TESTS],
        capture_output=True, text=True, cwd=REPO_ROOT)
    caps[7] = {
        "passed": bool(stable and suite.returncode == 0),
        "transactional_hashes_stable": stable,
        "unit_tests_returncode": suite.returncode,
        "unit_test_tail": suite.stdout.strip().splitlines()[-2:]
        if suite.stdout.strip() else suite.stderr.strip()
        .splitlines()[-2:],
    }

    # ---- capability 8: multi-task differentiation (G5) --------------------
    caps[8] = {"passed": bool(not m172["void"]
                              and m172["gates_all_passed"]),
               "gates": m172["gates"],
               "training_nondeterminism_min_cos":
                   m172["premise"]["training_nondeterminism_min_cos"]}

    # ---- capability 9: cold start (I4 fallback, measured) -----------------
    cold = m171["cold_start"]
    caps[9] = {
        "passed": bool(cold["audio"]["fallback_arm"] ==
                       "mean-mode-primitive"
                       and cold["cifar10"]["fallback_arm"] ==
                       "mean-mode-primitive"),
        "audio": cold["audio"],
        "cifar10": cold["cifar10"],
        "note": "gated on the deterministic fallback happening, not on "
                "beating specialists (registered)",
    }

    # ---- capability 10: repro-hash (I5) on every sealed decision ----------
    hash_ok = {}
    for name, path in SEALED.items():
        idx_path = path.parent / "artifact_index.json"
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        entry = next(e for e in idx["artifacts"]
                     if e["path"] == path.name)
        hash_ok[name] = bool(_sha256(path) == entry["sha256"])
    caps[10] = {"passed": bool(all(hash_ok.values())),
                "per_file": hash_ok}

    all_passed = bool(all(c["passed"] for c in caps.values()))
    evidence: dict[str, Any] = {
        "milestone": "M173",
        "cell": "MVP acceptance: capabilities 1-10",
        "admissible_as_evidence": not inadmissible,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "question": config["question"],
        "interpretation_registered_before_running":
            config["interpretation_registered_before_running"],
        "capabilities": caps,
        "all_capabilities_passed": all_passed,
        "g4_continuity": "DEFERRED (no sweep families; registered pending)",
        "m163_corpus_decision": "pending (registered, not an acceptance "
                                "blocker)",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"M173 complete -> {output_dir / 'evidence.json'}", flush=True)
    print(json.dumps({"all_capabilities_passed": all_passed,
                      "per_capability": {str(k): v["passed"]
                                         for k, v in caps.items()}},
                     indent=1), flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m173(args.config, args.output)


if __name__ == "__main__":
    main()
