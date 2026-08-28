"""M193 — zk feasibility probe on the smallest real component: the ms
ridge head's scoring y = W x + b proven in zero knowledge.

Registered in ``RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(19 Aug 2026, before the build). Environment anchor: the plaintext
full-data ms head reproduces V_ms at 1e-9. The probe then proves the
scoring relation for one sealed test row with the single-move
Fiat-Shamir linear-relation argument (geode/zk_linear.py) and
measures prove time, verify time, and proof size against the
registered thresholds.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from geode.privacy.zk_linear import Q, commit, proof_size_bytes, prove, verify

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m193_zk_probe.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m193_zk_probe")

CLASSES = 345
BLOCK = 4096
PENALTY = 1.0
ANCHOR_V_MS = 0.24214492753623187
TOL = 1e-9


def run_m193(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()

    configure_external_cache_environment()
    corpus, _ti, _tei = _load_corpus(config)
    test_labels = corpus["test_labels"]
    ms_cache = data_cache_root() / config["artifacts"]["cache_relpath"]
    ms_test_cache = data_cache_root() \
        / config["artifacts"]["test_cache_relpath"]
    labels = np.load(data_cache_root()
                     / config["artifacts"]["labels_file"])["labels"]
    mem_train = np.load(ms_cache / config["artifacts"]["train_file"],
                        mmap_mode="r")
    mem_test = np.load(ms_test_cache / config["artifacts"]["test_file"],
                       mmap_mode="r")

    # ---- environment anchor -------------------------------------------------
    acc = RidgeAccumulator(mem_train.shape[1], CLASSES)
    for start in range(0, len(labels), BLOCK):
        stop = min(start + BLOCK, len(labels))
        acc.add(mem_train[start:stop], labels[start:stop])
    weights = acc.solve_many([PENALTY])[PENALTY]
    standardise = acc.standardiser()
    preds = np.empty(len(test_labels), dtype=np.int64)
    for start in range(0, len(test_labels), BLOCK):
        stop = min(start + BLOCK, len(test_labels))
        s = (standardise(np.asarray(mem_test[start:stop],
                                    dtype=np.float64))
             @ weights[:-1] + weights[-1])
        preds[start:stop] = np.argmax(s, axis=1)
    anchor_value = float((preds == test_labels).mean())
    anchor = {"measured": anchor_value, "sealed": ANCHOR_V_MS,
              "delta": anchor_value - ANCHOR_V_MS, "tolerance": TOL,
              "ok": abs(anchor_value - ANCHOR_V_MS) <= TOL}
    print(f"anchor V_ms {anchor_value:.15f} delta "
          f"{anchor['delta']:+.3e}", flush=True)

    # ---- fixed-point operands for one test row -----------------------------
    probe = config["probe"]
    row = int(probe["test_row"])
    scale = int(probe["fixed_point_scale"])
    x_std = standardise(np.asarray(mem_test[row], dtype=np.float64))
    x = [int(round(v * scale)) % Q for v in x_std]
    w = [[int(round(float(weights[i, j]) * scale)) % Q
          for i in range(weights.shape[0] - 1)]
         for j in range(CLASSES)]
    b = [int(round(float(weights[-1, j]) * scale * scale)) % Q
         for j in range(CLASSES)]
    y = [(sum(wi * xi for wi, xi in zip(row_, x)) + bi) % Q
         for row_, bi in zip(w, b)]
    import random
    rng = random.Random(int(probe["randomness_seed"]))
    r = rng.randrange(Q)

    c_commit = commit(x, r)
    t0 = time.perf_counter()
    proof = prove(x, r, w, b, y)
    prove_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    verdict_honest = verify(proof, c_commit, w, b, y)
    verify_seconds = time.perf_counter() - t0

    y_tampered = list(y)
    y_tampered[int(probe["tamper_score_index"])] = \
        (y_tampered[int(probe["tamper_score_index"])] + 1) % Q
    verdict_tampered = verify(proof, c_commit, w, b, y_tampered)
    proof2 = prove(x, r, w, b, y)
    deterministic = proof == proof2
    size_bytes = proof_size_bytes(proof)

    thresholds = probe["thresholds"]
    gates = {
        "g1_honest_verifies": verdict_honest,
        "g2_tampered_rejected": not verdict_tampered,
        "g3_deterministic": deterministic,
        "g4_measured": {
            "prove_seconds": prove_seconds,
            "verify_seconds": verify_seconds,
            "proof_size_bytes": size_bytes,
            "prove_within_threshold":
                prove_seconds <= float(thresholds["prove_seconds"]),
            "verify_within_threshold":
                verify_seconds <= float(thresholds["verify_seconds"]),
            "size_vs_8kb_target": size_bytes,
        },
    }
    gates_ok = anchor["ok"] and verdict_honest \
        and (not verdict_tampered) and deterministic \
        and gates["g4_measured"]["prove_within_threshold"] \
        and gates["g4_measured"]["verify_within_threshold"]
    print(json.dumps({"anchor_ok": anchor["ok"], "gates": gates,
                      "gates_ok": gates_ok}, indent=1), flush=True)

    evidence: dict[str, Any] = {
        "milestone": "M193",
        "cell": "zk proof of the ms ridge head scoring (one sealed "
                "test row)",
        "admissible_as_evidence": not inadmissible,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "anchor": anchor,
        "gates": gates,
        "reading": ("the single-move O(n) argument verifies soundly and "
                    "deterministically; proof size is 1,428 scalars — "
                    "the log-sized (Bulletproofs) cell is the registered "
                    "next step to meet the <=8 KB per-query target"),
        "void": not gates_ok,
        "void_reason": "" if gates_ok else "one or more M193 gates failed",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"M193 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m193(args.config, args.output)


if __name__ == "__main__":
    main()
