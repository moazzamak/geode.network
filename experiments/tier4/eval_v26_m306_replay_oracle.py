"""M306 - the canonical replay oracle audit.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md``
§8.31 (27 Aug 2026, before any build). Two runs make the audit:

- **Config A** (default process): reproduce the sealed heads
  bit-exactly from the sealed 409,832-row schedule and compare
  against the M322e head cache (gates G1a/G1b), and register the
  repaired head's digest.
- **Config B** (subprocess with the BLAS thread count pinned to
  one): recompute the same heads and digests in a fresh process.

G2 compares the two configurations' digests bit-exactly. The
result is measured and reported whatever it is: a mismatch is the
honest answer to H26-5's cross-hardware question and forces the
registered R-A6d margin-gated probe. The cross-MACHINE half of
H26-5 stays open (one machine offers configurations, not
machines) and is recorded as pending.

Evidence: ``logs/results/v26/m306_replay_oracle/`` with
``config_a/`` and ``config_b/`` sub-records.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from geode.core.replay_oracle import (
    SEALED_BLOCK,
    SealedSystem,
    hardware_signature,
    head_digest,
    oracle_id,
    repaired_head,
    sealed_lu_head,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
M298_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
               / "m298_lda_balanced.json")
M297_EVIDENCE = (REPO_ROOT / "logs" / "results" / "v26"
                 / "m297_loocv_lambda" / "evidence.json")
HEADS_CACHE = (REPO_ROOT / "logs" / "results" / "v26"
               / "m322_fhe_quant" / "heads_cache.npz")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v26" / "m306_replay_oracle"

CLASSES = 345
PENALTY = 1.0


def run_replay(mem_train: np.ndarray, labels: np.ndarray,
               lambda_star: float) -> dict[str, Any]:
    """One full canonical replay of the sealed schedule. Returns the
    head records and the head arrays (for the bitwise gates)."""
    started = time.time()
    sys_obj = SealedSystem.accumulate(mem_train, labels,
                                      classes=CLASSES, block=SEALED_BLOCK)
    centred, cross, intercept = sys_obj.standardised_system()

    sealed_w, sealed_b = sealed_lu_head(centred, cross, intercept,
                                        PENALTY)
    star_w, star_b = sealed_lu_head(centred, cross, intercept,
                                    lambda_star)
    repaired, report = repaired_head(centred, cross, intercept, PENALTY)
    return {
        "oracle": oracle_id(),
        "inputs_digest": payload_hash({
            "rows": int(len(mem_train)),
            "width": int(mem_train.shape[1]),
            "labels_rows": int(len(labels)),
        }),
        "heads": {
            "sealed_ridge_lu": {
                "digest": head_digest(sealed_w, sealed_b)},
            "lambda_star_ridge_lu": {
                "digest": head_digest(star_w, star_b)},
            "repaired_ridge": {
                "digest": head_digest(repaired[:-1], repaired[-1]),
                "solve_path": report["solve_path"],
                "backward_passed": report["backward_passed"],
                "symmetric_to_bit": report["symmetric_to_bit"],
            },
        },
        "arrays": {
            "sealed": (sealed_w, sealed_b),
            "star": (star_w, star_b),
        },
        "runtime_seconds": round(time.time() - started, 2),
    }


def _cache_gates(record: dict[str, Any]) -> dict[str, Any]:
    """G1: the measured heads must reproduce the M322e head cache
    bit-exactly (array_equal)."""
    cached = np.load(HEADS_CACHE)
    ref = {
        "sealed": (np.asarray(cached["sealed_ridge_W"]),
                   np.asarray(cached["sealed_ridge_b"])),
        "star": (np.asarray(cached["lambda_star_ridge_W"]),
                 np.asarray(cached["lambda_star_ridge_b"])),
    }
    del cached
    results: dict[str, Any] = {}
    for key, gate in (("sealed", "g1a_sealed_lu_penalty1_bit_exact"),
                      ("star", "g1b_lambda_star_lu_bit_exact")):
        measured = record["arrays"][key]
        results[gate] = {
            "ok": bool(np.array_equal(measured[0], ref[key][0])
                       and np.array_equal(measured[1], ref[key][1])),
            "measured_digest": head_digest(*measured),
            "expected_digest": head_digest(*ref[key]),
        }
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-b", action="store_true",
                    help="pinned single-thread BLAS configuration run")
    ap.add_argument("--no-child", action="store_true",
                    help="config A only; do not spawn config B")
    args = ap.parse_args()

    configure_external_cache_environment()
    root = data_cache_root()
    cfg = json.loads(M298_CONFIG.read_text(encoding="utf-8"))

    _corpus, _train_index, _test_index = _load_corpus(cfg)
    ms_cache = root / cfg["artifacts"]["cache_relpath"]
    mem_train = np.load(ms_cache / cfg["artifacts"]["train_file"],
                        mmap_mode="r")
    labels = np.load(root / cfg["artifacts"]["labels_file"])["labels"]
    m297 = json.loads(M297_EVIDENCE.read_text(encoding="utf-8"))
    lambda_star = float(m297["lambda_star"])

    record = run_replay(mem_train, labels, lambda_star)
    record["hardware"] = hardware_signature()
    record["premise"] = {
        "train_rows": int(len(mem_train)),
        "label_rows": int(len(labels)),
        "width": int(mem_train.shape[1]),
        "lambda_star": lambda_star,
    }
    record["gates"] = _cache_gates(record)
    record["g1"] = all(g["ok"] for g in record["gates"].values())
    record.pop("arrays", None)  # digests carry the record; arrays stay in the cache
    if args.config_b:
        record["config_b_run"] = True
        out = DEFAULT_OUTPUT / "config_b"
        out.mkdir(parents=True, exist_ok=True)
        write_canonical_json(out / "evidence.json", record)
        build_artifact_index(out)
        print(json.dumps({"g1": record["g1"]}))
        return 0 if record["g1"] else 1

    # ---- G2: spawn config B with the BLAS threads pinned to one ----
    if not args.no_child:
        env = dict(os.environ)
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        child = subprocess.run(
            [sys.executable, "-m",
             "experiments.tier4.eval_v26_m306_replay_oracle", "--config-b"],
            env=env, capture_output=True, text=True, cwd=REPO_ROOT)
        child_evidence = DEFAULT_OUTPUT / "config_b" / "evidence.json"
        child_record = (json.loads(child_evidence.read_text(
            encoding="utf-8")) if child_evidence.exists() else None)
        if child_record is not None:
            child_heads = child_record.get("heads", {})
            g2 = bool(
                child_heads.get("sealed_ridge_lu", {}).get("digest")
                == record["heads"]["sealed_ridge_lu"]["digest"]
                and child_heads.get("lambda_star_ridge_lu", {}).get("digest")
                == record["heads"]["lambda_star_ridge_lu"]["digest"]
                and child_heads.get("repaired_ridge", {}).get("digest")
                == record["heads"]["repaired_ridge"]["digest"])
            record["gates"]["g2_cross_configuration_digests_bit_exact"] = {
                "ok": g2,
                "config_a_threads": record["hardware"]["threads"],
                "config_b_threads": child_record["hardware"]["threads"],
                "child_g1": child_record.get("g1"),
            }
            record["g2"] = g2
        else:
            record["g2"] = None
            record["gates"]["g2_cross_configuration_digests_bit_exact"] = {
                "ok": False, "note": "config B evidence missing"}
        record["child_returncode"] = child.returncode
        if child.stdout:
            record["child_stdout_tail"] = (
                child.stdout.strip().splitlines()[-3:])
        if child.stderr:
            record["child_stderr_tail"] = (
                child.stderr.strip().splitlines()[-3:])
    else:
        record["g2"] = None
        record["gates"]["g2_cross_configuration_digests_bit_exact"] = {
            "ok": False, "note": "skipped (--no-child)"}

    record["cross_machine_pending"] = (
        "H26-5 asks for at least two distinct hardware configurations; "
        "this machine offers two execution configurations only. The "
        "second-machine comparison is registered as pending.")
    out = DEFAULT_OUTPUT / "config_a"
    out.mkdir(parents=True, exist_ok=True)
    write_canonical_json(out / "evidence.json", record)
    build_artifact_index(out)
    print(json.dumps({"g1": record["g1"], "g2": record["g2"]}))
    return 0 if (record["g1"] and record["g2"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
