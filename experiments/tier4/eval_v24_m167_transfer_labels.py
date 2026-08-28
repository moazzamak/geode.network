"""M167 — the behavioral-transfer label protocol (v0 harness).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` (section 7
Phase A M167; the section 12 dispatch entry, 17 Aug 2026). Produces
measured transfer labels on the sealed SPM codes:

- similar: fit on domain-0 train rows, score on domain-1 test rows
  (same 345 classes, different domains);
- dissimilar control: the SAME fit on label-permuted domain-0 rows
  (permutation inside the selection destroys class structure);
- gate: similar - permuted >= 0.05 on the held-out rows, else the
  harness is VOID for label production. An own-domain sanity read is
  reported (fit domain-0, score domain-0 test rows), not gated.

Smoke declares inadmissibility and refuses the sealed output directory.
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
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator, _score
from experiments.tier4.eval_v16_m142_factorial import power_norm
from experiments.tier4.eval_v16_m109_trunk import _load_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m167_transfer_labels.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v24"
                  / "m167_transfer_labels")

CLASSES = 345


def _fit_rows(mem, labels, rows, power, block):
    acc = RidgeAccumulator(mem.shape[1], CLASSES)
    for start in range(0, len(rows), block):
        take = rows[start:start + block]
        acc.add(power_norm(mem[take], power), labels[take])
    solved = acc.solve_many([1.0])
    return solved[1.0], acc.standardiser()


def _score_rows(mem, labels, rows, weights, std, power, block):
    hits = 0
    for start in range(0, len(rows), block):
        take = rows[start:start + block]
        xs = std(power_norm(mem[take], power))
        hits += int(_score(weights, xs, labels[take]).sum())
    return hits / len(rows)


def run_m167(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    smoke = inadmissible
    skip_gate = bool(config.get("_smoke_skip_gate", False))
    smoke_train = int(config.get("_smoke_train_rows", 10 ** 9))
    smoke_test = int(config.get("_smoke_test_rows", 10 ** 9))

    configure_external_cache_environment()
    block = int(config["numerics"]["block"])
    power = float(config["sparse"]["power"])
    domain_a = int(config["pairs"]["domain_a"])
    domain_b = int(config["pairs"]["domain_b"])
    perm_seed = int(config["pairs"]["permutation_seed"])

    corpus, _, _ = _load_corpus(config)
    cache = data_cache_root() / config["artifacts"]["cache_relpath"]
    train_mem = np.load(cache / config["artifacts"]["spm_train_file"],
                        mmap_mode="r")
    test_mem = np.load(cache / config["artifacts"]["spm_test_file"],
                       mmap_mode="r")
    labels = np.load(cache / config["artifacts"]["labels_file"])["labels"]
    train_domains = corpus["train_domains"].astype(np.int64)
    test_domains = corpus["test_domains"].astype(np.int64)

    rows_a = np.where(train_domains == domain_a)[0][:smoke_train]
    rows_b_test = np.where(test_domains == domain_b)[0][:smoke_test]
    rows_a_test = np.where(test_domains == domain_a)[0][:smoke_test]
    print(f"rows: train@d{domain_a}={len(rows_a)} test@d{domain_b}="
          f"{len(rows_b_test)} test@d{domain_a}={len(rows_a_test)}",
          flush=True)

    # ---- similar: fit d_a, score d_b ---------------------------------------
    print("similar: fit d0 -> score d1", flush=True)
    w_sim, std_sim = _fit_rows(train_mem, labels, rows_a, power, block)
    test_labels = corpus["test_labels"]
    sim_acc = _score_rows(test_mem, test_labels, rows_b_test, w_sim,
                          std_sim, power, block)
    print(f"  similar {sim_acc:.6f}", flush=True)

    # ---- dissimilar: permuted labels ---------------------------------------
    print("dissimilar: permuted labels", flush=True)
    rng = np.random.default_rng(perm_seed)
    plabels = labels.copy()
    plabels[rows_a] = rng.permutation(labels[rows_a])
    w_perm, std_perm = _fit_rows(train_mem, plabels, rows_a, power, block)
    perm_acc = _score_rows(test_mem, test_labels, rows_b_test, w_perm,
                           std_perm, power, block)
    print(f"  permuted {perm_acc:.6f}", flush=True)

    # ---- own-domain sanity --------------------------------------------------
    print("sanity: fit d0 -> score d0", flush=True)
    own_acc = _score_rows(test_mem, test_labels, rows_a_test, w_sim,
                          std_sim, power, block)
    print(f"  own-domain {own_acc:.6f}", flush=True)

    # ---- two-tier gates (the M167a repair, registered) ---------------------
    validity_margin = float(config["gate"]["validity_margin"])
    transfer_margin = float(config["gate"]["transfer_margin"])
    validity_delta = own_acc - perm_acc
    transfer_delta = sim_acc - perm_acc
    validity_passed = bool(validity_delta >= validity_margin)
    transfer_passed = bool(transfer_delta >= transfer_margin)
    void = (not validity_passed) and (not skip_gate)
    evidence: dict[str, Any] = {
        "milestone": "M167",
        "cell": "behavioral-transfer label protocol v0 (positive controls)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
        "gate": config["gate"],
        "labels": {
            "similar": sim_acc,
            "permuted_control": perm_acc,
            "own_domain_sanity": own_acc,
            "validity_delta": validity_delta,
            "transfer_delta": transfer_delta,
            "validity_passed": validity_passed,
            "transfer_passed": transfer_passed,
        },
        "void": void,
        "void_reason": ("the validity gate failed: the harness cannot "
                        "order known-similar above known-dissimilar"
                        if void else ""),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM167 complete -> {output_dir / 'evidence.json'} "
          f"validity={validity_passed} transfer={transfer_passed}",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m167(args.config, args.output)


if __name__ == "__main__":
    main()
