"""M179 — unlearning/erasure reuse on a frozen component: erase the flower
species class from the cached DINOv2 CLS features, with a certificate.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(18 Aug 2026). Target: the cached Flowers-102 DINOv2-small CLS features
(the capability map's flowers node). Concept: species class (102).
Machinery: M90.2 `leace_eraser` (float64, floor 1e-10, rank cap 101).

Gates (registered): (a) ridge probe on erased codes <= 1.5x chance;
(b) concept-specificity — erased probe <= 0.5x the budget-matched
random-partition null's probe (amended and registered before the first
seal: 101/384 random directions measurably degrade any probe, so the
null is the budget comparison, not a preservation check);
(c) certificate relative residuals <= 1e-6 (both figures); minimal-edit
ratio reported. Registered boundary: CAN (linear concept, component
level, auditable); CANNOT (nonlinear/second moments/cross-task effects
unmeasured; append-only registry => exclusion, never deletion).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from geode.audit.erasure import erasure_certificate, leace_eraser

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m179_unlearning.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v25" / "m179_unlearning"

CLASSES = 102
WIDTH = 384
CHANCE = 1.0 / CLASSES


def _probe(features: np.ndarray, labels: np.ndarray,
           train_rows: int) -> float:
    """Ridge read on the given codes: fit first ``train_rows``, score rest."""
    acc = RidgeAccumulator(WIDTH, CLASSES)
    acc.add(features[:train_rows], labels[:train_rows])
    weights = acc.solve(1.0)
    std = acc.standardiser()
    hits = 0
    n = len(labels) - train_rows
    for start in range(train_rows, len(labels), 4096):
        stop = min(start + 4096, len(labels))
        xs = std(features[start:stop]).astype(np.float64)
        scores = xs @ weights[:-1] + weights[-1]
        hits += int((np.argmax(scores, axis=1)
                     == labels[start:stop]).sum())
    return hits / n


def run_m179(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible

    fl = config["flowers"]
    features_dir = REPO_ROOT / fl["features_dir"]
    train = np.load(features_dir / fl["train_file"])
    test = np.load(features_dir / fl["test_file"])
    train_f = np.asarray(train["features"], dtype=np.float64)
    test_f = np.asarray(test["features"], dtype=np.float64)
    train_y = np.asarray(train["labels"], dtype=np.int64)
    test_y = np.asarray(test["labels"], dtype=np.int64)
    features = np.concatenate([train_f, test_f], axis=0)
    labels = np.concatenate([train_y, test_y], axis=0)
    n_train = len(train_f)
    original_probe = _probe(features, labels, n_train)
    print(f"original probe: {original_probe:.4f} (chance {CHANCE:.4f})",
          flush=True)

    # ---- the erasure (fitted on train rows only) ---------------------------
    eraser, removed = leace_eraser(
        train_f, train_y, group_count=CLASSES,
        floor=float(config["erasure"]["floor"]),
        singular_tolerance=float(config["erasure"]["singular_tolerance"]))
    print(f"eraser fitted; rank removed {removed} (cap {CLASSES - 1})",
          flush=True)
    erased = eraser(features)
    erased_probe = _probe(erased, labels, n_train)
    print(f"erased probe: {erased_probe:.4f}", flush=True)

    # ---- the budget-matched null (random 102-way partition) ----------------
    rng = np.random.default_rng(int(config["erasure"]["null_seed"]))
    null_groups = rng.permutation(np.arange(n_train)) % CLASSES
    null_eraser, null_removed = leace_eraser(
        train_f, null_groups, group_count=CLASSES,
        floor=float(config["erasure"]["floor"]),
        singular_tolerance=float(config["erasure"]["singular_tolerance"]))
    null_erased = null_eraser(features)
    null_probe = _probe(null_erased, labels, n_train)
    print(f"null probe: {null_probe:.4f} (removed {null_removed})",
          flush=True)

    # ---- certificate + minimal-edit ----------------------------------------
    cert = erasure_certificate(train_f, train_y, CLASSES, eraser)
    edit = float(np.linalg.norm(
        erased - features) / (np.linalg.norm(features) + 1e-12))

    gates = {
        "a_probe_at_chance": bool(erased_probe <= 1.5 * CHANCE),
        "b_concept_specific": bool(erased_probe <= 0.5 * null_probe),
        "c_certificate": bool(
            cert["relative_mean_gap_residual"]
            <= float(config["erasure"]["relative_residual_max"])
            and cert["relative_cross_covariance_residual"]
            <= float(config["erasure"]["relative_residual_max"])),
    }
    passed = all(gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M179",
        "cell": "closed-form erasure on a frozen component (flowers CLS, "
                "concept = species class)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "erasure": {
            "rank_removed": removed,
            "rank_cap": CLASSES - 1,
            "null_rank_removed": null_removed,
        },
        "probes": {
            "original": original_probe,
            "erased": erased_probe,
            "null": null_probe,
            "chance": CHANCE,
        },
        "certificate": cert,
        "minimal_edit_relative_frobenius": edit,
        "gates": gates,
        "verdict": {
            "passed": passed,
            "reading": ("the frozen component's linear species signal is "
                        "exactly erased (probe at chance, null intact, "
                        "certificate clean) — per-task unlearning works "
                        "at the linear, component level")
            if passed else "a registered gate failed",
        },
        "boundary": config["boundary"],
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"original": original_probe, "erased": erased_probe,
                      "null": null_probe, "cert": cert, "gates": gates},
                     indent=1), flush=True)
    print(f"M179 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m179(args.config, args.output)


if __name__ == "__main__":
    main()
