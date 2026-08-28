"""M152 — power-exponent refinement on cached codes (two-stage).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v23.md`` (section 4
M152; 16 Aug 2026). Two stages, registered to bound compute:

- STAGE 1 screens p in {0.25, 0.33, 0.66} at 138k against the sealed
  p=0.5/p=1.0 reads (the C4 protocol: penalty ladder {0.1, 1.0, 10.0};
  the 138k anchor is the penalty-1.0 read 0.2273623188405797, C4's
  cells_138k protocol).
- STAGE 2 promotes AT MOST ONE p — the stage-1 winner (best 138k cell
  above p=0.5's 138k read) — to the full-data fit, gated: best full-data
  cell >= 0.278550724637681 + 0.005 at identical cost, else archived as
  a scoped negative. Trained-head read at the promoted p (the C4
  protocol: SGD, 8 epochs, lr 0.001, seed 201).

If no new p beats p=0.5 at 138k, nothing is promoted and the cell closes
as a scoped negative at the screen (the p=0.5 recipe stands; the
full-data stage is not run). Smoke declares inadmissibility and refuses
the sealed output directory.

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v23_m152_pgrid
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m142_c4 import (
    _fit_power,
    _score_power,
    _trained_head_read,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v23"
                  / "m152_pgrid.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v23" / "m152_pgrid"

TOLERANCE = 1e-9
MARGIN = 0.005
FULL_BEST_REFERENCE = 0.27855072463768116
P_ANCHOR_138K = {0.5: 0.2273623188405797, 1.0: 0.2106376811594203}


def run_m152(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    smoke = inadmissible
    skip_anchors = bool(config.get("_smoke_skip_anchors", False))
    smoke_train = int(config.get("_smoke_train_rows", 10 ** 9))
    smoke_test = int(config.get("_smoke_test_rows", 10 ** 9))
    block = int(config["numerics"]["block"])

    torch.set_num_threads(int(config["numerics"]["torch_threads"]))
    configure_external_cache_environment()
    _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    cache = data_cache_root() / config["artifacts"]["m142_cache_relpath"]
    evidence: dict[str, Any] = {
        "milestone": "M152",
        "cell": "power-exponent refinement on cached codes",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
    }

    print("loading corpus + cached codes", flush=True)
    corpus, _train_index, _test_index = _load_corpus(config)
    test_labels = corpus["test_labels"][:smoke_test]
    test_domains = corpus["test_domains"][:smoke_test]
    labels = np.load(cache / config["artifacts"]["labels_file"])["labels"]
    train_mem = np.load(cache / config["artifacts"]["spm_train_file"],
                        mmap_mode="r")
    test_mem = np.load(cache / config["artifacts"]["spm_test_file"],
                       mmap_mode="r")[:smoke_test]
    n_full = min(len(labels), len(train_mem), smoke_train)
    n_138 = min(int(config["cell"]["n_138k"]), n_full)
    penalties = [float(q) for q in config["cell"]["penalty_ladder"]]
    p_screen = [float(p) for p in config["cell"]["p_screen"]]
    p_anchors = [float(p) for p in config["cell"]["p_anchors"]]
    print(f"rows: full {n_full} / 138k {n_138} / test {len(test_labels)}",
          flush=True)

    # ---- stage 1: 138k screen (anchors + new p values) ----------------------
    anchors: dict[str, Any] = {}
    cells_138k: dict[str, Any] = {}
    p_best_138k: dict[float, float] = {}
    for p in sorted(set(p_anchors + p_screen)):
        print(f"stage 1: p={p} at 138k", flush=True)
        solved, std = _fit_power(train_mem, labels, p, penalties, n_138,
                                 block, transform=True)
        best = -1.0
        for q in penalties:
            acc = _score_power(test_mem, test_labels, test_domains, p,
                               solved[str(q)], std, block, transform=True)
            cells_138k[f"p{p}_lambda{q}"] = acc
            best = max(best, acc)
        p_best_138k[p] = best
        if p in p_anchors:
            anchored = P_ANCHOR_138K[p]
            anchors[f"p{p}_138k_lambda1"] = {
                "measured": cells_138k[f"p{p}_lambda1.0"],
                "sealed": anchored,
                "delta": cells_138k[f"p{p}_lambda1.0"] - anchored,
                "tolerance": TOLERANCE,
            }
            print(f"  anchor p={p} lambda1: "
                  f"{cells_138k[f'p{p}_lambda1.0']:.6f} "
                  f"(delta {cells_138k[f'p{p}_lambda1.0'] - anchored:+.3e})",
                  flush=True)
            if not skip_anchors and abs(cells_138k[f"p{p}_lambda1.0"]
                                        - anchored) > TOLERANCE:
                evidence.update({"void": True,
                                 "void_reason": f"p={p} 138k anchor failed",
                                 "anchors": anchors})
                _write(output_dir, evidence)
                return evidence

    # ---- promotion decision ------------------------------------------------
    p05_best = p_best_138k[0.5]
    winners = [p for p in p_screen if p_best_138k[p] > p05_best]
    evidence["stage1"] = {"cells_138k": cells_138k,
                          "p_best_138k": {str(p): v for p, v in
                                          p_best_138k.items()},
                          "p05_best_138k": p05_best,
                          "winners": [str(p) for p in winners]}

    if not winners:
        print("no new p beats p=0.5 at 138k; nothing promoted", flush=True)
        evidence.update({
            "anchors": anchors,
            "stage2": {"promoted": None,
                       "note": "no promotion; the p=0.5 recipe stands and "
                               "the full-data stage is not run"},
            "gate": {"registered": config["gate"]["registered"],
                     "fired": True,
                     "consequence": "scoped negative at the screen: the "
                                    "p=0.5 recipe stands"},
            "runtime_seconds": round(time.time() - started, 2),
        })
        _write(output_dir, evidence)
        print(f"\nM152 complete -> {output_dir / 'evidence.json'}", flush=True)
        return evidence

    promoted = max(winners, key=lambda p: p_best_138k[p])
    print(f"stage 2: promoting p={promoted} to full data", flush=True)
    solved, std = _fit_power(train_mem, labels, promoted, penalties, n_full,
                             block, transform=True)
    cells_full: dict[str, Any] = {}
    best_key, best_acc = None, -1.0
    for q in penalties:
        acc = _score_power(test_mem, test_labels, test_domains, promoted,
                           solved[str(q)], std, block, transform=True)
        key = f"p{promoted}_lambda{q}"
        cells_full[key] = acc
        print(f"  {key}: {acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc, best_key = acc, key

    print(f"trained-head read at p={promoted}", flush=True)
    trained = _trained_head_read(train_mem, labels, test_mem, test_labels,
                                 promoted, block, 8, 0.001, 201, device)

    gain = best_acc - FULL_BEST_REFERENCE
    fired = gain < MARGIN
    both_fail = fired and trained < FULL_BEST_REFERENCE + MARGIN
    evidence.update({
        "anchors": anchors,
        "stage2": {"promoted": promoted, "cells_full": cells_full,
                   "best_cell": best_key, "best_accuracy": best_acc,
                   "trained_head_read": trained},
        "gate": {"registered": config["gate"]["registered"],
                 "incumbent": FULL_BEST_REFERENCE,
                 "gain": gain,
                 "required": MARGIN,
                 "fired": fired,
                 "both_reads_fail": bool(both_fail),
                 "consequence": ("scoped negative: no refined p beats the "
                                 "sealed p=0.5 recipe" if both_fail
                                 else "the refined p is promoted")},
        "runtime_seconds": round(time.time() - started, 2),
    })
    _write(output_dir, evidence)
    print(f"\nM152 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def _write(output_dir: Path, evidence: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m152(args.config, args.output)


if __name__ == "__main__":
    main()
