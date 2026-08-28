"""M156 smoke-premise probe (premise-only; no accuracy claims, no gates).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v23.md`` section 6
(16 Aug 2026): the first M156 smoke stopped on its premise gate because
at 20k train rows the SPM+sqrt head interpolates its training rows
(0 error rows). This probe fits the SPM+sqrt head (the M155 protocol:
``_fit_power`` p=0.5, penalty 0.1, block 4096) at n in
{20000, 40000, 60000, 80000, 100000} and records the n-row head's
train error-row count on its own n rows. The smallest probed n with a
nonzero population becomes the registered smoke row count. Only error
COUNTS are read.
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
from experiments.tier4.eval_v16_m142_c4 import _fit_power
from experiments.tier4.eval_v16_m142_factorial import power_norm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v23"
                  / "m156_smoke_premise_probe")
CLASSES = 345
BLOCK = 4096
NS = [20000, 40000, 60000, 80000, 100000]


def run_probe(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    configure_external_cache_environment()
    root = data_cache_root()
    m142_cache = root / "v16" / "m142_c2"
    spm_train = np.load(m142_cache / "spm1923_fulltrain.npy", mmap_mode="r")
    labels = np.load(m142_cache / "m142_c2_fulltrain_labels.npz")["labels"]
    out: dict[str, Any] = {}
    print(f"probing n in {NS}", flush=True)
    for n in NS:
        print(f"n={n}: fit SPM+sqrt head", flush=True)
        solved, std = _fit_power(spm_train, labels, 0.5, [0.1], n, BLOCK,
                                 transform=True)
        weights = solved["0.1"]
        hits = 0
        for start in range(0, n, BLOCK):
            stop = min(start + BLOCK, n)
            xs = std(power_norm(np.asarray(spm_train[start:stop]), 0.5))
            scores = xs @ weights[:-1] + weights[-1]
            hits += int((np.argmax(scores, axis=1)
                         == labels[start:stop]).sum())
        n_err = n - hits
        out[str(n)] = {"train_accuracy": hits / n,
                       "n_error_rows": int(n_err)}
        print(f"n={n}: train errors {n_err} ({hits / n:.4f})", flush=True)
    evidence: dict[str, Any] = {
        "kind": "premise probe (no accuracy claim, no gate)",
        "registered_in": "analysis/RESEARCH_IMPLEMENTATION_PLAN_v23.md "
                         "section 6 (16 Aug 2026)",
        "question": "at which n does the n-row SPM+sqrt head (p=0.5, "
                    "penalty 0.1, the M155 protocol) leave a nonzero "
                    "train error population?",
        "per_n": out,
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nprobe complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_probe(args.output)


if __name__ == "__main__":
    main()
