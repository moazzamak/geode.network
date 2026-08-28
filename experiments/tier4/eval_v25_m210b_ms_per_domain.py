"""M210b — the ms family's per-domain accuracies, measured on the
sealed test with the M180 bit-exact ridge path, so every orchestration
arm carries task-level data instead of the aggregate fallback.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(19 Aug 2026, before the build). Anchor: the aggregate must reproduce
the sealed V_ms = 0.24214492753623187 at 1e-9.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m210b_ms_per_domain.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m210b_ms_per_domain")

CLASSES = 345
DOMAINS = 6
BLOCK = 4096
PENALTY = 1.0
ANCHOR_V_MS = 0.24214492753623187
TOL = 1e-9


def run_m210b(config_path: Path, output_dir: Path) -> dict[str, Any]:
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
    test_domains = corpus["test_domains"]

    ms_cache = data_cache_root() / config["artifacts"]["cache_relpath"]
    ms_test_cache = data_cache_root() \
        / config["artifacts"].get("test_cache_relpath",
                                  config["artifacts"]["cache_relpath"])
    labels = np.load(data_cache_root()
                     / config["artifacts"]["labels_file"])["labels"]
    mem_train = np.load(ms_cache / config["artifacts"]["train_file"],
                        mmap_mode="r")
    mem_test = np.load(ms_test_cache / config["artifacts"]["test_file"],
                       mmap_mode="r")
    width = mem_train.shape[1]
    if len(mem_train) != len(labels):
        raise SystemExit("M210b premise failure: ms train rows != labels")
    print(f"ms train {mem_train.shape}, test {mem_test.shape}",
          flush=True)

    acc = RidgeAccumulator(width, CLASSES)
    for start in range(0, len(labels), BLOCK):
        stop = min(start + BLOCK, len(labels))
        acc.add(mem_train[start:stop], labels[start:stop])
    weights = acc.solve_many([PENALTY])[PENALTY]
    standardise = acc.standardiser()

    per_domain_hits = np.zeros(DOMAINS, dtype=np.int64)
    per_domain_rows = np.zeros(DOMAINS, dtype=np.int64)
    total_hits = 0
    for start in range(0, len(test_labels), BLOCK):
        stop = min(start + BLOCK, len(test_labels))
        scores = (standardise(np.asarray(mem_test[start:stop]))
                  .astype(np.float64) @ weights[:-1] + weights[-1])
        preds = np.argmax(scores, axis=1)
        hits = preds == test_labels[start:stop]
        total_hits += int(hits.sum())
        for d in range(DOMAINS):
            mask = test_domains[start:stop] == d
            per_domain_hits[d] += int(hits[mask].sum())
            per_domain_rows[d] += int(mask.sum())

    aggregate = total_hits / len(test_labels)
    per_task = {f"d{d}": float(per_domain_hits[d] / per_domain_rows[d])
                for d in range(DOMAINS)}
    anchor = {"measured": aggregate, "sealed": ANCHOR_V_MS,
              "delta": aggregate - ANCHOR_V_MS, "tolerance": TOL,
              "ok": abs(aggregate - ANCHOR_V_MS) <= TOL}
    print(f"ms aggregate {aggregate:.15f} (anchor delta "
          f"{anchor['delta']:+.3e})", flush=True)
    print(f"per task: {per_task}", flush=True)

    evidence: dict[str, Any] = {
        "milestone": "M210b",
        "cell": "ms per-domain accuracies for orchestration routing",
        "admissible_as_evidence": not inadmissible,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "anchor": anchor,
        "per_task": per_task,
        "routing_registration": {
            "arm_id": "ms",
            "held_out_accuracy": {"all": aggregate, **per_task},
            "note": ("registered for future orchestration runs; the "
                     "sealed M210 evidence keeps the aggregate "
                     "fallback for the ms arm"),
        },
        "void": not anchor["ok"],
        "void_reason": "" if anchor["ok"] else
        "V_ms anchor reproduction failed",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"anchor_ok": anchor["ok"], "per_task": per_task},
                     indent=1), flush=True)
    print(f"M210b complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m210b(args.config, args.output)


if __name__ == "__main__":
    main()
