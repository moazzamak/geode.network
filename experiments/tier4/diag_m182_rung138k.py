"""Diagnostic (not evidence): reproduce M117's sealed Q(6144, 138000)
cell with M117's exact call path, against the CURRENT cache files, to
localise why the M182 ladder rung n=138000 differs from the sealed
anchor by 2.6e-4 while rungs 34500/69000 match bit-exact.

Reads the same cache files M182 reads; runs M117's exact `_fit_and_score`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m117_scale import _fit_and_score

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
          / "m182_contributions.json")
SEALED = 0.2248695652173913


def main() -> None:
    configure_external_cache_environment()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    corpus, _ti, _tei = _load_corpus(config)
    cache = data_cache_root() / "v16" / "m117"
    mem_train = np.load(cache / "f6144_train.npy", mmap_mode="r")
    mem_test = np.load(cache / "f6144_test.npy", mmap_mode="r")
    print(f"train rows {mem_train.shape[0]}, test rows {mem_test.shape[0]}",
          flush=True)
    r = _fit_and_score(mem_train, mem_test, corpus["train_labels"],
                       corpus["test_labels"], corpus["test_domains"],
                       345, 138000)
    acc = r["accuracy_by_penalty"]["1.0"]
    print(f"M117 path on current cache: {acc:.15f}", flush=True)
    print(f"sealed M117 value:          {SEALED:.15f}", flush=True)
    print(f"delta: {acc - SEALED:+.3e}", flush=True)
    print(f"per_domain: "
          f"{[round(v, 6) for v in r['per_domain_correct']['1.0'].tolist()]}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
