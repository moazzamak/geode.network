"""Diagnostic (not evidence): bit-exact re-encode of sampled positions of
the sealed f6144_train cache, to localise the M182 rung-138000 anchor
miss (2.6e-4 while rungs 34500/69000 match bit-exact).

Re-encodes a few row windows with the M117 exact pipeline (M108 whitener
+ seeded dictionary 11 -> 6144 atoms) and compares bitwise against the
cache at those positions. The M142 t1 checks already confirmed this
re-encode path is bit-reproducible on this machine (delta 0.0) for the
FIRST rows; this probe extends the check to the MIDDLE and TAIL.
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
from experiments.tier4.eval_v16_m113_learned import (
    _build_whitener_and_candidates,
    _random_dictionary,
)
from experiments.tier4.eval_v16_m115_lofi import _write_frozen_codes
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
M117_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m117_scale.json"
WINDOWS = [(0, 256), (34400, 34656), (68800, 69056), (103200, 103456),
           (137744, 138000)]


def main() -> None:
    configure_external_cache_environment()
    config = json.loads(M117_CONFIG.read_text(encoding="utf-8"))
    corpus, _ti, _tei = _load_corpus(config)
    torch.set_num_threads(4)
    device_report = _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    print(f"device: {device_report}", flush=True)

    print("building whitener + candidate pool (M108 exact)", flush=True)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)
    dictionary = _random_dictionary(candidates, len(candidates),
                                    int(config["sparse"]["dictionary_seed"]),
                                    6144)
    del candidates

    sealed = np.load(data_cache_root() / "v16" / "m117" / "f6144_train.npy",
                     mmap_mode="r")
    print(f"sealed train rows: {sealed.shape[0]}", flush=True)
    for lo, hi in WINDOWS:
        rows = np.arange(lo, hi)
        check = _write_frozen_codes(corpus, dictionary, whitener, 2, device,
                                    rows, data_cache_root() / "v16" / "m117"
                                    / f"_diag_check_{lo}_{hi}.npy",
                                    split="train")
        ref = np.asarray(sealed[lo:hi], dtype=np.float64)
        got = np.asarray(check, dtype=np.float64)
        delta = float(np.abs(ref - got).max())
        print(f"rows {lo}:{hi}  max|delta| = {delta:.3e} "
              f"{'MATCH' if delta == 0.0 else 'MISMATCH'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
