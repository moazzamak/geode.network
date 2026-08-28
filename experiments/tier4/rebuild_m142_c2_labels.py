"""RECOVERY: rebuild ``v16/m142_c2/m142_c2_fulltrain_labels.npz``.

The 20 Aug cache cleanup deleted ``v16/m142_c2`` (derived cache of a
sealed milestone) — but the labels file inside it is the ms-code
alignment key referenced by many configs (M220/M228 and the v23 line).
The labels are re-derivable from the registered M142 cell-2 schedule
construction. RECOVERY GATE: the rebuilt labels must reproduce the M220
sealed anchor 0.24214492753623187 at 1e-9 (ms codes + labels -> ridge
penalty 1.0 -> sealed test) BEFORE the file is written; a mismatch means
the reconstruction is wrong and the file is not touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.tier4.eval_v15_m104_experts import _load_domainnet
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m140_data_extension import _extension_indices
from experiments.tier4.eval_v16_m141_data_full import _rest_extension_indices
from experiments.tier4.eval_v25_m222_dinov2_hybrid_pilot import _fit_and_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m228_dinov2_fullscale.json")

ANCHOR = 0.24214492753623187
TOL = 1e-9
FULL_ROWS = 409832


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--write", action="store_true",
                        help="write the file after the gate passes")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    configure_external_cache_environment()
    root = data_cache_root()

    print("loading corpus + raw (decoded cache)", flush=True)
    corpus, train_index, test_index = _load_corpus(config)
    raw = _load_domainnet(int(config["corpus"]["image_size"]))

    print("reconstructing the M142 cell-2 schedule labels", flush=True)
    ext600_indices, _ = _extension_indices(raw["train_labels"], train_index,
                                           600, 345)
    rest_indices = _rest_extension_indices(raw["train_labels"], train_index,
                                           345, per_class_take=200)
    labels = np.concatenate([
        corpus["train_labels"],                       # part 1: 138k subsample
        raw["train_labels"][ext600_indices],          # ext600: 69k
        raw["train_labels"][rest_indices],            # rest: 202,832
    ])
    print(f"labels: {len(labels)} rows "
          f"(part1 {len(corpus['train_labels'])}, "
          f"ext600 {len(ext600_indices)}, rest {len(rest_indices)})",
          flush=True)
    if len(labels) != FULL_ROWS:
        raise SystemExit(f"RECOVERY FAILED: {len(labels)} != {FULL_ROWS}")

    print("gate: ms codes + labels must reproduce the M220 anchor at 1e-9",
          flush=True)
    ms_cache = root / config["artifacts"]["cache_relpath"]
    ms_test_cache = root / config["artifacts"]["test_cache_relpath"]
    train_ms = np.asarray(np.load(
        ms_cache / config["artifacts"]["train_file"], mmap_mode="r"))
    test_ms = np.asarray(np.load(
        ms_test_cache / config["artifacts"]["test_file"], mmap_mode="r"))
    acc = _fit_and_score(train_ms, labels, test_ms, corpus["test_labels"],
                         [1.0])
    measured = acc["1.0"]
    delta = measured - ANCHOR
    print(f"measured {measured:.17f} | anchor {ANCHOR:.17f} | "
          f"delta {delta:.3e}", flush=True)
    if abs(delta) > TOL:
        raise SystemExit("RECOVERY FAILED: the anchor did not reproduce — "
                         "the reconstruction is wrong; nothing written.")

    print("gate passed - labels are bit-aligned with the ms codes", flush=True)
    if args.write:
        out_dir = root / "v16" / "m142_c2"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "m142_c2_fulltrain_labels.npz"
        np.savez(out_path, labels=labels)
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        print(f"wrote {out_path} (sha256 {digest[:16]}...)", flush=True)
    else:
        print("dry run: --write to persist", flush=True)


if __name__ == "__main__":
    main()
