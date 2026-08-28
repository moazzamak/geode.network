"""M142 cell C3b — power-normalisation on the cached multi-scale codes.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v22.md`` (execution
log, after the sealed C3 PASS; the registered follow-up free cell). C3b is
to C3 what C4 was to C2: the Fisher-vector post-processing (signed power p
+ per-row L2) applied to the CACHED ms357 codes, never re-encoded.

Question. Does power-normalisation lift the multi-scale construction the
way it lifted the SPM codes (the sealed C4 result: +1.8 at full data)?

Anchors: the raw MS refit (penalty 1.0, full data) must reproduce the
sealed C3 read Q_MS(409832) = 0.242145 within 0.002; p=1.0+L2 vs raw
same-fitter delta reported (C1b rule). Gate: best (p, penalty) cell at
full data >= 0.242145 + 0.005. Trained-head read (p=0.5) reported.

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m142_c3b
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
from experiments.tier4.eval_v16_m142_c3 import (
    SCALES,
    _append_scale_encode,
    _build_scale_whitener,
    _scale_dictionary,
)
from experiments.tier4.eval_v16_m142_c4 import (
    _fit_power,
    _score_power,
    _trained_head_read,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m142_c3b.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m142_c3b"

T1_TOLERANCE = 0.002
KS_MARGIN = 0.005


def run_m142_c3b(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    smoke = inadmissible
    smoke_skip = bool(config.get("_smoke_skip_gates", False))
    block = int(config["numerics"]["block"])
    torch.set_num_threads(int(config["numerics"]["torch_threads"]))
    configure_external_cache_environment()

    train_mem = np.load(
        data_cache_root() / config["artifacts"]["cache_relpath"]
        / config["artifacts"]["train_file"], mmap_mode="r")
    labels = np.load(
        data_cache_root() / "v16" / "m142_c2"
        / "m142_c2_fulltrain_labels.npz")["labels"][:len(train_mem)]

    corpus, _train_index, _test_index = _load_corpus(config)
    if smoke:
        train_mem = train_mem[:20000]
        labels = labels[:20000]

    # MS test codes to RAM (they were never persisted by the C3 run)
    _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    n_test = len(corpus["test_labels"]) if not smoke else int(
        config["_smoke_test_rows"])
    test_labels = corpus["test_labels"][:n_test]
    test_domains = corpus["test_domains"][:n_test]
    atoms_by_scale = {int(p): int(a)
                      for p, a in config["sparse"]["atoms_by_scale"].items()}
    width = 4 * sum(atoms_by_scale.values())
    print("building per-scale whiteners + dictionaries for the test encode",
          flush=True)
    test_mem = np.empty((n_test, width), dtype=np.float32)
    col = 0
    for patch in SCALES:
        whitener, candidates = _build_scale_whitener(config, corpus, patch)
        dictionary = _scale_dictionary(
            candidates, len(candidates),
            int(config["sparse"]["dictionary_seed"]), atoms_by_scale[patch])
        _append_scale_encode(corpus["test_images"], np.arange(n_test),
                             dictionary, whitener, device, test_mem, 0, col,
                             float(config["numerics"]
                                   ["encode_throttle_seconds"]))
        col += 4 * atoms_by_scale[patch]
    print(f"  test encode done ({time.time() - started:.0f}s so far)",
          flush=True)

    n_rows = len(train_mem)
    n_138 = int(config["cell_c3b"]["n_138k"]) if not smoke else n_rows
    p_ladder = [float(p) for p in config["cell_c3b"]["p_ladder"]]
    penalty_ladder = [float(q) for q in config["cell_c3b"]["penalty_ladder"]]

    raw_w, raw_std = _fit_power(train_mem, labels, 1.0, penalty_ladder,
                                n_rows, block, transform=False)
    raw_acc = {str(q): _score_power(test_mem, test_labels, test_domains,
                                    1.0, raw_w[str(q)], raw_std, block,
                                    transform=False)
               for q in penalty_ladder}
    raw_ref = raw_acc["1.0"]
    t1_delta = raw_ref - float(config["anchors"]["t1_reference"])
    print(f"t1 raw refit: {raw_ref:.4f} vs sealed C3 "
          f"{config['anchors']['t1_reference']} (delta {t1_delta:+.6f})",
          flush=True)
    if not smoke_skip and abs(t1_delta) > T1_TOLERANCE:
        raise SystemExit(f"t1 anchor reproduction failed (delta {t1_delta})")

    evidence: dict[str, Any] = {
        "milestone": "M142",
        "cell": "C3b power-normalisation on the cached multi-scale codes "
                "(free fit)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "anchors": {"raw_refit": raw_acc,
                    "t1_delta_vs_sealed_c3": t1_delta},
        "prior": config["cell_c3b"]["prior_note"],
    }

    cells: dict[str, Any] = {}
    for p in p_ladder:
        w, std = _fit_power(train_mem, labels, p, penalty_ladder, n_rows,
                            block)
        for q in penalty_ladder:
            acc = _score_power(test_mem, test_labels, test_domains, p,
                               w[str(q)], std, block)
            cells[f"p{p}_lambda{q}"] = {"accuracy": acc, "p": p,
                                        "penalty": q}
            print(f"  full p={p} lambda={q}: {acc:.4f}", flush=True)
    evidence["full_data_cells"] = cells

    cells_138: dict[str, Any] = {}
    for p in p_ladder:
        w, std = _fit_power(train_mem, labels, p, [1.0], n_138, block)
        cells_138[f"p{p}"] = _score_power(test_mem, test_labels, test_domains,
                                          p, w["1.0"], std, block)
        print(f"  138k p={p}: {cells_138[f'p{p}']:.4f}", flush=True)
    evidence["cells_138k"] = cells_138

    best_key = max(cells, key=lambda k: cells[k]["accuracy"])
    best = cells[best_key]
    gain = best["accuracy"] - raw_ref
    print(f"  best cell: {best_key} {best['accuracy']:.4f} "
          f"(raw ref {raw_ref:.4f}, gain {gain:+.4f})", flush=True)

    trained_acc = None
    if not smoke:
        print("trained-head read (p=0.5)", flush=True)
        trained_acc = _trained_head_read(
            train_mem, labels, test_mem, test_labels, 0.5, block,
            int(config["cell_c3b"]["trained_epochs"]),
            float(config["cell_c3b"]["trained_lr"]),
            int(config["cell_c3b"]["trained_seed"]),
            torch.device("cuda:0"))

    fired = (not smoke) and (gain < KS_MARGIN)
    both_fail = fired and (trained_acc is not None
                           and trained_acc < raw_ref + KS_MARGIN)
    evidence["reads"] = {"trained_head_read": trained_acc}
    evidence["gate"] = {
        "registered": config["cell_c3b"]["gate_registered"],
        "best_cell": best_key,
        "best_accuracy": best["accuracy"],
        "raw_same_fitter_reference": raw_ref,
        "gain": gain,
        "required": KS_MARGIN,
        "fired": fired,
        "consequence": (config["cell_c3b"]["consequence_fired"] if fired
                        else config["cell_c3b"]["consequence_passed"]),
        "closure_note": ("scoped negative requires BOTH reads to fail; "
                         f"both_fail={bool(both_fail)}"),
    }
    evidence["runtime_seconds"] = round(time.time() - started, 2)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM142 C3b complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m142_c3b(args.config, args.output)


if __name__ == "__main__":
    main()
