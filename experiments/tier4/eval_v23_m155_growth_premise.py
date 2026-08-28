"""M155 — growth premise at full-data scale (premise-only cell).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v23.md`` (section 4
M155, section 6; 16 Aug 2026). M155 measures NOTHING new: it refits the
two sealed full-data global heads from the cached codes, reproduces their
sealed reads (the anchors), and reports the train-split error-row counts
plus the floor-derived growth-budget ladders. Its output REGISTERS M156's
budgets (the M145 lesson: measure the population before registering
budgets).

Anchors (before any number is read, non-smoke):

- f6144 head: RidgeAccumulator over ``f6144_train.npy`` (409,832 x 24,576,
  the M141 cell-2 schedule), penalty 1.0 -> test read must reproduce
  Q(6144, 409832) = 0.2613623188405797 (tol 1e-9).
- promoted head: C4's ``_fit_power``/``_score_power`` over the cached SPM
  codes, p=0.5, penalty 0.1 -> test read must reproduce 0.278550724637681
  (tol 1e-9).

Premise outputs per head: train error-row count, floor-derived ladder
(``ceil(n_err / (4a)) >= 10`` over powers of two from 32 up). No gate;
no new accuracy claim.

Reproduce with::

    .\\.venv\\Scripts\\python.exe -m experiments.tier4.eval_v23_m155_growth_premise
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
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator, _load_domainnet
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m140_data_extension import _extension_indices
from experiments.tier4.eval_v16_m141_data_full import _rest_extension_indices
from experiments.tier4.eval_v16_m142_c4 import _fit_power, _score_power
from experiments.tier4.eval_v16_m142_factorial import power_norm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v23"
                  / "m155_growth_premise.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v23" / "m155_growth_premise"

CLASSES = 345
F6144_WIDTH = 24576
TOLERANCE = 1e-9
FLOOR = 10.0
POOL_BINS = 4          # 2x2 pooling: 4 pooled columns per atom
MIN_ATOMS = 32


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def _error_rows(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Positions where the prediction is wrong."""
    return np.flatnonzero(np.asarray(preds) != np.asarray(labels))


def _floor_ladder(n_error_rows: int, pool_bins: int = POOL_BINS,
                  floor: float = FLOOR, min_atoms: int = MIN_ATOMS
                  ) -> list[int]:
    """Powers of two whose fit ratio clears the floor, 32 up to the max."""
    ladder: list[int] = []
    a = min_atoms
    while True:
        ratio = (int(n_error_rows) + pool_bins * a - 1) // (pool_bins * a)
        if float(ratio) < floor:
            break
        ladder.append(a)
        a *= 2
    return ladder


def _block_accuracy(weights: np.ndarray, standardise, mem: np.ndarray,
                    labels: np.ndarray, block: int, n_rows: int,
                    transform: Any = None) -> tuple[float, int]:
    """Blocked accuracy over the first n_rows rows; returns (acc, n_err)."""
    hits = 0
    for start in range(0, n_rows, block):
        stop = min(start + block, n_rows)
        xs = mem[start:stop]
        if transform is not None:
            xs = transform(xs)
        scores = standardise(xs) @ weights[:-1] + weights[-1]
        hits += int((np.argmax(scores, axis=1)
                     == labels[start:stop]).sum())
    return hits / n_rows, n_rows - hits


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run_m155(config_path: Path, output_dir: Path) -> dict[str, Any]:
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

    configure_external_cache_environment()
    f6144_cache = data_cache_root() / config["artifacts"]["f6144_cache_relpath"]
    m142_cache = data_cache_root() / config["artifacts"]["m142_cache_relpath"]
    evidence: dict[str, Any] = {
        "milestone": "M155",
        "cell": "growth premise at full-data scale",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
    }

    print("loading corpus (test labels) + cached labels", flush=True)
    corpus, train_index, _test_index = _load_corpus(config)
    test_labels = corpus["test_labels"][:smoke_test]
    labels = np.load(m142_cache / config["artifacts"]["labels_file"])["labels"]
    n_train = min(len(labels), smoke_train)
    n_test = len(test_labels)
    print(f"rows: train {n_train} / test {n_test}", flush=True)

    anchors: dict[str, Any] = {}
    premise: dict[str, Any] = {}

    # ---- f6144 head (penalty 1.0, the M141 cell-2 accumulation) -------------
    print("f6144 head: accumulate + solve", flush=True)
    part1 = np.load(f6144_cache / config["artifacts"]["f6144_train_file"],
                    mmap_mode="r")
    mem_test = np.load(f6144_cache / config["artifacts"]["f6144_test_file"],
                       mmap_mode="r")[:smoke_test]
    acc = RidgeAccumulator(F6144_WIDTH, CLASSES)
    if smoke:
        parts = [(part1[:n_train], labels[:n_train])]
    else:
        raw = _load_domainnet(int(config["corpus"]["image_size"]))
        ext600 = np.load(data_cache_root() / "v16" / "m140"
                         / "f6144_ext600.npy", mmap_mode="r")
        rest = np.load(data_cache_root() / "v16" / "m141"
                       / "f6144_all_rest.npy", mmap_mode="r")
        ext_idx, _ = _extension_indices(raw["train_labels"], train_index,
                                        600, CLASSES)
        rest_idx = _rest_extension_indices(raw["train_labels"], train_index,
                                           CLASSES, per_class_take=200)
        if len(ext600) != 69000 or len(rest) != 202832:
            raise SystemExit("M155 premise failure: full-data part sizes")
        parts = [
            (part1, corpus["train_labels"]),
            (ext600, raw["train_labels"][ext_idx]),
            (rest, raw["train_labels"][rest_idx]),
        ]
    for mem, part_labels in parts:
        for start in range(0, len(part_labels), block):
            stop = min(start + block, len(part_labels))
            acc.add(mem[start:stop], part_labels[start:stop])
    weights = acc.solve_many([1.0])[1.0]
    standardise = acc.standardiser()
    test_acc, _ = _block_accuracy(weights, standardise, mem_test, test_labels,
                                  block, n_test)
    train_hits, train_n = 0, 0
    for mem, part_labels in parts:
        hits = 0
        for start in range(0, len(part_labels), block):
            stop = min(start + block, len(part_labels))
            scores = standardise(np.asarray(mem[start:stop])) @ weights[:-1] \
                + weights[-1]
            hits += int((np.argmax(scores, axis=1)
                         == part_labels[start:stop]).sum())
        train_hits += hits
        train_n += len(part_labels)
    train_acc = train_hits / train_n
    n_err = train_n - train_hits
    anchors["f6144"] = {"measured": test_acc,
                        "sealed": float(config["anchors"]["f6144_full_data"]),
                        "delta": test_acc
                        - float(config["anchors"]["f6144_full_data"]),
                        "tolerance": TOLERANCE}
    premise["f6144"] = {
        "train_accuracy": train_acc,
        "n_error_rows": int(n_err),
        "floor_ladder": _floor_ladder(n_err),
        "fit_rows": int(acc.rows),
    }
    print(f"  f6144 test {test_acc:.6f} (delta "
          f"{anchors['f6144']['delta']:+.3e}); train errors {n_err}",
          flush=True)
    if not skip_anchors and abs(anchors["f6144"]["delta"]) > TOLERANCE:
        evidence.update({"void": True,
                         "void_reason": "f6144 anchor reproduction failed",
                         "anchors": anchors})
        _write(output_dir, evidence)
        return evidence
    del part1, mem_test
    if not smoke:
        del ext600, rest, raw
    acc = None  # type: ignore[assignment]

    # ---- promoted head (C4 path, p=0.5, penalty 0.1) -----------------------
    print("SPM+sqrt head: C4 fitter", flush=True)
    spm_train = np.load(m142_cache / config["artifacts"]["spm_train_file"],
                        mmap_mode="r")
    spm_test = np.load(m142_cache / config["artifacts"]["spm_test_file"],
                       mmap_mode="r")[:smoke_test]
    solved, std = _fit_power(spm_train, labels, 0.5, [0.1], n_train, block,
                             transform=True)
    test_acc = _score_power(spm_test, test_labels, corpus["test_domains"]
                            [:smoke_test], 0.5, solved["0.1"], std, block,
                            transform=True)
    transform = lambda xs: power_norm(xs, 0.5)  # noqa: E731
    train_acc, n_err = _block_accuracy(solved["0.1"], std, spm_train, labels,
                                       block, n_train, transform=transform)
    anchors["spm_sqrt"] = {"measured": test_acc,
                           "sealed": float(config["anchors"]
                                           ["spm_sqrt_full_data"]),
                           "delta": test_acc
                           - float(config["anchors"]["spm_sqrt_full_data"]),
                           "tolerance": TOLERANCE}
    premise["spm_sqrt"] = {
        "train_accuracy": train_acc,
        "n_error_rows": int(n_err),
        "floor_ladder": _floor_ladder(n_err),
        "fit_rows": int(n_train),
    }
    print(f"  SPM+sqrt test {test_acc:.6f} (delta "
          f"{anchors['spm_sqrt']['delta']:+.3e}); train errors {n_err}",
          flush=True)
    if not skip_anchors and abs(anchors["spm_sqrt"]["delta"]) > TOLERANCE:
        evidence.update({"void": True,
                         "void_reason": "SPM+sqrt anchor reproduction failed",
                         "anchors": anchors})
        _write(output_dir, evidence)
        return evidence
    del spm_train, spm_test

    evidence.update({
        "anchors": anchors,
        "premise": premise,
        "registration_for_m156": {
            "base": "the promoted SPM+sqrt full-data head (section 6 "
                    "interpretation note)",
            "budgets": premise["spm_sqrt"]["floor_ladder"],
            "f6144_ladder_reported_alongside": premise["f6144"][
                "floor_ladder"],
        },
        "runtime_seconds": round(time.time() - started, 2),
    })
    _write(output_dir, evidence)
    print(f"\nM155 complete -> {output_dir / 'evidence.json'}", flush=True)
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
    run_m155(args.config, args.output)


if __name__ == "__main__":
    main()
