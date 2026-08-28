"""M182b — quantify the corrupt-tail impact on the M155/M156
growth-premise numbers by re-running the f6144 premise cell on the
REPAIRED view (registered 19 Aug in the plan execution log, commit
899cd41e).

The sealed M155 premise (n_error_rows 150289, train_accuracy
0.6332912..., floor_ladder [32..4096], test anchor 0.2613623188405797)
was measured while the f6144 train cache had 251 corrupt tail rows.
This run repeats exactly that cell with the registered in-memory patch
overlay and reports the genuine numbers; the M155/M156 budgets are
amended only if the floor ladder changes.
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
from experiments.tier4.eval_v23_m155_growth_premise import (
    _block_accuracy,
    _extension_indices,
    _floor_ladder,
    _load_domainnet,
    _rest_extension_indices,
)
from experiments.tier4.eval_v25_m182_contributions import (
    F6144_WIDTH,
    _load_repair_overlay,
    _part_block,
    _repair_gates,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m182b_growth_premise_repair.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m182b_growth_premise_repair")

CLASSES = 345
BLOCK = 4096


def run_m182b(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()

    configure_external_cache_environment()
    corpus, train_index, _test_index = _load_corpus(config)
    test_labels = corpus["test_labels"]
    n_test = len(test_labels)

    f6144_cache = data_cache_root() / config["artifacts"]["f6144_cache_relpath"]
    m142_cache = data_cache_root() / config["artifacts"]["m142_cache_relpath"]
    labels = np.load(m142_cache / config["artifacts"]["labels_file"])["labels"]

    part1 = np.load(f6144_cache / config["artifacts"]["f6144_train_file"],
                    mmap_mode="r")
    mem_test = np.load(f6144_cache / config["artifacts"]["f6144_test_file"],
                       mmap_mode="r")
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
        raise SystemExit("M182b premise failure: full-data part sizes")

    patch, patch_start = _load_repair_overlay(config)
    repair_gate = _repair_gates(config)

    parts = [
        (part1, corpus["train_labels"]),
        (ext600, raw["train_labels"][ext_idx]),
        (rest, raw["train_labels"][rest_idx]),
    ]

    # ---- f6144 head on the repaired view (penalty 1.0) ---------------------
    acc = RidgeAccumulator(F6144_WIDTH, CLASSES)
    for pi, (mem, part_labels) in enumerate(parts):
        for start in range(0, len(part_labels), BLOCK):
            stop = min(start + BLOCK, len(part_labels))
            acc.add(_part_block(mem, pi, start, stop, patch, patch_start),
                    part_labels[start:stop])
    weights = acc.solve_many([1.0])[1.0]
    standardise = acc.standardiser()
    test_acc, _ = _block_accuracy(weights, standardise, mem_test, test_labels,
                                  BLOCK, n_test)
    train_hits, train_n = 0, 0
    for pi, (mem, part_labels) in enumerate(parts):
        for start in range(0, len(part_labels), BLOCK):
            stop = min(start + BLOCK, len(part_labels))
            xs = _part_block(mem, pi, start, stop, patch, patch_start)
            scores = standardise(xs) @ weights[:-1] + weights[-1]
            train_hits += int((np.argmax(scores, axis=1)
                               == part_labels[start:stop]).sum())
        train_n += len(part_labels)
    train_acc = train_hits / train_n
    n_err = train_n - train_hits
    ladder = _floor_ladder(n_err)

    sealed = config["m155_sealed"]
    anchor = {
        "measured": test_acc,
        "sealed": float(config["anchors"]["f6144_genuine"]),
        "delta": test_acc - float(config["anchors"]["f6144_genuine"]),
        "tolerance": float(config["anchors"]["tolerance"]),
        "ok": abs(test_acc - float(config["anchors"]["f6144_genuine"]))
        <= float(config["anchors"]["tolerance"]),
    }
    print(f"  f6144 test {test_acc:.15f} (anchor delta "
          f"{anchor['delta']:+.3e}); train errors {n_err} "
          f"(sealed corrupt-era {sealed['n_error_rows']})", flush=True)
    print(f"  floor ladder repaired: {ladder}", flush=True)
    print(f"  floor ladder sealed:    {sealed['floor_ladder']}", flush=True)

    evidence: dict[str, Any] = {
        "milestone": "M182b",
        "cell": "corrupt-tail impact on the M155/M156 growth premise "
                "(f6144 cell on the repaired view)",
        "admissible_as_evidence": not inadmissible,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "anchor": anchor,
        "repaired_view": {
            "test_accuracy": test_acc,
            "train_accuracy": train_acc,
            "n_error_rows": int(n_err),
            "floor_ladder": ladder,
            "fit_rows": int(acc.rows),
        },
        "m155_sealed_corrupt_era": sealed,
        "impact": {
            "n_error_rows_delta": int(n_err) - int(sealed["n_error_rows"]),
            "train_accuracy_delta": train_acc
            - float(sealed["train_accuracy"]),
            "floor_ladder_changed": ladder != sealed["floor_ladder"],
            "reading": ("M155/M156 budgets stand as sealed"
                        if ladder == sealed["floor_ladder"] else
                        "M155/M156 floor ladder changed: budgets must be "
                        "amended from the repaired ladder"),
        },
        "repair_digests": repair_gate,
        "void": not anchor["ok"],
        "void_reason": "" if anchor["ok"] else
        "genuine-value anchor reproduction failed",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"anchor_ok": anchor["ok"],
                      "n_error_rows": int(n_err),
                      "floor_ladder": ladder,
                      "floor_ladder_changed":
                      ladder != sealed["floor_ladder"]}, indent=1),
          flush=True)
    print(f"M182b complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m182b(args.config, args.output)


if __name__ == "__main__":
    main()
