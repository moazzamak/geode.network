"""M182 — data-contribution measurement: the Q(n) marginal ladder and
group leave-one-domain-out on the sealed f6144 codes.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(18 Aug 2026) before building. Operands: the sealed f6144 memmaps
(m117 138k subsample + m140 ext600 + m141 all-rest, the M141 cell-2
row schedule), penalty 1.0, block 4096.

Cells:
1. Q(n) ladder at rungs {34500, 69000, 138000, 276000, 409832} as
   prefixes of the M141 schedule, anchored by Q(138000) =
   0.2248695652173913 (M117 exact) and Q(409832) =
   0.2613623188405797 (M155/M142 t2 exact), tol 1e-9.
2. Group LOO: one pass accumulates per-domain Grams (6 domains); each
   LOO fit is total-minus-domain (the closed-form Gram is additive, so
   no refitting). Reports the full-fit accuracy and each domain's
   marginal (accuracy drop) on the full and per-domain test.
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
    _extension_indices,
    _load_domainnet,
    _rest_extension_indices,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m182_contributions.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m182_contributions")

CLASSES = 345
DOMAINS = 6
BLOCK = 4096
PENALTY = 1.0
TOL = 1e-9
F6144_WIDTH = 24576


def _score(weights: np.ndarray, standardiser, mem_test: np.ndarray,
           test_labels: np.ndarray) -> np.ndarray:
    preds = np.empty(len(test_labels), dtype=np.int64)
    n_test = len(test_labels)
    for start in range(0, n_test, BLOCK):
        stop = min(start + BLOCK, n_test)
        scores = (standardiser(np.asarray(mem_test[start:stop]))
                  .astype(np.float64) @ weights[:-1] + weights[-1])
        preds[start:stop] = np.argmax(scores, axis=1)
    return preds


def _sha256_file(path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def _load_repair_overlay(config: dict[str, Any]):
    """The registered M182 premise repair (plan execution log, 19 Aug):
    rows 137749:137999 of the sealed f6144_train cache are corrupt, so
    the runner substitutes rows [patch_start, patch_start+rows) of
    part 1 from a 25 MB fresh-encode patch file. Returns
    (patch, patch_start) or (None, None). The config-registered
    digests are GATES: a mismatch refuses the run."""
    repair = config.get("repair")
    if not repair:
        return None, None
    patch_path = data_cache_root() / repair["patch_relpath"]
    patch = np.load(patch_path)
    if patch.shape != (int(repair["rows"]), F6144_WIDTH):
        raise SystemExit("M182 repair gate: patch shape mismatch")
    if int(repair["patch_start"]) != 137749:
        raise SystemExit("M182 repair gate: unexpected patch_start")
    for key in ("original_sha256", "patch_sha256", "test_sha256"):
        if key not in repair:
            raise SystemExit(f"M182 repair gate: missing {key}")
    return patch, int(repair["patch_start"])


def _repair_gates(config: dict[str, Any]) -> dict[str, Any]:
    """Measure the operand digests and gate them against the
    config-registered values. Runs on the repaired and the original
    path alike (the original run of 19 Aug has no repair block, so the
    gate set is empty)."""
    repair = config.get("repair")
    digests: dict[str, Any] = {}
    if not repair:
        return digests
    cache = data_cache_root() / config["artifacts"]["f6144_cache_relpath"]
    measured = {
        "original_sha256": _sha256_file(
            cache / config["artifacts"]["f6144_train_file"]),
        "test_sha256": _sha256_file(
            cache / config["artifacts"]["f6144_test_file"]),
        "patch_sha256": _sha256_file(data_cache_root()
                                     / repair["patch_relpath"]),
    }
    for key, value in measured.items():
        expected = str(repair[key])
        digests[key] = {"measured": value, "registered": expected,
                        "ok": value == expected}
        if value != expected:
            raise SystemExit(
                f"M182 repair gate FAILED: {key} digest mismatch — "
                f"the operand changed since registration")
    print("repair digests all match the registered values", flush=True)
    return digests


def _part_block(mem: np.ndarray, part_index: int, start: int, stop: int,
                patch, patch_start: int) -> np.ndarray:
    """Read a block of part `part_index`, overlaying the repair patch
    where it overlaps part 1 (the corrupt tail)."""
    block = np.asarray(mem[start:stop], dtype=np.float32)
    if patch is not None and part_index == 0:
        lo = max(start, patch_start)
        hi = min(stop, patch_start + len(patch))
        if hi > lo:
            block = block.copy()
            block[lo - start:hi - start] = patch[
                lo - patch_start:hi - patch_start]
    return block


def run_m182(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible
    smoke_test = int(config.get("_smoke_test_rows", 10 ** 9))

    configure_external_cache_environment()
    corpus, train_index, _test_index = _load_corpus(config)
    test_labels = corpus["test_labels"][:smoke_test]
    test_domains = corpus["test_domains"][:smoke_test]
    n_test = len(test_labels)

    f6144_cache = data_cache_root() / config["artifacts"]["f6144_cache_relpath"]
    part1 = np.load(f6144_cache / config["artifacts"]["f6144_train_file"],
                    mmap_mode="r")
    mem_test = np.load(f6144_cache / config["artifacts"]["f6144_test_file"],
                       mmap_mode="r")[:smoke_test]
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
        raise SystemExit("M182 premise failure: full-data part sizes")

    patch, patch_start = _load_repair_overlay(config)
    repair_gate = _repair_gates(config)

    schedule: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [
        (part1, corpus["train_labels"], corpus["train_domains"]),
        (ext600, raw["train_labels"][ext_idx], raw["train_domains"][ext_idx]),
        (rest, raw["train_labels"][rest_idx], raw["train_domains"][rest_idx]),
    ]
    total_rows = sum(len(labels) for _, labels, _ in schedule)
    print(f"schedule rows: {total_rows}; test {n_test}", flush=True)

    # ---- cell 1: Q(n) ladder ------------------------------------------------
    rungs = [int(n) for n in config["ladder"]["rungs"]]
    anchors_in = config["ladder"]["anchors"]
    ladder: dict[str, Any] = {"rungs": rungs, "points": {}, "anchors": {}}
    print("Q(n) ladder", flush=True)
    for n in rungs:
        acc = RidgeAccumulator(F6144_WIDTH, CLASSES)
        remaining = n
        for pi, (mem, part_labels, _dom) in enumerate(schedule):
            if remaining <= 0:
                break
            take = min(remaining, len(part_labels))
            for start in range(0, take, BLOCK):
                stop = min(start + BLOCK, take)
                acc.add(_part_block(mem, pi, start, stop, patch,
                                    patch_start),
                        part_labels[start:stop])
            remaining -= take
        weights = acc.solve_many([PENALTY])[PENALTY]
        preds = _score(weights, acc.standardiser(), mem_test, test_labels)
        acc_value = float((preds == test_labels).mean())
        ladder["points"][str(n)] = acc_value
        print(f"  n={n}: {acc_value:.10f}", flush=True)
        if str(n) in anchors_in:
            sealed = float(anchors_in[str(n)]["value"])
            delta = acc_value - sealed
            ladder["anchors"][str(n)] = {"measured": acc_value,
                                         "sealed": sealed,
                                         "delta": delta,
                                         "tolerance": TOL,
                                         "ok": abs(delta) <= TOL}
            print(f"    anchor n={n}: delta {delta:+.3e} "
                  f"ok={ladder['anchors'][str(n)]['ok']}", flush=True)
        del acc, weights, preds

    # ---- cell 2: group LOO (one pass, per-domain Grams) ---------------------
    domain_accs = [RidgeAccumulator(F6144_WIDTH, CLASSES)
                   for _ in range(DOMAINS)]
    for pi, (mem, part_labels, part_domains) in enumerate(schedule):
        for start in range(0, len(part_labels), BLOCK):
            stop = min(start + BLOCK, len(part_labels))
            block_domains = part_domains[start:stop]
            block = _part_block(mem, pi, start, stop, patch, patch_start)
            block_labels = part_labels[start:stop]
            for d in range(DOMAINS):
                mask = block_domains == d
                if mask.any():
                    domain_accs[d].add(block[mask], block_labels[mask])
    full = RidgeAccumulator(F6144_WIDTH, CLASSES)
    for d in range(DOMAINS):
        acc = domain_accs[d]
        # add the domain's accumulated raw system into the total
        full.gram += acc.gram
        full.column_sum += acc.column_sum
        full.cross += acc.cross
        full.class_count += acc.class_count
        full.rows += acc.rows
    print(f"LOO full rows: {full.rows}", flush=True)

    full_w = full.solve_many([PENALTY])[PENALTY]
    full_preds = _score(full_w, full.standardiser(), mem_test, test_labels)
    full_acc = float((full_preds == test_labels).mean())
    loo: dict[str, Any] = {"full_accuracy": full_acc,
                           "per_domain": {}, "marginals": {}}
    print(f"  LOO full: {full_acc:.10f}", flush=True)
    for d in range(DOMAINS):
        acc = RidgeAccumulator(F6144_WIDTH, CLASSES)
        acc.gram = full.gram - domain_accs[d].gram
        acc.column_sum = full.column_sum - domain_accs[d].column_sum
        acc.cross = full.cross - domain_accs[d].cross
        acc.class_count = full.class_count - domain_accs[d].class_count
        acc.rows = full.rows - domain_accs[d].rows
        weights = acc.solve_many([PENALTY])[PENALTY]
        preds = _score(weights, acc.standardiser(), mem_test, test_labels)
        acc_value = float((preds == test_labels).mean())
        per_domain = [float((preds[test_domains == dd]
                             == test_labels[test_domains == dd]).mean())
                      if (test_domains == dd).any() else 0.0
                      for dd in range(DOMAINS)]
        loo["per_domain"][str(d)] = {"accuracy": acc_value,
                                     "rows_left_out": int(
                                         domain_accs[d].rows),
                                     "per_domain_test": per_domain}
        loo["marginals"][str(d)] = full_acc - acc_value
        print(f"  LOO domain {d}: {acc_value:.10f} "
              f"(marginal {full_acc - acc_value:+.6f}, "
              f"left out {domain_accs[d].rows} rows)", flush=True)
        del acc, weights, preds

    anchors_ok = all(a["ok"] for a in ladder["anchors"].values())
    if not anchors_ok:
        print("M182 VOID: a ladder anchor failed", flush=True)

    evidence: dict[str, Any] = {
        "milestone": "M182",
        "cell": "data-contribution measurement (Q(n) ladder + group LOO)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "ladder": ladder,
        "group_loo": loo,
        "repair_digests": repair_gate,
        "void": not anchors_ok,
        "void_reason": "" if anchors_ok else "ladder anchor reproduction "
        "failed at 1e-9",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"anchors_ok": anchors_ok,
                      "ladder": ladder["points"],
                      "loo_marginals": loo["marginals"]}, indent=1),
          flush=True)
    print(f"M182 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m182(args.config, args.output)


if __name__ == "__main__":
    main()
