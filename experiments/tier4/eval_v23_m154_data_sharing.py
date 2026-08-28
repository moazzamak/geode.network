"""M154 — data-sharing specialists (score-level): domain-gated fusion.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v23.md`` (section 4
M154 / section 6; 16 Aug 2026). Fits only, on the M143/M143b score
caches.

Construction (registered): the flat 7-arm concat (6 specialists + the
global arm = 2,415 features, the M143b layout) plus each of the SAME 7
arms' 345-dim score vector GATED per domain — 7 arms x 6 domains =
42 gated copies (14,490 features) — so the stacking weights may depend
on the row's domain. This is the score-level data-sharing proxy;
head-level sharing stays M159 (needs codes).

Anchor (before any new number): the M143b flat-stacking protocol
reproduced from the same caches (fused 0.22431884057971013, global
0.22460869565217392, tol 1e-9). Gate: fused(interaction) >= global +
0.005 on the sealed test scores, else archived as a scoped negative.
Smoke declares inadmissibility and refuses the sealed output directory.
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
    sha256_file,
    write_canonical_json,
)
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m143_integration import (
    _select_penalty,
    _stacking_fit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v23"
                  / "m154_data_sharing.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v23"
                  / "m154_data_sharing")

CLASSES = 345
DOMAINS = 6
ARMS = 7
TOLERANCE = 1e-9
MARGIN = 0.005
VALID_FRAC = 0.8
VALID_SEED = 55
LADDER = [1.0, 10.0, 100.0, 1000.0, 10000.0]


def _flat_features(arm_scores: np.ndarray) -> np.ndarray:
    """(7, n, C) arm scores -> (n, 7*C) in the M143b arm order."""
    return arm_scores.transpose(1, 0, 2).reshape(arm_scores.shape[1], -1)


def _gated_features(arm_scores: np.ndarray, domains: np.ndarray
                    ) -> np.ndarray:
    """(n, 7, C) arm scores -> (n, 7*6*C) domain-gated copies."""
    n, arms_n, classes = arm_scores.shape
    onehot = np.zeros((n, DOMAINS), dtype=np.float32)
    onehot[np.arange(n), np.asarray(domains, dtype=np.int64)] = 1.0
    out = np.empty((n, arms_n * DOMAINS * classes), dtype=np.float32)
    for a in range(arms_n):
        for d in range(DOMAINS):
            out[:, (a * DOMAINS + d) * classes:(a * DOMAINS + d + 1)
                * classes] = arm_scores[:, a, :] * onehot[:, d:d + 1]
    return out


def run_m154(config_path: Path, output_dir: Path) -> dict[str, Any]:
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

    configure_external_cache_environment()
    root = data_cache_root()
    m143 = np.load(root / config["score_caches"]["m143"], allow_pickle=False)
    m143b = np.load(root / config["score_caches"]["m143b"], allow_pickle=False)
    corpus, _ti, _vi = _load_corpus(config)
    train_domains = corpus["train_domains"][:smoke_train]
    specialist_train = m143b["specialist_train"][:, :smoke_train, :]
    global_train = m143b["global_train"][:smoke_train, :]
    train_labels = m143b["train_labels"][:smoke_train]
    specialist_test = m143["specialist_scores"][:, :smoke_test, :]
    global_test = m143["global_scores"][:smoke_test, :]
    test_labels = m143["test_labels"][:smoke_test]
    test_domains = m143["test_domains"][:smoke_test]
    n_train = len(train_labels)
    n_test = len(test_labels)
    if len(train_domains) != n_train:
        raise SystemExit("M154 premise failure: train domain count mismatch")
    evidence: dict[str, Any] = {
        "milestone": "M154",
        "cell": "data-sharing specialists (score-level, domain-gated fusion)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
    }

    # ---- anchor: the M143b flat stacking ------------------------------------
    print("anchor: M143b flat stacking", flush=True)
    arm_train = np.concatenate([specialist_train, global_train[None, :, :]],
                               axis=0)          # (7, n_train, 345)
    arm_test = np.concatenate([specialist_test, global_test[None, :, :]],
                              axis=0)           # (7, n_test, 345)
    # The M143b layout: 6 specialists then the global arm, 2,415 columns.
    flat_train = _flat_features(arm_train)
    flat_test = _flat_features(arm_test)
    if flat_train.shape[1] != ARMS * CLASSES:
        raise SystemExit("M154 instrument failure: flat concat width is "
                         f"{flat_train.shape[1]}, not {ARMS * CLASSES} "
                         "(the M143b layout)")
    order = np.random.default_rng(VALID_SEED).permutation(n_train)
    cut = int(VALID_FRAC * n_train)
    ft, fv = order[:cut], order[cut:]

    def _metric(feats, penalty):
        predict = _stacking_fit(feats[ft], train_labels[ft], penalty)
        return float((predict(feats[fv]) == train_labels[fv]).mean())

    penalty, ladder_scores = _select_penalty(
        lambda p: _metric(flat_train, p), LADDER)
    stacking = _stacking_fit(flat_train, train_labels, penalty)
    fused_flat = float((stacking(flat_test) == test_labels).mean())
    global_acc = float(
        (np.argmax(global_test, axis=1) == test_labels).mean())
    anchors = {
        "m143b_fused": {"measured": fused_flat,
                        "sealed": float(config["anchors"]["m143b_fused"]),
                        "delta": fused_flat
                        - float(config["anchors"]["m143b_fused"]),
                        "tolerance": TOLERANCE},
        "m143b_global": {"measured": global_acc,
                         "sealed": float(config["anchors"]["m143b_global"]),
                         "delta": global_acc
                         - float(config["anchors"]["m143b_global"]),
                         "tolerance": TOLERANCE},
    }
    print(f"  flat fused {fused_flat:.6f}; global {global_acc:.6f}",
          flush=True)
    if not skip_anchors and (abs(anchors["m143b_fused"]["delta"]) > TOLERANCE
                             or abs(anchors["m143b_global"]["delta"])
                             > TOLERANCE):
        evidence.update({"void": True,
                         "void_reason": "M143b anchor reproduction failed",
                         "anchors": anchors})
        _write(output_dir, evidence)
        return evidence

    # ---- domain-gated fusion --------------------------------------------------
    # The 16,905-wide stack cannot be fit in RAM at 138k rows (the sealed
    # stacking helper materialises float64 copies ~30 GB per call). The
    # interaction stack is spilled to disk memmaps; the feature values and
    # the stacking protocol are bitwise unchanged (section 6 amendment).
    print("domain-gated fusion", flush=True)
    width = ARMS * CLASSES + ARMS * DOMAINS * CLASSES
    gated_train = _gated_features(arm_train.transpose(1, 0, 2),
                                  train_domains)
    gated_test = _gated_features(arm_test.transpose(1, 0, 2), test_domains)
    full_test = np.concatenate([flat_test, gated_test], axis=1)
    scratch = (data_cache_root() / "v23" / "m154_features"
               / output_dir.name)
    scratch.mkdir(parents=True, exist_ok=True)
    full_mem = np.lib.format.open_memmap(
        scratch / "full_train.npy", mode="w+", dtype=np.float32,
        shape=(n_train, width))
    full_mem[:, :ARMS * CLASSES] = flat_train
    full_mem[:, ARMS * CLASSES:] = gated_train
    del flat_train, gated_train, arm_train, specialist_train, global_train
    ft_mem = np.lib.format.open_memmap(
        scratch / "ft.npy", mode="w+", dtype=np.float32,
        shape=(len(ft), width))
    for start in range(0, len(ft), 4096):
        stop = min(start + 4096, len(ft))
        ft_mem[start:stop] = full_mem[ft[start:stop]]
    fv_mem = np.lib.format.open_memmap(
        scratch / "fv.npy", mode="w+", dtype=np.float32,
        shape=(len(fv), width))
    for start in range(0, len(fv), 4096):
        stop = min(start + 4096, len(fv))
        fv_mem[start:stop] = full_mem[fv[start:stop]]
    del arm_test, specialist_test, global_test, flat_test, gated_test

    def _metric_disk(penalty):
        predict = _stacking_fit(ft_mem, train_labels[ft], penalty)
        return float((predict(fv_mem) == train_labels[fv]).mean())

    pen_i, ladder_i = _select_penalty(_metric_disk, LADDER)
    stacking_i = _stacking_fit(full_mem, train_labels, pen_i)
    fused_i = float((stacking_i(full_test) == test_labels).mean())
    spilled = []
    for name in ("full_train.npy", "ft.npy", "fv.npy"):
        path = scratch / name
        spilled.append({"relpath": str(path.relative_to(data_cache_root())),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path)})
    gain = fused_i - global_acc
    passed = gain >= MARGIN
    print(f"  gated fused {fused_i:.6f} (gain {gain:+.6f}, penalty "
          f"{pen_i})", flush=True)

    evidence.update({
        "anchors": anchors,
        "flat_stacking": {"fused": fused_flat, "penalty": penalty,
                          "ladder": ladder_scores},
        "gated_fusion": {"fused": fused_i, "penalty": pen_i,
                         "ladder": ladder_i,
                         "n_features": int(width),
                         "spilled_features": spilled,
                         "spill_note": "the [flat | gated] stack and its "
                                       "valid-slice partition, spilled to "
                                       "disk memmaps (section 6 amendment); "
                                       "values and the stacking protocol "
                                       "unchanged."},
        "global_accuracy": global_acc,
        "gate": {
            "registered": config["gate"]["registered"],
            "gain": gain,
            "required": MARGIN,
            "passed": bool(passed),
            "consequence": ("scoped negative: score-level data sharing "
                            "adds no measured value" if not passed
                            else "the gated fusion exceeds the global arm"),
        },
        "runtime_seconds": round(time.time() - started, 2),
    })
    _write(output_dir, evidence)
    print(f"\nM154 complete -> {output_dir / 'evidence.json'}", flush=True)
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
    run_m154(args.config, args.output)


if __name__ == "__main__":
    main()
