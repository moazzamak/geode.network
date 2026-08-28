"""M153 — routing granularity: class-group splits + fused readout.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v23.md`` (section 4
M153; 16 Aug 2026). Fits only, on the M143/M143b score caches.

Construction (registered): class groups = seeded 2-means over the 345
class score profiles (the global_train class-means); child-k scores = the
GLOBAL head's score vector with non-group classes masked to -1e30;
fusion = stacking over [child1..childK masked vectors, global] fit on the
train scores with the M143b valid-slice penalty protocol, evaluated on
the test scores. NO new head is fit, hence no section 5.3 floor applies.
Control: K random class partitions, same fusion.

Anchor (before any new number): the M143b protocol reproduced from the
same caches — fused test read 0.22431884057971013 and the global arm
0.22460869565217392 (tol 1e-9).

Gate per K: fused(kmeans) >= global + 0.005 AND fused(kmeans) >=
fused(random); otherwise the M143b single-head verdict extends to finer
granularity. Smoke declares inadmissibility and refuses the sealed
output directory.
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
from experiments.tier4.eval_v16_m143_integration import (
    _select_penalty,
    _stacking_fit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v23"
                  / "m153_routing_granularity.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v23"
                  / "m153_routing_granularity")

CLASSES = 345
TOLERANCE = 1e-9
MARGIN = 0.005
VALID_FRAC = 0.8
VALID_SEED = 55
LADDER = [1.0, 10.0, 100.0, 1000.0, 10000.0]


def _class_profiles(global_scores: np.ndarray, labels: np.ndarray
                    ) -> np.ndarray:
    """(classes, classes) matrix of per-class mean score vectors."""
    out = np.zeros((CLASSES, global_scores.shape[1]), dtype=np.float64)
    for c in range(CLASSES):
        rows = labels == c
        out[c] = (global_scores[rows].mean(axis=0) if rows.any()
                  else 0.0)
    return out


def _kmeans_groups(profiles: np.ndarray, k: int, seed: int,
                   runs: int) -> np.ndarray:
    """Seeded 2-means over class profiles; returns (classes,) group ids."""
    rng = np.random.default_rng(seed)
    best_labels: np.ndarray | None = None
    best_inertia = float("inf")
    for run in range(runs):
        init = rng.choice(len(profiles), size=k, replace=False)
        centres = profiles[init].copy()
        for _ in range(100):
            dist = ((profiles[:, None, :] - centres[None, :, :]) ** 2).sum(
                axis=2)
            labels = np.argmin(dist, axis=1)
            new_centres = np.stack(
                [profiles[labels == g].mean(axis=0) if (labels == g).any()
                 else centres[g] for g in range(k)])
            if np.array_equal(new_centres, centres):
                centres = new_centres
                break
            centres = new_centres
        inertia = float(((profiles - centres[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
    return np.asarray(best_labels, dtype=np.int64)


def _child_vectors(global_scores: np.ndarray, groups: np.ndarray, k: int,
                   mask_value: float) -> list[np.ndarray]:
    """Masked global score vectors per group: (n, classes) each."""
    out = []
    for g in range(k):
        masked = np.array(global_scores, dtype=np.float64, copy=True)
        mask = np.ones(len(groups), dtype=bool)
        mask[groups == g] = False
        masked[:, mask] = mask_value
        out.append(masked)
    return out


def _group_fusion(children: list[np.ndarray], global_scores: np.ndarray,
                  train_labels: np.ndarray, n_train: int,
                  test_labels: np.ndarray) -> tuple[float, float, float]:
    """Stacking over [children..., global]; returns (fused, penalty, val)."""
    train_feats = np.concatenate(
        [c[:n_train] for c in children] + [global_scores[:n_train]], axis=1)
    test_feats = np.concatenate(
        [c[n_train:] for c in children] + [global_scores[n_train:]], axis=1)
    n_tr = len(train_labels)
    order = np.random.default_rng(VALID_SEED).permutation(n_tr)
    cut = int(VALID_FRAC * n_tr)
    ft, fv = order[:cut], order[cut:]

    def metric(penalty):
        predict = _stacking_fit(train_feats[ft], train_labels[ft], penalty)
        return float((predict(train_feats[fv])
                      == train_labels[fv]).mean())

    penalty, ladder_scores = _select_penalty(metric, LADDER)
    stacking = _stacking_fit(train_feats, train_labels, penalty)
    preds = stacking(test_feats)
    return float((preds == test_labels).mean()), float(penalty), ladder_scores


def run_m153(config_path: Path, output_dir: Path) -> dict[str, Any]:
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
    specialist_train = m143b["specialist_train"][:, :smoke_train, :]
    global_train = m143b["global_train"][:smoke_train, :]
    train_labels = m143b["train_labels"][:smoke_train]
    specialist_test = m143["specialist_scores"][:, :smoke_test, :]
    global_test = m143["global_scores"][:smoke_test, :]
    test_labels = m143["test_labels"][:smoke_test]
    n_train = len(train_labels)
    n_test = len(test_labels)
    global_all = np.concatenate([global_train, global_test], axis=0)
    evidence: dict[str, Any] = {
        "milestone": "M153",
        "cell": "routing granularity (class-group splits)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
    }

    # ---- anchor: the M143b protocol reproduced ------------------------------
    print("anchor: M143b stacking reproduction", flush=True)
    train_concat = np.concatenate(
        [specialist_train.reshape(6, n_train, CLASSES).transpose(1, 0, 2)
            .reshape(n_train, -1), global_train], axis=1)
    test_concat = np.concatenate(
        [specialist_test.reshape(6, n_test, CLASSES).transpose(1, 0, 2)
            .reshape(n_test, -1), global_test], axis=1)
    order = np.random.default_rng(VALID_SEED).permutation(n_train)
    cut = int(VALID_FRAC * n_train)
    ft, fv = order[:cut], order[cut:]

    def metric(penalty):
        predict = _stacking_fit(train_concat[ft], train_labels[ft], penalty)
        return float((predict(train_concat[fv])
                      == train_labels[fv]).mean())

    penalty, _ladder = _select_penalty(metric, LADDER)
    stacking = _stacking_fit(train_concat, train_labels, penalty)
    fused_m143b = float((stacking(test_concat) == test_labels).mean())
    global_acc = float(
        (np.argmax(global_test, axis=1) == test_labels).mean())
    anchors = {
        "m143b_fused": {"measured": fused_m143b,
                        "sealed": float(config["anchors"]["m143b_fused"]),
                        "delta": fused_m143b
                        - float(config["anchors"]["m143b_fused"]),
                        "tolerance": TOLERANCE},
        "m143b_global": {"measured": global_acc,
                         "sealed": float(config["anchors"]["m143b_global"]),
                         "delta": global_acc
                         - float(config["anchors"]["m143b_global"]),
                         "tolerance": TOLERANCE},
    }
    print(f"  fused {fused_m143b:.6f}; global {global_acc:.6f}", flush=True)
    if not skip_anchors and (abs(anchors["m143b_fused"]["delta"]) > TOLERANCE
                             or abs(anchors["m143b_global"]["delta"])
                             > TOLERANCE):
        evidence.update({"void": True,
                         "void_reason": "M143b anchor reproduction failed",
                         "anchors": anchors})
        _write(output_dir, evidence)
        return evidence

    # ---- class-group splits --------------------------------------------------
    profiles = _class_profiles(global_train, train_labels)
    mask_value = float(config["cell"]["mask_value"])
    results: dict[str, Any] = {}
    gate: dict[str, Any] = {}
    for k in config["cell"]["k_values"]:
        k = int(k)
        print(f"K={k}: 2-means groups + fusion", flush=True)
        groups = _kmeans_groups(profiles, k, int(config["cell"]
                                                 ["kmeans_seed"]),
                                int(config["cell"]["kmeans_runs"]))
        children = _child_vectors(global_all, groups, k, mask_value)
        fused_k, pen_k, ladder_k = _group_fusion(
            children, global_all, train_labels, n_train, test_labels)
        rng = np.random.default_rng(int(config["cell"]
                                       ["random_control_seed"]))
        rand_groups = np.arange(CLASSES)
        rng.shuffle(rand_groups)
        rand_groups = rand_groups % k
        rand_children = _child_vectors(global_all, rand_groups, k, mask_value)
        fused_rand, _pen, _ladder = _group_fusion(
            rand_children, global_all, train_labels, n_train, test_labels)
        results[str(k)] = {
            "fused_kmeans": fused_k,
            "fused_random": fused_rand,
            "penalty_kmeans": pen_k,
            "group_sizes": [int((groups == g).sum()) for g in range(k)],
        }
        passed = (fused_k >= global_acc + MARGIN
                  and fused_k >= fused_rand)
        gate[str(k)] = {"passed": bool(passed),
                        "beats_global": bool(fused_k >= global_acc + MARGIN),
                        "beats_random": bool(fused_k >= fused_rand)}
        print(f"  kmeans {fused_k:.6f} vs random {fused_rand:.6f} "
              f"vs global {global_acc:.6f} -> passed={passed}", flush=True)

    evidence.update({
        "anchors": anchors,
        "results": results,
        "global_accuracy": global_acc,
        "gate": {
            "registered": config["gate"]["registered"],
            "per_k": gate,
            "fired": not any(g["passed"] for g in gate.values()),
            "consequence": ("the single-head verdict of M143b extends to "
                            "class-group granularity"
                            if not any(g["passed"] for g in gate.values())
                            else "a finer granularity passes"),
        },
        "runtime_seconds": round(time.time() - started, 2),
    })
    _write(output_dir, evidence)
    print(f"\nM153 complete -> {output_dir / 'evidence.json'}", flush=True)
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
    run_m153(args.config, args.output)


if __name__ == "__main__":
    main()
