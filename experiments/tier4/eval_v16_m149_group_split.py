"""M149 — group splitting (split-and-rebuild), the transactional registry split.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v22.md`` (section 7
splitting, section 9 M149, 14 Aug 2026).

Question (registered before measurement). When a registered task group turns
out to contain two subpopulations, can the registry split it into two
closed-form specialists plus a fused readout WITHOUT losing accuracy — and is
the win from real structure rather than from the proposal step itself?

Protocol (registered):
- Input: the M143 cached score matrices (data_cache_root()/v16/m143/scores.npz),
  never recomputed. A group = the test rows of one domain (the incumbent
  specialist's group). The experiment runs entirely on held-out rows, exactly
  like M143's stacking protocol.
- Per group A: rows split 50/50 (fit / eval, seeded). On the FIT half:
  incumbent A_ridge fit on all rows; proposal = seeded 2-means on the
  standardised 2415-dim score vectors -> children X, Y; child ridges fit on
  their cluster rows; fusion = stacking over [X_scores, Y_scores] with the
  M143 penalty ladder selected on a validation slice of the fit half.
  On the EVAL half: incumbent accuracy, fused(2-means) accuracy, fused(random)
  accuracy (seeded random split, same recipe).
- Row floor: a group with fewer than 2 x floor rows on the fit half is
  SKIPPED, not failed (data starvation, E10).
- Gate (kill switch, registered): a domain PASSES iff fused(2-means) >=
  incumbent + margin AND fused(2-means) >= fused(random) on its eval half.
  M149 fires (closes as a scoped negative) iff ZERO domains pass; otherwise
  PASS with the per-domain table as evidence.

Reproduce with::

    .\\.venv\\Scripts\\python.exe -m experiments.tier4.eval_v16_m149_group_split
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.data_cache import data_cache_root
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v16_m143_integration import (
    CLASSES,
    DOMAINS,
    _select_penalty,
    _split_indices,
    _stacking_fit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m149_group_split.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m149_group_split"


def _kmeans_2(features: np.ndarray, seed: int, iterations: int = 20
              ) -> np.ndarray:
    """Seeded 2-means over standardised rows. Returns (n,) cluster labels."""
    xs = np.asarray(features, dtype=np.float64)
    centre = xs.mean(axis=0)
    scale = xs.std(axis=0)
    scale[scale < 1e-12] = 1.0
    xs = (xs - centre) / scale
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xs), size=2, replace=False)
    centroids = xs[idx].copy()
    labels = np.zeros(len(xs), dtype=np.int64)
    for _ in range(iterations):
        dists = ((xs[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dists, axis=1)
        for c in range(2):
            members = labels == c
            if members.any():
                centroids[c] = xs[members].mean(axis=0)
    return labels


def _fusion_on_children(
    rows: np.ndarray, group_scores: np.ndarray, labels: np.ndarray,
    cluster_labels: np.ndarray, ladder: list[float],
) -> tuple[float, float, dict[str, float]]:
    """Split *rows* by cluster labels, fit children + stacking; used by the
    runner's inline recipe instead of a separate fit/eval split."""
    child_0 = rows[cluster_labels == 0]
    child_1 = rows[cluster_labels == 1]
    n = len(rows)
    fit_features = np.zeros((n, 2 * CLASSES), dtype=np.float64)
    pred_0 = _fit_child(child_0, group_scores, labels)
    pred_1 = _fit_child(child_1, group_scores, labels)
    fit_features[:, :CLASSES] = pred_0(group_scores[rows])
    fit_features[:, CLASSES:] = pred_1(group_scores[rows])

    def metric(penalty):
        predict = _stacking_fit(fit_features, labels[rows], penalty)
        return float((predict(fit_features) == labels[rows]).mean())

    best, ladder_scores = _select_penalty(metric, ladder)
    return best, ladder_scores, 2 * CLASSES


def _fit_child(rows: np.ndarray, group_scores: np.ndarray,
               labels: np.ndarray):
    """Ridge on one child's rows -> 345-way score predictions for any rows."""
    sub = group_scores[rows]
    centre = sub.mean(axis=0)
    scale = sub.std(axis=0)
    scale[scale < 1e-12] = 1.0
    xs = (sub - centre) / scale
    targets = np.zeros((len(rows), CLASSES), dtype=np.float64)
    targets[np.arange(len(rows)), np.asarray(labels[rows], dtype=np.int64)] = 1.0
    w = np.linalg.solve(xs.T @ xs + 1.0 * np.eye(xs.shape[1]), xs.T @ targets)
    b = targets.mean(axis=0)

    def predict(block: np.ndarray) -> np.ndarray:
        z = (np.asarray(block, dtype=np.float64) - centre) / scale
        return z @ w + b

    return predict


def run_m149(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    scores_file = data_cache_root() / config["score_cache"]["cache_relpath"] / "scores.npz"
    payload = np.load(scores_file, allow_pickle=False)
    specialist_scores = payload["specialist_scores"]
    global_scores = payload["global_scores"]
    test_labels = payload["test_labels"]
    test_domains = payload["test_domains"]
    n_test = len(test_labels)

    # row representation for clustering = the full concatenated score vector
    concat = np.concatenate(
        [specialist_scores.reshape(DOMAINS, n_test, CLASSES)
            .transpose(1, 0, 2).reshape(n_test, -1),
         global_scores], axis=1)

    margin = float(config["gate"]["margin"])
    floor = int(config["gate"]["row_floor"])
    ladder = [float(x) for x in config["phase2"]["penalty_ladder"]]
    split_seed = int(config["phase2"]["split_seed"])
    kmeans_seed = int(config["phase2"]["kmeans_seed"])
    random_seed = int(config["phase2"]["random_seed"])

    domain_report: dict[str, Any] = {}
    for domain in range(DOMAINS):
        rows_d = np.where(test_domains == domain)[0]
        n_d = len(rows_d)
        print(f"domain {domain}: {n_d} rows", flush=True)
        if n_d < 2 * floor + 100:
            domain_report[str(domain)] = {"skipped": True,
                                          "reason": "row floor", "rows": int(n_d)}
            continue

        fit_idx, eval_idx = _split_indices(n_d, split_seed + domain)
        abs_fit = rows_d[fit_idx]
        abs_eval = rows_d[eval_idx]
        # incumbent: ridge on the fit half's own score vectors
        incumbent = _fit_child(abs_fit, concat, test_labels)
        inc_eval = np.argmax(incumbent(concat[abs_eval]), axis=1)
        inc_acc = float((inc_eval == test_labels[abs_eval]).mean())

        # proposal: 2-means on the fit-half score vectors
        cluster_labels = _kmeans_2(concat[abs_fit], kmeans_seed + domain)
        if min(int((cluster_labels == 0).sum()), int((cluster_labels == 1).sum())) < floor:
            domain_report[str(domain)] = {"skipped": True,
                                          "reason": "degenerate cluster", "rows": int(n_d)}
            continue

        # fused over the 2-means children
        fit_features = np.zeros((len(abs_fit), 2 * CLASSES), dtype=np.float64)
        pred_0 = _fit_child(abs_fit[cluster_labels == 0], concat, test_labels)
        pred_1 = _fit_child(abs_fit[cluster_labels == 1], concat, test_labels)
        fit_features[:, :CLASSES] = pred_0(concat[abs_fit])
        fit_features[:, CLASSES:] = pred_1(concat[abs_fit])

        # validation slice inside the fit half for penalty selection
        rng = np.random.default_rng(config["phase2"]["valid_seed"] + domain)
        order = rng.permutation(len(fit_idx))
        cut = int(config["phase2"]["valid_frac"] * len(fit_idx))
        ft = fit_idx[order[:cut]]
        fv = fit_idx[order[cut:]]
        abs_ft = rows_d[ft]
        abs_fv = rows_d[fv]

        def metric(penalty, children=(pred_0, pred_1)):
            feat = np.zeros((len(abs_fv), 2 * CLASSES), dtype=np.float64)
            feat[:, :CLASSES] = children[0](concat[abs_fv])
            feat[:, CLASSES:] = children[1](concat[abs_fv])
            predict = _stacking_fit(fit_features[order[:cut]],
                                    test_labels[abs_ft], penalty)
            return float((predict(feat) == test_labels[abs_fv]).mean())

        best_penalty, ladder_scores = _select_penalty(metric, ladder)
        stacking = _stacking_fit(fit_features, test_labels[abs_fit],
                                 best_penalty)
        eval_feat = np.zeros((len(abs_eval), 2 * CLASSES), dtype=np.float64)
        eval_feat[:, :CLASSES] = pred_0(concat[abs_eval])
        eval_feat[:, CLASSES:] = pred_1(concat[abs_eval])
        fused_preds = stacking(eval_feat)
        fused_acc = float((fused_preds == test_labels[abs_eval]).mean())

        # control: random split, identical recipe
        rng_r = np.random.default_rng(random_seed + domain)
        rand_labels = rng_r.integers(0, 2, size=len(abs_fit))
        fit_features_r = np.zeros((len(abs_fit), 2 * CLASSES), dtype=np.float64)
        pred_r0 = _fit_child(abs_fit[rand_labels == 0], concat, test_labels)
        pred_r1 = _fit_child(abs_fit[rand_labels == 1], concat, test_labels)
        fit_features_r[:, :CLASSES] = pred_r0(concat[abs_fit])
        fit_features_r[:, CLASSES:] = pred_r1(concat[abs_fit])

        def metric_r(penalty, children=(pred_r0, pred_r1)):
            feat = np.zeros((len(abs_fv), 2 * CLASSES), dtype=np.float64)
            feat[:, :CLASSES] = children[0](concat[abs_fv])
            feat[:, CLASSES:] = children[1](concat[abs_fv])
            predict = _stacking_fit(fit_features_r[order[:cut]],
                                    test_labels[abs_ft], penalty)
            return float((predict(feat) == test_labels[abs_fv]).mean())

        best_penalty_r, ladder_scores_r = _select_penalty(metric_r, ladder)
        stacking_r = _stacking_fit(fit_features_r, test_labels[abs_fit],
                                   best_penalty_r)
        eval_feat_r = np.zeros((len(abs_eval), 2 * CLASSES), dtype=np.float64)
        eval_feat_r[:, :CLASSES] = pred_r0(concat[abs_eval])
        eval_feat_r[:, CLASSES:] = pred_r1(concat[abs_eval])
        fused_r_acc = float((stacking_r(eval_feat_r)
                             == test_labels[abs_eval]).mean())

        passed = (fused_acc >= inc_acc + margin) and (fused_acc >= fused_r_acc)
        domain_report[str(domain)] = {
            "rows": int(n_d),
            "incumbent_eval_accuracy": inc_acc,
            "fused_2means_eval_accuracy": fused_acc,
            "fused_random_eval_accuracy": fused_r_acc,
            "selected_penalty": best_penalty,
            "random_selected_penalty": best_penalty_r,
            "ladder_scores": ladder_scores,
            "passed": passed,
        }
        print(f"  incumbent {inc_acc:.4f}; fused(2means) {fused_acc:.4f}; "
              f"fused(random) {fused_r_acc:.4f} -> passed={passed}", flush=True)

    passing = [d for d, r in domain_report.items()
               if r.get("passed")]
    fired = len(passing) == 0 and not inadmissible
    evidence: dict[str, Any] = {
        "milestone": "M149",
        "admissible_as_evidence": not inadmissible,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": ("can a transactional split into two closed-form "
                     "specialists plus a fused readout beat the incumbent "
                     "group without accuracy loss, and is the win real "
                     "structure rather than the proposal step?"),
        "domain_report": domain_report,
        "passing_domains": passing,
        "gate": {
            "registered": config["gate"]["registered"],
            "fired": fired,
            "consequence": (config["gate"]["consequence_fired"] if fired
                            else config["gate"]["consequence_passed"]),
        },
        "score_cache": {"path": str(scores_file)},
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM149 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m149(args.config, args.output)


if __name__ == "__main__":
    main()
