"""M136 — the head objective axis on the sealed 6144-atom codes.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v21.md`` section 1 and
``experiments/configs/v16/m136_margin_head.json``.

Question. M128 showed accuracy lives in a thin positive-margin tail (the correct
class loses the argmax for ~78% of test rows). The head is a closed-form ridge at
lambda = 1.0 chosen once (M107). Does frozen-code accuracy depend on the head
OBJECTIVE?

Cells (all reuse the SEALED M117 6144-atom code memmaps; no encode, no GPU):

1. Ridge penalty ladder lambda in {0.01, 0.1, 0.3, 1.0, 10.0} at n = 138000.
   lambda >= 1.0 is sealed at M108 (monotone decline); lambda < 1.0 is unmeasured.
   t1: lambda = 1.0 must reproduce M117's sealed 0.2248695652173913 (0.002).
2. Smoothed-target ridge (epsilon 0.1, n = 138000). Registered analytic
   expectation: for the standardised ridge with intercept, the epsilon/345
   uniform term cancels the centring exactly, so cross_s = (1 - epsilon) * cross
   and the intercept softens to (1 - epsilon) * p_hat + epsilon/345 — a
   per-row constant score shift, hence predictions IDENTICAL to lambda = 1.0.
   The cell verifies the lemma numerically.
3. Batch multi-class hinge at n = 34500, lambda in {1e-4, 1e-3}, 8 epochs,
   deterministic subgradient (see config). t2: the same-cell ridge (n = 34500)
   must reproduce M117's sealed 0.09057971014492754 (0.002).

Kill switch: if NEITHER any lambda < 1.0 cell NOR any hinge cell beats its own
same-cell ridge by >= +0.003, the head-objective axis is CLOSED and lambda = 1.0
ridge remains the head (negative sealed). Margins (f_true - max_other quantiles)
are reported for the hinge arm and its ridge control.

No pixel-identity verification is performed here because no encode happens: the
codes are the sealed memmaps reused byte-for-byte, and the t1/t2 anchors verify
the codes-to-labels pairing end-to-end (registered reasoning, v21 section 3).

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m136_margin_head
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator

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
from experiments.tier4.eval_v15_m107_dense import _solve_and_score
from experiments.tier4.eval_v16_m109_trunk import _load_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m136_margin_head.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m136_margin_head"

T1_TOLERANCE = 0.002
KS_MARGIN = 0.003
CLASSES = 345


# --------------------------------------------------------------------------
# head fits (pure, tested)
# --------------------------------------------------------------------------
def _fit_ridge(mem_train: np.ndarray, labels: np.ndarray, n: int,
               block: int) -> RidgeAccumulator:
    """Streaming ridge accumulator over the first *n* rows of the memmap."""
    acc = RidgeAccumulator(int(mem_train.shape[1]), CLASSES)
    for start in range(0, n, block):
        stop = min(start + block, n)
        acc.add(np.asarray(mem_train[start:stop]), labels[start:stop])
    return acc


def _add_smoothed(acc: RidgeAccumulator, features: np.ndarray, labels: np.ndarray,
                  classes: int, epsilon: float) -> None:
    """Accumulate one block with smoothed one-hot targets Y_s = (1-e)Y + e/C."""
    block_rows = np.asarray(features, dtype=np.float64)
    n_rows = len(block_rows)
    targets = np.full((n_rows, classes), epsilon / classes, dtype=np.float64)
    targets[np.arange(n_rows), labels] += 1.0 - epsilon
    acc.gram += block_rows.T @ block_rows
    acc.column_sum += block_rows.sum(axis=0)
    acc.cross += block_rows.T @ targets
    acc.class_count += targets.sum(axis=0)
    acc.rows += n_rows


def _fit_smoothed(mem_train: np.ndarray, labels: np.ndarray, n: int,
                  epsilon: float, block: int) -> RidgeAccumulator:
    """Streaming accumulator with smoothed targets over the first *n* rows."""
    acc = RidgeAccumulator(int(mem_train.shape[1]), CLASSES)
    for start in range(0, n, block):
        stop = min(start + block, n)
        _add_smoothed(acc, mem_train[start:stop], labels[start:stop], CLASSES, epsilon)
    return acc


def _test_blocks(mem_test: np.ndarray, test_labels: np.ndarray,
                 test_domains: np.ndarray, block: int
                 ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    for start in range(0, len(test_labels), block):
        stop = min(start + block, len(test_labels))
        yield (np.asarray(mem_test[start:stop]),
               test_labels[start:stop], test_domains[start:stop])


def _predictions(acc: RidgeAccumulator, penalty: float, mem_test: np.ndarray,
                 test_labels: np.ndarray, block: int) -> np.ndarray:
    """Per-row argmax predictions of a fitted ridge on the full test set."""
    standardise = acc.standardiser()
    weights = acc.solve_many([penalty])[penalty]
    out: list[np.ndarray] = []
    for start in range(0, len(test_labels), block):
        stop = min(start + block, len(test_labels))
        scores = standardise(np.asarray(mem_test[start:stop])) @ weights[:-1] + weights[-1]
        out.append(np.argmax(scores, axis=1).astype(np.int64))
    return np.concatenate(out)


def _standardised_train_block(mem_train: np.ndarray, n: int,
                              block: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardised first-n train rows (float32) plus the (centre, scale) stats."""
    width = int(mem_train.shape[1])
    total = np.zeros(width, dtype=np.float64)
    sq = np.zeros(width, dtype=np.float64)
    for start in range(0, n, block):
        stop = min(start + block, n)
        x = np.asarray(mem_train[start:stop], dtype=np.float64)
        total += x.sum(axis=0)
        sq += np.square(x).sum(axis=0)
    centre = total / n
    scale = np.sqrt(np.maximum(sq / n - np.square(centre), 0.0)) + 1e-8
    xs = np.empty((n, width), dtype=np.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        xs[start:stop] = (mem_train[start:stop] - centre) / scale
    return xs, centre.astype(np.float32), scale.astype(np.float32)


def _fit_batch_hinge(xs: np.ndarray, labels: np.ndarray, classes: int, lam: float,
                     epochs: int, block: int) -> tuple[np.ndarray, list[float]]:
    """Batch multi-class hinge subgradient on the standardised rows.

    Step rule (registered): G accumulates +x on the true class and -x on the
    best WRONG class for every row with margin < 1, averaged over rows;
    W = (1 - 1/t) * W + (1/(lambda*t)) * G with t the epoch index. Corpus
    order, fully deterministic.
    """
    width = xs.shape[1]
    n = len(labels)
    weights = np.zeros((width, classes), dtype=np.float64)
    objective: list[float] = []
    for epoch in range(1, epochs + 1):
        grad = np.zeros_like(weights)
        hinge_loss = 0.0
        for start in range(0, n, block):
            stop = min(start + block, n)
            scores = xs[start:stop].astype(np.float64) @ weights
            y = labels[start:stop]
            wrong = scores.copy()
            wrong[np.arange(stop - start), y] = -np.inf
            yhat = np.argmax(wrong, axis=1)
            margin = scores[np.arange(stop - start), y] - wrong[np.arange(stop - start), yhat]
            violators = margin < 1.0
            if violators.any():
                idx = np.arange(start, stop)[violators]
                gy = y[violators]
                gh = yhat[violators]
                rows = xs[idx].T.astype(np.float64)  # (width, n_viol)
                np.add.at(grad, (slice(None), gy), rows)
                np.subtract.at(grad, (slice(None), gh), rows)
                hinge_loss += float(np.clip(1.0 - margin[violators], 0.0, None).sum())
        grad /= n
        shrink = 1.0 - 1.0 / epoch
        weights = shrink * weights + (1.0 / (lam * epoch)) * grad
        objective.append(hinge_loss / n)
    return weights, objective


def _hinge_scores(xs_test: np.ndarray, weights: np.ndarray, prior: np.ndarray,
                  block: int) -> tuple[np.ndarray, np.ndarray]:
    """Scores and predictions of a hinge weight matrix on standardised test rows."""
    predictions: list[np.ndarray] = []
    margins_rows: list[np.ndarray] = []
    for start in range(0, len(xs_test), block):
        stop = min(start + block, len(xs_test))
        scores = xs_test[start:stop].astype(np.float64) @ weights + prior
        predictions.append(np.argmax(scores, axis=1).astype(np.int64))
        margins_rows.append(scores[np.arange(stop - start), predictions[-1]])
    return np.concatenate(predictions), np.concatenate(margins_rows)


def _margin_quantiles(scores_true: np.ndarray, max_other: np.ndarray) -> dict[str, float]:
    """Margins f_true - max_other: quantiles and positive share (the M128 object)."""
    margins = scores_true - max_other
    return {
        "q25": float(np.quantile(margins, 0.25)),
        "q50": float(np.quantile(margins, 0.50)),
        "q75": float(np.quantile(margins, 0.75)),
        "q95": float(np.quantile(margins, 0.95)),
        "positive_share": float((margins > 0).mean()),
        "margin_mean": float(margins.mean()),
    }


def _accuracy_per_domain(predictions: np.ndarray, test_labels: np.ndarray,
                         test_domains: np.ndarray) -> tuple[float, list[float]]:
    correct = predictions == test_labels
    per_domain: list[float] = []
    for domain in range(6):
        mask = test_domains == domain
        per_domain.append(float(correct[mask].mean()) if mask.any() else 0.0)
    return float(correct.mean()), per_domain


def _ridge_margin_quantiles(acc: RidgeAccumulator, penalty: float, mem_test: np.ndarray,
                            test_labels: np.ndarray, block: int) -> dict[str, float]:
    """Margins of a fitted ridge on the test set (same object as M128's)."""
    standardise = acc.standardiser()
    weights = acc.solve_many([penalty])[penalty]
    true_scores: list[float] = []
    other_max: list[float] = []
    for start in range(0, len(test_labels), block):
        stop = min(start + block, len(test_labels))
        scores = standardise(np.asarray(mem_test[start:stop])) @ weights[:-1] + weights[-1]
        wrong = scores.copy()
        wrong[np.arange(stop - start), test_labels[start:stop]] = -np.inf
        true_scores.append(scores[np.arange(stop - start), test_labels[start:stop]])
        other_max.append(wrong.max(axis=1))
    return _margin_quantiles(np.concatenate(true_scores), np.concatenate(other_max))


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def run_m136(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if "_smoke_note" in config and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    configure_external_cache_environment()
    block = int(config["numerics"]["block"])
    smoke = bool(config.get("_smoke_skip_gates", False))

    print("loading corpus labels (subsample digest verified by _load_corpus)", flush=True)
    started = time.time()
    corpus, _, _ = _load_corpus(config)
    train_labels = corpus["train_labels"]
    test_labels = corpus["test_labels"]
    test_domains = corpus["test_domains"]

    print("opening sealed 6144-atom code memmaps", flush=True)
    codes_dir = data_cache_root() / config["sealed_codes"]["cache_relpath"]
    train_path = codes_dir / config["sealed_codes"]["train_file"]
    test_path = codes_dir / config["sealed_codes"]["test_file"]
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(f"sealed code memmaps missing under {codes_dir}")
    mem_train = np.load(train_path, mmap_mode="r")
    mem_test = np.load(test_path, mmap_mode="r")
    width = int(config["sealed_codes"]["width"])
    expected_shape = (len(train_labels), width)
    if mem_train.shape != expected_shape:
        raise SystemExit(
            f"code memmap shape {mem_train.shape} != expected {expected_shape}")
    test_rows = len(test_labels)

    head = config["head"]
    evidence: dict[str, Any] = {
        "milestone": "M136",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "codes": {
            "dir": str(codes_dir),
            "train_shape": list(mem_train.shape),
            "test_shape": list(mem_test.shape),
        },
    }

    # ---- cell 1: ridge penalty ladder at full data -------------------------
    ladder = [float(p) for p in head["ridge"]["penalty_ladder"]]
    n_full = int(head["ridge"]["n"])
    print(f"ridge ladder {ladder} at n={n_full}", flush=True)
    acc_full = _fit_ridge(mem_train, train_labels, n_full, block)
    ridge_result = _solve_and_score(
        acc_full, ladder, _test_blocks(mem_test, test_labels, test_domains, block))
    evidence["ridge_ladder"] = {
        str(p): {
            "accuracy": ridge_result["accuracy_by_penalty"][str(p)],
            "per_domain": [
                c / r for c, r in zip(
                    ridge_result["per_domain_correct"][str(p)],
                    ridge_result["per_domain_rows"][str(p)])
            ],
        }
        for p in ladder
    }
    ridge_full_1 = evidence["ridge_ladder"]["1.0"]["accuracy"]

    if not smoke:
        ref_full = 0.2248695652173913
        t1_delta = ridge_full_1 - ref_full
        evidence["gates"] = {"t1_delta": t1_delta,
                             "t1_tolerance": T1_TOLERANCE}
        if abs(t1_delta) > T1_TOLERANCE:
            print(f"  t1 FAILED: {ridge_full_1:.6f} vs sealed {ref_full} "
                  f"(delta {t1_delta:+.6f})", flush=True)
            evidence["void"] = True
            evidence["void_reason"] = "t1 anchor reproduction failed"
            write_canonical_json(output_dir / "evidence.json", evidence)
            build_artifact_index(output_dir)
            return evidence
        print(f"  t1 anchor delta {t1_delta:+.6f} (<= {T1_TOLERANCE})", flush=True)

    # ---- cell 2: smoothed-target ridge -------------------------------------
    epsilon = float(head["smoothed"]["epsilon"])
    print(f"smoothed ridge epsilon={epsilon} at n={n_full}", flush=True)
    acc_smooth = _fit_smoothed(mem_train, train_labels, n_full, epsilon, block)
    smooth_result = _solve_and_score(
        acc_smooth, [1.0], _test_blocks(mem_test, test_labels, test_domains, block))
    pred_ridge = _predictions(acc_full, 1.0, mem_test, test_labels, block)
    pred_smooth = _predictions(acc_smooth, 1.0, mem_test, test_labels, block)
    evidence["smoothed"] = {
        "epsilon": epsilon,
        "accuracy": smooth_result["accuracy_by_penalty"]["1.0"],
        "per_domain": [
            c / r for c, r in zip(
                smooth_result["per_domain_correct"]["1.0"],
                smooth_result["per_domain_rows"]["1.0"])
        ],
        "exact_prediction_match_share_vs_ridge_1_0": float((pred_ridge == pred_smooth).mean()),
        "analytic_note": config["head"]["smoothed"]["analytic_note"],
    }

    # ---- cell 3: same-cell ridge control + batch hinge ---------------------
    n_small = int(head["hinge"]["n"])
    print(f"same-cell ridge at n={n_small} + standardisation", flush=True)
    acc_small = _fit_ridge(mem_train, train_labels, n_small, block)
    small_result = _solve_and_score(
        acc_small, [1.0], _test_blocks(mem_test, test_labels, test_domains, block))
    ridge_small_1 = small_result["accuracy_by_penalty"]["1.0"]
    evidence["same_cell_ridge"] = {
        "n": n_small,
        "penalty": 1.0,
        "accuracy": ridge_small_1,
        "per_domain": [
            c / r for c, r in zip(
                small_result["per_domain_correct"]["1.0"],
                small_result["per_domain_rows"]["1.0"])
        ],
        "margins": _ridge_margin_quantiles(acc_small, 1.0, mem_test, test_labels, block),
    }
    if not smoke:
        ref_small = 0.09057971014492754
        t2_delta = ridge_small_1 - ref_small
        evidence["gates"]["t2_delta"] = t2_delta
        evidence["gates"]["t2_tolerance"] = T1_TOLERANCE
        if abs(t2_delta) > T1_TOLERANCE:
            print(f"  t2 FAILED: {ridge_small_1:.6f} vs sealed {ref_small} "
                  f"(delta {t2_delta:+.6f})", flush=True)
            evidence["void"] = True
            evidence["void_reason"] = "t2 anchor reproduction failed"
            write_canonical_json(output_dir / "evidence.json", evidence)
            build_artifact_index(output_dir)
            return evidence
        print(f"  t2 anchor delta {t2_delta:+.6f} (<= {T1_TOLERANCE})", flush=True)

    print(f"batch hinge at n={n_small}, lambdas={head['hinge']['lambdas']}", flush=True)
    xs, centre, scale = _standardised_train_block(mem_train, n_small, block)
    prior = acc_small.class_count / acc_small.rows  # same intercept as the ridge head
    xs_test = np.empty((test_rows, width), dtype=np.float32)
    for start in range(0, test_rows, block):
        stop = min(start + block, test_rows)
        xs_test[start:stop] = (mem_test[start:stop] - centre) / scale

    evidence["hinge"] = {}
    for lam in head["hinge"]["lambdas"]:
        lam = float(lam)
        weights, objective = _fit_batch_hinge(
            xs, train_labels[:n_small], CLASSES, lam,
            int(head["hinge"]["epochs"]), block)
        predictions, _ = _hinge_scores(xs_test, weights, prior, block)
        accuracy, per_domain = _accuracy_per_domain(
            predictions, test_labels, test_domains)
        true_scores: list[float] = []
        other_max: list[float] = []
        for start in range(0, test_rows, block):
            stop = min(start + block, test_rows)
            scores = xs_test[start:stop].astype(np.float64) @ weights + prior
            wrong = scores.copy()
            wrong[np.arange(stop - start), test_labels[start:stop]] = -np.inf
            true_scores.append(scores[np.arange(stop - start), test_labels[start:stop]])
            other_max.append(wrong.max(axis=1))
        margins = _margin_quantiles(np.concatenate(true_scores), np.concatenate(other_max))
        evidence["hinge"][str(lam)] = {
            "epochs": int(head["hinge"]["epochs"]),
            "accuracy": accuracy,
            "per_domain": per_domain,
            "final_objective": objective[-1],
            "margins": margins,
        }
        print(f"  hinge lambda={lam}: {accuracy:.4f} (ridge cell {ridge_small_1:.4f})",
              flush=True)

    # ---- kill switch -------------------------------------------------------
    if not smoke:
        best_ridge_alt = max(evidence["ridge_ladder"][str(p)]["accuracy"]
                             for p in ladder if p < 1.0)
        best_hinge = max(cell["accuracy"] for cell in evidence["hinge"].values())
        margin_over_ridge = max(best_ridge_alt - ridge_full_1, best_hinge - ridge_small_1)
        fired = margin_over_ridge < KS_MARGIN
        evidence["gates"]["kill_switch_objective"] = {
            "registered": config["gate"]["kill_switch_objective"],
            "best_lambda_under_1": best_ridge_alt,
            "lambda_1_reference": ridge_full_1,
            "best_hinge": best_hinge,
            "same_cell_ridge": ridge_small_1,
            "best_margin_over_same_cell_ridge": margin_over_ridge,
            "required": KS_MARGIN,
            "fired": fired,
            "consequence": (
                "head-objective axis CLOSED: lambda=1.0 ridge remains the head "
                "(negative sealed and reported)" if fired else
                "a head objective beats its same-cell ridge by the registered "
                "margin: the objective is a measured lever; escalate to full data"),
        }

    evidence["runtime_seconds"] = round(time.time() - started, 2)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"wrote {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    run_m136(Path(args.config), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
