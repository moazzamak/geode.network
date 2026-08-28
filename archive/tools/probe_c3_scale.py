"""Reconcile the v12 AUROC record against the v13 corpus by varying task width.

The v12 milestones recorded novelty AUROC between 0.902 and 0.972 on a task with
eight known classes and two unknown ones. The C3 probe records roughly 0.59 on a
hundred known classes and twenty-eight unknown ones. Both cannot describe the
same detector, but they need not conflict: a detector that only has to fence off
eight known regions faces a far easier problem than one that has to fence off a
hundred, because every added known class enlarges the accepted volume and gives
an unknown sample another way to look familiar.

This probe measures that directly. It sweeps the number of known classes over a
fixed feature space and reports how AUROC decays. Each width is repeated over
several disjointly drawn class subsets, because a single draw at width eight is
a small sample over classes and can be lucky. The spread across draws is the
quantity that tells us whether 0.95 was a detector property or a draw property.

This is a diagnostic, not a registered experiment. It opens no final labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.common.v12_metric_fields import initialize_metric_fields  # noqa: E402

CORPUS = REPO_ROOT / "logs/results/v13/domainnet_large"

GEOMETRY_PER_CLASS = 536
EVALUATION_PER_CLASS = 40
UNKNOWN_RATIO = 0.25
RANK = 16
SEED = 11


def _auroc(known: np.ndarray, unknown: np.ndarray) -> float:
    targets = np.concatenate(
        [np.zeros(len(known), dtype=np.int64), np.ones(len(unknown), dtype=np.int64)]
    )
    return float(roc_auc_score(targets, np.concatenate([known, unknown])))


def _knn_distance(queries: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Distance to the nearest reference row, blocked to bound peak memory."""
    reference_square = np.einsum("ij,ij->i", reference, reference)
    out = np.empty(len(queries), dtype=np.float64)
    for start in range(0, len(queries), 256):
        block = queries[start : start + 256]
        squared = (
            np.einsum("ij,ij->i", block, block)[:, None]
            + reference_square[None, :]
            - 2.0 * (block @ reference.T)
        )
        out[start : start + len(block)] = np.sqrt(
            np.maximum(squared.min(axis=1), 0.0)
        )
    return out


def _one_draw(
    features: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    known_count: int,
    unknown_count: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    drawn = rng.permutation(classes)
    known = drawn[:known_count]
    unknown = drawn[known_count : known_count + unknown_count]

    fit_rows: list[int] = []
    known_rows: list[int] = []
    for label in known:
        rows = rng.permutation(np.flatnonzero(labels == label))
        fit_rows.extend(rows[:GEOMETRY_PER_CLASS].tolist())
        known_rows.extend(
            rows[GEOMETRY_PER_CLASS : GEOMETRY_PER_CLASS + EVALUATION_PER_CLASS].tolist()
        )

    unknown_rows: list[int] = []
    for label in unknown:
        rows = rng.permutation(np.flatnonzero(labels == label))
        unknown_rows.extend(rows[:EVALUATION_PER_CLASS].tolist())

    fit_x = features[fit_rows]
    fit_y = labels[fit_rows]
    known_x = features[known_rows]
    unknown_x = features[unknown_rows]

    knn = _auroc(
        _knn_distance(known_x, fit_x), _knn_distance(unknown_x, fit_x)
    )
    state = initialize_metric_fields(fit_x, fit_y, rank=RANK)
    geometric = _auroc(
        np.min(state.scores(known_x), axis=1), np.min(state.scores(unknown_x), axis=1)
    )
    return {"knn": knn, "geometric": geometric}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 100],
        help="Numbers of known classes to sweep.",
    )
    parser.add_argument(
        "--draws", type=int, default=5, help="Class subsets drawn per width."
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "logs/results/v13/c3_scale.json"
    )
    args = parser.parse_args()

    features = np.load(CORPUS / "arrays/features.npy").astype(np.float64)
    labels = np.load(CORPUS / "arrays/labels.npy")
    classes = np.unique(labels)

    print(
        f"corpus {len(features)} rows, {len(classes)} classes, rank {RANK}, "
        f"{args.draws} draws per width\n"
    )
    header = f"{'known':>6} {'unknown':>8}  {'kNN AUROC':>22}  {'geometric AUROC':>22}"
    print(header)
    print("-" * len(header))

    results: dict[str, dict[str, list[float] | float]] = {}
    for width in args.widths:
        unknown_count = max(2, int(round(width * UNKNOWN_RATIO)))
        if width + unknown_count > len(classes):
            print(f"{width:>6}  skipped - not enough classes")
            continue
        draws = [
            _one_draw(features, labels, classes, width, unknown_count, SEED + index)
            for index in range(args.draws)
        ]
        knn = np.array([draw["knn"] for draw in draws])
        geometric = np.array([draw["geometric"] for draw in draws])
        results[str(width)] = {
            "unknown_count": unknown_count,
            "knn": knn.tolist(),
            "geometric": geometric.tolist(),
            "knn_mean": float(knn.mean()),
            "geometric_mean": float(geometric.mean()),
        }
        print(
            f"{width:>6} {unknown_count:>8}  "
            f"{knn.mean():.4f} [{knn.min():.4f}, {knn.max():.4f}]  "
            f"{geometric.mean():.4f} [{geometric.min():.4f}, {geometric.max():.4f}]"
        )

    args.output.write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nwrote {args.output.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
