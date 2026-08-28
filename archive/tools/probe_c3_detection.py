"""Diagnostic probe for branch criterion C3 — can unseen classes be detected at all?

Prior evidence is contradictory. The v12 milestones recorded AUROC between 0.902
and 0.972 for the geometric novelty score, while M78 recorded unknown-class
recall of 0.50% to 4.24% at a matched false-alarm rate. An AUROC of 0.95 at 8%
false alarm implies recall near 85%, not near zero, so at most one of those
numbers describes the detector.

This probe separates the two possibilities on the v13 corpus, where rank 32 is
adequately sampled for the first time (536 samples per class, 16.75 per fitted
dimension against a floor of 10):

  * If AUROC is high and recall at threshold is low, the ranking is fine and the
    threshold policy is admitting too much volume. Fixable.
  * If AUROC is near 0.5 while the kNN and nearest-class-mean controls on the
    identical features are high, the geometric score fails to rank structure
    that is present. Fixable in principle.
  * If the controls are also near chance, unknowns lie inside known-class
    structure in this feature space and no geometric score recovers them. Fatal
    to the branch under the registered kill criterion.

This is a diagnostic, not a registered experiment. It opens no final labels; it
reads only the training corpus and holds out whole classes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.common.v12_metric_fields import initialize_metric_fields  # noqa: E402

CORPUS = REPO_ROOT / "logs/results/v13/domainnet_large"

KNOWN_CLASS_COUNT = 100
GEOMETRY_PER_CLASS = 536
EVALUATION_PER_CLASS = 40
RANKS = (4, 8, 16, 32)
SEED = 11


def _split(features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Hold out whole classes as unknown, and split known classes by row."""
    rng = np.random.default_rng(SEED)
    classes = np.unique(labels)
    known = classes[:KNOWN_CLASS_COUNT]
    unknown = classes[KNOWN_CLASS_COUNT:]

    fit_rows: list[int] = []
    evaluation_rows: list[int] = []
    for label in known:
        rows = rng.permutation(np.flatnonzero(labels == label))
        fit_rows.extend(rows[:GEOMETRY_PER_CLASS].tolist())
        evaluation_rows.extend(
            rows[GEOMETRY_PER_CLASS : GEOMETRY_PER_CLASS + EVALUATION_PER_CLASS].tolist()
        )

    unknown_rows: list[int] = []
    for label in unknown:
        rows = rng.permutation(np.flatnonzero(labels == label))
        unknown_rows.extend(rows[:EVALUATION_PER_CLASS].tolist())

    return {
        "fit": np.asarray(sorted(fit_rows), dtype=np.int64),
        "known_evaluation": np.asarray(sorted(evaluation_rows), dtype=np.int64),
        "unknown_evaluation": np.asarray(sorted(unknown_rows), dtype=np.int64),
        "known_classes": known,
        "unknown_classes": unknown,
    }


def _recall_at_false_alarm(
    known_scores: np.ndarray, unknown_scores: np.ndarray, false_alarm: float
) -> float:
    """Fraction of unknowns flagged when the threshold admits 1 - false_alarm of knowns."""
    threshold = float(np.quantile(known_scores, 1.0 - false_alarm, method="higher"))
    return float(np.mean(unknown_scores > threshold))


def _report(name: str, known: np.ndarray, unknown: np.ndarray) -> dict[str, float]:
    targets = np.concatenate(
        [np.zeros(len(known), dtype=np.int64), np.ones(len(unknown), dtype=np.int64)]
    )
    auroc = float(roc_auc_score(targets, np.concatenate([known, unknown])))
    row = {
        "auroc": auroc,
        "recall_at_10pct_fa": _recall_at_false_alarm(known, unknown, 0.10),
        "recall_at_5pct_fa": _recall_at_false_alarm(known, unknown, 0.05),
    }
    print(
        f"  {name:<28} AUROC {row['auroc']:.4f}   "
        f"recall@10%FA {row['recall_at_10pct_fa'] * 100:6.2f}%   "
        f"recall@5%FA {row['recall_at_5pct_fa'] * 100:6.2f}%"
    )
    return row


def main() -> None:
    global GEOMETRY_PER_CLASS, EVALUATION_PER_CLASS

    features = np.load(CORPUS / "arrays/features.npy").astype(np.float64)
    labels = np.load(CORPUS / "arrays/labels.npy")

    # Optional single-domain restriction. The corpus mixes six DomainNet domains,
    # and 61% of it is quickdraw, so domain variation may swamp class variation.
    # Restricting to one domain separates "unknown class is undetectable" from
    # "unknown class is undetectable once six domains are mixed together".
    domain_label = None
    if len(sys.argv) > 1:
        domain_label = int(sys.argv[1])
        # Single-domain depth is shallower than the mixed corpus, so the budget
        # shrinks. Rank 32 is under-sampled here; the comparison across ranks is
        # the point, not a gated adequacy claim.
        GEOMETRY_PER_CLASS, EVALUATION_PER_CLASS = 300, 50
        manifest = json.loads(
            (CORPUS / "selection_manifest.json").read_text(encoding="utf-8")
        )
        domains = np.array(
            [entry["domain"] for entry in manifest["selection"]], dtype=np.int64
        )
        if len(domains) != len(labels):
            raise ValueError("Manifest is not row-aligned with the feature array.")
        keep = domains == domain_label
        features, labels = features[keep], labels[keep]

        # Keep only classes that clear the fit-plus-evaluation budget.
        needed = GEOMETRY_PER_CLASS + EVALUATION_PER_CLASS
        counts = {int(c): int(np.sum(labels == c)) for c in np.unique(labels)}
        usable = sorted(c for c, n in counts.items() if n >= needed)
        if len(usable) < KNOWN_CLASS_COUNT + 10:
            raise ValueError(
                f"Domain {domain_label} has {len(usable)} classes with >= {needed} "
                f"samples; need at least {KNOWN_CLASS_COUNT + 10}."
            )
        keep = np.isin(labels, usable)
        features, labels = features[keep], labels[keep]
        print(
            f"restricted to domain {domain_label}: {len(features)} rows, "
            f"{len(usable)} usable classes\n"
        )

    split = _split(features, labels)

    fit_x = features[split["fit"]]
    fit_y = labels[split["fit"]]
    known_x = features[split["known_evaluation"]]
    unknown_x = features[split["unknown_evaluation"]]

    print(
        f"known classes {len(split['known_classes'])}, "
        f"unknown classes {len(split['unknown_classes'])}, "
        f"fit rows {len(fit_x)}, known eval {len(known_x)}, "
        f"unknown eval {len(unknown_x)}\n"
    )

    results: dict[str, dict[str, float]] = {}

    # Control 1 - nearest class mean. Trivially composable, bounded memory.
    centers = np.stack(
        [fit_x[fit_y == label].mean(axis=0) for label in split["known_classes"]]
    )
    ncm_known = np.min(
        np.linalg.norm(known_x[:, None, :] - centers[None, :, :], axis=2), axis=1
    )
    ncm_unknown = np.min(
        np.linalg.norm(unknown_x[:, None, :] - centers[None, :, :], axis=2), axis=1
    )
    print("controls (identical features):")
    results["nearest_class_mean"] = _report("nearest class mean", ncm_known, ncm_unknown)

    # Control 2 - kNN distance to the nearest fitted sample. The free-composability bar.
    fit_square = np.einsum("ij,ij->i", fit_x, fit_x)

    def knn_distance(queries: np.ndarray) -> np.ndarray:
        out = np.empty(len(queries), dtype=np.float64)
        for start in range(0, len(queries), 256):
            block = queries[start : start + 256]
            block_square = np.einsum("ij,ij->i", block, block)
            squared = (
                block_square[:, None] + fit_square[None, :] - 2.0 * (block @ fit_x.T)
            )
            out[start : start + len(block)] = np.sqrt(np.maximum(squared.min(axis=1), 0.0))
        return out

    results["knn"] = _report(
        "kNN nearest distance", knn_distance(known_x), knn_distance(unknown_x)
    )

    # Positive control - synthetic far-field points must be detected near perfectly.
    # If they are not, the measurement itself is broken and no verdict is admissible.
    rng = np.random.default_rng(SEED + 1)
    scale = float(np.linalg.norm(fit_x, axis=1).mean())
    far_field = rng.normal(size=(1000, fit_x.shape[1])) * scale
    print("\npositive control (must be near 1.0):")
    _report("NCM on far-field noise", ncm_known, np.min(
        np.linalg.norm(far_field[:, None, :] - centers[None, :, :], axis=2), axis=1
    ))
    _report("kNN on far-field noise", knn_distance(known_x), knn_distance(far_field))

    # Negative control - held-out KNOWN samples must sit near chance, since they
    # are not unknown. A value far from 0.5 would indicate a leaking split.
    held_out = known_x[: len(unknown_x)]
    print("\nnegative control (must be near 0.5):")
    _report("kNN on held-out knowns", knn_distance(known_x), knn_distance(held_out))

    # The geometric head, at each rank, on adequately sampled bases.
    print("\ngeometric novelty score:")
    for rank in RANKS:
        state = initialize_metric_fields(fit_x, fit_y, rank=rank)
        known_scores = np.min(state.scores(known_x), axis=1)
        unknown_scores = np.min(state.scores(unknown_x), axis=1)
        results[f"geometric_rank_{rank}"] = _report(
            f"rank {rank} (n/dim {GEOMETRY_PER_CLASS / rank:.1f})",
            known_scores,
            unknown_scores,
        )

    print("\nverdict inputs:")
    best_control = max(
        results["knn"]["auroc"], results["nearest_class_mean"]["auroc"]
    )
    best_geometric = max(
        results[f"geometric_rank_{rank}"]["auroc"] for rank in RANKS
    )
    print(f"  best control AUROC   {best_control:.4f}")
    print(f"  best geometric AUROC {best_geometric:.4f}")
    print(f"  difference           {best_geometric - best_control:+.4f}")

    (REPO_ROOT / "logs/results/v13/c3_probe.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
