"""Probe 2: do sub-populations have lower effective rank than the whole?

Not a milestone. Not sealed. Scoping evidence for a registration decision.

Probe 1 measured that RankMe effective rank grows as roughly atoms^0.43 in this
pipeline: doubling the atom count doubles the compute and the ambient dimension
but buys only ~1.35x the useful directions. The fraction of ambient dimensions
that carry signal falls from 0.145 at 64 atoms to 0.022 at 2048.

That is a measured argument for keeping any single model small, and it is the
argument a "network of small specialists" rests on. But it only pays off if the
specialists face EASIER sub-problems than the generalist. If every class needs
as many directions as the full ten-class problem, splitting the problem buys
nothing and a network of small models is strictly worse than one large one.

So this probe asks the decisive question directly:

    is the effective rank of a single class materially lower than the
    effective rank of the full corpus, on identical features?

Encoding happens ONCE and every subset is measured on those same features, so
no subset gets a different representation. Row counts are reported beside every
figure because effective rank is bounded above by the row count, and a subset
with fewer rows could look lower-rank for that reason alone. The
row-matched control exists to separate those two explanations: it draws a
random subset of the same size, ignoring class. If the class subsets and the
row-matched control have the same effective rank, then the drop is an artifact
of having fewer rows and the specialisation argument fails.

Instrument is RankMe (arXiv:2210.02885), used unmodified, as in probe 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from experiments.common.data_cache import configure_external_cache_environment  # noqa: E402

configure_external_cache_environment()

from experiments.tier4.eval_v15_m103_atoms import (  # noqa: E402
    Whitener,
    _contrast_normalise,
    _extract_patches,
    _fit_zca,
    _load_cifar10,
    encode,
)

PATCH = 6
STRIDE = 1
POOL_GRID = 2
CONTRAST_EPSILON = 10.0
ZCA_EPSILON = 0.1
WHITENING_PATCHES = 400_000
ROWS = 10_000
BUDGET = 512
SEED = 11


def rankme(features: np.ndarray, epsilon: float = 1e-7) -> float:
    singular = np.linalg.svd(features, compute_uv=False)
    total = np.abs(singular).sum()
    if total <= 0.0:
        return 0.0
    p = np.abs(singular) / total + epsilon
    return float(np.exp(-(p * np.log(p)).sum()))


def main() -> None:
    corpus = _load_cifar10(ROWS, 1)
    images = corpus["train_images"]
    labels = corpus["train_labels"]

    rng = np.random.default_rng(SEED)
    patches = _extract_patches(images, PATCH, STRIDE)
    grid = (images.shape[1] - PATCH) // STRIDE + 1
    sample = patches[rng.choice(len(patches), WHITENING_PATCHES, replace=False)]
    sample = _contrast_normalise(sample, CONTRAST_EPSILON)
    mean, whiten = _fit_zca(sample, ZCA_EPSILON)
    whitener = Whitener(PATCH, STRIDE, CONTRAST_EPSILON, mean, whiten, grid)

    pool = _contrast_normalise(
        patches[rng.choice(len(patches), BUDGET, replace=False)],
        CONTRAST_EPSILON,
    )
    dictionary = (pool - mean) @ whiten

    features = encode(images, dictionary, whitener, POOL_GRID)
    ambient = features.shape[1]
    print(f"features {features.shape}  atoms {BUDGET}", flush=True)

    whole = rankme(features)
    print(f"\nwhole corpus   rows {len(features):>6}  RankMe {whole:9.3f}",
          flush=True)

    per_class = []
    for label in range(int(labels.max()) + 1):
        subset = features[labels == label]
        value = rankme(subset)
        per_class.append({
            "class": label,
            "rows": int(len(subset)),
            "rankme": value,
            "ratio_to_whole": value / whole,
        })
        print(f"  class {label}      rows {len(subset):>6}  "
              f"RankMe {value:9.3f}   x{value / whole:.3f} of whole",
              flush=True)

    control_rows = int(np.median([r["rows"] for r in per_class]))
    controls = []
    for trial in range(3):
        index = np.random.default_rng(9000 + trial).choice(
            len(features), control_rows, replace=False,
        )
        value = rankme(features[index])
        controls.append(value)
        print(f"  row-matched control {trial}  rows {control_rows:>6}  "
              f"RankMe {value:9.3f}   x{value / whole:.3f} of whole",
              flush=True)

    mean_class = float(np.mean([r["rankme"] for r in per_class]))
    mean_control = float(np.mean(controls))
    print(f"\nmean over classes          {mean_class:9.3f}", flush=True)
    print(f"mean row-matched control   {mean_control:9.3f}", flush=True)
    print(f"specialisation ratio       {mean_class / mean_control:9.4f}",
          flush=True)
    print("  (below 1.0 means a class really is lower-rank than a random "
          "subset of the same size; at 1.0 the drop is only the row count)",
          flush=True)

    out = {
        "probe": "rankme_effective_rank_of_subpopulations",
        "instrument": "RankMe, arXiv:2210.02885, used unmodified",
        "note": "scoping evidence, not sealed, not a milestone",
        "atoms": BUDGET,
        "ambient_dimension": ambient,
        "rows_used": ROWS,
        "seed": SEED,
        "whole_corpus_rankme": whole,
        "per_class": per_class,
        "row_matched_controls": controls,
        "control_rows": control_rows,
        "mean_class_rankme": mean_class,
        "mean_control_rankme": mean_control,
        "specialisation_ratio": mean_class / mean_control,
    }
    target = REPO / 'logs' / 'results' / 'v15' / 'rank_probes' / 'probe2_rank_by_class.json'
    target.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(f"\nwrote {target}", flush=True)


if __name__ == "__main__":
    main()
