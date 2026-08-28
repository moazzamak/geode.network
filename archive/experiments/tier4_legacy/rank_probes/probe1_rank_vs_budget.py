"""Probe: does RankMe effective rank predict where M103's margin vanished?

Not a milestone. Not sealed. This is scoping evidence for a registration
decision and is reported as such.

The question. M103 measured that a discriminatively grown dictionary reaches a
random dictionary's accuracy with half the atoms, and that the advantage
narrows to roughly one seed spread by 1024 atoms. The random-features
literature (Avron et al. 2017, arXiv:1804.09893; Li et al. 2019,
arXiv:1806.09178) predicts exactly that narrowing: the advantage of non-uniform
selection over uniform sampling is bounded by the ratio of the ambient budget to
the effective dimension, and both converge as the budget grows past the number
of directions the data actually has.

If that explanation is right, then the effective rank of the pooled feature
matrix should SATURATE near the budget where M103's margin collapsed. If the
effective rank keeps climbing linearly with the atom count, the explanation is
wrong and the narrowing needs another cause.

The instrument is RankMe (Garrido, Balestriero, Najman & LeCun, ICML 2023,
arXiv:2210.02885), used unmodified. It is label-free and has no tunable
parameters, which is why it is used here rather than anything built for this
probe:

    p_k       = sigma_k / ||sigma||_1 + epsilon
    RankMe(Z) = exp(-sum_k p_k log p_k)

sigma are the singular values of the feature matrix. This is the exponentiated
Shannon entropy of the normalised spectrum, i.e. the effective rank of Roy &
Vetterli (2007).

Arm (a) — random patches — is the arm measured, because it is M103's null and
the reference the 2x figure is quoted against.
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
BUDGETS = [64, 128, 256, 512, 1024, 2048]
SEED = 11


def rankme(features: np.ndarray, epsilon: float = 1e-7) -> float:
    """RankMe effective rank, arXiv:2210.02885 equation 2."""
    singular = np.linalg.svd(features, compute_uv=False)
    total = np.abs(singular).sum()
    if total <= 0.0:
        return 0.0
    p = np.abs(singular) / total + epsilon
    return float(np.exp(-(p * np.log(p)).sum()))


def main() -> None:
    corpus = _load_cifar10(ROWS, 1)
    images = corpus["train_images"]
    print(f"images {images.shape}", flush=True)

    rng = np.random.default_rng(SEED)
    patches = _extract_patches(images, PATCH, STRIDE)
    grid = (images.shape[1] - PATCH) // STRIDE + 1
    sample = patches[rng.choice(len(patches), WHITENING_PATCHES, replace=False)]
    sample = _contrast_normalise(sample, CONTRAST_EPSILON)
    mean, whiten = _fit_zca(sample, ZCA_EPSILON)
    whitener = Whitener(PATCH, STRIDE, CONTRAST_EPSILON, mean, whiten, grid)

    pool = _contrast_normalise(
        patches[rng.choice(len(patches), max(BUDGETS), replace=False)],
        CONTRAST_EPSILON,
    )
    pool = (pool - mean) @ whiten

    rows = []
    for budget in BUDGETS:
        dictionary = pool[:budget]
        features = encode(images, dictionary, whitener, POOL_GRID)
        effective = rankme(features)
        ambient = features.shape[1]
        rows.append({
            "atoms": budget,
            "ambient_dimension": ambient,
            "rankme": effective,
            "fraction_of_ambient": effective / ambient,
        })
        print(
            f"atoms {budget:>5}  ambient {ambient:>5}  "
            f"RankMe {effective:10.3f}  frac {effective / ambient:.4f}",
            flush=True,
        )

    growth = []
    for prev, cur in zip(rows, rows[1:]):
        growth.append({
            "from_atoms": prev["atoms"],
            "to_atoms": cur["atoms"],
            "atom_ratio": cur["atoms"] / prev["atoms"],
            "rankme_ratio": cur["rankme"] / prev["rankme"],
        })
        print(
            f"  {prev['atoms']:>5} -> {cur['atoms']:>5}   "
            f"atoms x{cur['atoms'] / prev['atoms']:.2f}   "
            f"RankMe x{cur['rankme'] / prev['rankme']:.3f}",
            flush=True,
        )

    out = {
        "probe": "rankme_effective_rank_vs_atom_budget",
        "instrument": "RankMe, arXiv:2210.02885, used unmodified",
        "arm": "(a) random patches — M103's null",
        "rows_used": ROWS,
        "seed": SEED,
        "note": "scoping evidence, not sealed, not a milestone",
        "curve": rows,
        "growth": growth,
    }
    target = REPO / 'logs' / 'results' / 'v15' / 'rank_probes' / 'probe1_rank_vs_budget.json'
    target.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(f"\nwrote {target}", flush=True)


if __name__ == "__main__":
    main()
