"""Probe 4: intrinsic fingerprints, and whether a router survives its own growth.

Not a milestone. Not sealed. Scoping evidence for a registration decision.

Two questions are being scoped, and they are not independent.

Q1 — what should the router read? Probe 3 measured that a domain's effective
rank varies 5.4x across data types (quickdraw 8.75, real 52.56). That was a
statistic of a POPULATION. If the same quantity can be computed for a SINGLE
IMAGE, from the image alone, with no labels and no training, then it is a
candidate fingerprint that costs one small eigendecomposition per input.

Q2 — must the fingerprint be stable as the system grows? A network of
specialists is built by ADDING specialists. If adding expert K+1 changes the
fingerprints that experts 1..K were selected by, then every addition
invalidates the previous ones and construction is O(K^2) rather than additive.
That would defeat the thesis the whole program rests on. So this probe measures
the drift directly rather than assuming either answer.

The distinction that matters, and which this probe is built to separate:

  * an INTRINSIC fingerprint is a function of the input alone. Adding an expert
    cannot change it. Drift is zero BY CONSTRUCTION, and the probe reports the
    measurement anyway rather than asserting it.
  * a DISCRIMINATIVE router is fitted to separate the experts that exist when
    it is fitted. Adding an expert changes what it was fitted to. The probe
    measures how many previously-correct assignments move.

The instruments:

  per-image spectrum   for image i, take its own whitened patch matrix
                       (grid*grid patches x patch dimension) and compute the
                       eigenvalues of its Gram matrix. All four fingerprint
                       features below are read off that one spectrum.

  RankMe               exp(-sum p log p), p = sigma / ||sigma||_1
                       (Garrido, Balestriero, Najman & LeCun, arXiv:2210.02885),
                       used unmodified, here applied per image rather than per
                       population.

  alpha                power-law decay exponent of the eigenspectrum, the
                       alpha-ReQ instrument (Agrawal et al., NeurIPS 2022),
                       fitted by least squares on log rank vs log eigenvalue.

Both are label-free and training-free. Neither is invented here.

The drift experiment:

  router A   multinomial logistic fitted on domains {0,1,2,3}
  router B   multinomial logistic fitted on domains {0,1,2,3,4,5}

  For held-out images belonging to domains 0-3 only, compare A's argmax with
  B's argmax restricted to {0,1,2,3}. Any disagreement is an assignment that
  moved because the system grew, not because the input changed. This is the
  quantity that decides whether routing composes additively.
"""

from __future__ import annotations

import collections
import io
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
    encode,
)

CACHE = Path(r"D:\geode-ml\data\cache\domainnet\repository\data")
DOMAIN_NAMES = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
PER_DOMAIN = 1000
IMAGE_SIZE = 32
PATCH = 6
STRIDE = 1
POOL_GRID = 2
CONTRAST_EPSILON = 10.0
ZCA_EPSILON = 0.1
WHITENING_PATCHES = 300_000
BUDGET = 512
SEED = 11
SEEN_DOMAINS = [0, 1, 2, 3]


def load_domainnet() -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq
    from PIL import Image

    need = {d: PER_DOMAIN for d in range(len(DOMAIN_NAMES))}
    groups: dict[int, list[tuple[Path, int]]] = collections.defaultdict(list)
    for path in sorted(CACHE.glob("train-*.parquet")):
        source = pq.ParquetFile(path)
        for group in range(source.metadata.num_row_groups):
            if not any(v > 0 for v in need.values()):
                break
            table = source.read_row_group(group, columns=["domain"])
            present = set(table.column("domain").to_pylist())
            if len(present) != 1:
                continue
            domain = int(next(iter(present)))
            if need.get(domain, 0) <= 0:
                continue
            groups[domain].append((path, group))
            need[domain] -= table.num_rows
        if not any(v > 0 for v in need.values()):
            break

    images: list[np.ndarray] = []
    labels: list[int] = []
    for domain, entries in sorted(groups.items()):
        taken = 0
        for path, group in entries:
            if taken >= PER_DOMAIN:
                break
            table = pq.ParquetFile(path).read_row_group(group, columns=["image"])
            for record in table.column("image").to_pylist():
                if taken >= PER_DOMAIN:
                    break
                picture = Image.open(io.BytesIO(record["bytes"]))
                picture = picture.convert("RGB").resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR,
                )
                images.append(np.asarray(picture, dtype=np.uint8))
                labels.append(domain)
                taken += 1
        print(f"  {DOMAIN_NAMES[domain]:<10} {taken}", flush=True)
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def intrinsic_fingerprints(whitened: np.ndarray) -> np.ndarray:
    """Four label-free scalars per image, read off its own patch spectrum.

    whitened is (n_images, patches_per_image, patch_dimension).
    """
    count = len(whitened)
    out = np.empty((count, 4), dtype=np.float64)
    for i in range(count):
        block = whitened[i].astype(np.float64)
        gram = block.T @ block
        eigenvalues = np.linalg.eigvalsh(gram)
        eigenvalues = np.clip(eigenvalues[::-1], 0.0, None)
        singular = np.sqrt(eigenvalues)
        total = singular.sum()
        if total <= 0.0:
            out[i] = 0.0
            continue
        p = singular / total + 1e-7
        rank = float(np.exp(-(p * np.log(p)).sum()))

        keep = eigenvalues[eigenvalues > eigenvalues[0] * 1e-6]
        if len(keep) >= 8:
            ranks = np.arange(1, len(keep) + 1, dtype=np.float64)
            slope = np.polyfit(np.log(ranks), np.log(keep), 1)[0]
        else:
            slope = 0.0

        out[i, 0] = rank
        out[i, 1] = slope
        out[i, 2] = float(np.sqrt(eigenvalues.sum() / len(block)))
        out[i, 3] = float(eigenvalues[:10].sum() / max(eigenvalues.sum(), 1e-12))
    return out


def fit_probe(train_x, train_y, test_x, test_y, classes):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(max_iter=3000, C=1.0)
    model.fit(scaler.transform(train_x), train_y)
    predicted = model.predict(scaler.transform(test_x))
    accuracy = float((predicted == test_y).mean())
    return model, scaler, accuracy


def main() -> None:
    print("loading DomainNet", flush=True)
    images, domains = load_domainnet()
    print(f"images {images.shape}", flush=True)

    rng = np.random.default_rng(SEED)
    patches = _extract_patches(images, PATCH, STRIDE)
    grid = (IMAGE_SIZE - PATCH) // STRIDE + 1
    per_image = grid * grid

    count = min(WHITENING_PATCHES, len(patches))
    sample = _contrast_normalise(
        patches[rng.choice(len(patches), count, replace=False)], CONTRAST_EPSILON,
    )
    mean, whiten = _fit_zca(sample, ZCA_EPSILON)
    whitener = Whitener(PATCH, STRIDE, CONTRAST_EPSILON, mean, whiten, grid)

    print("computing intrinsic fingerprints", flush=True)
    normalised = _contrast_normalise(patches, CONTRAST_EPSILON)
    whitened = ((normalised - mean) @ whiten).reshape(len(images), per_image, -1)
    fingerprints = intrinsic_fingerprints(whitened)
    del normalised, whitened, patches

    print("\nper-domain mean of each intrinsic feature", flush=True)
    print(f"  {'domain':<10} {'RankMe':>9} {'alpha':>9} {'scale':>9} {'top10':>9}",
          flush=True)
    per_domain_means = {}
    for domain, name in enumerate(DOMAIN_NAMES):
        rows = fingerprints[domains == domain]
        per_domain_means[name] = rows.mean(axis=0).tolist()
        print(f"  {name:<10} {rows[:, 0].mean():9.3f} {rows[:, 1].mean():9.3f} "
              f"{rows[:, 2].mean():9.3f} {rows[:, 3].mean():9.3f}", flush=True)

    pooled = encode(images, ((_contrast_normalise(
        _extract_patches(images, PATCH, STRIDE)[
            rng.choice(len(images) * per_image, BUDGET, replace=False)
        ], CONTRAST_EPSILON) - mean) @ whiten), whitener, POOL_GRID)
    print(f"\npooled features {pooled.shape}", flush=True)

    order = np.random.default_rng(SEED).permutation(len(images))
    split = int(0.7 * len(order))
    train_index, test_index = order[:split], order[split:]

    _, _, intrinsic_accuracy = fit_probe(
        fingerprints[train_index], domains[train_index],
        fingerprints[test_index], domains[test_index], len(DOMAIN_NAMES),
    )
    _, _, pooled_accuracy = fit_probe(
        pooled[train_index], domains[train_index],
        pooled[test_index], domains[test_index], len(DOMAIN_NAMES),
    )
    chance = 1.0 / len(DOMAIN_NAMES)
    print(f"\ndomain identification, 6 domains, chance {chance:.4f}", flush=True)
    print(f"  4 intrinsic scalars      {intrinsic_accuracy:.4f}", flush=True)
    print(f"  {pooled.shape[1]} pooled features    {pooled_accuracy:.4f}",
          flush=True)

    print("\n--- drift under growth ---", flush=True)
    seen = np.isin(domains, SEEN_DOMAINS)
    train_seen = train_index[seen[train_index]]
    test_seen = test_index[seen[test_index]]

    drift = {}
    for label, matrix in (("pooled", pooled), ("intrinsic", fingerprints)):
        model_a, scaler_a, _ = fit_probe(
            matrix[train_seen], domains[train_seen],
            matrix[test_seen], domains[test_seen], len(SEEN_DOMAINS),
        )
        model_b, scaler_b, _ = fit_probe(
            matrix[train_index], domains[train_index],
            matrix[test_index], domains[test_index], len(DOMAIN_NAMES),
        )
        assign_a = model_a.predict(scaler_a.transform(matrix[test_seen]))
        scores_b = model_b.decision_function(scaler_b.transform(matrix[test_seen]))
        columns = [list(model_b.classes_).index(c) for c in SEEN_DOMAINS]
        assign_b = np.asarray(SEEN_DOMAINS)[scores_b[:, columns].argmax(axis=1)]
        agreement = float((assign_a == assign_b).mean())
        drift[label] = {
            "assignment_agreement": agreement,
            "assignments_moved": float(1.0 - agreement),
            "rows": int(len(test_seen)),
        }
        print(f"  {label:<10} agreement {agreement:.4f}   "
              f"moved {1.0 - agreement:.4f}", flush=True)

    print("\n  representation stability under growth:", flush=True)
    print("    intrinsic fingerprint values are a function of the image alone,", flush=True)
    print("    so the FEATURES cannot move when a domain is added; only the", flush=True)
    print("    small decision rule on top of them is refitted.", flush=True)

    out = {
        "probe": "intrinsic_fingerprint_and_router_drift",
        "note": "scoping evidence, not sealed, not a milestone",
        "instruments": [
            "RankMe per image, arXiv:2210.02885, unmodified",
            "alpha-ReQ spectral decay exponent, NeurIPS 2022",
        ],
        "corpus": "DomainNet, 32x32, 1000 images per domain",
        "atoms": BUDGET,
        "seed": SEED,
        "domain_identification": {
            "chance": chance,
            "intrinsic_4_scalars": intrinsic_accuracy,
            "pooled_features": pooled_accuracy,
            "pooled_dimension": int(pooled.shape[1]),
        },
        "per_domain_intrinsic_means": per_domain_means,
        "seen_domains": SEEN_DOMAINS,
        "drift_under_growth": drift,
    }
    target = REPO / 'logs' / 'results' / 'v15' / 'rank_probes' / 'probe4_intrinsic_fingerprint.json'
    target.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(f"\nwrote {target}", flush=True)


if __name__ == "__main__":
    main()
