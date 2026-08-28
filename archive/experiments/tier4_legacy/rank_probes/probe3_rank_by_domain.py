"""Probe 3: does DOMAIN specialisation buy more rank reduction than CLASS?

Not a milestone. Not sealed. Scoping evidence for a registration decision.

Probe 2 measured that on CIFAR-10 a single class needs 0.9107 of the effective
rank of a random subset of the same size. Class specialisation buys about 9%,
which is a real effect (the row-matched controls agree to within 0.63 while the
gap is 6.3) but a small one. A network of ten class-specialists that each need
91% of the generalist's directions is not obviously cheaper than one generalist.

But CIFAR-10 classes are all 32x32 natural photographs. They are semantically
different and visually similar. The proposal being scoped is a network of
specialists split by TASK and DATA TYPE, not by class, and v14's M90.2 measured
that in a frozen backbone domain dominates class geometrically: a domain probe
reads 0.8946 and domain structure survives complete linear erasure.

So this probe repeats probe 2 on DomainNet, splitting by DOMAIN instead of by
class. DomainNet's six domains are clipart, infograph, painting, quickdraw,
real and sketch — quickdraw is line art and real is photographs, so these are
genuinely different data types rather than different subjects.

Identical instrument, identical pipeline, identical control design as probe 2:
one encoding shared by every subset, and a row-matched random control that
ignores domain. If domain subsets drop no further below their control than
class subsets did below theirs, then splitting by data type buys no more than
splitting by class, and the specialisation argument does not improve by
choosing a better axis.

Images are resized to 32x32 so the patch pipeline, the whitener and the atom
count are the ones probes 1 and 2 used. That makes the ratio comparable across
probes, which is the only quantity being compared.
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
WHITENING_PATCHES = 400_000
BUDGET = 512
SEED = 11


def rankme(features: np.ndarray, epsilon: float = 1e-7) -> float:
    singular = np.linalg.svd(features, compute_uv=False)
    total = np.abs(singular).sum()
    if total <= 0.0:
        return 0.0
    p = np.abs(singular) / total + epsilon
    return float(np.exp(-(p * np.log(p)).sum()))


def load_domainnet() -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq
    from PIL import Image

    needed = {d: PER_DOMAIN for d in range(len(DOMAIN_NAMES))}
    chosen: dict[int, list[tuple[Path, int]]] = collections.defaultdict(list)
    for path in sorted(CACHE.glob("train-*.parquet")):
        source = pq.ParquetFile(path)
        for group in range(source.metadata.num_row_groups):
            if not any(v > 0 for v in needed.values()):
                break
            table = source.read_row_group(group, columns=["domain"])
            domains = set(table.column("domain").to_pylist())
            if len(domains) != 1:
                continue
            domain = int(next(iter(domains)))
            if needed.get(domain, 0) <= 0:
                continue
            chosen[domain].append((path, group))
            needed[domain] -= table.num_rows
        if not any(v > 0 for v in needed.values()):
            break

    images: list[np.ndarray] = []
    labels: list[int] = []
    for domain, groups in sorted(chosen.items()):
        taken = 0
        for path, group in groups:
            if taken >= PER_DOMAIN:
                break
            source = pq.ParquetFile(path)
            table = source.read_row_group(group, columns=["image"])
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
        print(f"  {DOMAIN_NAMES[domain]:<10} {taken} images", flush=True)
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def main() -> None:
    print("loading DomainNet", flush=True)
    images, domains = load_domainnet()
    print(f"images {images.shape}", flush=True)

    rng = np.random.default_rng(SEED)
    patches = _extract_patches(images, PATCH, STRIDE)
    grid = (IMAGE_SIZE - PATCH) // STRIDE + 1
    count = min(WHITENING_PATCHES, len(patches))
    sample = patches[rng.choice(len(patches), count, replace=False)]
    sample = _contrast_normalise(sample, CONTRAST_EPSILON)
    mean, whiten = _fit_zca(sample, ZCA_EPSILON)
    whitener = Whitener(PATCH, STRIDE, CONTRAST_EPSILON, mean, whiten, grid)

    pool = _contrast_normalise(
        patches[rng.choice(len(patches), BUDGET, replace=False)],
        CONTRAST_EPSILON,
    )
    dictionary = (pool - mean) @ whiten

    features = encode(images, dictionary, whitener, POOL_GRID)
    print(f"features {features.shape}", flush=True)

    whole = rankme(features)
    print(f"\nwhole corpus   rows {len(features):>6}  RankMe {whole:9.3f}",
          flush=True)

    per_domain = []
    for domain, name in enumerate(DOMAIN_NAMES):
        subset = features[domains == domain]
        if len(subset) == 0:
            continue
        value = rankme(subset)
        per_domain.append({
            "domain": name,
            "rows": int(len(subset)),
            "rankme": value,
            "ratio_to_whole": value / whole,
        })
        print(f"  {name:<10} rows {len(subset):>6}  RankMe {value:9.3f}   "
              f"x{value / whole:.3f} of whole", flush=True)

    control_rows = int(np.median([r["rows"] for r in per_domain]))
    controls = []
    for trial in range(3):
        index = np.random.default_rng(9000 + trial).choice(
            len(features), control_rows, replace=False,
        )
        value = rankme(features[index])
        controls.append(value)
        print(f"  row-matched control {trial}  rows {control_rows:>6}  "
              f"RankMe {value:9.3f}", flush=True)

    mean_domain = float(np.mean([r["rankme"] for r in per_domain]))
    mean_control = float(np.mean(controls))
    ratio = mean_domain / mean_control
    print(f"\nmean over domains          {mean_domain:9.3f}", flush=True)
    print(f"mean row-matched control   {mean_control:9.3f}", flush=True)
    print(f"specialisation ratio       {ratio:9.4f}", flush=True)
    print(f"  CIFAR-10 class ratio was    0.9107 (probe 2)", flush=True)

    out = {
        "probe": "rankme_effective_rank_by_domain",
        "instrument": "RankMe, arXiv:2210.02885, used unmodified",
        "note": "scoping evidence, not sealed, not a milestone",
        "corpus": "DomainNet, resized to 32x32 to match probes 1 and 2",
        "atoms": BUDGET,
        "per_domain_rows": PER_DOMAIN,
        "seed": SEED,
        "whole_corpus_rankme": whole,
        "per_domain": per_domain,
        "row_matched_controls": controls,
        "control_rows": control_rows,
        "mean_domain_rankme": mean_domain,
        "mean_control_rankme": mean_control,
        "specialisation_ratio": ratio,
        "cifar10_class_specialisation_ratio": 0.9107,
    }
    target = REPO / 'logs' / 'results' / 'v15' / 'rank_probes' / 'probe3_rank_by_domain.json'
    target.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(f"\nwrote {target}", flush=True)


if __name__ == "__main__":
    main()
