"""M104 — does sizing an expert to its sub-population's effective rank beat
sizing it uniformly?

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v15.md`` section 7.10.

Section 2.9.7 probe 3 measured a **6.32x spread** in effective rank across
DomainNet's six data types. Every mixture of experts in section 8.10 — Switch,
Mixtral, DeepSeekMoE — gives every expert the same capacity; DeepSeekMoE's
contribution was making experts *finer*, and they stayed uniform. A uniform
allocation therefore overspends on quickdraw by roughly 5x and starves the four
domains that sit above the mixed control. M104 asks whether, at matched total
inference MACs, allocating capacity in proportion to measured effective rank
beats allocating it equally.

Registration notes carried by this runner:

* **N104.1 — only the SIZE varies.** Every expert in every arm holds a prefix of
  the same seeded permutation of one shared candidate pool, whitened by one
  shared ZCA basis fitted on train patches only. Two experts of the same size in
  different arms hold the *same atoms*. The operand is therefore the allocation
  rule and nothing else.
* **N104.2 — the MAC match is the row-weighted atom sum.** Under oracle routing
  an image is encoded by exactly one expert, so matching total inference MACs
  reduces exactly to matching ``sum_e f_e * A_e``. That is *not* the plain atom
  sum, because DomainNet's domains differ in size by 3.6x. Section 7.10 design
  item 1 says "same total atoms" and design item 2 says "matched on inference
  MACs"; item 2 governs, and the realised plain atom sum is reported beside it.
* **N104.3 — arm (e) was added before measurement.** Reading the corpus revealed
  a confound section 7.10 does not control: under a MAC match a rule can win by
  moving capacity off high-traffic domains, where it is expensive, and onto
  low-traffic ones, where it is cheap. Arm (e) is the allocation that maximises
  that confound while carrying no rank information. Adding a *harder* null
  before measurement tightens the test; it is recorded here so the record shows
  the design was tightened rather than tuned.
* **N104.4 — both generalists are run.** Design item 1(c)'s "same total atoms"
  and design item 2's MAC match disagree for a generalist by a factor of six.
  Both are run, because kill switch 3 is decided differently by each and
  choosing between them after seeing the result would be choosing the answer.
* **N104.5 — the sample floor binds per expert, on that expert's own rows.**
  Section 7.10 design item 6. The per-class reading of section 5.3 is computed
  and reported beside it; see the config's ``sample_adequacy`` note for why the
  per-fitted-dimension reading is the correct one for a single multi-output
  linear head, and why the stricter reading bounds every arm equally.
* **N104.6 — oracle routing is an upper bound.** Section 7.10 design item 3 and
  restriction 4. No figure here is a system result; M105 measures the system.
* **N104.7 — no novelty.** Mixtures of experts, conditional computation,
  effective rank and spectral sizing are all established (section 8.10). The
  contribution is a comparison of allocation rules. RankMe is Garrido et al.'s
  and is used unmodified (prohibition 25).
* **N104.8 — corpus isolation.** Section 7.10 restriction 1 and prohibition 24:
  no M104 figure may be compared to any M103 or section 2.9.3 CIFAR-10 figure in
  either direction.

Reproduce with::

    .\\.venv\\Scripts\\python.exe -m experiments.tier4.eval_v15_m104_experts
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.common.data_cache import configure_external_cache_environment
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m103_atoms import (
    Whitener,
    _contrast_normalise,
    _extract_patches,
    _fit_zca,
    _pool,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v15" / "m104_experts.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v15" / "m104_experts"

# The distance matrix is (patch rows x atoms) and is the run's memory ceiling.
DISTANCE_BYTES_TARGET = 1_200_000_000


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------
def _cache_root() -> Path:
    import os

    return Path(os.environ.get("GEODE_CACHE_DIR", r"D:\geode-ml\data\cache"))


def _decode_split(split: str, size: int) -> dict[str, np.ndarray]:
    """Decode one DomainNet split to (n, size, size, 3) uint8, in file order."""
    import pyarrow.parquet as pq
    from PIL import Image

    source_dir = _cache_root() / "domainnet" / "repository" / "data"
    files = sorted(source_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet under {source_dir}")

    images: list[np.ndarray] = []
    labels: list[int] = []
    domains: list[int] = []
    for path in files:
        handle = pq.ParquetFile(path)
        for group in range(handle.metadata.num_row_groups):
            table = handle.read_row_group(group, columns=["image", "label", "domain"])
            blobs = table.column("image").to_pylist()
            labels.extend(table.column("label").to_pylist())
            domains.extend(table.column("domain").to_pylist())
            for record in blobs:
                picture = Image.open(io.BytesIO(record["bytes"]))
                picture = picture.convert("RGB").resize(
                    (size, size), Image.BILINEAR
                )
                images.append(np.asarray(picture, dtype=np.uint8))
        print(f"    {path.name}: {len(images)} images", flush=True)
    return {
        "images": np.stack(images),
        "labels": np.asarray(labels, dtype=np.int64),
        "domains": np.asarray(domains, dtype=np.int64),
    }


def _load_domainnet(size: int) -> dict[str, np.ndarray]:
    """Decode once, then reuse. The decode is ~35 minutes; the run is not.

    The cache lives under the external cache directory, never in the repository:
    it is 1.8 GB of derived data and re-deriving it is deterministic.
    """
    cache = _cache_root() / "domainnet_decoded" / f"size{size}.npz"
    if cache.exists():
        print(f"  reusing decoded corpus {cache}", flush=True)
        with np.load(cache) as stored:
            return {k: stored[k] for k in stored.files}

    print("  decoding DomainNet (first run only)", flush=True)
    packed: dict[str, np.ndarray] = {}
    for split in ("train", "test"):
        print(f"  {split}:", flush=True)
        part = _decode_split(split, size)
        for key, value in part.items():
            packed[f"{split}_{key}"] = value
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **packed)
    print(f"  wrote {cache}", flush=True)
    return packed


def _stratified_subsample(domains: np.ndarray, rows: int | None, seed: int
                          ) -> np.ndarray:
    """Indices of a domain-stratified subsample at the natural proportions.

    Returns ``arange(n)`` untouched when ``rows`` is None or already covers the
    split, so the full-split path is the identity and cannot reorder anything.
    """
    total = len(domains)
    if rows is None or rows >= total:
        return np.arange(total)
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for domain in np.unique(domains):
        member = np.flatnonzero(domains == domain)
        take = int(round(rows * len(member) / total))
        take = max(1, min(take, len(member)))
        keep.append(rng.choice(member, size=take, replace=False))
    return np.sort(np.concatenate(keep))


def _corpus_digest(corpus: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(corpus):
        digest.update(key.encode("utf-8"))
        digest.update(str(corpus[key].shape).encode("utf-8"))
        digest.update(corpus[key].tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------
# instrument
# --------------------------------------------------------------------------
def rankme(features: np.ndarray, epsilon: float) -> float:
    """RankMe (arXiv:2210.02885), used unmodified. Prohibition 25."""
    singular = np.linalg.svd(features, compute_uv=False)
    total = np.abs(singular).sum()
    if total <= 0.0:
        return 0.0
    p = np.abs(singular) / total + epsilon
    return float(np.exp(-(p * np.log(p)).sum()))


def _encode_block(images: np.ndarray, table: torch.Tensor, whitener: Whitener,
                  pool_grid: int) -> np.ndarray:
    white = torch.from_numpy(np.ascontiguousarray(whitener(images))).to(torch.float32)
    with torch.no_grad():
        distances = torch.cdist(white, table)
        activation = torch.clamp(
            distances.mean(dim=1, keepdim=True) - distances, min=0.0
        )
        pooled = _pool(activation, len(images), whitener.grid, pool_grid)
    return pooled.to(torch.float32).numpy()


def _chunk_rows(atoms: int, grid: int, total: int) -> int:
    per_image = grid * grid
    return max(1, min(total, int(DISTANCE_BYTES_TARGET / (4 * atoms * per_image))))


def encode_all(images: np.ndarray, dictionary: np.ndarray, whitener: Whitener,
               pool_grid: int) -> np.ndarray:
    atoms = len(dictionary)
    table = torch.from_numpy(np.ascontiguousarray(dictionary)).to(torch.float32)
    step = _chunk_rows(atoms, whitener.grid, len(images))
    out = np.empty((len(images), pool_grid * pool_grid * atoms), dtype=np.float32)
    for start in range(0, len(images), step):
        block = images[start:start + step]
        out[start:start + step] = _encode_block(block, table, whitener, pool_grid)
    return out


# --------------------------------------------------------------------------
# allocation
# --------------------------------------------------------------------------
def allocate(weights: np.ndarray, shares: np.ndarray, budget: int,
             caps: np.ndarray, minimum: int) -> dict[str, Any]:
    """Sizes proportional to ``weights`` matching ``sum_e shares_e * A_e``.

    The target is arm (a)'s row-weighted total, which for a uniform allocation
    is exactly ``budget`` because the shares sum to one. Experts that would
    exceed their section 5.3 cap are pinned at it and the freed budget is
    redistributed among the rest, which is why this is a loop rather than one
    division. The identical procedure runs for every arm, so the capping cannot
    favour one allocation rule over another.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights <= 0):
        raise ValueError("allocation weights must be positive")
    target = float(budget)
    pinned = np.zeros(len(weights), dtype=bool)
    sizes = np.zeros(len(weights), dtype=np.int64)
    capped_at: list[str] = []
    for _ in range(len(weights) + 1):
        free = ~pinned
        denominator = float(np.sum(shares[free] * weights[free]))
        if denominator <= 0:
            break
        scale = (target - float(np.sum(shares[pinned] * sizes[pinned]))) / denominator
        proposal = np.rint(scale * weights).astype(np.int64)
        proposal = np.clip(proposal, minimum, None)
        sizes[free] = proposal[free]
        over = free & (sizes > caps)
        if not np.any(over):
            break
        sizes[over] = caps[over]
        pinned |= over
        capped_at.extend(np.flatnonzero(over).tolist())
    sizes = np.clip(sizes, minimum, caps)
    return {
        "atoms": sizes.tolist(),
        "row_weighted_atoms": float(np.sum(shares * sizes)),
        "plain_atom_sum": int(sizes.sum()),
        "capped_experts": sorted(set(capped_at)),
    }


def _weights_for(arm: str, config: dict[str, Any], ranks: np.ndarray,
                 shares: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    kind = config["arms"][arm]["weights"]
    if kind == "uniform":
        return np.ones(len(shares))
    if kind == "rankme_train_only":
        return ranks.copy()
    if kind == "dirichlet":
        concentration = config["arms"][arm]["dirichlet_concentration"]
        return rng.dirichlet(np.full(len(shares), concentration))
    if kind == "inverse_traffic":
        return 1.0 / shares
    raise ValueError(f"unknown weight rule {kind!r}")


# --------------------------------------------------------------------------
# head
# --------------------------------------------------------------------------
class RidgeAccumulator:
    """Streaming normal equations for a multi-output ridge.

    The design matrix is never materialised. At arm (c2)'s 3,072 atoms it would
    be 12,289 columns over 409,832 rows, which is 20 GB in float32; the Gram it
    contributes to is 1.2 GB in float64 and is what the solve actually needs.

    The accumulation is over the **raw** features, and the standardisation is
    recovered from the raw Gram afterwards rather than measured in a separate
    pass: ``sum_j F[j,i]`` and ``sum_j F[j,i]^2`` are already the last column
    and the diagonal of that Gram. One pass over the rows therefore produces
    the centring, the scaling and the normal equations together, which matters
    because the pass is an encode and the encode is the run's whole cost.
    """

    def __init__(self, width: int, classes: int):
        self.width = width
        self.classes = classes
        self.gram = np.zeros((width, width), dtype=np.float64)
        self.column_sum = np.zeros(width, dtype=np.float64)
        self.cross = np.zeros((width, classes), dtype=np.float64)
        self.class_count = np.zeros(classes, dtype=np.float64)
        self.rows = 0

    def add(self, features: np.ndarray, labels: np.ndarray) -> None:
        block = np.asarray(features, dtype=np.float64)
        targets = np.zeros((len(block), self.classes), dtype=np.float64)
        targets[np.arange(len(block)), labels] = 1.0
        self.gram += block.T @ block
        self.column_sum += block.sum(axis=0)
        self.cross += block.T @ targets
        self.class_count += targets.sum(axis=0)
        self.rows += len(block)

    def standardiser(self) -> "Standardiser":
        centre = self.column_sum / self.rows
        variance = np.diag(self.gram) / self.rows - np.square(centre)
        scale = np.sqrt(np.maximum(variance, 0.0)) + 1e-8
        return Standardiser(centre.astype(np.float32), scale.astype(np.float32))

    def _standardised_system(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normal equations for the centred, scaled design, in closed form.

        Centring makes the design orthogonal to the intercept column, so the
        intercept separates out exactly and is simply the class prior. Nothing
        here is an approximation: it is the same system a second standardised
        pass would have accumulated.
        """
        standardiser = self.standardiser()
        centre = standardiser.centre.astype(np.float64)
        inverse = 1.0 / standardiser.scale.astype(np.float64)
        centred = self.gram - np.outer(self.column_sum, centre)
        centred *= inverse[:, None]
        centred *= inverse[None, :]
        cross = (self.cross - np.outer(centre, self.class_count)) * inverse[:, None]
        intercept = self.class_count / self.rows
        return centred, cross, intercept

    def solve(self, penalty: float) -> np.ndarray:
        return self.solve_many([penalty])[penalty]

    def solve_many(self, penalties: list[float]) -> dict[float, np.ndarray]:
        """One standardised system, one solve per penalty.

        Building the centred, scaled system costs two full ``width x width``
        passes, which at arm (c2)'s 12,288 columns is 1.2 GB of float64 traffic.
        Doing it once for the whole grid rather than once per penalty is what
        makes a five-constant sweep cost five solves instead of five rebuilds
        and five solves.
        """
        centred, cross, intercept = self._standardised_system()
        out: dict[float, np.ndarray] = {}
        for penalty in penalties:
            centred.flat[:: self.width + 1] += penalty
            weights = np.linalg.solve(centred, cross)
            centred.flat[:: self.width + 1] -= penalty
            out[penalty] = np.vstack([weights, intercept[None, :]])
        return out


def _score(weights: np.ndarray, features: np.ndarray, labels: np.ndarray
           ) -> np.ndarray:
    scores = features @ weights[:-1] + weights[-1]
    return np.asarray(np.argmax(scores, axis=1) == labels)


class Standardiser:
    def __init__(self, centre: np.ndarray, scale: np.ndarray):
        self.centre = centre
        self.scale = scale

    def __call__(self, features: np.ndarray) -> np.ndarray:
        return (features - self.centre) / self.scale


# --------------------------------------------------------------------------
# compute ledger
# --------------------------------------------------------------------------
def _inference_macs(atoms: int, grid: int, dimension: int, pool_grid: int,
                    classes: int) -> dict[str, int]:
    patches = grid * grid
    whitening = patches * dimension * dimension
    encoding = patches * atoms * dimension
    head = pool_grid * pool_grid * atoms * classes
    return {
        "whitening": int(whitening),
        "encoding": int(encoding),
        "head": int(head),
        "total": int(whitening + encoding + head),
        "_excluded": (
            "patch extraction and per-patch contrast normalisation are NOT "
            "counted, exactly as section 2.9.3 limitation (iv) discloses"
        ),
    }


def _training_macs(rows: int, atoms: int, grid: int, dimension: int,
                   pool_grid: int, classes: int) -> int:
    """Encode of the train rows plus the Gram and cross-product accumulation."""
    width = pool_grid * pool_grid * atoms + 1
    encode = rows * grid * grid * (dimension * dimension + atoms * dimension)
    gram = rows * width * width
    cross = rows * width * classes
    solve = width ** 3 // 3
    return int(encode + gram + cross + solve)


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def _fit_expert(train_images: np.ndarray, train_labels: np.ndarray,
                test_images: np.ndarray, test_labels: np.ndarray,
                dictionary: np.ndarray, whitener: Whitener, pool_grid: int,
                classes: int, penalties: list[float],
                validation_rows: int) -> dict[str, Any]:
    """Encode, accumulate, solve, score. One expert, one budget, one pass.

    ``penalties`` holds every constant to solve for. The Gram is accumulated
    once and reused for all of them, so the selection sweep costs one encode
    rather than five, and the rows are encoded exactly once for the fit, once
    for validation and once for test.

    Two models come out of a single accumulator. The **selection** model is
    solved from the first ``fit_rows`` rows and is what the held-out rows
    score, so the constant is never chosen on rows the selection model saw.
    The held-out rows are then added to the same accumulator and the **final**
    model is solved from every row the expert owns. That costs no extra encode
    -- the held-out rows are already in memory, standardised, at that point --
    and it is what makes the section 5.3 sample floor exactly enforceable: the
    atom cap is computed from an expert's full row count, so a model fitted on
    90% of them could be capped and still fall below the floor. The smoke run
    showed exactly that, voiding two of arm (b)'s six experts while every cap
    was respected.

    ``train_images`` must already be in the run's fixed shuffled order, so the
    validation rows carved off the end are a random sample of the expert's rows
    rather than whatever the corpus file order happened to put last.
    """
    atoms = len(dictionary)
    width = pool_grid * pool_grid * atoms
    table = torch.from_numpy(np.ascontiguousarray(dictionary)).to(torch.float32)
    step = _chunk_rows(atoms, whitener.grid, len(train_images))
    selection_rows = len(train_images) - validation_rows

    accumulator = RidgeAccumulator(width, classes)
    for start in range(0, selection_rows, step):
        stop = min(start + step, selection_rows)
        accumulator.add(
            _encode_block(train_images[start:stop], table, whitener, pool_grid),
            train_labels[start:stop],
        )
    selection_standardise = accumulator.standardiser()
    selection = accumulator.solve_many(penalties)

    validation: dict[str, float] = {}
    if validation_rows > 0:
        hits = {penalty: 0 for penalty in penalties}
        for start in range(selection_rows, len(train_images), step):
            stop = min(start + step, len(train_images))
            raw = _encode_block(train_images[start:stop], table, whitener,
                                pool_grid)
            block = selection_standardise(raw)
            for penalty in penalties:
                hits[penalty] += int(
                    _score(selection[penalty], block, train_labels[start:stop]).sum()
                )
            accumulator.add(raw, train_labels[start:stop])
        validation = {
            str(penalty): hits[penalty] / validation_rows for penalty in penalties
        }

    standardise = accumulator.standardiser()
    solutions = accumulator.solve_many(penalties)
    fit_rows = accumulator.rows

    correct = {penalty: 0 for penalty in penalties}
    test_step = _chunk_rows(atoms, whitener.grid, max(1, len(test_images)))
    for start in range(0, len(test_images), test_step):
        stop = min(start + test_step, len(test_images))
        block = standardise(
            _encode_block(test_images[start:stop], table, whitener, pool_grid)
        )
        for penalty in penalties:
            correct[penalty] += int(
                _score(solutions[penalty], block, test_labels[start:stop]).sum()
            )
    return {
        "atoms": atoms,
        "features": width,
        "fit_rows": int(fit_rows),
        "selection_fit_rows": int(selection_rows),
        "validation_rows": int(validation_rows),
        "test_rows": int(len(test_images)),
        "validation_accuracy_by_penalty": validation,
        "correct_by_penalty": {str(k): v for k, v in correct.items()},
    }


def _domain_slices(domains: np.ndarray, count: int, seed: int
                   ) -> list[np.ndarray]:
    """Row indices per domain, shuffled once under a fixed seed.

    The shuffle is load-bearing and is fixed rather than per-seed. The corpus
    arrives in parquet file order, whose row groups are single-domain and are
    not guaranteed to be class-shuffled within a domain, so carving the
    validation split off the end of the natural order could hand an expert a
    validation set drawn from a few classes. Fixing the permutation means every
    arm and every seed sees the identical train/validation split and the only
    thing the run's seeds vary is the dictionary draw, which is what they are
    for.
    """
    rng = np.random.default_rng(seed)
    return [rng.permutation(np.flatnonzero(domains == d)) for d in range(count)]


def _expert_order(seed: int, expert: int, pool_size: int) -> np.ndarray:
    """A nested atom order for one expert.

    Each expert draws its own permutation of the shared pool, so a mixture is a
    mixture of genuinely different dictionaries rather than six copies of one.
    Because an expert always takes a PREFIX of its own order, its 773-atom
    dictionary in arm (b) contains its 512-atom dictionary in arm (a) exactly.
    Between arms an expert therefore differs in how many atoms it has and in
    nothing else, which is the whole of what M104 measures.
    """
    return np.random.default_rng([seed, expert]).permutation(pool_size)


def run_m104(config_path: Path, output_dir: Path, progress: bool = True
             ) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    torch.set_num_threads(config["numerics"]["torch_threads"])
    configure_external_cache_environment()

    corpus_config = config["corpus"]
    names = config["domains"]
    representation = config["representation"]
    pool_grid = representation["pool_grid"]
    patch = representation["patch"]
    stride = representation["stride"]
    size = corpus_config["image_size"]

    print("loading DomainNet", flush=True)
    raw = _load_domainnet(size)
    train_index = _stratified_subsample(
        raw["train_domains"], corpus_config["train_rows"],
        corpus_config["stratify_seed"],
    )
    test_index = _stratified_subsample(
        raw["test_domains"], corpus_config["test_rows"],
        corpus_config["stratify_seed"],
    )
    corpus = {
        "train_images": raw["train_images"][train_index],
        "train_labels": raw["train_labels"][train_index],
        "train_domains": raw["train_domains"][train_index],
        "test_images": raw["test_images"][test_index],
        "test_labels": raw["test_labels"][test_index],
        "test_domains": raw["test_domains"][test_index],
    }
    del raw
    digest = _corpus_digest(corpus)
    expected = corpus_config.get("expected_sha256")
    if expected and expected != digest:
        raise RuntimeError(
            f"M104 corpus digest {digest} does not match the pinned "
            f"{expected}; the corpus changed and no figure from it is "
            "comparable to a previously sealed one."
        )
    classes = int(corpus["train_labels"].max()) + 1
    domain_count = len(names)
    shuffle_seed = corpus_config["stratify_seed"]
    train_by_domain = _domain_slices(corpus["train_domains"], domain_count,
                                     shuffle_seed)
    test_by_domain = _domain_slices(corpus["test_domains"], domain_count,
                                    shuffle_seed)
    all_train = np.random.default_rng(shuffle_seed).permutation(
        len(corpus["train_images"])
    )
    all_test = np.arange(len(corpus["test_images"]))
    train_counts = np.array([len(i) for i in train_by_domain], dtype=np.int64)
    test_counts = np.array([len(i) for i in test_by_domain], dtype=np.int64)
    train_shares = train_counts / train_counts.sum()
    test_shares = test_counts / test_counts.sum()
    print(
        f"  train {len(train_index)}  test {len(test_index)}  classes {classes}",
        flush=True,
    )
    for d, name in enumerate(names):
        print(f"    {name:<10} train {train_counts[d]:>7}  test {test_counts[d]:>7}"
              f"  share {train_shares[d]:.4f}", flush=True)

    print("fitting the shared whitener on TRAIN patches only", flush=True)
    rng = np.random.default_rng(representation["zca_fit_seed"])
    whiten_rows = min(len(corpus["train_images"]), 20_000)
    sample_images = corpus["train_images"][
        rng.choice(len(corpus["train_images"]), whiten_rows, replace=False)
    ]
    patches = _extract_patches(sample_images, patch, stride)
    grid = (size - patch) // stride + 1
    take = min(representation["zca_fit_patches"], len(patches))
    sample = _contrast_normalise(
        patches[rng.choice(len(patches), take, replace=False)],
        representation["contrast_epsilon"],
    )
    mean, whiten = _fit_zca(sample, representation["zca_epsilon"])
    whitener = Whitener(patch, stride, representation["contrast_epsilon"],
                        mean, whiten, grid)
    dimension = patch * patch * 3
    patch_pool = sample
    del patches, sample_images

    determinism = bool(
        np.array_equal(
            whitener(corpus["train_images"][:64]),
            whitener(corpus["train_images"][:64]),
        )
    )
    if not determinism:
        raise RuntimeError(
            "M104 instrument failure: the whitener is not bitwise repeatable "
            "within a single process, so no figure from it is reproducible."
        )

    floor = config["sample_adequacy"]["fit_samples_per_fitted_dimension_floor"]
    caps = (train_counts // (pool_grid * pool_grid * floor)).astype(np.int64)
    minimum = config["allocation"]["minimum_atoms_per_expert"]
    rank_config = config["rank_measurement"]
    head_config = config["head"]
    penalties = [float(p) for p in head_config["regularisation_grid"]]
    pool_size = config["candidate_pool"]["size"]

    output_dir.mkdir(parents=True, exist_ok=True)
    seeds_payload: list[dict[str, Any]] = []
    chosen_penalty: dict[int, dict[str, Any]] = {}

    for seed in config["seeds"]:
        seed_rng = np.random.default_rng(seed)
        candidates = ((patch_pool[
            seed_rng.choice(len(patch_pool), pool_size, replace=False)
        ] - mean) @ whiten).astype(np.float32)
        # One nested atom order per expert, plus two more for the generalists,
        # so no two experts hold the same dictionary and every expert's larger
        # dictionary contains its smaller one.
        orders = [
            _expert_order(seed, e, pool_size) for e in range(domain_count + 2)
        ]

        # ---- effective rank, TRAIN ROWS ONLY, equal rows per domain --------
        started = time.time()
        probe_atoms = rank_config["probe_atoms"]
        probe_rows = rank_config["rows_per_domain"]
        probe_dictionary = candidates[orders[0][:probe_atoms]]
        ranks = np.zeros(domain_count, dtype=np.float64)
        rank_rows = np.zeros(domain_count, dtype=np.int64)
        rank_macs = 0
        for d in range(domain_count):
            member = train_by_domain[d]
            take_rows = min(probe_rows, len(member))
            picked = np.sort(member[:take_rows])
            features = encode_all(
                corpus["train_images"][picked], probe_dictionary,
                whitener, pool_grid,
            )
            ranks[d] = rankme(features, rank_config["epsilon"])
            rank_rows[d] = take_rows
            rank_macs += take_rows * grid * grid * (
                dimension * dimension + probe_atoms * dimension
            )
            if progress:
                print(f"  seed {seed}  rank {names[d]:<10} rows {take_rows:>5}  "
                      f"RankMe {ranks[d]:9.3f}", flush=True)
        rank_seconds = time.time() - started
        rank_reference = rank_config["external_reference"]["ratio_to_control"]
        rank_order_agrees = bool(
            np.array_equal(
                np.argsort(ranks),
                np.argsort([rank_reference[n] for n in names]),
            )
        )

        # ---- allocations ---------------------------------------------------
        allocation_rng = np.random.default_rng(1000 + seed)
        budgets_payload: list[dict[str, Any]] = []
        for budget in config["budgets"]:
            allocations: dict[str, dict[str, Any]] = {}
            for arm in config["arms"]:
                spec = config["arms"][arm]
                if spec["weights"] == "single_expert":
                    atoms = (budget if arm.startswith("c1")
                             else budget * domain_count)
                    allocations[arm] = {
                        "atoms": [atoms],
                        "row_weighted_atoms": float(atoms),
                        "plain_atom_sum": atoms,
                        "capped_experts": [],
                        "experts": ["all"],
                    }
                    continue
                weights = _weights_for(arm, config, ranks, train_shares,
                                       allocation_rng)
                allocated = allocate(weights, train_shares, budget, caps, minimum)
                allocated["experts"] = list(names)
                allocated["weights"] = [float(w) for w in weights]
                allocations[arm] = allocated

            # ---- head constant, chosen once on the NULL arm, first seed ----
            selection_arm = head_config["selection_arm"]
            arms_payload: list[dict[str, Any]] = []
            for arm in config["arms"]:
                allocated = allocations[arm]
                single = config["arms"][arm]["weights"] == "single_expert"
                started = time.time()
                experts: list[dict[str, Any]] = []
                train_macs = 0
                for position, atoms in enumerate(allocated["atoms"]):
                    if single:
                        rows_in = all_train
                        rows_out = all_test
                        label = "all"
                        order = orders[domain_count + (0 if arm.startswith("c1")
                                                       else 1)]
                    else:
                        rows_in = train_by_domain[position]
                        rows_out = test_by_domain[position]
                        label = names[position]
                        order = orders[position]
                    validation = int(round(
                        head_config["validation_fraction"] * len(rows_in)
                    ))
                    dictionary = candidates[order[:atoms]]
                    record = _fit_expert(
                        corpus["train_images"][rows_in],
                        corpus["train_labels"][rows_in],
                        corpus["test_images"][rows_out],
                        corpus["test_labels"][rows_out],
                        dictionary, whitener, pool_grid, classes,
                        penalties, validation,
                    )
                    features = pool_grid * pool_grid * atoms
                    record["expert"] = label
                    record["rows_per_fitted_dimension"] = (
                        record["fit_rows"] / features
                    )
                    record["rows_per_fitted_dimension_per_class"] = (
                        record["fit_rows"] / (features * classes)
                    )
                    record["sample_adequate"] = bool(
                        record["rows_per_fitted_dimension"] >= floor
                    )
                    record["inference_macs_per_image"] = _inference_macs(
                        atoms, grid, dimension, pool_grid, classes
                    )
                    train_macs += _training_macs(
                        record["fit_rows"], atoms, grid, dimension, pool_grid,
                        classes,
                    )
                    experts.append(record)
                    if progress:
                        print(f"  seed {seed}  {budget:>5}  {arm:<28} "
                              f"{label:<10} atoms {atoms:>5}  "
                              f"rows/dim {record['rows_per_fitted_dimension']:7.2f}"
                              f"{'' if record['sample_adequate'] else '  [VOID]'}",
                              flush=True)
                if arm == "b_rank_sized":
                    train_macs += rank_macs
                arms_payload.append({
                    "arm": arm,
                    "label": config["arms"][arm]["label"],
                    "allocation": allocated,
                    "experts": experts,
                    "sample_adequate": all(e["sample_adequate"] for e in experts),
                    "training_macs": int(train_macs),
                    "rank_measurement_macs": int(
                        rank_macs if arm == "b_rank_sized" else 0
                    ),
                    "seconds": time.time() - started,
                })

            # the constant is chosen on the null arm, at the first seed only
            if budget not in chosen_penalty:
                null = next(a for a in arms_payload if a["arm"] == selection_arm)
                scores = {
                    str(p): float(np.average(
                        [e["validation_accuracy_by_penalty"][str(p)]
                         for e in null["experts"]],
                        weights=[e["validation_rows"] for e in null["experts"]],
                    ))
                    for p in penalties
                }
                best = max(scores, key=lambda k: scores[k])
                chosen_penalty[budget] = {
                    "regularisation": float(best),
                    "validation_accuracy_by_penalty": scores,
                    "selection_arm": selection_arm,
                    "selection_seed": config["seeds"][0],
                    "grid_edge_selected": bool(
                        float(best) in (penalties[0], penalties[-1])
                    ),
                }

            constant = str(chosen_penalty[budget]["regularisation"])
            for record in arms_payload:
                per_domain: dict[str, float] = {}
                hits = 0
                rows = 0
                for expert in record["experts"]:
                    expert["test_accuracy"] = (
                        expert["correct_by_penalty"][constant] / expert["test_rows"]
                    )
                    per_domain[expert["expert"]] = expert["test_accuracy"]
                    hits += expert["correct_by_penalty"][constant]
                    rows += expert["test_rows"]
                record["pooled_test_accuracy"] = hits / rows
                record["per_domain_test_accuracy"] = per_domain
                record["regularisation"] = float(constant)
                sizes = record["allocation"]["atoms"]
                if len(sizes) == 1:
                    realised = float(sizes[0])
                    parameters = sum(
                        e["features"] * classes + classes for e in record["experts"]
                    )
                else:
                    realised = float(np.sum(test_shares * np.array(sizes)))
                    parameters = sum(
                        e["features"] * classes + classes for e in record["experts"]
                    )
                record["realised_test_row_weighted_atoms"] = realised
                record["head_parameters"] = int(parameters)
                record["inference_macs_per_image_expected"] = float(np.sum(
                    (test_shares if len(sizes) > 1 else np.array([1.0]))
                    * np.array([
                        _inference_macs(a, grid, dimension, pool_grid,
                                        classes)["total"]
                        for a in sizes
                    ])
                ))
                if progress:
                    print(f"  seed {seed}  {budget:>5}  {record['arm']:<28} "
                          f"pooled {record['pooled_test_accuracy']:.4f}  "
                          f"MACs/img {record['inference_macs_per_image_expected']:.3e}",
                          flush=True)

            budgets_payload.append({
                "budget": budget,
                "regularisation_selection": chosen_penalty[budget],
                "arms": arms_payload,
            })
            write_canonical_json(
                output_dir / "partial_seeds.json",
                {"seeds": seeds_payload + [{
                    "seed": seed, "budgets": budgets_payload,
                }]},
            )

        seeds_payload.append({
            "seed": seed,
            "effective_rank": {
                "per_domain": {names[d]: float(ranks[d]) for d in range(domain_count)},
                "rows_per_domain": {
                    names[d]: int(rank_rows[d]) for d in range(domain_count)
                },
                "probe_atoms": probe_atoms,
                "seconds": rank_seconds,
                "macs": int(rank_macs),
                "order_agrees_with_probe3": rank_order_agrees,
            },
            "budgets": budgets_payload,
        })

    evidence: dict[str, Any] = {
        "milestone": "M104",
        "registered_in": config["registered_in"],
        "config": config,
        "corpus_sha256": digest,
        "class_count": classes,
        "patch_grid": grid,
        "patch_dimension": dimension,
        "train_rows": int(len(train_index)),
        "test_rows": int(len(test_index)),
        "train_rows_by_domain": {
            names[d]: int(train_counts[d]) for d in range(domain_count)
        },
        "test_rows_by_domain": {
            names[d]: int(test_counts[d]) for d in range(domain_count)
        },
        "train_row_shares": {
            names[d]: float(train_shares[d]) for d in range(domain_count)
        },
        "sample_floor_atom_caps": {
            names[d]: int(caps[d]) for d in range(domain_count)
        },
        "whitener_bitwise_repeatable": determinism,
        "torch_threads": config["numerics"]["torch_threads"],
        "seeds": seeds_payload,
        "routing": config["routing"]["mode"],
    }
    evidence["gate"] = _build_gate(evidence, config)
    evidence["payload_sha256"] = payload_hash(evidence)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    return evidence


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------
def _arm_curve(evidence: dict[str, Any], budget: int, arm: str) -> list[float]:
    values = []
    for seed in evidence["seeds"]:
        for entry in seed["budgets"]:
            if entry["budget"] != budget:
                continue
            for record in entry["arms"]:
                if record["arm"] == arm:
                    values.append(record["pooled_test_accuracy"])
    return values


def _per_domain(evidence: dict[str, Any], budget: int, arm: str
                ) -> dict[str, list[float]]:
    out: dict[str, list[float]] = collections.defaultdict(list)
    for seed in evidence["seeds"]:
        for entry in seed["budgets"]:
            if entry["budget"] != budget:
                continue
            for record in entry["arms"]:
                if record["arm"] != arm:
                    continue
                for name, value in record["per_domain_test_accuracy"].items():
                    out[name].append(value)
    return dict(out)


def _build_gate(evidence: dict[str, Any], config: dict[str, Any]
                ) -> dict[str, Any]:
    """Every kill switch, evaluated from the run's own arms and nothing else."""
    gate_config = config["gate"]
    out: dict[str, Any] = {"budgets": {}}
    for budget in config["budgets"]:
        uniform = _arm_curve(evidence, budget, "a_uniform")
        ranked = _arm_curve(evidence, budget, "b_rank_sized")
        random_sized = _arm_curve(evidence, budget, "d_random_sized")
        traffic = _arm_curve(evidence, budget, "e_traffic_inverse")
        generalist1 = _arm_curve(evidence, budget, "c1_generalist_mac_matched")
        generalist2 = _arm_curve(evidence, budget, "c2_generalist_atom_matched")
        if not uniform or not ranked:
            continue
        uniform_spread = float(np.max(uniform) - np.min(uniform))
        margin = float(np.mean(ranked) - np.mean(uniform))

        ranked_domains = _per_domain(evidence, budget, "b_rank_sized")
        uniform_domains = _per_domain(evidence, budget, "a_uniform")
        low_rank = ["quickdraw", "sketch"]
        margins = {
            name: float(np.mean(ranked_domains[name]) - np.mean(uniform_domains[name]))
            for name in ranked_domains if name in uniform_domains
        }
        low = [v for k, v in margins.items() if k in low_rank]
        high = [v for k, v in margins.items() if k not in low_rank]
        low_mean = float(np.mean(low)) if low else float("nan")
        high_mean = float(np.mean(high)) if high else float("nan")
        tolerance = gate_config["mechanism_uniformity_tolerance"]
        if low_mean <= 0:
            mechanism = "unsupported: no margin on the low-rank domains"
        elif high_mean > low_mean:
            mechanism = "contradicted: the margin is larger on the HIGH-rank domains"
        elif high_mean > tolerance * low_mean:
            mechanism = (
                "unsupported: the margin is close to uniform across domains, so "
                "the stated mechanism is not what produced it"
            )
        else:
            mechanism = "supported: the margin is concentrated in the low-rank domains"

        entry: dict[str, Any] = {
            "uniform_pooled_mean": float(np.mean(uniform)),
            "uniform_pooled_spread": uniform_spread,
            "rank_sized_pooled_mean": float(np.mean(ranked)),
            "margin_b_minus_a": margin,
            "kill_switch_1_fired": bool(margin <= uniform_spread),
            "kill_switch_1_text": gate_config["kill_switch_1"],
            "per_domain_margin_b_minus_a": margins,
            "low_rank_domain_mean_margin": low_mean,
            "high_rank_domain_mean_margin": high_mean,
            "registered_mechanism_verdict": mechanism,
        }
        tolerance2 = gate_config["kill_switch_2_tolerance"]
        if random_sized:
            entry["random_sized_pooled_mean"] = float(np.mean(random_sized))
            entry["kill_switch_2_fired"] = bool(
                float(np.mean(ranked) - np.mean(random_sized)) <= tolerance2
            )
        if traffic:
            entry["traffic_inverse_pooled_mean"] = float(np.mean(traffic))
            entry["kill_switch_4_fired"] = bool(
                float(np.mean(ranked) - np.mean(traffic)) <= tolerance2
            )
        best = max(np.mean(uniform), np.mean(ranked),
                   np.mean(random_sized) if random_sized else -np.inf,
                   np.mean(traffic) if traffic else -np.inf)
        if generalist1:
            entry["generalist_mac_matched_pooled_mean"] = float(np.mean(generalist1))
            entry["kill_switch_3_mac_matched_fired"] = bool(
                float(np.mean(generalist1)) >= best - tolerance2
            )
        if generalist2:
            entry["generalist_atom_matched_pooled_mean"] = float(np.mean(generalist2))
            entry["kill_switch_3_atom_matched_fired"] = bool(
                float(np.mean(generalist2)) >= best - tolerance2
            )
        entry["void_arms"] = sorted({
            record["arm"]
            for seed in evidence["seeds"]
            for item in seed["budgets"] if item["budget"] == budget
            for record in item["arms"] if not record["sample_adequate"]
        })
        out["budgets"][str(budget)] = entry
    out["_reading"] = (
        "Oracle routing. Section 7.10 design item 3 and restriction 4: these are "
        "upper bounds and not system results, and every sentence quoting one "
        "carries the word 'oracle'. Section 7.10 restriction 1 and prohibition "
        "24 forbid comparing any figure here to any M103 or section 2.9.3 "
        "CIFAR-10 figure in either direction."
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="M104 expert sizing")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    evidence = run_m104(arguments.config, arguments.output,
                        progress=not arguments.quiet)
    gate = evidence["gate"]
    print("\n=== M104 gate (ORACLE routing) ===", flush=True)
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
