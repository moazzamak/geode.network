"""M103 — is a grown dictionary better than a drawn one?

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v15.md`` section 7.9.

Section 2.9.3 measured a backprop-free patch representation on CIFAR-10 and
found that *learning* the dictionary by k-means is **worse** than drawing it at
random at every budget. Section 2.9.4 then measured that this ordering has a
second reading — k-means optimises reconstruction, not discrimination — and
that a dictionary selected against a discriminative residual beats random
selection. Both are unsealed scoping probes and are inadmissible as operands
(section 2.4, section 11.2 item 23). M103 is the milestone that settles the
question under seal, at full scale, with the nulls the plan requires.

**The registered primary operand is efficiency at matched accuracy**: the
smallest atom count at which each arm reaches the accuracy arm (a) reaches at
the reference budget. M103 measures whether the thesis buys *fewer atoms*, not
whether it buys accuracy at matched size, because section 10.2's question is
about needing fewer patches.

Registration notes carried by this runner:

* **N103.1 — arms (a) and (c) share one candidate pool.** Section 7.9
  restriction 6, added from section 2.9.4 limitation (iii). A larger search
  space is itself an advantage; if arm (c) could see more atoms than arm (a),
  a win would be ambiguous between "the discriminative criterion works" and
  "looking at more atoms works".
* **N103.2 — the 2048 rung is registered void before the run.** 50,000 rows and
  4 x 2048 = 8,192 features is 6.1 rows per fitted dimension, below section
  5.3's floor of 10. It is run and reported ``void``, not negative, exactly as
  M102 reported its d=64 arm. Registering this in advance is the direct lesson
  of correction C102.2, which had to be made after the fact.
* **N103.3 — both compute columns, never netted.** Arm (c) pays a selection
  cost that arms (a) and (d) do not pay at all. Section 11.2 item 10 forbids
  netting training against inference, so both are reported and neither is
  subtracted from the other.
* **N103.4 — no novelty.** Patch dictionaries (Coates et al. 2011), random
  features (Rahimi & Recht 2007), greedy dictionary construction (Pati et al.
  1993) and random-beats-learned on this exact pipeline (Thiry et al. 2021) are
  all established. M103's contribution is a comparison, not an invention.
* **N103.5 — the instrument is checked before any arm is read, on internal
  conditions only.** Section 7.9 design item 4 as corrected in section 2.9.6:
  arm (b)'s accuracy must rise with atom count, must exceed section 2.9.3's own
  1024-atom reading, and the encode must be bitwise repeatable. The Coates
  figure is reported beside the curve as an anchor at 4000 features and gates
  nothing, because R7 states that external figures are anchors and never
  operands and because M103's top readable rung is 1024 atoms.
* **N103.6 — arm (d) carries the patch pool's norm distribution.** Section
  2.9.4 measured that atom-norm distribution is the mechanism separating these
  dictionaries. Arm (d) is therefore isotropic in *direction* while resampling
  its norms from the whitened patch pool, so it differs from arm (a) in
  direction only. Unit-norm directions would have made the second kill switch
  a test of norm variance as well as of provenance.
* **N103.7 — corpus isolation.** Section 7.9 restriction 1: no figure produced
  here may be compared to any v13, v14 or v15 DomainNet figure in either
  direction. CIFAR-10 has 10 classes; DomainNet-128 has 128.

Reproduce with::

    .\\.venv\\Scripts\\python.exe -m experiments.tier4.eval_v15_m103_atoms
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v15" / "m103_atoms.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v15" / "m103_atoms"

# The distance matrix is (patch rows x atoms) and is the run's memory ceiling.
# Chunks are sized to hold it near this many bytes rather than to a fixed image
# count, so the 64-atom and 2048-atom rungs both run near the same footprint.
DISTANCE_BYTES_TARGET = 1_200_000_000


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------
def _load_cifar10(train_rows: int, test_rows: int) -> dict[str, np.ndarray]:
    from datasets import load_dataset

    data = load_dataset("uoft-cs/cifar10")

    def pack(split, rows):
        images = np.stack([np.asarray(r) for r in split["img"][:rows]])
        return (
            np.ascontiguousarray(images, dtype=np.uint8),
            np.asarray(split["label"][:rows], dtype=np.int64),
        )

    train_images, train_labels = pack(data["train"], train_rows)
    test_images, test_labels = pack(data["test"], test_rows)
    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }


def _corpus_digest(corpus: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(corpus):
        digest.update(key.encode("utf-8"))
        digest.update(str(corpus[key].shape).encode("utf-8"))
        digest.update(corpus[key].tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------
# representation
# --------------------------------------------------------------------------
def _extract_patches(images: np.ndarray, patch: int, stride: int) -> np.ndarray:
    """Return (n_images * grid * grid, patch*patch*channels) float32.

    Channel-last within the patch, matching the ordering section 2.9.3 used, so
    the whitening basis fitted here is the same object that pipeline whitened
    with.
    """
    scaled = images.astype(np.float32) / 255.0
    windows = np.lib.stride_tricks.sliding_window_view(
        scaled, (patch, patch), axis=(1, 2)
    )[:, ::stride, ::stride]
    windows = np.transpose(windows, (0, 1, 2, 4, 5, 3))
    return np.ascontiguousarray(windows).reshape(-1, patch * patch * scaled.shape[3])


def _contrast_normalise(patches: np.ndarray, epsilon: float) -> np.ndarray:
    centred = patches - patches.mean(axis=-1, keepdims=True)
    return centred / np.sqrt(patches.var(axis=-1, keepdims=True) + epsilon)


def _fit_zca(patches: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    mean = patches.mean(axis=0)
    centred = patches - mean
    covariance = centred.T @ centred / len(centred)
    values, vectors = np.linalg.eigh(covariance.astype(np.float64))
    whiten = vectors @ np.diag(1.0 / np.sqrt(values + epsilon)) @ vectors.T
    return mean.astype(np.float32), whiten.astype(np.float32)


class Whitener:
    """Patch extraction, contrast normalisation and ZCA in one place.

    Every arm and every seed shares one instance, so no arm gets a different
    representation and only the dictionary varies.
    """

    def __init__(self, patch: int, stride: int, contrast_epsilon: float,
                 mean: np.ndarray, whiten: np.ndarray, grid: int):
        self.patch = patch
        self.stride = stride
        self.contrast_epsilon = contrast_epsilon
        self.mean = mean
        self.whiten = whiten
        self.grid = grid

    def __call__(self, images: np.ndarray) -> np.ndarray:
        patches = _extract_patches(images, self.patch, self.stride)
        patches = _contrast_normalise(patches, self.contrast_epsilon)
        return (patches - self.mean) @ self.whiten


def _pool(activation: torch.Tensor, count: int, grid: int, pool_grid: int
          ) -> torch.Tensor:
    """2x2 (or pool_grid x pool_grid) sum pooling over the patch grid."""
    atoms = activation.shape[1]
    activation = activation.reshape(count, grid, grid, atoms)
    edges = [round(grid * i / pool_grid) for i in range(pool_grid + 1)]
    blocks = []
    for iy in range(pool_grid):
        for ix in range(pool_grid):
            blocks.append(
                activation[:, edges[iy]:edges[iy + 1], edges[ix]:edges[ix + 1]]
                .sum(dim=(1, 2))
            )
    return torch.cat(blocks, dim=1)


def encode(images: np.ndarray, dictionary: np.ndarray, whitener: Whitener,
           pool_grid: int, dtype: torch.dtype = torch.float32) -> np.ndarray:
    """Triangle-encode and pool, holding only (n_images, pool^2 * atoms).

    The whole point of doing the pooling inside the chunk loop is that the
    unpooled activation for the full train split at stride 1 would be hundreds
    of gigabytes. Nothing here ever materialises it.
    """
    atoms = len(dictionary)
    grid = whitener.grid
    per_image = grid * grid
    chunk = max(1, min(len(images),
                       int(DISTANCE_BYTES_TARGET / (4 * atoms * per_image))))
    table = torch.from_numpy(np.ascontiguousarray(dictionary)).to(dtype)
    out = np.empty((len(images), pool_grid * pool_grid * atoms), dtype=np.float32)
    for start in range(0, len(images), chunk):
        block = images[start:start + chunk]
        white = torch.from_numpy(
            np.ascontiguousarray(whitener(block))
        ).to(dtype)
        with torch.no_grad():
            distances = torch.cdist(white, table)
            activation = torch.clamp(
                distances.mean(dim=1, keepdim=True) - distances, min=0.0
            )
            pooled = _pool(activation, len(block), grid, pool_grid)
        out[start:start + chunk] = pooled.to(torch.float32).numpy()
    return out


# --------------------------------------------------------------------------
# arm (c): additive selection against a discriminative residual
# --------------------------------------------------------------------------
def select_discriminative(features: np.ndarray, labels: np.ndarray,
                          atom_count: int, budget: int, pool_grid: int
                          ) -> tuple[np.ndarray, int]:
    """Group OMP over the candidate pool. Returns (order, multiply_adds).

    Each candidate owns the ``pool_grid**2`` pooled columns it contributes, so
    atom ``a`` owns ``{a, A+a, 2A+a, ...}``. The residual starts as the centred
    one-hot label matrix; the atom whose block best explains it is taken and
    regressed out, so the next atom is chosen against what the selected ones
    leave unexplained. This is the plan's construction principle with the model
    linear and the component an atom.

    The order is nested by construction, so every budget in the sweep is a
    prefix of this one call.
    """
    groups = np.stack(
        [np.arange(atom_count) + q * atom_count for q in range(pool_grid ** 2)],
        axis=1,
    )
    rows = len(features)
    classes = int(labels.max()) + 1
    work = features.astype(np.float32, copy=True)
    work -= work.mean(axis=0, keepdims=True)
    work /= work.std(axis=0, keepdims=True) + 1e-8

    residual = np.zeros((rows, classes), dtype=np.float32)
    residual[np.arange(rows), labels] = 1.0
    residual -= residual.mean(axis=0, keepdims=True)

    order: list[int] = []
    available = np.ones(atom_count, dtype=bool)
    macs = 0
    block_width = pool_grid ** 2
    for _ in range(budget):
        correlation = work.T @ residual
        macs += work.shape[0] * work.shape[1] * classes
        score = np.zeros(atom_count, dtype=np.float32)
        for q in range(block_width):
            score += np.sum(correlation[groups[:, q]] ** 2, axis=1)
        score[~available] = -np.inf
        best = int(np.argmax(score))
        order.append(best)
        available[best] = False
        block = work[:, groups[best]]
        coefficients, *_ = np.linalg.lstsq(block, residual, rcond=None)
        residual = residual - block @ coefficients
        macs += 2 * rows * block_width * classes
    return np.asarray(order, dtype=np.int64), macs


# --------------------------------------------------------------------------
# head
# --------------------------------------------------------------------------
def _standardise(train_x: np.ndarray, test_x: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Centre and scale in place.

    At 2048 atoms these matrices are gigabytes and a temporary copy of each
    would double the run's memory ceiling for no benefit.
    """
    centre = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True) + 1e-8
    train_x -= centre
    train_x /= scale
    test_x -= centre
    test_x /= scale
    return train_x, test_x


def _choose_regularisation(train_z: np.ndarray, train_y: np.ndarray,
                           config: dict[str, Any]) -> dict[str, Any]:
    """Rank the grid on a validation split carved out of TRAIN.

    Run once per budget, on the null arm, at the first seed. The chosen value
    is then applied unchanged to every arm and every seed at that budget, so no
    arm — least of all arm (c) — gets a constant tuned to itself.
    """
    grid = config["regularisation_grid"]
    split = len(train_z) - config["validation_rows"]
    rows = min(config["selection_rows"], split)
    inner_x, inner_y = train_z[:rows], train_y[:rows]
    validation_x = train_z[split:]
    validation_y = train_y[split:]

    scores = []
    for penalty in grid:
        model = LogisticRegression(max_iter=config["max_iter"], C=penalty)
        model.fit(inner_x, inner_y)
        scores.append(float(model.score(validation_x, validation_y)))
    chosen = grid[int(np.argmax(scores))]
    return {
        "regularisation_c": chosen,
        "validation_accuracy_by_c": dict(zip(map(str, grid), scores)),
        "selection_rows": rows,
        "validation_rows": config["validation_rows"],
        "grid_top_selected": bool(chosen == grid[-1]),
    }


def _fit_head(train_z: np.ndarray, train_y: np.ndarray, test_z: np.ndarray,
              test_y: np.ndarray, penalty: float, max_iter: int
              ) -> dict[str, Any]:
    """Refit on all train rows at the chosen constant and score on test."""
    final = LogisticRegression(max_iter=max_iter, C=penalty)
    final.fit(train_z, train_y)
    iterations = int(np.max(final.n_iter_))
    return {
        "test_accuracy": float(final.score(test_z, test_y)),
        "regularisation_c": penalty,
        "solver_iterations": iterations,
        "converged": bool(iterations < max_iter),
        "head_parameters": int(final.coef_.size + final.intercept_.size),
    }


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


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def run_m103(config_path: Path, output_dir: Path,
             progress: bool = True) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    torch.set_num_threads(config["numerics"]["torch_threads"])

    corpus_config = config["corpus"]
    corpus = _load_cifar10(corpus_config["train_rows"], corpus_config["test_rows"])
    digest = _corpus_digest(corpus)
    expected = corpus_config.get("expected_sha256")
    if expected is not None and expected != digest:
        raise ValueError(
            f"M103 corpus mismatch: config pins {expected}, loaded {digest}"
        )

    representation = config["representation"]
    patch = representation["patch"]
    stride = representation["stride"]
    pool_grid = representation["pool_grid"]
    height = corpus["train_images"].shape[1]
    grid = (height - patch) // stride + 1
    dimension = patch * patch * corpus["train_images"].shape[3]

    pool_rng = np.random.default_rng(representation["zca_fit_seed"])
    sample_images = corpus["train_images"][
        pool_rng.choice(len(corpus["train_images"]), size=2000, replace=False)
    ]
    raw = _contrast_normalise(
        _extract_patches(sample_images, patch, stride),
        representation["contrast_epsilon"],
    )
    raw = raw[pool_rng.choice(len(raw), size=representation["zca_fit_rows"],
                              replace=False)]
    mean, whiten = _fit_zca(raw, representation["zca_epsilon"])
    patch_pool = (raw - mean) @ whiten
    pool_norms = np.linalg.norm(patch_pool, axis=1)
    pool_mean_norm = float(pool_norms.mean())
    whitener = Whitener(patch, stride, representation["contrast_epsilon"],
                        mean, whiten, grid)

    # Determinism self-check: the encode is multi-threaded, so prove within the
    # run that it is bitwise repeatable rather than assuming it.
    probe_dictionary = patch_pool[:64]
    probe_images = corpus["test_images"][:64]
    first = encode(probe_images, probe_dictionary, whitener, pool_grid)
    second = encode(probe_images, probe_dictionary, whitener, pool_grid)
    determinism = bool(np.array_equal(first, second))
    if not determinism:
        raise ValueError(
            "M103 instrument failure: the encode is not bitwise repeatable "
            "within a single process, so no figure from it is reproducible."
        )

    floor = config["sample_adequacy"]["fit_samples_per_fitted_dimension_floor"]
    classes = int(corpus["train_labels"].max()) + 1
    budgets = config["budgets"]
    pool_size = config["candidate_pool"]["size"]
    selection_rows = config["arms"]["c_discriminative"]["selection_rows"]
    kmeans_config = config["arms"]["b_kmeans"]
    head_config = config["head"]

    seeds_payload: list[dict[str, Any]] = []
    seed_state: dict[int, dict[str, Any]] = {}
    for seed in config["seeds"]:
        rng = np.random.default_rng(seed)
        candidates = patch_pool[
            rng.choice(len(patch_pool), size=pool_size, replace=False)
        ]
        # Arm (a) draws from the SAME pool, so any arm (c) advantage cannot be
        # an artefact of pool size (section 7.9 restriction 6).
        draw_order = rng.permutation(pool_size)

        selection_index = rng.choice(
            len(corpus["train_images"]), size=selection_rows, replace=False
        )
        selection_started = time.time()
        selection_features = encode(
            corpus["train_images"][selection_index], candidates, whitener, pool_grid
        )
        selection_encode_macs = (
            selection_rows
            * (grid * grid * (dimension * dimension + pool_size * dimension))
        )
        # The selection order is nested, so one call covers every budget in the
        # sweep and the whole sweep pays this cost once per seed.
        selection_order, selection_macs = select_discriminative(
            selection_features,
            corpus["train_labels"][selection_index],
            pool_size,
            max(budgets),
            pool_grid,
        )
        del selection_features
        seed_state[seed] = {
            "candidates": candidates,
            "draw_order": draw_order,
            "selection_order": selection_order,
            "selection_macs": int(selection_encode_macs + selection_macs),
            "selection_seconds": time.time() - selection_started,
        }
        seeds_payload.append({
            "seed": seed,
            "candidate_pool_size": pool_size,
            "selection_seconds": seed_state[seed]["selection_seconds"],
            "budgets": [],
        })
        if progress:
            print(f"  seed {seed}: candidate pool {pool_size}, selection order "
                  f"of {max(budgets)} atoms in "
                  f"{seed_state[seed]['selection_seconds']:.0f}s", flush=True)

    # Budgets outer, seeds inner: a run that is stopped early then holds
    # COMPLETE rungs at the cheap, readable budgets rather than one complete
    # seed and nothing else, and the reference budget is reached before the
    # rung the sample floor already registers as void.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chosen_penalty: dict[int, dict[str, Any]] = {}
    for budget in budgets:
        ratio = corpus_config["train_rows"] / (pool_grid * pool_grid * budget)
        adequate = ratio >= floor
        for position, seed in enumerate(config["seeds"]):
            state = seed_state[seed]
            budget_started = time.time()
            dictionaries: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}

            dictionaries["a_random_patches"] = (
                state["candidates"][state["draw_order"][:budget]],
                {"training_macs": 0, "training_seconds": 0.0,
                 "candidate_pool_size": pool_size},
            )

            started = time.time()
            kmeans = MiniBatchKMeans(
                n_clusters=budget,
                random_state=seed,
                n_init=kmeans_config["n_init"],
                batch_size=kmeans_config["batch_size"],
                max_iter=kmeans_config["max_iter"],
            ).fit(patch_pool)
            dictionaries["b_kmeans"] = (
                kmeans.cluster_centers_.astype(np.float32),
                {
                    "training_macs": int(
                        kmeans.n_iter_ * kmeans_config["batch_size"]
                        * budget * dimension
                    ),
                    "training_seconds": time.time() - started,
                    "kmeans_iterations": int(kmeans.n_iter_),
                },
            )

            dictionaries["c_discriminative"] = (
                state["candidates"][state["selection_order"][:budget]],
                {
                    "training_macs": state["selection_macs"],
                    "training_seconds": state["selection_seconds"],
                    "candidate_pool_size": pool_size,
                    "selection_rows": selection_rows,
                    "_note": (
                        "the selection cost is paid ONCE per seed for the whole "
                        "sweep because the order is nested; it is charged in "
                        "full to every budget rather than amortised, which is "
                        "the reading against this arm"
                    ),
                },
            )

            projection_rng = np.random.default_rng(seed * 1000 + budget)
            directions = projection_rng.standard_normal((budget, dimension))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            # Section 2.9.4 measured that atom-norm distribution is the
            # mechanism separating these dictionaries, so arm (d) is given the
            # patch pool's empirical norm distribution and differs from arm (a)
            # in DIRECTION only. Fixing every atom to one norm instead would
            # have made the second kill switch a test of norm variance as well
            # as of provenance, which is not what section 7.9 registers.
            borrowed_norms = projection_rng.choice(pool_norms, size=budget,
                                                   replace=False)
            dictionaries["d_random_projections"] = (
                (directions * borrowed_norms[:, None]).astype(np.float32),
                {
                    "training_macs": 0,
                    "training_seconds": 0.0,
                    "norms_resampled_from_patch_pool": True,
                    "pool_mean_norm": pool_mean_norm,
                    "_note": (
                        "isotropic directions carrying the patch pool's "
                        "empirical norm distribution, so arm (d) differs from "
                        "arm (a) in direction only"
                    ),
                },
            )

            arms_payload = []
            for name in ("a_random_patches", "b_kmeans", "c_discriminative",
                         "d_random_projections"):
                dictionary, ledger = dictionaries[name]
                started = time.time()
                train_x = encode(corpus["train_images"], dictionary, whitener,
                                 pool_grid)
                test_x = encode(corpus["test_images"], dictionary, whitener,
                                pool_grid)
                encode_seconds = time.time() - started
                train_z, test_z = _standardise(train_x, test_x)
                if (budget not in chosen_penalty
                        and name == head_config["selection_arm"]
                        and position == 0):
                    chosen_penalty[budget] = _choose_regularisation(
                        train_z, corpus["train_labels"], head_config
                    )
                penalty = chosen_penalty[budget]["regularisation_c"]
                head = _fit_head(train_z, corpus["train_labels"], test_z,
                                 corpus["test_labels"], penalty,
                                 head_config["max_iter"])
                norms = np.linalg.norm(dictionary, axis=1)
                arms_payload.append({
                    "arm": name,
                    "atoms": budget,
                    "features": pool_grid * pool_grid * budget,
                    "sample_adequate": bool(adequate),
                    "fit_samples_per_fitted_dimension": float(ratio),
                    **head,
                    "regularisation_selection": chosen_penalty[budget],
                    "atom_norms": {
                        "mean": float(norms.mean()),
                        "median": float(np.median(norms)),
                        "p5": float(np.percentile(norms, 5)),
                        "p95": float(np.percentile(norms, 95)),
                    },
                    "inference_macs_per_image": _inference_macs(
                        budget, grid, dimension, pool_grid, classes
                    ),
                    "training_ledger": ledger,
                    "encode_seconds": encode_seconds,
                })
                del train_x, test_x, train_z, test_z
                if progress:
                    print(
                        f"  seed {seed}  {budget:>5} atoms  {name:>22}  "
                        f"acc {arms_payload[-1]['test_accuracy']:.4f}  "
                        f"{'' if adequate else '[VOID: below sample floor]'}",
                        flush=True,
                    )
            seeds_payload[position]["budgets"].append({
                "atoms": budget,
                "features": pool_grid * pool_grid * budget,
                "sample_adequate": bool(adequate),
                "fit_samples_per_fitted_dimension": float(ratio),
                "arms": arms_payload,
                "seconds": time.time() - budget_started,
            })
        # Partial dumps after each completed rung, so a run interrupted at any
        # point leaves a readable artifact rather than nothing.
        write_canonical_json(
            output_dir / "partial_seeds.json", {"seeds": seeds_payload}
        )

    evidence: dict[str, Any] = {
        "milestone": "M103",
        "registered_in": config["registered_in"],
        "config": config,
        "corpus_sha256": digest,
        "patch_grid": grid,
        "patch_dimension": dimension,
        "class_count": classes,
        "patch_pool_mean_norm": pool_mean_norm,
        "encode_bitwise_repeatable": determinism,
        "torch_threads": config["numerics"]["torch_threads"],
        "regularisation_by_budget": {
            str(budget): record for budget, record in chosen_penalty.items()
        },
        "seeds": seeds_payload,
        "_corpus_isolation": (
            "CIFAR-10. Section 7.9 restriction 1: no figure here may be "
            "compared to any v13, v14 or v15 DomainNet figure in either "
            "direction."
        ),
    }
    evidence["float64_control"] = _float64_control(config, corpus, whitener,
                                                   patch_pool, pool_grid)
    evidence["instrument"] = _instrument_check(evidence, config)
    evidence["gate"] = _build_gate(evidence, config)
    evidence["payload_hash"] = payload_hash(evidence["seeds"])

    write_canonical_json(output_dir / "evidence.json", evidence)
    (output_dir / "partial_seeds.json").unlink(missing_ok=True)
    build_artifact_index(output_dir)
    return evidence


def _float64_control(config: dict[str, Any], corpus: dict[str, np.ndarray],
                     whitener: Whitener, patch_pool: np.ndarray,
                     pool_grid: int) -> dict[str, Any]:
    """Re-encode one cell in double precision and report the difference.

    The sweep runs in float32 because that is what makes it affordable. This
    control measures what that costs rather than asserting it costs nothing.
    """
    budget = config["numerics"]["float64_control_budget"]
    seed = config["numerics"]["float64_control_seed"]
    rng = np.random.default_rng(seed)
    dictionary = patch_pool[
        rng.choice(len(patch_pool), size=budget, replace=False)
    ]
    images = corpus["test_images"][:2000]
    single = encode(images, dictionary, whitener, pool_grid, torch.float32)
    double = encode(images, dictionary, whitener, pool_grid, torch.float64)
    scale = float(np.abs(double).max()) or 1.0
    return {
        "budget": budget,
        "seed": seed,
        "rows": len(images),
        "max_absolute_difference": float(np.abs(single - double).max()),
        "max_relative_difference": float(np.abs(single - double).max() / scale),
        "_note": (
            "float32 against float64 on the same dictionary and rows. A "
            "difference at the level of float32 rounding is expected; anything "
            "larger would mean the encode is numerically unstable and the "
            "sweep's precision choice would have to be revisited."
        ),
    }


def _accuracy(evidence: dict[str, Any], seed_row: dict[str, Any], arm: str,
              budget: int) -> float | None:
    rung = next((b for b in seed_row["budgets"] if b["atoms"] == budget), None)
    if rung is None:
        return None
    record = next((a for a in rung["arms"] if a["arm"] == arm), None)
    return None if record is None else record["test_accuracy"]


def _readable_budgets(config: dict[str, Any]) -> list[int]:
    floor = config["sample_adequacy"]["fit_samples_per_fitted_dimension_floor"]
    rows = config["corpus"]["train_rows"]
    pool_grid = config["representation"]["pool_grid"]
    return [
        b for b in config["budgets"]
        if rows / (pool_grid * pool_grid * b) >= floor
    ]


def _atoms_to_reach(curve: list[tuple[int, float]], target: float
                    ) -> dict[str, Any]:
    """Smallest atom count reaching ``target``, plus a linear interpolant.

    The rung figure is the registered operand; the interpolant is reported
    beside it because the sweep is coarse and a rung answer alone would hide
    how far inside the interval the crossing sits.
    """
    reached = next((atoms for atoms, value in curve if value >= target), None)
    interpolated = None
    for (low_atoms, low), (high_atoms, high) in zip(curve, curve[1:]):
        if low < target <= high and high > low:
            interpolated = low_atoms + (high_atoms - low_atoms) * (
                (target - low) / (high - low)
            )
            break
    if reached is not None and curve and curve[0][1] >= target:
        interpolated = float(curve[0][0])
    return {
        "rung": reached,
        "interpolated": None if interpolated is None else float(interpolated),
        "reached_within_sweep": reached is not None,
    }


def _instrument_check(evidence: dict[str, Any], config: dict[str, Any]
                      ) -> dict[str, Any]:
    """Section 7.9 design item 4 as corrected in section 2.9.6.

    Three internal conditions and one anchor. The anchor gates nothing: R7
    states that external figures are anchors and never operands, and the check
    as first registered broke that rule while also being unsatisfiable at any
    rung the sample floor allows M103 to read.
    """
    check = config["instrument_check"]
    readable = _readable_budgets(config)
    arm = check["monotonicity_arm"]

    curve = []
    for budget in readable:
        values = [
            _accuracy(evidence, seed_row, arm, budget)
            for seed_row in evidence["seeds"]
        ]
        values = [v for v in values if v is not None]
        if values:
            curve.append((budget, float(np.mean(values))))

    monotone = all(low[1] <= high[1] for low, high in zip(curve, curve[1:]))

    floor_budget = check["floor_budget"]
    floor_values = [
        _accuracy(evidence, seed_row, arm, floor_budget)
        for seed_row in evidence["seeds"]
    ]
    floor_values = [v for v in floor_values if v is not None]
    floor_reading = float(np.mean(floor_values)) if floor_values else None
    clears_floor = (
        floor_reading is not None and floor_reading > check["floor_accuracy"]
    )

    deterministic = bool(evidence["encode_bitwise_repeatable"])
    ok = bool(monotone and clears_floor and deterministic)
    return {
        "arm": arm,
        "accuracy_by_atoms": {str(atoms): value for atoms, value in curve},
        "monotone_in_atom_count": bool(monotone),
        "floor_budget": floor_budget,
        "floor_accuracy": check["floor_accuracy"],
        "floor_source": check["floor_source"],
        "floor_reading": floor_reading,
        "clears_floor": bool(clears_floor),
        "encode_bitwise_repeatable": deterministic,
        "anchor_accuracy": check["anchor_accuracy"],
        "anchor_features": check["anchor_features"],
        "anchor_source": check["anchor_source"],
        "anchor_gates_nothing": True,
        "verdict": "ok" if ok else "instrument_reported_broken",
        "_note": (
            "R7: the Coates figure is an ANCHOR at 4000 features and never an "
            "operand; M103's top readable rung is 1024 atoms and the two are "
            "not comparable. If any of the three internal conditions fails, "
            "section 7.9 design item 4 requires the instrument to be reported "
            "broken and NO arm to be read."
        ),
    }


def _build_gate(evidence: dict[str, Any], config: dict[str, Any]
                ) -> dict[str, Any]:
    """Compute M103's verdict from the artifact, never from prose."""
    reference_budget = config["reference_budget"]
    readable = _readable_budgets(config)
    tolerance = config["gate"]["kill_switch_2_tolerance"]
    arms = ("a_random_patches", "b_kmeans", "c_discriminative",
            "d_random_projections")

    per_seed: dict[str, dict[int, list[float]]] = {a: {} for a in arms}
    for seed_row in evidence["seeds"]:
        for arm in arms:
            for budget in config["budgets"]:
                value = _accuracy(evidence, seed_row, arm, budget)
                if value is not None:
                    per_seed[arm].setdefault(budget, []).append(value)

    curves = {
        arm: {
            str(budget): {
                "mean": float(np.mean(values)),
                "spread": float(np.max(values) - np.min(values)),
                "per_seed": values,
                "readable": budget in readable,
            }
            for budget, values in sorted(per_seed[arm].items())
        }
        for arm in arms
    }

    target = float(np.mean(per_seed["a_random_patches"][reference_budget]))
    efficiency = {}
    for arm in arms:
        curve = [
            (budget, float(np.mean(per_seed[arm][budget])))
            for budget in readable if budget in per_seed[arm]
        ]
        efficiency[arm] = _atoms_to_reach(curve, target)

    # Kill switch 1 is required at every seed, not only on the mean, because a
    # mean can clear a bar that no individual seed clears.
    per_seed_reach = []
    for index in range(len(evidence["seeds"])):
        curve = [
            (budget, per_seed["c_discriminative"][budget][index])
            for budget in readable if budget in per_seed["c_discriminative"]
        ]
        seed_target = per_seed["a_random_patches"][reference_budget][index]
        per_seed_reach.append(_atoms_to_reach(curve, seed_target)["rung"])

    c_beats_a = all(
        r is not None and r < reference_budget for r in per_seed_reach
    )

    d_matches_a = all(
        abs(float(np.mean(per_seed["d_random_projections"][budget]))
            - float(np.mean(per_seed["a_random_patches"][budget]))) <= tolerance
        for budget in readable
        if budget in per_seed["d_random_projections"]
        and budget in per_seed["a_random_patches"]
    )

    instrument_ok = evidence["instrument"]["verdict"] == "ok"
    if not instrument_ok:
        verdict = "void_instrument_reported_broken"
    elif c_beats_a:
        verdict = "confirmed"
    else:
        verdict = "refuted_atom_count_dominates_atom_choice"

    return {
        "milestone": "M103",
        "reference_budget": reference_budget,
        "reference_accuracy": target,
        "readable_budgets": readable,
        "void_budgets": [b for b in config["budgets"] if b not in readable],
        "accuracy_curves": curves,
        "atoms_to_reach_reference": efficiency,
        "c_reaches_reference_per_seed_rung": per_seed_reach,
        "kill_switch_1_c_beats_a_at_every_seed": bool(c_beats_a),
        "kill_switch_2_d_matches_a": bool(d_matches_a),
        "instrument_ok": bool(instrument_ok),
        "verdict": verdict,
        "_verdict_note": (
            "'refuted_atom_count_dominates_atom_choice' is a substantive "
            "negative about the plan's central thesis, pre-registered in "
            "section 7.9, and section 11.1 requires it to be reported as the "
            "headline rather than as a footnote. Void budgets are reported and "
            "are NOT read: below section 5.3's floor an arm is void, not "
            "negative."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    evidence = run_m103(arguments.config, arguments.output)

    gate = evidence["gate"]
    instrument = evidence["instrument"]
    print("\nM103 — efficiency at matched accuracy")
    print(f"  instrument: {instrument['verdict']} "
          f"(monotone {instrument['monotone_in_atom_count']}, "
          f"floor {instrument['floor_reading']} > "
          f"{instrument['floor_accuracy']} = {instrument['clears_floor']}, "
          f"deterministic {instrument['encode_bitwise_repeatable']})")
    print(f"  anchor (gates nothing): Coates {instrument['anchor_accuracy']} at "
          f"{instrument['anchor_features']} features")
    print(f"  reference: arm (a) at {gate['reference_budget']} atoms = "
          f"{gate['reference_accuracy']:.4f}")
    for arm, reach in gate["atoms_to_reach_reference"].items():
        print(f"  {arm:>22}  reaches it at {reach['rung']} atoms "
              f"(interpolated {reach['interpolated']})")
    print(f"  kill switch 1 (c beats a at every seed): "
          f"{gate['kill_switch_1_c_beats_a_at_every_seed']}")
    print(f"  kill switch 2 (d matches a): {gate['kill_switch_2_d_matches_a']}")
    print(f"  verdict: {gate['verdict']}")
    print(f"  void budgets (below the sample floor): {gate['void_budgets']}")


if __name__ == "__main__":
    main()
