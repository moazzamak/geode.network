"""M107 — the dense comparator: what does a sparse model's compute buy against
a dense one's?

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v15.md`` section 7.14.

Section 7.13 records the defect this milestone exists to close: **no milestone
in this program has ever compared a sparse model against a dense one on the
same corpus under the same protocol**, yet section 3.2 Q2 is an efficiency
question about exactly that. Prohibition 27 was the honest response to the
defect. It is not a measurement. This is.

The comparison is a *representation* comparison and only that. Both families
run in one identical protocol — frozen features, the same multi-output ridge
head, the same penalty grid, the same rows — and the only thing that differs
between arms is what produced the features. That is the standard linear-probe
protocol DINOv2 is itself published under, so the dense arm sits at its
intended operating point rather than at one chosen to make it lose.

Registration notes carried by this runner:

* **N107.1 — the head constant is chosen on the SPARSE side.** Section 7.14
  design item 5. It is selected once, on the sparse generalist at the smallest
  budget, against a held-out split, and then applied unchanged to every other
  arm in both families. This is against interest: it denies the dense arms a
  constant tuned to them. The full penalty grid is recorded for every arm as
  *sensitivity*, and the reported figure is the pre-registered constant. Reading
  the best constant per arm off that grid would be choosing the answer.
* **N107.2 — three asymmetries bind every figure this runner emits.** DINOv2 is
  pre-trained on LVD-142M and the sparse dictionary is drawn from this corpus's
  own train patches (favours dense); arms (d1)-(d4) read original-resolution
  images and the sparse arms read 32x32 (favours dense); the sparse mixture is
  scored under **oracle** routing (favours sparse). All three are written into
  the evidence payload so no downstream reader can quote a figure without them.
* **N107.3 — arm (d5) makes the resolution asymmetry measurable.** Section 7.14
  design item 2b: the identical DINOv2-small at the identical 224 tokens, fed
  the identical 32x32 tensors the sparse arms see, bilinearly upsampled.
  ``(d1) - (d5)`` is what the extra pixels are worth; ``(d5) - sparse`` is what
  the architecture is worth. Restriction 7 binds the two to be reported
  together.
* **N107.4 — analytic MACs only.** Section 7.14 design item 4 and restriction 5.
  Wall-clock is recorded, and it is not an operand: the ratio between these two
  families would measure onnxruntime against numpy.
* **N107.5 — the resolution sweep is a LOWER BOUND on dense.** DINOv2 is trained
  at 224 and interpolates its position embeddings everywhere else, so arms at
  other resolutions understate what a dense network designed for that budget
  would reach. Restriction 2.
* **N107.6 — no novelty.** Linear probing of frozen self-supervised features is
  the standard protocol and is not this program's idea; DINOv2 is Oquab et
  al.'s, used unmodified from a published ONNX export (restriction 1).
* **N107.7 — execution-time amendment 1, registered before measurement.** The
  registered resolution sweep {140, 98, 70, 42} leaves the two ladders
  overlapping at essentially one point. Two resolutions, **28 and 56**, are
  added. This runs *against* the sparse side: it puts more dense points inside
  the sparse ladder's own MAC range, where the sparse side has to win. See
  section 7.14.

Reproduce with::

    .\\.venv\\Scripts\\python.exe -m experiments.tier4.eval_v15_m107_dense
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Iterator

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
)
from experiments.tier4.eval_v15_m104_experts import (
    RidgeAccumulator,
    _cache_root,
    _chunk_rows,
    _encode_block,
    _inference_macs,
    _load_domainnet,
    _score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v15" / "m107_dense.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v15" / "m107_dense"

# DINOv2's published preprocessing. Anchors, used unmodified (restriction 1).
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
PATCH = 14


# --------------------------------------------------------------------------
# corpus: one fixed subsample shared by every arm
# --------------------------------------------------------------------------
def _class_subsample(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Indices of exactly ``per_class`` rows for every class, sorted ascending.

    Sorted because the parquet stream that materialises the original-resolution
    pixels walks row groups in order and can only move forwards; the fitting
    order is imposed separately by an explicit permutation, so sorting here
    costs nothing and buys a single-pass decode.
    """
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for label in range(int(labels.max()) + 1):
        member = np.flatnonzero(labels == label)
        if len(member) < per_class:
            raise RuntimeError(
                f"M107 corpus failure: class {label} has {len(member)} rows but "
                f"the registered subsample needs {per_class}. The subsample is "
                "shared by every arm and may not be made ragged."
            )
        keep.append(rng.choice(member, size=per_class, replace=False))
    return np.sort(np.concatenate(keep))


def _index_digest(index: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(index):
        digest.update(key.encode("utf-8"))
        digest.update(str(index[key].shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(index[key]).tobytes())
    return digest.hexdigest()


def _row_group_blobs(split: str) -> Iterator[tuple[int, int, Any]]:
    """(start, count, reader) per parquet row group, in the corpus's own order.

    The decoded 32x32 cache was built by walking ``sorted(glob)`` and then row
    groups in order, so a row's position in that cache is its position here.
    That identity is what lets a 32x32 row index address an original-resolution
    image, and it is the only reason the two resolutions can be guaranteed to
    describe the same picture.
    """
    import pyarrow.parquet as pq

    source_dir = _cache_root() / "domainnet" / "repository" / "data"
    files = sorted(source_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet under {source_dir}")
    cursor = 0
    for path in files:
        handle = pq.ParquetFile(path)
        for group in range(handle.metadata.num_row_groups):
            count = handle.metadata.row_group(group).num_rows
            yield cursor, count, (handle, group)
            cursor += count


def _verify_pixel_identity(split: str, index: np.ndarray, images: np.ndarray,
                           size: int, count: int) -> dict[str, Any]:
    """Prove that a 32x32 row index addresses the same picture in the parquet.

    Every dense figure M107 produces is a figure about the same images the
    sparse arms saw, and the only thing that makes that true is the claim that
    the decoded 32x32 cache and the parquet stream enumerate rows in the same
    order. The claim is cheap to *state* and load-bearing enough that stating
    it is not good enough: this decodes the first ``count`` selected rows
    straight from the parquet at 32x32 and requires them to be **bitwise**
    equal to the cached tensors the sparse arms will encode.

    It costs a couple of row-group reads because the selected indices are
    sorted and the first few live at the front of the stream.
    """
    from PIL import Image

    wanted = index[:count]
    seen = 0
    for start, rows, (handle, group) in _row_group_blobs(split):
        if seen >= len(wanted) or wanted[seen] >= start + rows:
            continue
        blobs = handle.read_row_group(group, columns=["image"]).column("image")
        blobs = blobs.to_pylist()
        while seen < len(wanted) and wanted[seen] < start + rows:
            record = blobs[int(wanted[seen]) - start]
            picture = Image.open(io.BytesIO(record["bytes"])).convert("RGB")
            fresh = np.asarray(
                picture.resize((size, size), Image.BILINEAR), dtype=np.uint8
            )
            if not np.array_equal(fresh, images[seen]):
                raise RuntimeError(
                    f"M107 instrument failure: {split} row {int(wanted[seen])} "
                    "decoded from the parquet does not match the same row of "
                    "the decoded cache. The dense arms would be reading "
                    "different pictures from the sparse arms and no comparison "
                    "between the two families would mean anything."
                )
            seen += 1
        if seen >= len(wanted):
            break
    if seen != len(wanted):
        raise RuntimeError(
            f"M107 instrument failure: only {seen} of {len(wanted)} {split} "
            "rows could be located in the parquet stream."
        )
    return {"split": split, "rows_checked": int(seen), "bitwise_identical": True}


def _materialise_original(split: str, index: np.ndarray, sizes: list[int],
                          tag: str) -> dict[int, Path]:
    """Decode the selected rows once, at every dense resolution, to memmaps.

    172,500 images at 224 is 26 GB, so nothing is held in memory: each
    resolution gets a ``.npy`` on the external cache volume and every dense arm
    memory-maps the one it needs. The decode is the expensive part and is done
    **once for all resolutions**, not once per arm.

    ``tag`` carries the subsample digest, so a changed subsample cannot silently
    reuse pixels drawn for a different one.
    """
    from PIL import Image

    if not sizes:
        return {}
    root = _cache_root() / "domainnet_m107" / tag
    root.mkdir(parents=True, exist_ok=True)
    paths = {size: root / f"{split}_{size}.npy" for size in sizes}
    done = root / f"{split}.done"
    if done.exists() and all(p.exists() for p in paths.values()):
        print(f"  reusing original-resolution cache {root} ({split})", flush=True)
        return paths

    writers = {
        size: np.lib.format.open_memmap(
            paths[size], mode="w+", dtype=np.uint8,
            shape=(len(index), size, size, 3),
        )
        for size in sizes
    }
    pointer = 0
    started = time.time()
    for start, count, (handle, group) in _row_group_blobs(split):
        if pointer >= len(index) or index[pointer] >= start + count:
            continue
        blobs = handle.read_row_group(group, columns=["image"]).column("image")
        blobs = blobs.to_pylist()
        while pointer < len(index) and index[pointer] < start + count:
            record = blobs[int(index[pointer]) - start]
            picture = Image.open(io.BytesIO(record["bytes"])).convert("RGB")
            for size in sizes:
                writers[size][pointer] = np.asarray(
                    picture.resize((size, size), Image.BILINEAR), dtype=np.uint8
                )
            pointer += 1
        if pointer % 20000 < count:
            rate = pointer / max(1e-9, time.time() - started)
            print(f"    {split}: {pointer}/{len(index)}  {rate:.0f} img/s",
                  flush=True)
    if pointer != len(index):
        raise RuntimeError(
            f"M107 corpus failure: materialised {pointer} of {len(index)} "
            f"{split} rows; the parquet order no longer matches the decoded "
            "cache and no dense figure from it would describe the same images "
            "as the sparse arms."
        )
    for writer in writers.values():
        writer.flush()
    del writers
    done.write_text("ok\n", encoding="utf-8", newline="\n")
    return paths


# --------------------------------------------------------------------------
# dense features
# --------------------------------------------------------------------------
def _dinov2_path(name: str) -> Path:
    root = _cache_root() / "huggingface" / "hub"
    found = sorted(root.glob(
        f"models--onnx-community--dinov2-{name}-ONNX/snapshots/*/onnx/model.onnx"
    ))
    if not found:
        raise FileNotFoundError(
            f"DINOv2-{name} ONNX export not found under {root}. M107 uses a "
            "published export unmodified (section 7.14 restriction 1)."
        )
    return found[0]


def _dinov2_geometry(name: str) -> dict[str, int]:
    """Depth, width and MLP ratio read off the graph, not quoted from memory.

    R7: an external figure is an anchor, never an operand. These drive the
    analytic MAC ledger, which *is* an operand, so they are measured from the
    initializer shapes of the very file being run.
    """
    import onnx

    model = onnx.load(str(_dinov2_path(name)), load_external_data=False)
    width = int(model.graph.output[0].type.tensor_type.shape.dim[2].dim_value)
    square = sum(
        1 for i in model.graph.initializer if tuple(i.dims) == (width, width)
    )
    hidden = {
        tuple(i.dims)[1] for i in model.graph.initializer
        if len(i.dims) == 2 and i.dims[0] == width and i.dims[1] != width
    }
    if len(hidden) != 1 or square % 4 != 0:
        raise RuntimeError(
            f"M107 instrument failure: DINOv2-{name}'s graph does not have the "
            "4-projection, single-MLP-width block structure the analytic MAC "
            "ledger assumes; the ledger would be wrong."
        )
    return {
        "width": width,
        "depth": square // 4,
        "mlp_hidden": int(next(iter(hidden))),
    }


def _transformer_macs(geometry: dict[str, int], resolution: int, classes: int
                      ) -> dict[str, int]:
    """Analytic multiply-accumulates for one image. Section 7.14 design item 4."""
    side = resolution // PATCH
    patches = side * side
    tokens = patches + 1
    width = geometry["width"]
    depth = geometry["depth"]
    hidden = geometry["mlp_hidden"]
    embed = patches * PATCH * PATCH * 3 * width
    projections = 4 * tokens * width * width
    attention = 2 * tokens * tokens * width
    mlp = 2 * tokens * width * hidden
    head = 2 * width * classes
    total = embed + depth * (projections + attention + mlp) + head
    return {
        "tokens": int(tokens),
        "patch_embedding": int(embed),
        "projections": int(depth * projections),
        "attention": int(depth * attention),
        "mlp": int(depth * mlp),
        "head": int(head),
        "total": int(total),
        "_excluded": (
            "layer norms, GELU and the softmax are NOT counted, matching the "
            "sparse ledger's exclusion of patch extraction and contrast "
            "normalisation; both exclusions are elementwise and both favour "
            "the arm they are excluded from"
        ),
    }


class DenseEncoder:
    """A frozen DINOv2 ONNX session behind a fixed feature definition.

    The feature is the CLS token concatenated with the mean of the patch
    tokens, which is DINOv2's own linear-evaluation feature (section 7.14
    design item 2), giving ``2 * width`` columns.
    """

    def __init__(self, name: str, threads: int):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(_dinov2_path(name)), options,
            providers=["CPUExecutionProvider"],
        )
        self.name = name
        self.geometry = _dinov2_geometry(name)
        self.width = 2 * self.geometry["width"]

    def __call__(self, images: np.ndarray) -> np.ndarray:
        block = np.asarray(images, dtype=np.float32) / 255.0
        block = (block - IMAGENET_MEAN) / IMAGENET_STD
        block = np.ascontiguousarray(block.transpose(0, 3, 1, 2))
        tokens = self.session.run(None, {"pixel_values": block})[0]
        return np.concatenate(
            [tokens[:, 0, :], tokens[:, 1:, :].mean(axis=1)], axis=1
        ).astype(np.float32)


def _upsample(images: np.ndarray, resolution: int) -> np.ndarray:
    """32x32 uint8 tensors to ``resolution``, bilinear. Arm (d5) only."""
    block = torch.from_numpy(
        np.ascontiguousarray(images.transpose(0, 3, 1, 2))
    ).to(torch.float32)
    with torch.no_grad():
        grown = torch.nn.functional.interpolate(
            block, size=(resolution, resolution), mode="bilinear",
            align_corners=False,
        )
    return grown.clamp(0.0, 255.0).round().to(torch.uint8).numpy().transpose(
        0, 2, 3, 1
    )


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
def _solve_and_score(accumulator: RidgeAccumulator, penalties: list[float],
                     test_features: Any) -> dict[str, Any]:
    standardise = accumulator.standardiser()
    solutions = accumulator.solve_many(penalties)
    correct = {penalty: 0 for penalty in penalties}
    per_domain: dict[float, np.ndarray] = {}
    seen = 0
    for block, labels, domains in test_features:
        standardised = standardise(block)
        for penalty in penalties:
            hits = _score(solutions[penalty], standardised, labels)
            correct[penalty] += int(hits.sum())
            bucket = per_domain.setdefault(penalty, np.zeros((2, 6), dtype=np.int64))
            np.add.at(bucket[0], domains, hits.astype(np.int64))
            np.add.at(bucket[1], domains, 1)
        seen += len(labels)
    return {
        "fit_rows": int(accumulator.rows),
        "features": int(accumulator.width),
        "test_rows": int(seen),
        "correct_by_penalty": {str(k): int(v) for k, v in correct.items()},
        "accuracy_by_penalty": {str(k): v / seen for k, v in correct.items()},
        "per_domain_correct": {
            str(k): v[0].tolist() for k, v in per_domain.items()
        },
        "per_domain_rows": {
            str(k): v[1].tolist() for k, v in per_domain.items()
        },
        "rows_per_fitted_dimension": accumulator.rows / accumulator.width,
    }


def _dense_arm(encoder: DenseEncoder, train_pixels: Any, test_pixels: Any,
               corpus: dict[str, np.ndarray], penalties: list[float],
               batch: int, classes: int, progress: bool) -> dict[str, Any]:
    """One dense arm: stream train through the encoder, solve, stream test.

    Exactly two passes over the pixels, one per split. The encode is the whole
    cost of this milestone and no row is encoded twice.

    The train rows are read in **corpus order**, not in the run's shuffled
    order, because the accumulator only ever sums: a Gram is order-invariant
    and a dense arm has no validation split to carve off. That turns a random
    walk over a 21 GB memory-mapped file into a sequential scan, which is the
    difference between a disk-bound arm and a compute-bound one.
    """
    rows_total = len(corpus["train_labels"])
    accumulator = RidgeAccumulator(encoder.width, classes)
    started = time.time()
    for start in range(0, rows_total, batch):
        stop = min(start + batch, rows_total)
        accumulator.add(encoder(train_pixels[start:stop]),
                        corpus["train_labels"][start:stop])
        if progress and start % (batch * 100) == 0:
            rate = stop / max(1e-9, time.time() - started)
            print(f"      train {stop}/{rows_total}  {rate:.1f} img/s", flush=True)
    train_seconds = time.time() - started

    def _test_blocks() -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for begin in range(0, len(corpus["test_labels"]), batch):
            stop = min(begin + batch, len(corpus["test_labels"]))
            yield (
                encoder(test_pixels[begin:stop]),
                corpus["test_labels"][begin:stop],
                corpus["test_domains"][begin:stop],
            )

    started = time.time()
    result = _solve_and_score(accumulator, penalties, _test_blocks())
    result["wall_clock_seconds"] = {
        "_not_an_operand": (
            "section 7.14 restriction 5: no wall-clock comparison between "
            "families; this measures onnxruntime against numpy"
        ),
        "train_encode": round(train_seconds, 2),
        "test_encode_and_score": round(time.time() - started, 2),
    }
    return result


def _sparse_features(images: np.ndarray, dictionary: np.ndarray,
                     whitener: Whitener, pool_grid: int, rows: np.ndarray,
                     batch: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    table = torch.from_numpy(np.ascontiguousarray(dictionary)).to(torch.float32)
    step = min(batch, _chunk_rows(len(dictionary), whitener.grid, len(rows)))
    for start in range(0, len(rows), step):
        take = rows[start:start + step]
        yield _encode_block(images[take], table, whitener, pool_grid), take


def _sparse_arm(corpus: dict[str, np.ndarray], dictionary: np.ndarray,
                whitener: Whitener, pool_grid: int, order: np.ndarray,
                penalties: list[float], classes: int, validation_rows: int
                ) -> dict[str, Any]:
    """One sparse generalist. Two models from one accumulator, as in M104.

    ``validation_rows`` is non-zero for exactly one arm in the whole run — the
    generalist at the smallest budget, where section 7.14 design item 5 says
    the head constant is chosen. Every other arm fits on all of its rows and
    inherits that constant.
    """
    width = pool_grid * pool_grid * len(dictionary)
    accumulator = RidgeAccumulator(width, classes)
    selection_rows = len(order) - validation_rows
    fitted = order[:selection_rows]
    for block, rows in _sparse_features(corpus["train_images"], dictionary,
                                        whitener, pool_grid, fitted, 4096):
        accumulator.add(block, corpus["train_labels"][rows])

    validation: dict[str, float] = {}
    if validation_rows > 0:
        standardise = accumulator.standardiser()
        selection = accumulator.solve_many(penalties)
        hits = {penalty: 0 for penalty in penalties}
        held = order[selection_rows:]
        for block, rows in _sparse_features(corpus["train_images"], dictionary,
                                            whitener, pool_grid, held, 4096):
            standardised = standardise(block)
            for penalty in penalties:
                hits[penalty] += int(
                    _score(selection[penalty], standardised,
                           corpus["train_labels"][rows]).sum()
                )
            accumulator.add(block, corpus["train_labels"][rows])
        validation = {
            str(penalty): hits[penalty] / validation_rows for penalty in penalties
        }

    test_order = np.arange(len(corpus["test_labels"]))

    def _test_blocks() -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for block, rows in _sparse_features(corpus["test_images"], dictionary,
                                            whitener, pool_grid, test_order,
                                            4096):
            yield block, corpus["test_labels"][rows], corpus["test_domains"][rows]

    result = _solve_and_score(accumulator, penalties, _test_blocks())
    result["validation_accuracy_by_penalty"] = validation
    result["selection_fit_rows"] = int(selection_rows)
    return result


def _mixture_arm(corpus: dict[str, np.ndarray], dictionaries: list[np.ndarray],
                 whitener: Whitener, pool_grid: int, order: np.ndarray,
                 penalties: list[float], classes: int) -> dict[str, Any]:
    """Six uniform experts under ORACLE routing. Section 7.10 restriction 4.

    Every expert holds the same number of atoms, so the row-weighted atom sum
    equals that number and the mixture is MAC-matched to the generalist at the
    same budget while holding six times the parameters. That asymmetry is
    M104's own arm (a) versus arm (c1) and is reported, not matched away.
    """
    experts: list[dict[str, Any]] = []
    correct = {penalty: 0 for penalty in penalties}
    rows_total = 0
    for domain, dictionary in enumerate(dictionaries):
        train_rows = order[corpus["train_domains"][order] == domain]
        test_rows = np.flatnonzero(corpus["test_domains"] == domain)
        width = pool_grid * pool_grid * len(dictionary)
        accumulator = RidgeAccumulator(width, classes)
        for block, rows in _sparse_features(corpus["train_images"], dictionary,
                                            whitener, pool_grid, train_rows,
                                            4096):
            accumulator.add(block, corpus["train_labels"][rows])
        standardise = accumulator.standardiser()
        solutions = accumulator.solve_many(penalties)
        hits = {penalty: 0 for penalty in penalties}
        for block, rows in _sparse_features(corpus["test_images"], dictionary,
                                            whitener, pool_grid, test_rows,
                                            4096):
            standardised = standardise(block)
            for penalty in penalties:
                hits[penalty] += int(
                    _score(solutions[penalty], standardised,
                           corpus["test_labels"][rows]).sum()
                )
        for penalty in penalties:
            correct[penalty] += hits[penalty]
        rows_total += len(test_rows)
        experts.append({
            "domain": domain,
            "atoms": int(len(dictionary)),
            "fit_rows": int(accumulator.rows),
            "test_rows": int(len(test_rows)),
            "rows_per_fitted_dimension": accumulator.rows / width,
            "correct_by_penalty": {str(k): int(v) for k, v in hits.items()},
        })
    return {
        "routing": "oracle",
        "experts": experts,
        "test_rows": int(rows_total),
        "correct_by_penalty": {str(k): int(v) for k, v in correct.items()},
        "accuracy_by_penalty": {
            str(k): v / rows_total for k, v in correct.items()
        },
        "rows_per_fitted_dimension": min(
            e["rows_per_fitted_dimension"] for e in experts
        ),
    }


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def run_m107(config_path: Path, output_dir: Path, progress: bool = True
             ) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    # A config that names itself inadmissible must never be able to leave its
    # numbers where the sealed run's are read from. This is not hypothetical:
    # the smoke config was run with --config but without --output and wrote a
    # 2,760-row evidence.json straight into the sealed directory, where the
    # verifier block written later would have read it as the milestone.
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible as evidence (_smoke_note), and would write to the "
            f"SEALED output directory {DEFAULT_OUTPUT}. Pass --output with a "
            "separate directory. Section 11.2 item 23: no figure from an "
            "inadmissible configuration may sit where an operand is read from."
        )

    torch.set_num_threads(config["numerics"]["torch_threads"])
    configure_external_cache_environment()

    corpus_config = config["corpus"]
    representation = config["representation"]
    pool_grid = representation["pool_grid"]
    patch = representation["patch"]
    stride = representation["stride"]
    size = corpus_config["image_size"]
    penalties = [float(p) for p in config["head"]["regularisation_grid"]]

    print("loading DomainNet at 32x32", flush=True)
    raw = _load_domainnet(size)
    train_index = _class_subsample(
        raw["train_labels"], corpus_config["train_rows_per_class"],
        corpus_config["subsample_seed"],
    )
    test_index = _class_subsample(
        raw["test_labels"], corpus_config["test_rows_per_class"],
        corpus_config["subsample_seed"],
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
    digest = _index_digest({"train": train_index, "test": test_index})
    expected = corpus_config.get("expected_subsample_sha256")
    if expected and expected != digest:
        raise RuntimeError(
            f"M107 subsample digest {digest} does not match the pinned "
            f"{expected}; the shared subsample changed and no arm's figure is "
            "comparable to any other arm's."
        )
    classes = int(corpus["train_labels"].max()) + 1
    print(f"  train {len(train_index)}  test {len(test_index)}  "
          f"classes {classes}  digest {digest[:16]}", flush=True)

    identity_checks = [
        _verify_pixel_identity(
            split, index, corpus[f"{split}_images"], size,
            corpus_config["pixel_identity_rows"],
        )
        for split, index in (("train", train_index), ("test", test_index))
    ]
    print(f"  pixel identity verified bitwise on "
          f"{identity_checks[0]['rows_checked']} train and "
          f"{identity_checks[1]['rows_checked']} test rows", flush=True)

    order = np.random.default_rng(corpus_config["shuffle_seed"]).permutation(
        len(train_index)
    )
    validation_rows = int(round(
        len(order) * config["head"]["selection_validation_fraction"]
    ))

    # ---- the sparse side, which also fixes the head constant -------------
    print("fitting the shared whitener on TRAIN patches only", flush=True)
    rng = np.random.default_rng(representation["zca_fit_seed"])
    sample_images = corpus["train_images"][
        rng.choice(len(corpus["train_images"]),
                   min(len(corpus["train_images"]), 20_000), replace=False)
    ]
    patches = _extract_patches(sample_images, patch, stride)
    grid = (size - patch) // stride + 1
    take = min(representation["zca_fit_patches"], len(patches))
    patch_pool = _contrast_normalise(
        patches[rng.choice(len(patches), take, replace=False)],
        representation["contrast_epsilon"],
    )
    mean, whiten = _fit_zca(patch_pool, representation["zca_epsilon"])
    whitener = Whitener(patch, stride, representation["contrast_epsilon"],
                        mean, whiten, grid)
    dimension = patch * patch * 3
    del patches, sample_images

    seed = config["sparse"]["dictionary_seed"]
    pool_size = config["sparse"]["candidate_pool_size"]
    seed_rng = np.random.default_rng(seed)
    candidates = ((patch_pool[
        seed_rng.choice(len(patch_pool), pool_size, replace=False)
    ] - mean) @ whiten).astype(np.float32)
    generalist_order = np.random.default_rng([seed, 100]).permutation(pool_size)
    expert_orders = [
        np.random.default_rng([seed, d]).permutation(pool_size) for d in range(6)
    ]

    budgets = list(config["sparse"]["budgets"])
    floor = config["head"]["fit_samples_per_fitted_dimension_floor"]
    arms: dict[str, Any] = {}
    partial = output_dir / "partial_arms.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    def _record(name: str, payload: dict[str, Any]) -> None:
        # Section 5.3 is never waived, and reporting a violation is not
        # enforcing one. M104's execution-time amendment 5 exists because a cap
        # that was computed but not enforced voided nothing; the same mistake
        # is not available here. An arm below the floor is VOID, not negative,
        # and the gate must not read it.
        ratio = payload.get("rows_per_fitted_dimension")
        if ratio is not None and ratio < floor:
            payload["void"] = True
            payload["void_reason"] = (
                f"section 5.3: {ratio:.2f} fit rows per fitted dimension is "
                f"below the floor of {floor}. The arm is void, not negative, "
                "and contributes to no kill switch."
            )
            print(f"    VOID: {name} at {ratio:.2f} rows per fitted dimension",
                  flush=True)
        arms[name] = payload
        write_canonical_json(partial, {"arms": arms})

    chosen_penalty: float | None = None
    for position, budget in enumerate(budgets):
        label = f"s_generalist_{budget}"
        print(f"  sparse generalist {budget} atoms", flush=True)
        dictionary = candidates[generalist_order[:budget]]
        selecting = validation_rows if position == 0 else 0
        payload = _sparse_arm(corpus, dictionary, whitener, pool_grid, order,
                              penalties, classes, selecting)
        payload["family"] = "sparse"
        payload["atoms"] = budget
        payload["macs"] = _inference_macs(budget, grid, dimension, pool_grid,
                                          classes)
        if selecting:
            best = max(payload["validation_accuracy_by_penalty"].items(),
                       key=lambda kv: (kv[1], -float(kv[0])))
            chosen_penalty = float(best[0])
            payload["_note"] = (
                "section 7.14 design item 5: the head constant is chosen HERE, "
                "on the sparse side, on held-out train rows, and applied "
                "unchanged to every other arm in both families"
            )
            print(f"    head constant chosen on the sparse side: "
                  f"{chosen_penalty}", flush=True)
        _record(label, payload)
    if chosen_penalty is None:
        raise RuntimeError("M107 instrument failure: no head constant was chosen")
    if arms[f"s_generalist_{budgets[0]}"].get("void"):
        raise RuntimeError(
            "M107 instrument failure: the arm section 7.14 design item 5 "
            "chooses the head constant on is itself below the section 5.3 "
            "floor, so the constant every other arm inherits was chosen on an "
            "inadmissible fit."
        )

    for budget in config["sparse"].get("mixture_budgets", budgets):
        label = f"s_mixture_{budget}"
        print(f"  sparse mixture (oracle) {budget} atoms per expert", flush=True)
        payload = _mixture_arm(
            corpus, [candidates[o[:budget]] for o in expert_orders],
            whitener, pool_grid, order, penalties, classes,
        )
        payload["family"] = "sparse"
        payload["atoms"] = budget
        payload["macs"] = _inference_macs(budget, grid, dimension, pool_grid,
                                          classes)
        payload["parameter_note"] = (
            "MAC-matched to the generalist at the same budget and holding six "
            "times the atoms; the parameter excess is reported, not matched away"
        )
        _record(label, payload)

    # ---- the dense side --------------------------------------------------
    dense_config = config["dense"]
    resolutions = sorted({
        int(r) for arm in dense_config["arms"] if arm["pixels"] == "original"
        for r in [arm["resolution"]]
    })
    tag = digest[:16]
    train_pixels = _materialise_original("train", train_index, resolutions, tag)
    test_pixels = _materialise_original("test", test_index, resolutions, tag)

    encoders: dict[str, DenseEncoder] = {}
    for arm in dense_config["arms"]:
        label = arm["name"]
        model = arm["model"]
        resolution = int(arm["resolution"])
        print(f"  dense {label}: dinov2-{model} at {resolution}", flush=True)
        if model not in encoders:
            encoders[model] = DenseEncoder(model, config["numerics"]["onnx_threads"])
        encoder = encoders[model]
        if arm["pixels"] == "original":
            train_source = np.load(train_pixels[resolution], mmap_mode="r")
            test_source = np.load(test_pixels[resolution], mmap_mode="r")
        else:
            train_source = _Upsampled(corpus["train_images"], resolution)
            test_source = _Upsampled(corpus["test_images"], resolution)
        payload = _dense_arm(encoder, train_source, test_source, corpus,
                             penalties, dense_config["batch"], classes, progress)
        payload["family"] = "dense"
        payload["model"] = model
        payload["resolution"] = resolution
        payload["pixels"] = arm["pixels"]
        payload["geometry"] = encoder.geometry
        payload["macs"] = _transformer_macs(encoder.geometry, resolution, classes)
        payload["disclosure"] = arm["disclosure"]
        _record(label, payload)

    evidence = {
        "milestone": "M107",
        "question": (
            "section 3.2 Q2: how does test accuracy per inference MAC compare "
            "between a frozen sparse dictionary code and a frozen dense "
            "transformer feature, on one corpus under one protocol?"
        ),
        "registered_in": "analysis/RESEARCH_IMPLEMENTATION_PLAN_v15.md section 7.14",
        "admissible_as_evidence": not inadmissible,
        "config_file": Path(config_path).name,
        "config": config,
        "corpus": {
            "train_rows": int(len(train_index)),
            "test_rows": int(len(test_index)),
            "classes": classes,
            "subsample_sha256": digest,
            "pixel_identity": identity_checks,
            "train_rows_per_domain": np.bincount(
                corpus["train_domains"], minlength=6).tolist(),
            "test_rows_per_domain": np.bincount(
                corpus["test_domains"], minlength=6).tolist(),
        },
        "head": {
            "chosen_penalty": chosen_penalty,
            "chosen_on": f"s_generalist_{budgets[0]}",
            "fit_samples_per_fitted_dimension_floor": floor,
            "_rule": (
                "section 7.14 design item 5 and N107.1: chosen once on the "
                "SPARSE side and applied unchanged to every arm. The full grid "
                "is recorded per arm as sensitivity only; reading the best "
                "constant per arm off it would be choosing the answer"
            ),
        },
        "asymmetries": {
            "training_data": (
                "DINOv2 is pre-trained on LVD-142M, 142 million curated images; "
                "the sparse dictionary is drawn from this corpus's own train "
                "patches and nothing else. Favours dense. Section 7.14 "
                "restriction 4 requires this in every sentence"
            ),
            "resolution": (
                "arms d1-d4 read original-resolution images; the sparse arms "
                "read the 32x32 downsample. Favours dense. Arm d5 measures it"
            ),
            "routing": (
                "the sparse mixture is scored under ORACLE routing, an upper "
                "bound. Favours sparse. Section 7.10 restriction 4"
            ),
            "subsample": (
                "138,000 train rows, a third of DomainNet's train split. "
                "Affects both families identically; no M107 accuracy may be "
                "quoted as this program's best. Section 7.14 restriction 8"
            ),
        },
        "arms": arms,
    }
    evidence["gate"] = _build_gate(evidence)
    evidence["payload_sha256"] = payload_hash(evidence)
    path = output_dir / "evidence.json"
    write_canonical_json(path, evidence)
    build_artifact_index(output_dir)
    return evidence


class _Upsampled:
    """A read-only view of 32x32 tensors at a transformer's resolution.

    Arm (d5) only. Materialising 172,500 upsampled 224s would be 26 GB, and the
    dense encoder consumes them in batches anyway, so the upsample happens on
    the slice.
    """

    def __init__(self, images: np.ndarray, resolution: int):
        self.images = images
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, key: Any) -> np.ndarray:
        return _upsample(self.images[key], self.resolution)


def _curve(evidence: dict[str, Any], family: str, penalty: float,
           prefix: str | None = None) -> list[list[float]]:
    """Accuracy against analytic MACs for one family, sorted by MACs.

    Voided arms are excluded. A voided arm is void, not zero: leaving it in at
    whatever accuracy it happened to reach would let a section 5.3 violation
    contribute to a kill switch.
    """
    points = []
    for name, arm in evidence["arms"].items():
        if arm["family"] != family or arm.get("void"):
            continue
        if prefix is not None and not name.startswith(prefix):
            continue
        points.append([float(arm["macs"]["total"]),
                       arm["accuracy_by_penalty"][str(penalty)]])
    return sorted(points)


def _build_gate(evidence: dict[str, Any]) -> dict[str, Any]:
    """The three registered kill switches of section 7.14, evaluated here.

    A kill switch is evaluated by the instrument and not by whoever writes the
    result up, so the write-up cannot decide it did not fire.
    """
    penalty = evidence["head"]["chosen_penalty"]
    dense = _curve(evidence, "dense", penalty)
    sparse = _curve(evidence, "sparse", penalty)
    generalist = _curve(evidence, "sparse", penalty, "s_generalist")
    mixture = _curve(evidence, "sparse", penalty, "s_mixture")

    overlap: list[list[float]] = []
    bounds = None
    if dense and sparse:
        low = max(dense[0][0], sparse[0][0])
        high = min(dense[-1][0], sparse[-1][0])
        bounds = [low, high]
        overlap = [[m, a] for m, a in sparse if low <= m <= high]

    def _dense_at_or_below(macs: float) -> float | None:
        below = [a for m, a in dense if m <= macs]
        return max(below) if below else None

    comparable = [
        (m, a, _dense_at_or_below(m)) for m, a in overlap
        if _dense_at_or_below(m) is not None
    ]
    # An empty overlap is NOT a quiet "not fired". It means the two ladders
    # never meet on the MAC axis, so M107 has measured nothing about which
    # family is more efficient, and section 5.10's rule -- a failing arm is
    # void, not negative -- applies to the comparison itself.
    admissible = len(comparable) > 0
    dominated = [reference > accuracy for _, accuracy, reference in comparable]
    sparse_wins = [[m, a] for m, a, reference in comparable if a >= reference]

    matched = dict(
        (g[0], (g[1], x[1])) for g, x in zip(generalist, mixture)
        if g[0] == x[0]
    )
    generalist_beats_mixture = [
        macs for macs, (g, x) in matched.items() if g > x
    ]

    return {
        "comparison_admissible": admissible,
        "_admissibility_note": (
            "kill switches 1 and 2 are decidable only where the two ladders "
            "overlap on the MAC axis. If they do not overlap, M107 is VOID for "
            "those two switches and may not be reported as either supporting "
            "or refuting section 3.2 Q2; it is not a 'not fired'"
        ),
        "overlap_macs": bounds,
        "overlap_points": len(comparable),
        "voided_arms": sorted(
            name for name, arm in evidence["arms"].items() if arm.get("void")
        ),
        "kill_switch_1_dense_dominates_everywhere": {
            "fired": bool(admissible and all(dominated)),
            "decidable": admissible,
            "consequence": (
                "section 3.2 Q2's efficiency claim is REFUTED at this scale on "
                "this corpus; the admissible reading of M104-M106 collapses to "
                "'efficiency relative to a uniform mixture'. Section 7.14 kill "
                "switch 1: this is the headline under section 11.1 and may not "
                "be reported as a footnote, a limitation, or future work"
            ),
        },
        "kill_switch_2_sparse_wins_somewhere": {
            "fired": bool(admissible and sparse_wins),
            "decidable": admissible,
            "points": sparse_wins,
            "consequence": (
                "the admissible claim is bounded to those budgets, this corpus "
                "and the accuracy quoted beside them, and still carries the "
                "LVD-142M and oracle-routing caveats. A crossing at an accuracy "
                "nobody would deploy is not an efficiency result"
            ),
        },
        "kill_switch_3_generalist_beats_mixture": {
            "fired": bool(generalist_beats_mixture),
            "decidable": bool(matched),
            "budgets": generalist_beats_mixture,
            "compared_at_macs": sorted(matched),
            "_narrowing_note": (
                "section 7.14 execution-time amendment 5: the section 5.3 floor "
                "caps a mixture expert at 280 atoms on this subsample, because "
                "the smallest domain holds 11,224 of the 138,000 train rows. The "
                "mixture ladder therefore runs at 128 and 256 atoms only, and "
                "this switch is decidable at TWO budgets instead of six. Every "
                "sentence reporting this switch must say so. The generalist "
                "ladder is unaffected and spans the full range"
            ),
            "consequence": (
                "M104's partition is not buying anything here either; M107 must "
                "say so beside M104's own kill switch 3 rather than leaving the "
                "two unreconciled"
            ),
        },
        "_comparison_rule": (
            "a sparse point at M MACs is compared against the best dense point "
            "at or below M MACs; comparing it against a dense point above it "
            "would let the sparse side win by being cheaper"
        ),
        "curves": {"dense": dense, "sparse_generalist": generalist,
                   "sparse_mixture": mixture},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    evidence = run_m107(args.config, args.output, progress=not args.quiet)
    gate = evidence["gate"]
    print("\ngate:", flush=True)
    if not gate["comparison_admissible"]:
        print("  THE LADDERS DO NOT OVERLAP ON THE MAC AXIS. M107 has measured "
              "nothing about\n  which family is more efficient. Switches 1 and 2 "
              "are VOID, not negative.", flush=True)
    for key, value in gate.items():
        if isinstance(value, dict) and "fired" in value:
            if not value.get("decidable", True):
                verdict = "VOID (undecidable -- NOT a negative result)"
            else:
                verdict = "FIRED" if value["fired"] else "not fired"
            print(f"  {key}: {verdict}", flush=True)
    if gate["voided_arms"]:
        print(f"  arms voided by the section 5.3 floor: "
              f"{', '.join(gate['voided_arms'])}", flush=True)


if __name__ == "__main__":
    main()
