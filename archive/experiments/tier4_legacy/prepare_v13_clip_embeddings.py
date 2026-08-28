"""Embed every v13 corpus image and the sealed vocabulary with CLIP.

Amendment R9 requires CLIP to enter M82 as a hashed input artifact produced
outside the frozen replay environment, in exactly the way the DINOv2 backbone
enters as a frozen ONNX graph rather than as a live model. This script is that
producer. It runs in ``.venv-rocm`` on the RX 9070 XT; the measurement that
consumes its output runs in the frozen ``.venv`` and never imports CLIP.

Two design choices are worth stating, because both were available in a cheaper
form and the cheaper form would have been wrong.

*Every corpus image is embedded, not just the exemplars of some dictionary.*
Exemplar sets depend on the dictionary, hence on the seed, and R9's promoted
primary operand is stability under exemplar resampling. An artifact built from
one seed's exemplars could not answer the question it exists to serve, and
rebuilding it per resampling would make the artifact a function of the analysis
it feeds.

*Images are fetched by (source_file, source_row) and then checked.* The stored
``image_path`` of every fetched row is compared against the manifest's. An
off-by-one in row-group indexing would otherwise embed the wrong images and
leave no trace: naming would still emit plausible words, merely for the wrong
atoms.

Run with::

    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.prepare_v13_clip_embeddings
"""

from __future__ import annotations

import argparse
import io
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v13" / "m82_clip_embeddings.json"
)


def _resolve_inside_repo(path: str) -> Path:
    resolved = (REPO_ROOT / Path(path)).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError("M82 output paths must remain inside the repository")
    return resolved


# --------------------------------------------------------------------------
# Preprocessing. Written out rather than loaded, because the cached snapshot
# carries no preprocessor_config.json and because a sealed artifact should
# define its own transform instead of inheriting a library default.
# --------------------------------------------------------------------------


def preprocess_image(image: Any, settings: dict[str, Any]) -> np.ndarray:
    from PIL import Image

    resample = {
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
    }[settings["resample"]]

    image = image.convert(settings["convert_mode"])
    target = int(settings["resize_shortest_side"])
    width, height = image.size
    if width <= height:
        new_width, new_height = target, max(target, round(height * target / width))
    else:
        new_height, new_width = target, max(target, round(width * target / height))
    image = image.resize((new_width, new_height), resample)

    crop = int(settings["center_crop"])
    left = (new_width - crop) // 2
    top = (new_height - crop) // 2
    image = image.crop((left, top, left + crop, top + crop))

    array = np.asarray(image, dtype=np.float32) * float(settings["rescale"])
    array = (array - np.asarray(settings["mean"], dtype=np.float32)) / np.asarray(
        settings["std"], dtype=np.float32
    )
    return np.transpose(array, (2, 0, 1))


# --------------------------------------------------------------------------
# Corpus image access
# --------------------------------------------------------------------------


def _row_group_offsets(parquet_file: Any) -> list[int]:
    offsets = [0]
    metadata = parquet_file.metadata
    for index in range(metadata.num_row_groups):
        offsets.append(offsets[-1] + metadata.row_group(index).num_rows)
    return offsets


def iterate_corpus_images(
    manifest_rows: list[dict[str, Any]],
    parquet_root: Path,
    *,
    verify_paths: bool,
) -> Iterator[tuple[int, Any]]:
    """Yield ``(corpus_row, PIL image)`` for every manifest row.

    Rows are grouped by shard and read one row group at a time, so peak memory
    is a single row group rather than a whole shard. DomainNet's row groups
    hold 100 rows each.
    """
    import pyarrow.parquet as pq
    from PIL import Image

    shards = {path.name: path for path in parquet_root.rglob("*.parquet")}

    by_shard: defaultdict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for corpus_row, row in enumerate(manifest_rows):
        by_shard[row["source_file"]].append(
            (int(row["source_row"]), corpus_row, row["image_path"])
        )

    for shard_name in sorted(by_shard):
        if shard_name not in shards:
            raise FileNotFoundError(f"Manifest references missing shard {shard_name}")
        parquet_file = pq.ParquetFile(shards[shard_name])
        offsets = _row_group_offsets(parquet_file)

        wanted: defaultdict[int, list[tuple[int, int, str]]] = defaultdict(list)
        for source_row, corpus_row, image_path in by_shard[shard_name]:
            group = int(np.searchsorted(offsets, source_row, side="right") - 1)
            wanted[group].append((source_row - offsets[group], corpus_row, image_path))

        for group in sorted(wanted):
            table = parquet_file.read_row_group(group, columns=["image", "image_path"])
            images = table.column("image").to_pylist()
            stored_paths = table.column("image_path").to_pylist()
            for local_row, corpus_row, expected_path in wanted[group]:
                if verify_paths and stored_paths[local_row] != expected_path:
                    raise ValueError(
                        "Parquet row does not match the manifest: "
                        f"{shard_name} row {local_row} in group {group} holds "
                        f"{stored_paths[local_row]!r}, manifest expects {expected_path!r}"
                    )
                yield corpus_row, Image.open(io.BytesIO(images[local_row]["bytes"]))


# --------------------------------------------------------------------------
# CLIP
# --------------------------------------------------------------------------


def load_clip(config: dict[str, Any], device: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import CLIPModel, CLIPTokenizerFast

    snapshot = Path(config["model"]["snapshot_dir"]).expanduser()
    if not snapshot.is_dir():
        raise FileNotFoundError(
            f"CLIP snapshot not found at {snapshot}. R9 requires the cached "
            "checkpoint; nothing here may download one."
        )
    model = CLIPModel.from_pretrained(
        snapshot, dtype=torch.float32, local_files_only=True
    ).eval()
    tokenizer = CLIPTokenizerFast.from_pretrained(snapshot, local_files_only=True)
    return model.to(device), tokenizer, torch


def _prompt(name: str, template: str) -> str:
    return template.format(name.replace("_", " "))


def embed_vocabulary(
    model: Any,
    tokenizer: Any,
    torch: Any,
    vocabulary: dict[str, Any],
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed every term as the mean of its template ensemble, then normalise.

    This is standard CLIP zero-shot practice: the ensemble is averaged in
    embedding space and the mean is renormalised, so a term is one vector
    rather than a set the caller has to reduce.
    """

    def encode(prompts: list[str]) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                tokens = tokenizer(
                    prompts[start : start + batch_size],
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                features = model.get_text_features(**tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                outputs.append(features.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs)

    def ensemble(names: list[str], templates: list[str]) -> np.ndarray:
        prompts = [_prompt(name, template) for name in names for template in templates]
        embedded = encode(prompts).reshape(len(names), len(templates), -1)
        averaged = embedded.mean(axis=1)
        return averaged / np.linalg.norm(averaged, axis=1, keepdims=True)

    objects = ensemble(
        [term["name"] for term in vocabulary["object_terms"]],
        vocabulary["object_templates"],
    )
    styles = ensemble(
        [term["phrase"] for term in vocabulary["style_terms"]],
        vocabulary["style_templates"],
    )
    return objects.astype(np.float32), styles.astype(np.float32)


def embed_images(
    model: Any,
    torch: Any,
    manifest_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    device: str,
    report_every: int = 8192,
) -> np.ndarray:
    dimension = int(config["model"]["projection_dim"])
    embeddings = np.zeros((len(manifest_rows), dimension), dtype=np.float32)
    written = np.zeros(len(manifest_rows), dtype=bool)

    batch_size = int(config["batch_size"])
    settings = config["preprocessing"]
    parquet_root = Path(config["corpus"]["parquet_root"])

    pending_rows: list[int] = []
    pending_arrays: list[np.ndarray] = []
    started = time.time()
    done = 0

    def flush() -> None:
        nonlocal done
        if not pending_rows:
            return
        batch = torch.from_numpy(np.stack(pending_arrays)).to(device)
        with torch.no_grad():
            features = model.get_image_features(pixel_values=batch)
            features = features / features.norm(dim=-1, keepdim=True)
        embeddings[pending_rows] = features.cpu().numpy().astype(np.float32)
        written[pending_rows] = True
        done += len(pending_rows)
        pending_rows.clear()
        pending_arrays.clear()

    for corpus_row, image in iterate_corpus_images(
        manifest_rows,
        parquet_root,
        verify_paths=bool(config["integrity"]["verify_image_path_matches_manifest"]),
    ):
        pending_rows.append(corpus_row)
        pending_arrays.append(preprocess_image(image, settings))
        if len(pending_rows) >= batch_size:
            flush()
            if done % report_every < batch_size:
                rate = done / max(time.time() - started, 1e-9)
                print(f"  embedded {done}/{len(manifest_rows)}  {rate:.0f} img/s")
    flush()

    if not written.all():
        raise ValueError(f"{int((~written).sum())} corpus rows were never embedded")
    return embeddings


def cpu_agreement_control(
    config: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    gpu_embeddings: np.ndarray,
    objects: np.ndarray,
) -> dict[str, Any]:
    """Re-embed a registered sample on the CPU and compare.

    The v13 corpus builder found the DirectML provider disagreeing with CPU by
    11.7 percent relative, which is why the DINOv2 extraction is CPU-locked.
    That precedent makes accelerator agreement something this program measures
    rather than assumes.
    """
    settings = config["cpu_agreement_control"]
    generator = np.random.default_rng(int(settings["draw_seed"]))
    sample = np.sort(
        generator.choice(len(manifest_rows), size=int(settings["sample_rows"]), replace=False)
    )

    model, _, torch = load_clip(config, "cpu")
    sampled_rows = [manifest_rows[int(row)] for row in sample]
    cpu_embeddings = embed_images(
        model, torch, sampled_rows, config, device="cpu", report_every=1 << 30
    )
    del model

    gpu_sample = gpu_embeddings[sample]
    cosine = np.sum(gpu_sample * cpu_embeddings, axis=1)
    gpu_terms = np.argmax(gpu_sample @ objects.T, axis=1)
    cpu_terms = np.argmax(cpu_embeddings @ objects.T, axis=1)
    agreement = float(np.mean(gpu_terms == cpu_terms))

    deviation = float(np.max(1.0 - cosine))
    return {
        "sample_rows": int(len(sample)),
        "draw_seed": int(settings["draw_seed"]),
        "minimum_cosine": float(np.min(cosine)),
        "mean_cosine": float(np.mean(cosine)),
        "max_cosine_deviation": deviation,
        "tolerance": float(settings["max_cosine_deviation"]),
        "nearest_term_agreement": agreement,
        "passes": bool(
            deviation <= float(settings["max_cosine_deviation"]) and agreement == 1.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal the M82 CLIP embeddings.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Embed only the first N corpus rows. For smoke runs; the artifact is not sealed.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    arguments = parser.parse_args()

    from experiments.common.v5_artifacts import (
        build_artifact_index,
        payload_hash,
        sha256_file,
        write_canonical_json,
    )

    config = json.loads(arguments.config.read_text(encoding="utf-8"))

    vocabulary_path = _resolve_inside_repo(config["vocabulary"]["path"])
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    if vocabulary["vocabulary_hash"] != config["vocabulary"]["expected_hash"]:
        raise ValueError(
            "Vocabulary hash mismatch. The vocabulary was sealed before naming "
            "and may not change underneath this artifact."
        )

    manifest_path = _resolve_inside_repo(config["corpus"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_rows = manifest["selection"]
    if len(manifest_rows) != config["corpus"]["expected_rows"]:
        raise ValueError(
            f"Expected {config['corpus']['expected_rows']} corpus rows, "
            f"manifest holds {len(manifest_rows)}"
        )
    if arguments.limit:
        manifest_rows = manifest_rows[: arguments.limit]

    print(f"loading CLIP on {arguments.device}")
    model, tokenizer, torch = load_clip(config, arguments.device)

    print("embedding vocabulary")
    objects, styles = embed_vocabulary(
        model,
        tokenizer,
        torch,
        vocabulary,
        device=arguments.device,
        batch_size=int(config["text_batch_size"]),
    )
    print(f"  objects {objects.shape}  styles {styles.shape}")

    print(f"embedding {len(manifest_rows)} corpus images")
    started = time.time()
    images = embed_images(
        model, torch, manifest_rows, config, device=arguments.device
    )
    elapsed = time.time() - started
    print(f"  done in {elapsed / 60:.1f} min ({len(manifest_rows) / elapsed:.0f} img/s)")

    del model
    if arguments.device.startswith("cuda"):
        torch.cuda.empty_cache()

    if arguments.limit:
        print("smoke run: artifact not sealed")
        return

    print("running the CPU agreement control")
    control = cpu_agreement_control(config, manifest_rows, images, objects)
    print(
        f"  max cosine deviation {control['max_cosine_deviation']:.2e} "
        f"(tolerance {control['tolerance']:.0e}), "
        f"nearest-term agreement {control['nearest_term_agreement']:.4f}"
    )
    if not control["passes"]:
        raise ValueError(
            "The GPU does not agree with the CPU within the registered tolerance. "
            "Per the config's gating clause the artifact is not sealed."
        )

    output_dir = _resolve_inside_repo(config["output_dir"])
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    np.save(arrays_dir / "image_embeddings.npy", images)
    np.save(arrays_dir / "text_object_embeddings.npy", objects)
    np.save(arrays_dir / "text_style_embeddings.npy", styles)

    evidence = {
        "schema_version": config["schema_version"],
        "artifact": config["artifact"],
        "milestone": "M82",
        "produced_outside_frozen_venv": True,
        "device": arguments.device,
        "dtype": config["precision"]["dtype"],
        "model": config["model"],
        "preprocessing": config["preprocessing"],
        "rows": int(images.shape[0]),
        "dimension": int(images.shape[1]),
        "object_terms": int(objects.shape[0]),
        "style_terms": int(styles.shape[0]),
        "vocabulary_hash": vocabulary["vocabulary_hash"],
        "corpus_index_sha256": sha256_file(
            _resolve_inside_repo(config["corpus"]["index_path"])
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "elapsed_seconds": round(elapsed, 1),
        "images_per_second": round(len(manifest_rows) / elapsed, 1),
        "cpu_agreement_control": control,
        "embedding_hash": payload_hash(
            {
                "images": [round(float(value), 6) for value in images[:: max(1, len(images) // 512)].ravel()[:4096]],
                "objects": [round(float(value), 6) for value in objects.ravel()[:4096]],
            }
        ),
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"sealed to {output_dir}")


if __name__ == "__main__":
    main()
