"""Extract bounded Oxford Flowers-102 features for M19 identity spaces."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from torchvision.datasets import Flowers102

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.experiment_manifest import canonical_json
from experiments.common.v5_frozen_representations import (
    FeatureCacheMetadata,
    RepresentationManifest,
    compute_objective_hash,
    compute_preprocessing_digest,
    compute_split_hash,
)
from experiments.tier4.prepare_v5_frozen_features import (
    cache_features,
    extract_features_batch,
)


def _image_id(path: Path) -> int:
    match = re.fullmatch(r"image_(\d+)\.jpg", path.name)
    if match is None:
        raise ValueError(f"Unexpected Flowers-102 image name: {path.name}.")
    return int(match.group(1))


def _select_split(
    dataset: Flowers102,
    per_class: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    labels = np.asarray(dataset._labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in range(102):
        candidates = np.flatnonzero(labels == class_id)
        if len(candidates) < per_class:
            raise ValueError(
                f"Flowers class {class_id} has {len(candidates)} samples; "
                f"{per_class} required."
            )
        selected.extend(rng.choice(candidates, per_class, replace=False).tolist())
    selected.sort(key=lambda index: _image_id(dataset._image_files[index]))
    images = [np.asarray(dataset[index][0].convert("RGB")) for index in selected]
    image_ids = np.asarray(
        [_image_id(dataset._image_files[index]) for index in selected],
        dtype=np.int64,
    )
    return images, labels[selected], image_ids


def _reuse_cache(
    cache_path: Path,
    *,
    labels: np.ndarray,
    image_ids: np.ndarray,
    representation_hash: str,
    output_dimension: int,
    split: str,
) -> FeatureCacheMetadata | None:
    """Reuse a complete cache only when its full binding still matches."""
    if not cache_path.exists():
        return None
    with np.load(cache_path) as cache:
        features = cache["features"]
        cached_labels = cache["labels"]
        cached_ids = cache["indices"]
    if (
        features.shape != (len(labels), output_dimension)
        or not np.isfinite(features).all()
        or not np.array_equal(cached_labels, labels)
        or not np.array_equal(cached_ids, image_ids)
    ):
        return None
    return FeatureCacheMetadata(
        representation_hash=representation_hash,
        feature_file_hash=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        n_samples=len(labels),
        feature_dimension=output_dimension,
        split_name=split,
    )


def run_extraction(
    config_path: str | Path = "experiments/configs/v5/m19_flowers102_s1.json",
    output_dir: str | Path = "data/v5/features/m19_flowers102_s1",
) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    backbone_config = json.loads(
        (PROJECT_ROOT / config["backbone_config"]).read_text(encoding="utf-8")
    )
    datasets = {
        split: Flowers102(
            root=PROJECT_ROOT / config["dataset_root"],
            split=official_split,
        )
        for split, official_split in config["official_split_mapping"].items()
    }
    selected = {
        split: _select_split(
            dataset,
            config["samples_per_class"][split],
            config["seed"] + offset,
        )
        for offset, (split, dataset) in enumerate(datasets.items())
    }
    split_hashes = {
        split: compute_split_hash(image_ids)
        for split, (_, _, image_ids) in selected.items()
    }
    results: dict[str, Any] = {
        "dataset": config["dataset"],
        "split_hashes": split_hashes,
        "representations": {},
    }
    output_dir = Path(output_dir)

    for backbone in backbone_config["backbones"]:
        if backbone.get("status") == "blocked":
            continue
        backbone_id = backbone["id"]
        model_path = PROJECT_ROOT / backbone["onnx_path"]
        model_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if model_digest != backbone["weights_sha256"]:
            raise ValueError(f"Backbone weights hash mismatch for {backbone_id}.")
        preprocessor_path = PROJECT_ROOT / backbone["preprocessor_config"]
        preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        manifest = RepresentationManifest(
            backbone_id=backbone_id,
            upstream_weights_digest=model_digest,
            preprocessing_digest=compute_preprocessing_digest(preprocessor_path),
            interface_architecture="identity",
            interface_weights_digest="none",
            objective_hash=compute_objective_hash((0.0, 0.0, 0.0), "identity"),
            training_split_hash=split_hashes["train"],
            output_dimension=backbone["output_dimension"],
            checkpoint_source=backbone["checkpoint_source"],
            checkpoint_license=backbone["checkpoint_license"],
            token_pooling_policy=backbone["token_pooling_policy"],
        )
        cache_metadata = {}
        for split, (images, labels, image_ids) in selected.items():
            cache_path = (
                output_dir
                / backbone_id
                / f"features_{split}_{manifest.representation_hash[:16]}.npz"
            )
            existing = _reuse_cache(
                cache_path,
                labels=labels,
                image_ids=image_ids,
                representation_hash=manifest.representation_hash,
                output_dimension=backbone["output_dimension"],
                split=split,
            )
            if existing is not None:
                print(f"Reusing {backbone_id}/{split}: {len(images)} images.")
                cache_metadata[split] = existing.to_dict()
                continue
            print(f"Extracting {backbone_id}/{split}: {len(images)} images...")
            features = extract_features_batch(
                images,
                backbone_id,
                str(model_path),
                preprocessor,
                backbone["token_pooling_policy"],
                batch_size=backbone.get("extraction_batch_size", 32),
            )
            cache_metadata[split] = cache_features(
                features,
                labels,
                image_ids,
                split,
                manifest.representation_hash,
                output_dir / backbone_id,
            ).to_dict()
        results["representations"][backbone_id] = {
            "status": "extracted",
            "representation_hash": manifest.representation_hash,
            "output_dimension": backbone["output_dimension"],
            "manifest": manifest.to_dict(),
            "cache_metadata": cache_metadata,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "extraction_summary.json").write_text(
        canonical_json(results) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return results


if __name__ == "__main__":
    run_extraction()
