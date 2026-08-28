"""Extract and cache frozen backbone features for M19 frozen-space study.

Uses ONNX Runtime CPU with deterministic PIL preprocessing. Features are
keyed by the complete upstream representation hash. Correctly applies
each backbone's preprocessor config (resize, crop, normalize).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.experiment_manifest import canonical_json
from experiments.common.v5_frozen_representations import (
    FeatureCacheMetadata,
    RepresentationManifest,
    compute_preprocessing_digest,
    compute_objective_hash,
    compute_split_hash,
)


# ---------------------------------------------------------------------------
# Deterministic PIL preprocessing
# ---------------------------------------------------------------------------


def _resize_shortest_edge(image: Image.Image, shortest_edge: int) -> Image.Image:
    """Resize so the shortest edge equals the target, using BICUBIC (resample=3)."""
    w, h = image.size
    if h < w:
        new_h = shortest_edge
        new_w = int(round(w * shortest_edge / h))
    else:
        new_w = shortest_edge
        new_h = int(round(h * shortest_edge / w))
    return image.resize((new_w, new_h), Image.BICUBIC)


def _resize_fixed(
    image: Image.Image,
    height: int,
    width: int,
    resample: int = 3,
) -> Image.Image:
    """Resize to exact (width, height) using a declared PIL resampling mode."""
    resampling = {
        2: Image.Resampling.BILINEAR,
        3: Image.Resampling.BICUBIC,
    }
    if resample not in resampling:
        raise ValueError(f"Unsupported PIL resampling mode: {resample}.")
    return image.resize((width, height), resampling[resample])


def _center_crop(image: Image.Image, height: int, width: int) -> Image.Image:
    """Center crop to (height, width)."""
    w, h = image.size
    left = (w - width) // 2
    top = (h - height) // 2
    return image.crop((left, top, left + width, top + height))


def preprocess_image_dinov2(
    image: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """DINOv2 preprocessing: resize shortest edge, center crop, rescale, normalize."""
    img = Image.fromarray(image).convert("RGB")

    # Resize shortest edge
    shortest_edge = config["size"]["shortest_edge"]
    img = _resize_shortest_edge(img, shortest_edge)

    # Center crop
    crop_h = config["crop_size"]["height"]
    crop_w = config["crop_size"]["width"]
    img = _center_crop(img, crop_h, crop_w)

    # To float array and rescale
    arr = np.asarray(img, dtype=np.float64) * config["rescale_factor"]

    # Normalize
    mean = np.array(config["image_mean"], dtype=np.float64)
    std = np.array(config["image_std"], dtype=np.float64)
    arr = (arr - mean) / std

    # HWC -> CHW, add batch dim
    arr = arr.transpose(2, 0, 1)
    return arr.astype(np.float32)


def preprocess_image_siglip(
    image: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """SigLIP preprocessing: resize to fixed size, rescale, normalize."""
    img = Image.fromarray(image).convert("RGB")

    # Resize to fixed size
    height = config["size"]["height"]
    width = config["size"]["width"]
    img = _resize_fixed(img, height, width)

    # To float array and rescale
    arr = np.asarray(img, dtype=np.float64) * config["rescale_factor"]

    # Normalize
    mean = np.array(config["image_mean"], dtype=np.float64)
    std = np.array(config["image_std"], dtype=np.float64)
    arr = (arr - mean) / std

    # HWC -> CHW, add batch dim
    arr = arr.transpose(2, 0, 1)
    return arr.astype(np.float32)


def preprocess_image_ijepa(
    image: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """I-JEPA preprocessing: bilinear fixed resize, rescale, and normalize."""
    img = Image.fromarray(image).convert("RGB")
    height = config["size"]["height"]
    width = config["size"]["width"]
    img = _resize_fixed(img, height, width, resample=config["resample"])

    arr = np.asarray(img, dtype=np.float64) * config["rescale_factor"]
    mean = np.array(config["image_mean"], dtype=np.float64)
    std = np.array(config["image_std"], dtype=np.float64)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1).astype(np.float32)


def _pool_features(features: np.ndarray, policy: str) -> np.ndarray:
    """Apply the representation's declared token pooling policy."""
    if policy == "cls_token":
        if features.ndim != 3:
            raise ValueError("CLS-token pooling requires rank-3 model output.")
        return features[:, 0, :]
    if policy == "mean_patch_tokens":
        if features.ndim != 3:
            raise ValueError("Mean patch-token pooling requires rank-3 model output.")
        return features.mean(axis=1)
    if policy == "pooler_output":
        if features.ndim != 2:
            raise ValueError("Pooler output must have rank 2.")
        return features
    raise ValueError(f"Unknown token pooling policy: {policy}.")


def _select_model_output(
    outputs: dict[str, np.ndarray],
    policy: str,
) -> np.ndarray:
    """Select the named ONNX output required by a pooling policy."""
    output_name = "pooler_output" if policy == "pooler_output" else "last_hidden_state"
    if output_name not in outputs:
        raise ValueError(
            f"Pooling policy {policy!r} requires ONNX output {output_name!r}; "
            f"available outputs: {sorted(outputs)}."
        )
    return outputs[output_name]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features_batch(
    images: np.ndarray,
    backbone_id: str,
    onnx_path: str,
    config: dict[str, Any],
    token_pooling_policy: str,
    batch_size: int = 32,
    session: Any | None = None,
) -> np.ndarray:
    """Extract features for a batch of images using ONNX Runtime.

    Processes images one batch at a time to avoid memory issues.

    Args:
        session: An existing ``onnxruntime.InferenceSession`` to reuse. When
            omitted a session is constructed on the CPU execution provider,
            which is the sealed reference configuration. Callers that stream a
            corpus in many chunks pass a session here to avoid rebuilding the
            graph per chunk; this does not alter the computed features.
    """
    import onnxruntime as ort

    if session is None:
        session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    n = len(images)
    all_features: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_inputs = []
        for i in range(start, end):
            if backbone_id.startswith("dinov2"):
                processed = preprocess_image_dinov2(images[i], config)
            elif backbone_id.startswith("siglip"):
                processed = preprocess_image_siglip(images[i], config)
            elif backbone_id.startswith("ijepa"):
                processed = preprocess_image_ijepa(images[i], config)
            else:
                raise ValueError(f"Unknown backbone: {backbone_id}")
            batch_inputs.append(processed)

        batch_array = np.stack(batch_inputs, axis=0)
        raw_outputs = session.run(output_names, {input_name: batch_array})
        outputs = dict(zip(output_names, raw_outputs, strict=True))
        selected_output = _select_model_output(outputs, token_pooling_policy)
        features = _pool_features(selected_output, token_pooling_policy)
        all_features.append(features.astype(np.float64))

    return np.concatenate(all_features, axis=0)


# ---------------------------------------------------------------------------
# Split creation
# ---------------------------------------------------------------------------


def create_stratified_splits(
    labels: np.ndarray,
    train_per_class: int,
    dev_per_class: int,
    test_per_class: int,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic stratified train/dev/test splits with no leakage."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)

    train_indices: list[int] = []
    dev_indices: list[int] = []
    test_indices: list[int] = []

    for c in classes:
        class_indices = np.where(labels == c)[0]
        needed = train_per_class + dev_per_class + test_per_class
        if len(class_indices) < needed:
            raise ValueError(
                f"Class {c} has {len(class_indices)} samples but needs {needed}."
            )
        selected = rng.choice(class_indices, size=needed, replace=False)
        train_indices.extend(selected[:train_per_class].tolist())
        dev_indices.extend(selected[train_per_class:train_per_class + dev_per_class].tolist())
        test_indices.extend(selected[train_per_class + dev_per_class:].tolist())

    train_idx = np.array(sorted(train_indices), dtype=np.int64)
    dev_idx = np.array(sorted(dev_indices), dtype=np.int64)
    test_idx = np.array(sorted(test_indices), dtype=np.int64)

    # Verify no leakage
    all_idx = np.concatenate([train_idx, dev_idx, test_idx])
    if len(all_idx) != len(set(all_idx.tolist())):
        raise ValueError("Split leakage detected!")

    return train_idx, dev_idx, test_idx


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def cache_features(
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    split_name: str,
    representation_hash: str,
    output_dir: Path,
) -> FeatureCacheMetadata:
    """Save features to NPZ and return metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"features_{split_name}_{representation_hash[:16]}.npz"
    filepath = output_dir / filename

    np.savez_compressed(
        filepath,
        features=features.astype(np.float64),
        labels=labels.astype(np.int64),
        indices=indices.astype(np.int64),
    )

    # Compute file hash
    file_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()

    metadata = FeatureCacheMetadata(
        representation_hash=representation_hash,
        feature_file_hash=file_hash,
        n_samples=len(features),
        feature_dimension=features.shape[1],
        split_name=split_name,
    )
    return metadata


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


def run_extraction(
    config_path: str | Path = "experiments/configs/v5/m19_frozen_space_s1.json",
    cifar_path: str | Path = "data/tier4/cifar10_features.npz",
    output_dir: str | Path = "data/v5/features/m19_s1",
) -> dict[str, Any]:
    """Run the complete feature extraction pipeline.

    Returns a summary dict with representation hashes and cache metadata.
    """
    config_path = Path(config_path)
    cifar_path = Path(cifar_path)
    output_dir = Path(output_dir)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    # Load CIFAR-10 labels only first for split selection
    data = np.load(str(cifar_path))
    all_labels = data["labels"].astype(np.int64)

    # Create stratified splits
    split_cfg = config["split_protocol"]
    train_idx, dev_idx, test_idx = create_stratified_splits(
        all_labels,
        train_per_class=split_cfg["total_per_class_train"],
        dev_per_class=split_cfg["total_per_class_dev"],
        test_per_class=split_cfg["total_per_class_test"],
        seed=split_cfg["selection_seed"],
    )

    train_split_hash = compute_split_hash(train_idx)
    results: dict[str, Any] = {
        "split_hashes": {
            "train": train_split_hash,
            "dev": compute_split_hash(dev_idx),
            "test": compute_split_hash(test_idx),
        },
        "representations": {},
    }

    # Extract features for each available backbone
    for backbone_cfg in config["backbones"]:
        if backbone_cfg.get("status") == "blocked":
            results["representations"][backbone_cfg["id"]] = {
                "status": "blocked",
                "reason": backbone_cfg["reason"],
            }
            continue

        backbone_id = backbone_cfg["id"]
        onnx_path = str(PROJECT_ROOT / backbone_cfg["onnx_path"])
        preproc_path = PROJECT_ROOT / backbone_cfg["preprocessor_config"]

        if not Path(onnx_path).exists():
            results["representations"][backbone_id] = {
                "status": "blocked",
                "reason": f"ONNX model not found at {onnx_path}",
            }
            continue

        # Load preprocessor config
        with preproc_path.open("r", encoding="utf-8") as f:
            preproc_config = json.load(f)

        preproc_digest = compute_preprocessing_digest(preproc_path)
        weights_digest = backbone_cfg["weights_sha256"]

        # Load images for our splits (avoid loading all into transformed memory)
        all_needed_idx = np.concatenate([train_idx, dev_idx, test_idx])
        all_images = data["images"]  # memory-mapped access

        print(f"Extracting {backbone_id} features for {len(all_needed_idx)} images...")
        selected_images = all_images[all_needed_idx]

        actual_weights_digest = hashlib.sha256()
        with Path(onnx_path).open("rb") as model_file:
            for block in iter(lambda: model_file.read(1024 * 1024), b""):
                actual_weights_digest.update(block)
        if actual_weights_digest.hexdigest() != weights_digest:
            raise ValueError(
                f"Backbone weights hash mismatch for {backbone_id}: "
                f"expected {weights_digest}, got {actual_weights_digest.hexdigest()}."
            )

        features = extract_features_batch(
            selected_images,
            backbone_id,
            onnx_path,
            preproc_config,
            backbone_cfg["token_pooling_policy"],
            batch_size=backbone_cfg.get("extraction_batch_size", 32),
        )

        # Split features back
        n_train = len(train_idx)
        n_dev = len(dev_idx)
        train_features = features[:n_train]
        dev_features = features[n_train:n_train + n_dev]
        test_features = features[n_train + n_dev:]

        # Build identity representation manifest
        identity_obj_hash = compute_objective_hash((0.0, 0.0, 0.0), "identity")
        identity_manifest = RepresentationManifest(
            backbone_id=backbone_id,
            upstream_weights_digest=weights_digest,
            preprocessing_digest=preproc_digest,
            interface_architecture="identity",
            interface_weights_digest="none",
            objective_hash=identity_obj_hash,
            training_split_hash=train_split_hash,
            output_dimension=backbone_cfg["output_dimension"],
            checkpoint_source=backbone_cfg["checkpoint_source"],
            checkpoint_license=backbone_cfg["checkpoint_license"],
            token_pooling_policy=backbone_cfg["token_pooling_policy"],
        )

        rep_hash = identity_manifest.representation_hash
        backbone_output_dir = output_dir / backbone_id

        # Cache features
        train_meta = cache_features(
            train_features, all_labels[train_idx], train_idx,
            "train", rep_hash, backbone_output_dir,
        )
        dev_meta = cache_features(
            dev_features, all_labels[dev_idx], dev_idx,
            "dev", rep_hash, backbone_output_dir,
        )
        test_meta = cache_features(
            test_features, all_labels[test_idx], test_idx,
            "test", rep_hash, backbone_output_dir,
        )

        results["representations"][backbone_id] = {
            "status": "extracted",
            "representation_hash": rep_hash,
            "output_dimension": backbone_cfg["output_dimension"],
            "manifest": identity_manifest.to_dict(),
            "cache_metadata": {
                "train": train_meta.to_dict(),
                "dev": dev_meta.to_dict(),
                "test": test_meta.to_dict(),
            },
        }
        print(f"  {backbone_id} identity representation hash: {rep_hash[:16]}...")

    # Save extraction summary
    summary_path = output_dir / "extraction_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        canonical_json(results) + "\n", encoding="utf-8", newline="\n"
    )

    return results


if __name__ == "__main__":
    run_extraction()
