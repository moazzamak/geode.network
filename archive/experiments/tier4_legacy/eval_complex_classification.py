import argparse
import copy
import json
import numpy as np
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.common.moe_eval import fit_experts, k_fold_indices, split_train_test_indices
from experiments.common.experiment_manifest import (
    append_manifest,
    array_fingerprint,
    build_manifest,
)
from experiments.common.model_stats import model_structure_stats
from experiments.common.result_records import classification_result_record
from experiments.common.score_readouts import fit_all_readouts
from experiments.common.classification_baselines import fit_classification_baselines
from src.greedy_constructor import GreedyConstructor
from src.inference_engine import InferenceEngine
from src.model_fingerprint import InputSpec, ModelFingerprint, OutputSpec
from src.model_network import FittedModel
from src.sdf_engine import EllipsoidExpert


_MOBILENET_ONNX_CACHE: str | None = None  # path to exported ONNX model, set on first call
_RESNET18_ONNX_CACHE: str | None = None


def stratified_geometry_calibration_split(
    indices: np.ndarray,
    labels: np.ndarray,
    calibration_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices per class so readouts never train on geometry-fit samples."""
    if not 0.0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be in (0, 0.5).")
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels)
    if len(indices) != len(labels):
        raise ValueError("indices and labels must have the same length.")

    rng = np.random.default_rng(seed)
    geometry_parts = []
    calibration_parts = []
    for class_id in np.unique(labels):
        class_indices = indices[labels == class_id].copy()
        if len(class_indices) < 2:
            raise ValueError(
                f"Class {class_id!r} needs at least two samples for calibration.",
            )
        rng.shuffle(class_indices)
        calibration_size = min(
            len(class_indices) - 1,
            max(1, int(round(len(class_indices) * calibration_fraction))),
        )
        calibration_parts.append(class_indices[:calibration_size])
        geometry_parts.append(class_indices[calibration_size:])
    return np.concatenate(geometry_parts), np.concatenate(calibration_parts)


def stratified_geometry_carve_calibration_split(
    indices: np.ndarray,
    labels: np.ndarray,
    carve_fraction: float = 0.15,
    calibration_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create disjoint class-stratified geometry, carve, and calibration sets."""
    if carve_fraction <= 0 or calibration_fraction <= 0:
        raise ValueError("carve and calibration fractions must be positive.")
    if carve_fraction + calibration_fraction >= 0.5:
        raise ValueError("carve plus calibration fraction must be below 0.5.")
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels)
    if len(indices) != len(labels):
        raise ValueError("indices and labels must have the same length.")

    rng = np.random.default_rng(seed)
    geometry_parts, carve_parts, calibration_parts = [], [], []
    for class_id in np.unique(labels):
        class_indices = indices[labels == class_id].copy()
        if len(class_indices) < 3:
            raise ValueError(
                f"Class {class_id!r} needs at least three samples for ablation.",
            )
        rng.shuffle(class_indices)
        carve_size = max(1, int(round(len(class_indices) * carve_fraction)))
        calibration_size = max(
            1, int(round(len(class_indices) * calibration_fraction)),
        )
        if carve_size + calibration_size >= len(class_indices):
            calibration_size = 1
            carve_size = 1
        carve_parts.append(class_indices[:carve_size])
        calibration_parts.append(
            class_indices[carve_size:carve_size + calibration_size],
        )
        geometry_parts.append(class_indices[carve_size + calibration_size:])
    return (
        np.concatenate(geometry_parts),
        np.concatenate(carve_parts),
        np.concatenate(calibration_parts),
    )


def _get_mobilenet_onnx() -> str:
    """Export MobileNetV2 backbone to ONNX once and return the cached path.

    The ONNX file is written next to this source file so repeated runs skip
    the export entirely.  The model produces ``(B, 1280)`` feature embeddings
    for ``(B, 3, 224, 224)`` float32 inputs already normalised to ImageNet stats.
    """
    global _MOBILENET_ONNX_CACHE
    if _MOBILENET_ONNX_CACHE is not None:
        return _MOBILENET_ONNX_CACHE

    import contextlib
    import io
    import os

    cache_path = os.path.join(os.path.dirname(__file__), "mobilenetv2_backbone.onnx")
    if not os.path.exists(cache_path):
        import torch
        import torchvision.models as tvm

        weights  = tvm.MobileNet_V2_Weights.IMAGENET1K_V1
        backbone = tvm.mobilenet_v2(weights=weights)
        backbone.eval()

        class _Backbone(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
            def forward(self, x):
                x = self.net.features(x)
                x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
                return x.flatten(1)

        model  = _Backbone(backbone)
        model.eval()
        dummy  = torch.zeros(1, 3, 224, 224)

        # Redirect stdout/stderr into a StringIO so emoji in PyTorch's log
        # messages don't crash on Windows cp1252 terminals.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            torch.onnx.export(
                model, dummy, cache_path,
                input_names=["input"], output_names=["features"],
                # dynamic_shapes keys must match forward() arg names, not ONNX names
                dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=4096)}},
                opset_version=18,
            )

    _MOBILENET_ONNX_CACHE = cache_path
    return cache_path


def _extract_cnn_features(images: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Extract 1280-dim features using MobileNetV2 pretrained on ImageNet.

    The classification head is removed; features come from the global-average-pooled
    output of the final convolutional stage (backbone only).  This produces a
    semantically rich, 1280-dim embedding regardless of input image resolution
    or dataset — images are resized to 224×224 and normalised to ImageNet stats
    internally, so the extractor generalises to any natural-image dataset.

    Inference runs through ONNX Runtime with the DirectML execution provider
    (AMD GPU acceleration on Windows).  Falls back to CPU if DirectML is
    unavailable.

    Results are cached to disk keyed on a hash of the image array so that
    repeated runs (e.g. multiple CV experiments on the same dataset) skip
    extraction entirely.

    :param images: uint8 array of shape (N, H, W, 3).
    :param batch_size: Inference batch size.  DirectML handles large batches
        efficiently; reduce if VRAM is tight.
    :return: float64 array of shape (N, 1280).
    """
    import hashlib, os

    # ------------------------------------------------------------------
    # Disk cache: hash first 1000 images + shape to build a short key.
    # Full-array hashing would be slow for 15000 images; 1000 is enough
    # to detect dataset identity changes.
    # ------------------------------------------------------------------
    probe       = images[:1000].tobytes() + str(images.shape).encode()
    cache_key   = hashlib.md5(probe).hexdigest()[:16]
    cache_dir   = os.path.join(os.path.dirname(__file__), ".feat_cache")
    cache_path  = os.path.join(cache_dir, f"cnn_{cache_key}_{len(images)}.npz")

    if os.path.exists(cache_path):
        print(f"  CNN features loaded from cache ({cache_path})")
        return np.load(cache_path)["feats"]

    import onnxruntime as ort
    import torch
    import torch.nn.functional as F

    onnx_path = _get_mobilenet_onnx()

    # Prefer DirectML (AMD GPU) over CPU
    providers = (
        ["DmlExecutionProvider", "CPUExecutionProvider"]
        if "DmlExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(onnx_path, providers=providers)
    using_dml = providers[0] == "DmlExecutionProvider"

    # ImageNet normalisation constants
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    all_feats: list[np.ndarray] = []
    N = len(images)
    for start in range(0, N, batch_size):
        batch_np = images[start : start + batch_size]
        # (B, H, W, C) uint8 → (B, C, H, W) float32 in [0, 1]
        t = batch_np.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        # Resize to 224×224 using torch (bilinear, align_corners=False)
        t_t = torch.from_numpy(t)
        t_t = F.interpolate(t_t, size=(224, 224), mode="bilinear", align_corners=False)
        t = ((t_t.numpy() - mean) / std).astype(np.float32)
        feats = sess.run(["features"], {"input": t})[0]   # (B, 1280) float32
        all_feats.append(feats)
        print(f"  CNN batch {start + len(batch_np)}/{N}"
              + (" [DirectML]" if using_dml else " [CPU]"), end="\r")
    print()

    result = np.concatenate(all_feats, axis=0).astype(np.float64)

    # Save to disk for future runs
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_path, feats=result)

    return result


def _get_resnet18_onnx() -> str:
    global _RESNET18_ONNX_CACHE
    if _RESNET18_ONNX_CACHE is not None:
        return _RESNET18_ONNX_CACHE

    import contextlib
    import io

    cache_path = os.path.join(os.path.dirname(__file__), "resnet18_backbone.onnx")
    if not os.path.exists(cache_path):
        import torch
        import torchvision.models as tvm

        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = torch.nn.Identity()
        backbone.eval()
        dummy = torch.zeros(1, 3, 224, 224)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            torch.onnx.export(
                backbone,
                dummy,
                cache_path,
                input_names=["input"],
                output_names=["features"],
                dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=4096)}},
                opset_version=18,
            )
    _RESNET18_ONNX_CACHE = cache_path
    return cache_path


def _extract_resnet18_features(
    images: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    import hashlib

    probe = images[:1000].tobytes() + str(images.shape).encode()
    cache_key = hashlib.md5(probe).hexdigest()[:16]
    cache_dir = os.path.join(os.path.dirname(__file__), ".feat_cache")
    cache_path = os.path.join(
        cache_dir,
        f"resnet18_imagenet1k_v1_{cache_key}_{len(images)}.npz",
    )
    if os.path.exists(cache_path):
        print(f"  ResNet-18 features loaded from cache ({cache_path})")
        return np.load(cache_path)["feats"]

    import onnxruntime as ort
    import torch
    import torch.nn.functional as F

    providers = (
        ["DmlExecutionProvider", "CPUExecutionProvider"]
        if "DmlExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(_get_resnet18_onnx(), providers=providers)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    batches = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        tensor = batch.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        tensor = F.interpolate(
            torch.from_numpy(tensor),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).numpy()
        normalized = ((tensor - mean) / std).astype(np.float32)
        batches.append(session.run(["features"], {"input": normalized})[0])
    result = np.concatenate(batches).astype(np.float64)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_path, feats=result)
    return result


def _extract_hog_features(images: np.ndarray) -> np.ndarray:
    """Extract HOG features from a batch of images.

    :param images: uint8 or float array of shape (N, H, W, C).
    :return: float64 array of shape (N, hog_dim) where
             hog_dim = cells_x * cells_y * blocks_x * blocks_y * n_orient =
             for 32x32, (4,4) cells, (2,2) blocks, 8 orientations:
             7 * 7 * 2 * 2 * 8 = 1568.
    """
    from skimage.feature import hog as sk_hog

    imgs = images.astype(np.float64) / 255.0 if images.max() > 1.01 else images.astype(np.float64)
    feats = [
        sk_hog(
            img,
            orientations=8,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            channel_axis=-1,
            feature_vector=True,
        )
        for img in imgs
    ]
    return np.array(feats, dtype=np.float64)


def load_cifar_npz(dataset_path: str, max_samples: int, pca_components: int = 128,
                   seed: int = 42, feature_extractor: str = "cnn"):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Missing dataset: {dataset_path}\n"
            "Please download CIFAR-10 (public) and save data/tier4/cifar10_features.npz "
            "with keys 'images' shaped (N, H, W, C) and 'labels' shaped (N,)."
        )

    data = np.load(dataset_path)
    required = {"images", "labels"}
    if not required.issubset(set(data.files)):
        raise KeyError(f"Expected keys {sorted(required)} in {dataset_path}, found {sorted(data.files)}")

    images = data["images"]
    labels = data["labels"].astype(np.int32)
    if images.ndim != 4:
        raise ValueError("Expected images shape (N, H, W, C).")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(images))
    rng.shuffle(idx)
    idx = idx[: min(max_samples, len(idx))]
    images = images[idx]
    y = labels[idx]

    if feature_extractor == "cnn":
        print("Extracting CNN features (MobileNetV2 / ImageNet)...")
        X = _extract_cnn_features(images)
        print(f"CNN features: {X.shape[1]} dims")
    else:
        print("Extracting HOG features...")
        X = _extract_hog_features(images)
        print(f"HOG features: {X.shape[1]} dims")

    # --- Sample adequacy check ---
    # After LDA the final d = n_classes - 1 (= 9 for CIFAR-10). Report both
    # family budgets; the full-ellipsoid figures remain the conservative bound.
    # The transform is fitted per-fold inside run_cv_and_test_classification.
    n_classes = len(np.unique(y))
    d_final = n_classes - 1
    min_seed  = d_final * (d_final + 3) // 2
    sphere_min_seed = d_final + 2
    train_frac = 0.8
    n_min  = int(np.ceil(2  * min_seed * n_classes / train_frac))
    n_rec  = int(np.ceil(10 * min_seed * n_classes / train_frac))
    n_actual = len(X)
    status = ("OK" if n_actual >= n_rec
              else "LOW \u2014 RANSAC may underfit" if n_actual >= n_min
              else "CRITICAL \u2014 RANSAC will not run")
    per_class_train = train_frac * n_actual / n_classes
    d_max_min = int((-3 + (9 + 4 * per_class_train) ** 0.5) / 2)
    d_max_rec = int((-3 + (9 + 4 * per_class_train / 5) ** 0.5) / 2)
    print(f"Sample check  : {n_actual} samples, {n_classes} classes, d={d_final} (LDA)")
    print(
        f"  sphere min_seed={sphere_min_seed}  |  "
        f"full ellipsoid min_seed={min_seed}"
    )
    print(f"  full ellipsoid minimum={n_min}  recommended={n_rec}  [{status}]")
    print(f"  max d (minimum tier)={d_max_min}  max d (recommended)={d_max_rec}")
    print(f"  (Transform fitted per fold \u2014 no data leakage.)")

    return X, y


def _build_transform(X: np.ndarray, y: np.ndarray, pca_components: int, seed: int):
    """Fit PCA(whiten) \u2192 LDA \u2192 StandardScaler on training data only.
    Returns (pca, lda, scaler) for application to held-out data.
    """
    n_pca = min(pca_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_pca, random_state=seed, whiten=True)
    X_pca = pca.fit_transform(X)
    n_classes = len(np.unique(y))
    lda = LinearDiscriminantAnalysis(n_components=n_classes - 1)
    X_lda = lda.fit_transform(X_pca, y)
    scaler = StandardScaler()
    scaler.fit(X_lda)
    return pca, lda, scaler


def _apply_transform(X: np.ndarray, pca, lda, scaler) -> np.ndarray:
    """Apply a fitted PCA \u2192 LDA \u2192 StandardScaler transform to new data."""
    return scaler.transform(lda.transform(pca.transform(X)))


def fit_class_models(
    X: np.ndarray,
    y: np.ndarray,
    class_ids: np.ndarray,
    consensus_threshold: float,
    capture_threshold: float,
    alpha: float,
    max_iterations: int | None,
    nudge_iterations: int,
    nudge_learning_rate: float,
    use_gpu: bool = False,
    seed: int = 42,
    candidate_fitter: Callable[[np.ndarray, int], EllipsoidExpert] | None = None,
    candidate_seed_size: int | None = None,
    primitive_family: str = "sphere",
    gpu_candidate_fitting: bool = False,
):
    models = {}
    for class_position, class_id in enumerate(class_ids):
        class_points = X[y == class_id]
        other_points = X[y != class_id]  # used for discriminative RANSAC scoring
        models[int(class_id)] = fit_experts(
            points=class_points,
            exclude_points=other_points,
            consensus_threshold=consensus_threshold,
            capture_threshold=capture_threshold,
            alpha=alpha,
            max_iterations=max_iterations,
            nudge_iterations=nudge_iterations,
            nudge_learning_rate=nudge_learning_rate,
            use_gpu=use_gpu,
            seed=seed + class_position,
            candidate_fitter=candidate_fitter,
            candidate_seed_size=candidate_seed_size,
            primitive_family=primitive_family,
            gpu_candidate_fitting=gpu_candidate_fitting,
        )
    return models


# Module-level engine cache: one entry keyed on the models dict identity.
# Within a single fold all three scoring calls (scale, scores, predict) share
# the same GPUInferenceEngine — avoiding redundant kernel compilation and
# static buffer uploads. The cache is cleared whenever the models dict changes.
_gpu_engine_cache: dict = {}


def invalidate_gpu_engine_cache(models: dict | None = None) -> None:
    """Discard cached GPU parameters after in-place model mutation."""
    cached = _gpu_engine_cache.get("entry")
    if cached is not None and (models is None or cached[0] is models):
        _gpu_engine_cache.clear()


def _get_gpu_engine(models: dict, alpha: float):
    """Return a :class:`~src.gpu_engine.GPUInferenceEngine` for *models*, building it
    only when *models* has changed since the last call.

    Uses ``id(models)`` as a proxy for identity: the same dict object is reused
    across all scoring calls within one fold, so the engine is built once per fold.
    """
    from src.gpu_engine import GPUInferenceEngine
    cached = _gpu_engine_cache.get("entry")
    if cached is None or cached[0] is not models or cached[1] != alpha:
        class_ids = sorted(models.keys())
        ordered   = [models[cid] for cid in class_ids]
        engine    = GPUInferenceEngine(ordered, alpha=alpha)
        _gpu_engine_cache["entry"] = (models, alpha, class_ids, engine)
    _, _, class_ids, engine = _gpu_engine_cache["entry"]
    return class_ids, engine


def _gpu_class_sdfs(models: dict, X: np.ndarray, alpha: float) -> tuple[list, np.ndarray]:
    """Evaluate all class SDFs in one GPU pass.

    Returns ``(class_ids, sdf_matrix)`` where ``sdf_matrix`` is ``(N, C)`` float64.
    Reuses a cached :class:`~src.gpu_engine.GPUInferenceEngine` for consecutive
    calls on the same ``models`` dict (same fold), so static buffers are uploaded
    only once per fold rather than once per scoring call.
    """
    class_ids, engine = _get_gpu_engine(models, alpha)
    return class_ids, engine.class_sdfs(np.asarray(X, dtype=np.float64))


def compute_score_scales(
    models: dict, X: np.ndarray, alpha: float, use_gpu: bool = False,
    class_labels: np.ndarray | None = None,
) -> dict:
    """Per-class SDF normalisation scales computed from training data.

    Classes with large covariance ellipsoids produce more negative SDF values
    everywhere, biasing argmin toward them.  Dividing each class's scores by
    mean(|SDF|) makes scales comparable before argmin.

    Parameters
    ----------
    class_labels :
        When provided, each class scale is computed using **only that class's
        own training samples** (class-conditional normalisation).  This
        prevents rare classes from dominating argmin: without it, a rare class
        has a large mean|SDF| across *all* training points (most of which are
        far from its tiny ellipsoid), so its normalised scores are crushed
        toward 0 and it wins argmin against common classes — causing
        below-random accuracy.

        When ``None`` (default), the original global mean|SDF| behaviour is
        preserved for backward compatibility with Tier 4 / Tier 5.
    """
    # Class-conditional normalisation: compute scale from in-class samples only.
    if class_labels is not None:
        class_labels = np.asarray(class_labels)
        if use_gpu:
            class_ids, sdf_matrix = _gpu_class_sdfs(models, X, alpha)
            return {
                class_id: max(
                    float(np.mean(np.abs(sdf_matrix[class_labels == class_id, col]))),
                    1e-8,
                ) if models[class_id] and np.any(class_labels == class_id) else 1.0
                for col, class_id in enumerate(class_ids)
            }
        scales = {}
        for class_id, experts in models.items():
            if not experts:
                scales[class_id] = 1.0
                continue
            mask = class_labels == class_id
            if not np.any(mask):
                scales[class_id] = 1.0
                continue
            sdf = InferenceEngine(experts, alpha=alpha).get_fused_sdf(X[mask])
            scales[class_id] = max(float(np.mean(np.abs(sdf))), 1e-8)
        return scales

    # Original global normalisation (used by Tier 4 / Tier 5).
    if use_gpu:
        class_ids, sdf_matrix = _gpu_class_sdfs(models, X, alpha)
        return {
            cid: max(float(np.mean(np.abs(sdf_matrix[:, i]))), 1e-8) if models[cid] else 1.0
            for i, cid in enumerate(class_ids)
        }

    scales = {}
    for class_id, experts in models.items():
        if not experts:
            scales[class_id] = 1.0
            continue
        sdf = InferenceEngine(experts, alpha=alpha).get_fused_sdf(X)
        scales[class_id] = max(float(np.mean(np.abs(sdf))), 1e-8)
    return scales


def compute_raw_scores(
    models: dict,
    X: np.ndarray,
    alpha: float,
    score_scales: dict | None = None,
    use_gpu: bool = False,
) -> np.ndarray:
    """Return (N, n_classes) score matrix with inf replaced by 10.0 for calibration."""
    if use_gpu:
        class_ids, sdf_matrix = _gpu_class_sdfs(models, X, alpha)
        if score_scales is not None:
            scales = np.array([score_scales.get(cid, 1.0) for cid in class_ids])
            sdf_matrix = sdf_matrix / scales[np.newaxis, :]
        return np.where(np.isinf(sdf_matrix), 10.0, sdf_matrix)

    class_ids = sorted(models.keys())
    scores = np.full((len(X), len(class_ids)), np.inf, dtype=np.float64)
    for i, class_id in enumerate(class_ids):
        experts = models[class_id]
        if not experts:
            continue
        sdf = InferenceEngine(experts, alpha=alpha).get_fused_sdf(X)
        scale = score_scales[class_id] if score_scales is not None else 1.0
        scores[:, i] = sdf / scale
    return np.where(np.isinf(scores), 10.0, scores)


def _fit_calibrator(scores: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Platt scaling: logistic regression from SDF score matrix -> class label."""
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(scores, y)
    return clf


def _active_repair(
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    class_ids: np.ndarray,
    alpha: float,
    capture_threshold: float,
    use_gpu: bool = False,
    seed: int = 42,
    acceptance_X: np.ndarray | None = None,
    acceptance_y: np.ndarray | None = None,
    audit_trail: list[dict] | None = None,
    mdl_penalty_weight: float = 0.0,
    min_penalized_gain: float = 0.0,
    max_iterations: int | None = None,
    candidate_fitter: Callable[[np.ndarray, int], EllipsoidExpert] | None = None,
    primitive_family: str = "sphere",
    gpu_candidate_fitting: bool = False,
) -> None:
    """Second subtractive pass targeting points that are deep inside the wrong class.

    Points from other classes with SDF < -2*capture_threshold (well inside the
    additive volume) are the most harmful false positives.  Fitting subtractive
    ellipsoids specifically to these clusters carves the worst overlap first.
    """
    constructor = GreedyConstructor(
        consensus_threshold=0.0,
        capture_threshold=capture_threshold,
        task_type="regression",
        alpha=alpha,
        use_gpu=use_gpu,
        seed=seed,
        max_iterations=max_iterations,
        candidate_fitter=candidate_fitter,
        primitive_family=primitive_family,
        gpu_candidate_fitting=gpu_candidate_fitting,
    )
    n_repaired = 0
    for class_id in class_ids:
        experts = models.get(int(class_id), [])
        if not experts:
            continue
        other_mask = y != class_id
        other_pts = X[other_mask]
        if len(other_pts) == 0:
            continue
        sdf = InferenceEngine(experts, alpha=alpha).get_fused_sdf(other_pts)
        deep_inside = other_pts[sdf < -2.0 * capture_threshold]
        if len(deep_inside) < 2:
            continue
        for expert in experts:
            local_audit = []
            constructor.fit_subtractive_ellipsoids(
                expert,
                deep_inside,
                acceptance_positive_points=(
                    acceptance_X[acceptance_y == class_id]
                    if acceptance_X is not None else None
                ),
                acceptance_negative_points=(
                    acceptance_X[acceptance_y != class_id]
                    if acceptance_X is not None else None
                ),
                audit_trail=local_audit,
                mdl_penalty_weight=mdl_penalty_weight,
                min_penalized_gain=min_penalized_gain,
            )
            if audit_trail is not None:
                audit_trail.extend({
                    **record, "class_id": int(class_id), "phase": "active_repair",
                } for record in local_audit)
        n_repaired += len(deep_inside)
    print(f"  Active repair targeted {n_repaired} deeply misclassified points.")


def predict_labels(
    models, X: np.ndarray, alpha: float,
    score_scales: dict | None = None,
    calibrator: LogisticRegression | None = None,
    use_gpu: bool = False,
):
    class_ids = sorted(models.keys())

    if use_gpu:
        scores = compute_raw_scores(models, X, alpha, score_scales=score_scales, use_gpu=True)
        if calibrator is not None:
            return calibrator.predict(scores).astype(np.int32)
        best_idx = np.argmin(scores, axis=1)
        return np.array([class_ids[i] for i in best_idx], dtype=np.int32)

    # Fill with +inf: empty-expert classes are never preferred.
    scores = np.full((len(X), len(class_ids)), np.inf, dtype=np.float64)

    for i, class_id in enumerate(class_ids):
        experts = models[class_id]
        if not experts:
            continue
        sdf = InferenceEngine(experts, alpha=alpha).get_fused_sdf(X)
        # Use raw SDF (not |SDF|).  With covariance-based ellipsoids the class data
        # sits *inside* the ellipsoid (SDF < 0), so argmin(SDF) correctly picks the
        # class the point is deepest inside.  argmin(|SDF|) would pick the nearest
        # surface, which is typically a *different* class boundary — the sign error
        # that produces below-random accuracy.
        scale = score_scales[class_id] if score_scales is not None else 1.0
        scores[:, i] = sdf / scale

    if calibrator is not None:
        scores_clean = np.where(np.isinf(scores), 10.0, scores)
        return calibrator.predict(scores_clean).astype(np.int32)

    best_idx = np.argmin(scores, axis=1)
    predictions = np.array([class_ids[i] for i in best_idx], dtype=np.int32)
    return predictions


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray):
    return float(np.mean(y_true == y_pred))


def evaluate_score_readouts(
    *,
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_features: np.ndarray,
    evaluation_scores: np.ndarray,
    evaluation_labels: np.ndarray,
    evaluation_features: np.ndarray,
    class_ids: np.ndarray,
    dataset: str,
    split: str,
    representation: str,
    geometry_variant: str,
    model_stats: dict,
    geometry_sample_count: int,
    geometry_fit_seconds: float,
    seed: int,
    evaluation_indices: np.ndarray,
    bootstrap_resamples: int = 500,
    include_predictions: bool = False,
) -> dict[str, dict]:
    """Fit all readouts on calibration data and evaluate one fixed geometry."""
    calibration_scores = np.where(
        np.isfinite(calibration_scores), calibration_scores, 10.0,
    )
    evaluation_scores = np.where(
        np.isfinite(evaluation_scores), evaluation_scores, 10.0,
    )
    readout_fit_started = time.perf_counter()
    readouts = fit_all_readouts(
        calibration_scores,
        calibration_labels,
        class_ids,
        calibration_features,
        seed=seed,
    )
    readout_fit_seconds = time.perf_counter() - readout_fit_started

    records = {}
    for mode, readout in readouts.items():
        inference_started = time.perf_counter()
        probabilities = readout.predict_proba(
            evaluation_scores,
            evaluation_features if mode == "feature_logistic" else None,
        )
        inference_seconds = time.perf_counter() - inference_started
        records[mode] = classification_result_record(
            dataset=dataset,
            split=split,
            seed=seed,
            method="geode" if mode != "feature_logistic" else "logistic_regression",
            representation=representation,
            geometry_variant=geometry_variant,
            readout=mode,
            y_true=evaluation_labels,
            probabilities=probabilities,
            classes=class_ids,
            model_stats=model_stats,
            performance={
                "geometry_fit_seconds": float(geometry_fit_seconds),
                "readout_fit_seconds_all_modes": float(readout_fit_seconds),
                "readout_fit_iterations": int(readout.fit_iterations),
                "readout_iteration_limit": readout.iteration_limit,
                "readout_input_standardized": bool(
                    readout.classifier_mean is not None
                ),
                "inference_seconds": float(inference_seconds),
                "inference_samples_per_second": (
                    len(evaluation_labels) / inference_seconds
                    if inference_seconds > 0 else None
                ),
            },
            adequacy={
                "geometry_samples": int(geometry_sample_count),
                "calibration_samples": int(len(calibration_labels)),
                "minimum_calibration_class_count": int(min(
                    np.sum(calibration_labels == class_id) for class_id in class_ids
                )),
            },
            warnings=list(readout.fit_warnings),
            converged=readout.converged,
            bootstrap_resamples=bootstrap_resamples,
            split_hash=array_fingerprint(np.asarray(evaluation_indices, dtype=np.int64)),
            feature_hash=array_fingerprint(evaluation_features),
        )
        if include_predictions:
            records[mode]["predictions"] = readout.classes[
                probabilities.argmax(axis=1)
            ].tolist()
            records[mode]["targets"] = np.asarray(evaluation_labels).tolist()
    return records


def evaluate_classical_baselines(
    *,
    geometry_features: np.ndarray,
    geometry_labels: np.ndarray,
    evaluation_features: np.ndarray,
    evaluation_labels: np.ndarray,
    evaluation_indices: np.ndarray,
    class_ids: np.ndarray,
    geode_models: dict,
    dataset: str,
    split: str,
    representation: str,
    seed: int,
    rbf_sample_limit: int = 10_000,
    bootstrap_resamples: int = 500,
) -> dict[str, dict]:
    components_by_class = {
        int(class_id): max(1, sum(
            ellipsoid.polarity > 0
            for expert in geode_models.get(int(class_id), [])
            for ellipsoid in expert.ellipsoids
        ))
        for class_id in class_ids
    }
    baselines = fit_classification_baselines(
        geometry_features,
        geometry_labels,
        components_by_class=components_by_class,
        seed=seed,
        rbf_sample_limit=rbf_sample_limit,
    )
    split_hash = array_fingerprint(np.asarray(evaluation_indices, dtype=np.int64))
    feature_hash = array_fingerprint(evaluation_features)

    records = {}
    for name, baseline in baselines.items():
        inference_started = time.perf_counter()
        probabilities = baseline.predict_proba(evaluation_features)
        inference_seconds = time.perf_counter() - inference_started
        records[name] = classification_result_record(
            dataset=dataset,
            split=split,
            seed=seed,
            method=name,
            representation=representation,
            geometry_variant="none",
            readout="native_probability",
            y_true=evaluation_labels,
            probabilities=probabilities,
            classes=class_ids,
            model_stats={
                "matched_components_by_class": components_by_class,
                "training_samples": int(len(geometry_labels)),
            },
            performance={
                "fit_seconds": baseline.fit_seconds_,
                "inference_seconds": float(inference_seconds),
                "inference_samples_per_second": (
                    len(evaluation_labels) / inference_seconds
                    if inference_seconds > 0 else None
                ),
            },
            adequacy={
                "minimum_training_class_count": int(min(
                    np.sum(geometry_labels == class_id) for class_id in class_ids
                )),
            },
            bootstrap_resamples=bootstrap_resamples,
            split_hash=split_hash,
            feature_hash=feature_hash,
        )
    return records


def add_subtractive_ellipsoids(
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    class_ids: np.ndarray,
    capture_threshold: float = 0.08,
    alpha: float = 1.0,
    max_iterations: int | None = None,
    use_gpu: bool = False,
    seed: int = 42,
    acceptance_X: np.ndarray | None = None,
    acceptance_y: np.ndarray | None = None,
    audit_trail: list[dict] | None = None,
    mdl_penalty_weight: float = 0.0,
    min_penalized_gain: float = 0.0,
    candidate_fitter: Callable[[np.ndarray, int], EllipsoidExpert] | None = None,
    primitive_family: str = "sphere",
    gpu_candidate_fitting: bool = False,
) -> None:
    """Carve inter-class overlap out of each class model using subtractive ellipsoids.

    For each class, other-class points that are currently falsely captured by that
    class's experts are fitted with ellipsoids that are added with polarity=-1, creating
    CSG holes that shrink the expert's effective capture region at class boundaries.
    """
    constructor = GreedyConstructor(
        consensus_threshold=0.0,  # unused for subtractive fitting
        capture_threshold=capture_threshold,
        task_type="regression",
        alpha=alpha,
        max_iterations=max_iterations,
        use_gpu=use_gpu,
        seed=seed,
        candidate_fitter=candidate_fitter,
        primitive_family=primitive_family,
        gpu_candidate_fitting=gpu_candidate_fitting,
    )
    for class_id in class_ids:
        experts = models.get(int(class_id), [])
        if not experts:
            continue
        other_pts = X[y != class_id]
        if len(other_pts) == 0:
            continue
        for expert in experts:
            local_audit = []
            constructor.fit_subtractive_ellipsoids(
                expert,
                other_pts,
                acceptance_positive_points=(
                    acceptance_X[acceptance_y == class_id]
                    if acceptance_X is not None else None
                ),
                acceptance_negative_points=(
                    acceptance_X[acceptance_y != class_id]
                    if acceptance_X is not None else None
                ),
                audit_trail=local_audit,
                mdl_penalty_weight=mdl_penalty_weight,
                min_penalized_gain=min_penalized_gain,
            )
            if audit_trail is not None:
                audit_trail.extend({
                    **record, "class_id": int(class_id), "phase": "standard_excision",
                } for record in local_audit)


def build_csg_variants(
    additive_models: dict,
    geometry_X: np.ndarray,
    geometry_y: np.ndarray,
    carve_X: np.ndarray,
    carve_y: np.ndarray,
    class_ids: np.ndarray,
    capture_threshold: float,
    alpha: float,
    max_iterations: int | None,
    seed: int,
    use_gpu: bool = False,
    mdl_penalty_weight: float = 0.01,
    min_penalized_gain: float = 0.0,
    candidate_fitter: Callable[[np.ndarray, int], EllipsoidExpert] | None = None,
    primitive_family: str = "sphere",
    gpu_candidate_fitting: bool = False,
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Build cumulative A0/A1/A2 variants from one frozen additive model."""
    variants = {"A0": copy.deepcopy(additive_models)}
    audits = {"A0": [], "A1": [], "A2": []}

    variants["A1"] = copy.deepcopy(variants["A0"])
    add_subtractive_ellipsoids(
        variants["A1"],
        geometry_X,
        geometry_y,
        class_ids,
        capture_threshold=capture_threshold,
        alpha=alpha,
        max_iterations=max_iterations,
        use_gpu=use_gpu,
        seed=seed + 500,
        acceptance_X=carve_X,
        acceptance_y=carve_y,
        audit_trail=audits["A1"],
        mdl_penalty_weight=mdl_penalty_weight,
        min_penalized_gain=min_penalized_gain,
        candidate_fitter=candidate_fitter,
        primitive_family=primitive_family,
        gpu_candidate_fitting=gpu_candidate_fitting,
    )

    variants["A2"] = copy.deepcopy(variants["A1"])
    _active_repair(
        variants["A2"],
        geometry_X,
        geometry_y,
        class_ids,
        alpha=alpha,
        capture_threshold=capture_threshold,
        use_gpu=use_gpu,
        seed=seed + 600,
        acceptance_X=carve_X,
        acceptance_y=carve_y,
        audit_trail=audits["A2"],
        mdl_penalty_weight=mdl_penalty_weight,
        min_penalized_gain=min_penalized_gain,
        max_iterations=max_iterations,
        candidate_fitter=candidate_fitter,
        primitive_family=primitive_family,
        gpu_candidate_fitting=gpu_candidate_fitting,
    )
    return variants, audits

def run_cv_and_test_classification(
    X: np.ndarray,  # raw CNN/HOG features (not yet reduced)
    y: np.ndarray,
    seed: int,
    n_splits: int,
    max_iterations: int | None = None,
    alpha: float = 2.0,
    pca_components: int = 128,
    use_gpu: bool = False,
    calibration_fraction: float = 0.2,
    bootstrap_resamples: int = 500,
    baseline_rbf_sample_limit: int = 10_000,
):
    run_started = time.perf_counter()
    train_idx, test_idx = split_train_test_indices(len(X), test_fraction=0.2, seed=seed)
    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]
    class_ids = np.unique(y_train)

    fold_acc = []
    readout_records = []
    baseline_records = []
    for fold_i, (cv_train_idx, cv_val_idx) in enumerate(
        k_fold_indices(len(X_train_raw), n_splits=n_splits, seed=seed),
        start=1,
    ):
        fold_geometry_idx, fold_calibration_idx = stratified_geometry_calibration_split(
            cv_train_idx,
            y_train[cv_train_idx],
            calibration_fraction=calibration_fraction,
            seed=seed + fold_i * 1_000,
        )
        fold_fit_started = time.perf_counter()
        pca, lda, scaler = _build_transform(
            X_train_raw[fold_geometry_idx],
            y_train[fold_geometry_idx],
            pca_components,
            seed,
        )
        X_cv_geometry = _apply_transform(
            X_train_raw[fold_geometry_idx], pca, lda, scaler,
        )
        X_cv_calibration = _apply_transform(
            X_train_raw[fold_calibration_idx], pca, lda, scaler,
        )
        X_cv_val   = _apply_transform(X_train_raw[cv_val_idx],   pca, lda, scaler)

        models = fit_class_models(
            X=X_cv_geometry,
            y=y_train[fold_geometry_idx],
            class_ids=class_ids,
            consensus_threshold=0.12,
            capture_threshold=0.08,
            alpha=alpha,
            max_iterations=max_iterations,
            nudge_iterations=20,
            nudge_learning_rate=0.02,
            use_gpu=use_gpu,
            seed=seed + fold_i * 1_000,
        )
        add_subtractive_ellipsoids(
            models=models,
            X=X_cv_geometry,
            y=y_train[fold_geometry_idx],
            class_ids=class_ids,
            capture_threshold=0.08,
            alpha=alpha,
            max_iterations=max_iterations,
            use_gpu=use_gpu,
            seed=seed + fold_i * 1_000 + 500,
        )
        scales = compute_score_scales(
            models, X_cv_geometry, alpha=alpha, use_gpu=use_gpu,
        )
        calibration_scores = compute_raw_scores(
            models, X_cv_calibration, alpha=alpha,
            score_scales=scales, use_gpu=use_gpu,
        )
        validation_scores = compute_raw_scores(
            models, X_cv_val, alpha=alpha,
            score_scales=scales, use_gpu=use_gpu,
        )
        fold_records = evaluate_score_readouts(
            calibration_scores=calibration_scores,
            calibration_labels=y_train[fold_calibration_idx],
            calibration_features=X_cv_calibration,
            evaluation_scores=validation_scores,
            evaluation_labels=y_train[cv_val_idx],
            evaluation_features=X_cv_val,
            class_ids=class_ids,
            dataset="cifar10",
            split=f"cv_fold_{fold_i}",
            representation="mobilenetv2",
            geometry_variant="additive_subtractive",
            model_stats=model_structure_stats(models),
            geometry_sample_count=len(fold_geometry_idx),
            geometry_fit_seconds=time.perf_counter() - fold_fit_started,
            seed=seed + fold_i * 1_000,
            evaluation_indices=cv_val_idx,
            bootstrap_resamples=bootstrap_resamples,
        )
        readout_records.extend(fold_records.values())
        fold_baselines = evaluate_classical_baselines(
            geometry_features=X_cv_geometry,
            geometry_labels=y_train[fold_geometry_idx],
            evaluation_features=X_cv_val,
            evaluation_labels=y_train[cv_val_idx],
            evaluation_indices=cv_val_idx,
            class_ids=class_ids,
            geode_models=models,
            dataset="cifar10",
            split=f"cv_fold_{fold_i}",
            representation="mobilenetv2",
            seed=seed + fold_i * 1_000,
            rbf_sample_limit=baseline_rbf_sample_limit,
            bootstrap_resamples=bootstrap_resamples,
        )
        baseline_records.extend(fold_baselines.values())
        fold_score = fold_records["multinomial"]["metrics"]["accuracy"]
        fold_acc.append(fold_score)
        print(f"  Fold {fold_i}/{n_splits} validation accuracy: {fold_score:.4f}")

    geometry_idx, calibration_idx = stratified_geometry_calibration_split(
        np.arange(len(X_train_raw)),
        y_train,
        calibration_fraction=calibration_fraction,
        seed=seed + 100_000,
    )
    final_fit_started = time.perf_counter()
    pca, lda, scaler = _build_transform(
        X_train_raw[geometry_idx], y_train[geometry_idx], pca_components, seed,
    )
    X_geometry = _apply_transform(X_train_raw[geometry_idx], pca, lda, scaler)
    X_calibration = _apply_transform(X_train_raw[calibration_idx], pca, lda, scaler)
    X_test  = _apply_transform(X_test_raw,  pca, lda, scaler)

    final_models = fit_class_models(
        X=X_geometry,
        y=y_train[geometry_idx],
        class_ids=class_ids,
        consensus_threshold=0.12,
        capture_threshold=0.08,
        alpha=alpha,
        max_iterations=max_iterations,
        nudge_iterations=20,
        nudge_learning_rate=0.02,
        use_gpu=use_gpu,
        seed=seed + 100_000,
    )

    print("Fitting subtractive ellipsoids for boundary sharpening...")
    add_subtractive_ellipsoids(
        models=final_models,
        X=X_geometry,
        y=y_train[geometry_idx],
        class_ids=class_ids,
        capture_threshold=0.08,
        alpha=alpha,
        max_iterations=max_iterations,
        use_gpu=use_gpu,
        seed=seed + 100_500,
    )

    print("Active repair: targeting deeply misclassified training points...")
    _active_repair(
        final_models, X_geometry, y_train[geometry_idx], class_ids,
        alpha=alpha, capture_threshold=0.08, use_gpu=use_gpu,
        seed=seed + 100_600,
    )

    scales = compute_score_scales(
        final_models, X_geometry, alpha=alpha, use_gpu=use_gpu,
    )
    calibration_scores = compute_raw_scores(
        final_models, X_calibration, alpha=alpha,
        score_scales=scales, use_gpu=use_gpu,
    )
    test_scores = compute_raw_scores(
        final_models, X_test, alpha=alpha,
        score_scales=scales, use_gpu=use_gpu,
    )
    structure = model_structure_stats(final_models)
    final_records = evaluate_score_readouts(
        calibration_scores=calibration_scores,
        calibration_labels=y_train[calibration_idx],
        calibration_features=X_calibration,
        evaluation_scores=test_scores,
        evaluation_labels=y_test,
        evaluation_features=X_test,
        class_ids=class_ids,
        dataset="cifar10",
        split="test",
        representation="mobilenetv2",
        geometry_variant="additive_subtractive",
        model_stats=structure,
        geometry_sample_count=len(geometry_idx),
        geometry_fit_seconds=time.perf_counter() - final_fit_started,
        seed=seed + 100_000,
        evaluation_indices=test_idx,
        bootstrap_resamples=bootstrap_resamples,
    )
    readout_records.extend(final_records.values())
    final_baselines = evaluate_classical_baselines(
        geometry_features=X_geometry,
        geometry_labels=y_train[geometry_idx],
        evaluation_features=X_test,
        evaluation_labels=y_test,
        evaluation_indices=test_idx,
        class_ids=class_ids,
        geode_models=final_models,
        dataset="cifar10",
        split="test",
        representation="mobilenetv2",
        seed=seed + 100_000,
        rbf_sample_limit=baseline_rbf_sample_limit,
        bootstrap_resamples=bootstrap_resamples,
    )
    baseline_records.extend(final_baselines.values())
    test_acc = final_records["multinomial"]["metrics"]["accuracy"]

    return {
        "cv_mean_acc": float(np.mean(fold_acc)),
        "cv_std_acc": float(np.std(fold_acc)),
        "test_acc": test_acc,
        "class_count": len(class_ids),
        "n_experts": structure["experts"],
        "model_stats": structure,
        "performance": {
            **final_records["multinomial"]["performance"],
            "total_run_seconds": time.perf_counter() - run_started,
        },
        "readout_records": readout_records,
        "baseline_records": baseline_records,
        "comparison_records": readout_records + baseline_records,
    }


def build_fitted_model(
    X: np.ndarray,
    y: np.ndarray,
    task_name: str,
    class_names: dict | None = None,
    input_source: str = "raw_cnn",
    upstream_tasks: tuple = (),
    alpha: float = 2.0,
    pca_components: int = 128,
    max_iterations: int | None = None,
    seed: int = 42,
    use_gpu: bool = False,
) -> FittedModel:
    """Train a complete GEODE classifier and return it as a fingerprinted FittedModel.

    The returned model is ready to wire into a :class:`~src.model_network.ModelNetwork`.
    Two models produced with the same ``task_name``, ``input_source``, and same
    class set are considered swappable — they can replace each other in a network
    without rewiring.

    Parameters
    ----------
    X:
        Feature matrix (N, d).
        - For source models (``input_source="raw_cnn"`` / ``"raw_hog"``): raw
          CNN or HOG features.  PCA → LDA → StandardScaler is fitted and stored.
        - For downstream models (``input_source="sdf_scores"``): the column-
          concatenated SDF score matrices from upstream nodes.  No PCA/LDA is
          applied; only a StandardScaler is stored for scale consistency.
    y:
        Integer class labels (N,).
    task_name:
        Human-readable task identifier, e.g. ``"bird_detector"``.  This is the
        key used for swappability checks — two models with the same task_name,
        input_source, and class set are interchangeable.
    class_names:
        Optional ``{class_id: str}`` mapping for the fingerprint's output spec.
        Defaults to using the integer class IDs as names.
    input_source:
        One of ``"raw_cnn"``, ``"raw_hog"``, ``"sdf_scores"``, ``"passthrough"``.
    upstream_tasks:
        For ``input_source="sdf_scores"``, the task_names of nodes whose outputs
        this model expects.  Enforced during :meth:`ModelNetwork.validate`.
    alpha, pca_components, max_iterations, seed:
        Hyperparameters forwarded to the training pipeline.

    Returns
    -------
    FittedModel
        A trained model bundled with its fingerprint, transform pipeline,
        SDF experts, score scales, and Platt calibrator.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    class_ids = np.unique(y)

    # --- Transform ---
    if input_source in ("raw_cnn", "raw_hog", "passthrough"):
        pca, lda, scaler = _build_transform(X, y, pca_components, seed)
        X_feat = _apply_transform(X, pca, lda, scaler)
    else:
        # sdf_scores: skip PCA/LDA; only scale for capture_threshold consistency
        pca, lda = None, None
        scaler = StandardScaler()
        X_feat = scaler.fit_transform(X)

    # --- Fit class experts ---
    models = fit_class_models(
        X=X_feat,
        y=y,
        class_ids=class_ids,
        consensus_threshold=0.12,
        capture_threshold=0.08,
        alpha=alpha,
        max_iterations=max_iterations,
        nudge_iterations=20,
        nudge_learning_rate=0.02,
        use_gpu=use_gpu,
        seed=seed,
    )

    # --- Boundary sharpening ---
    add_subtractive_ellipsoids(
        models=models,
        X=X_feat,
        y=y,
        class_ids=class_ids,
        capture_threshold=0.08,
        alpha=alpha,
        max_iterations=max_iterations,
        use_gpu=use_gpu,
        seed=seed + 500,
    )
    _active_repair(
        models, X_feat, y, class_ids, alpha=alpha,
        capture_threshold=0.08, use_gpu=use_gpu, seed=seed + 600,
    )

    # --- Calibration ---
    scales = compute_score_scales(models, X_feat, alpha=alpha)
    raw_scores = compute_raw_scores(models, X_feat, alpha=alpha, score_scales=scales)
    calibrator = _fit_calibrator(raw_scores, y)

    # --- Fingerprint ---
    labels = (
        tuple(class_names[int(c)] for c in class_ids)
        if class_names is not None
        else tuple(int(c) for c in class_ids)
    )
    fingerprint = ModelFingerprint(
        task_name=task_name,
        input_spec=InputSpec(
            source=input_source,
            upstream_tasks=tuple(upstream_tasks),
            dim=X.shape[1],
        ),
        output_spec=OutputSpec(type="sdf_scores", classes=labels),
        alpha=alpha,
        pca_components=pca_components,
    )

    return FittedModel(
        fingerprint=fingerprint,
        class_models=models,
        score_scales=scales,
        calibrator=calibrator,
        pca=pca,
        lda=lda,
        scaler=scaler,
        use_gpu=use_gpu,
    )


def _load_config_defaults(path: str | None) -> dict:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must contain a JSON object.")
    return config


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args()
    config_defaults = _load_config_defaults(config_args.config)

    parser = argparse.ArgumentParser(description="Tier 4 complex image classification.")
    parser.add_argument("--config", help="JSON experiment configuration.")
    parser.add_argument(
        "--dataset-path",
        default="data/tier4/cifar10_features.npz",
        help="Path to npz file with keys images and labels.",
    )
    parser.add_argument("--max-samples", type=int, default=8000, help="Max sample count.")
    parser.add_argument("--pca-components", type=int, default=128, help="Intermediate PCA dim before LDA.")
    parser.add_argument("--feature-extractor", choices=["cnn", "hog"], default="cnn",
                        help="Feature extractor: 'cnn' (MobileNetV2, recommended) or 'hog'.")
    parser.add_argument("--n-splits", type=int, default=5, help="CV fold count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    parser.add_argument("--baseline-rbf-sample-limit", type=int, default=10000)
    parser.add_argument(
        "--use-gpu", action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument("--artifact-path", help="Append the run record to this JSONL file.")
    valid_destinations = {action.dest for action in parser._actions}
    unknown_keys = set(config_defaults) - valid_destinations
    if unknown_keys:
        raise ValueError(f"Unknown Tier 4 config keys: {sorted(unknown_keys)}")
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()

    X, y = load_cifar_npz(
        dataset_path=args.dataset_path,
        max_samples=args.max_samples,
        pca_components=args.pca_components,
        seed=args.seed,
        feature_extractor=args.feature_extractor,
    )

    print("--- Tier 4: Complex Image Classification ---")
    print(
        "Representation: pretrained MobileNetV2/ImageNet features."
        if args.feature_extractor == "cnn" else "Representation: HOG features."
    )
    print(f"Samples={len(X)}, dim={X.shape[1]}, classes={len(np.unique(y))}")

    metrics = run_cv_and_test_classification(
        X=X,
        y=y,
        seed=args.seed,
        n_splits=args.n_splits,
        max_iterations=args.max_iterations,
        alpha=args.alpha,
        pca_components=args.pca_components,
        use_gpu=args.use_gpu,
        calibration_fraction=args.calibration_fraction,
        bootstrap_resamples=args.bootstrap_resamples,
        baseline_rbf_sample_limit=args.baseline_rbf_sample_limit,
    )
    print(f"CV mean accuracy: {metrics['cv_mean_acc']:.4f} +/- {metrics['cv_std_acc']:.4f}")
    print(f"Held-out test accuracy: {metrics['test_acc']:.4f}")
    print(f"Class count modeled: {metrics['class_count']}")
    print(f"Experts fitted: {metrics['model_stats']['experts']}")

    if args.artifact_path:
        train_idx, test_idx = split_train_test_indices(
            len(X), test_fraction=0.2, seed=args.seed,
        )
        split_descriptor = np.concatenate([
            train_idx.astype(np.int64), np.array([-1], dtype=np.int64),
            test_idx.astype(np.int64),
        ])
        run_config = {
            key: value for key, value in vars(args).items()
            if key not in {"config", "artifact_path"}
        }
        manifest = build_manifest(
            config=run_config,
            seed=args.seed,
            repo_root=Path(__file__).resolve().parents[2],
            dataset_fingerprint=array_fingerprint(
                np.frombuffer(Path(args.dataset_path).read_bytes(), dtype=np.uint8),
            ),
            split_indices=split_descriptor,
            features=X,
            device="OpenCL" if args.use_gpu else "CPU",
        )
        manifest["metrics"] = metrics
        append_manifest(args.artifact_path, manifest)
        print(f"Artifact: {args.artifact_path}  id={manifest['experiment_id']}")


if __name__ == "__main__":
    main()
