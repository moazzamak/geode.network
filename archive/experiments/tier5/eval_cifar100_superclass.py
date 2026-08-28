"""
Tier 5: CIFAR-100 Superclass Classification
============================================
20-class (coarse superclass) image classification using GEODE.

Compared with Tier 4 (CIFAR-10, 10 classes, d=9 after LDA), this tier is
harder along every axis:

  * 2× more classes (20 coarse superclasses vs 10)
  * 2× higher LDA embedding dimension (d=19 vs d=9), so k_size = 209 vs 54
  * Coarse superclasses have high intra-class variation (e.g. "vehicles 1"
    contains bicycles, buses, motorcycles, pickup trucks, trains) — the class
    boundaries are not axis-aligned and require subtractive ellipsoids to carve
    clean separation surfaces.

Pipeline: MobileNetV2 features → PCA(128, whiten) → LDA(19) → StandardScaler
→ GEODE classifier with subtractive ellipsoids and active repair.
"""

import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.common.moe_eval import k_fold_indices, split_train_test_indices
from experiments.tier4.eval_complex_classification import (
    _extract_cnn_features,
    _build_transform,
    _apply_transform,
    fit_class_models,
    add_subtractive_ellipsoids,
    _active_repair,
    compute_score_scales,
    compute_raw_scores,
    _fit_calibrator,
    predict_labels,
    accuracy_score,
)

# Human-readable names for the 20 CIFAR-100 coarse labels (index = label value).
COARSE_LABEL_NAMES = [
    "aquatic mammals",
    "fish",
    "flowers",
    "food containers",
    "fruit and vegetables",
    "household electrical devices",
    "household furniture",
    "insects",
    "large carnivores",
    "large man-made outdoor things",
    "large natural outdoor scenes",
    "large omnivores and herbivores",
    "medium-sized mammals",
    "non-insect invertebrates",
    "people",
    "reptiles",
    "small mammals",
    "trees",
    "vehicles 1",
    "vehicles 2",
]


def load_cifar100_npz(
    dataset_path: str,
    max_samples: int,
    pca_components: int = 128,
    seed: int = 42,
):
    """Load CIFAR-100 superclass data and extract MobileNetV2 CNN features.

    :param dataset_path: Path to the NPZ produced by
        :func:`~experiments.common.dataset_utils.prepare_cifar100`.
    :param max_samples: Maximum number of images to use (randomly sampled).
    :param pca_components: Unused here; stored for caller documentation only —
        PCA is applied inside :func:`run_cv_and_test_classification`.
    :param seed: RNG seed for reproducible sampling.
    :return: ``(X, y)`` — ``X`` is ``(N, 1280)`` float64 CNN features,
        ``y`` is coarse label array in ``[0, 19]``.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Missing dataset: {dataset_path}\n"
            "Run verify_pipeline.py to auto-download, or call:\n"
            "    experiments.common.dataset_utils.prepare_cifar100()"
        )

    data = np.load(dataset_path)
    required = {"images", "coarse_labels"}
    if not required.issubset(set(data.files)):
        raise KeyError(
            f"Expected keys {sorted(required)} in {dataset_path}, "
            f"found {sorted(data.files)}"
        )

    images = data["images"]
    labels = data["coarse_labels"].astype(np.int32)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(images))
    rng.shuffle(idx)
    idx = idx[: min(max_samples, len(idx))]
    images = images[idx]
    y = labels[idx]

    print("Extracting CNN features (MobileNetV2 / ImageNet)...")
    X = _extract_cnn_features(images)
    print(f"CNN features: {X.shape[1]} dims")

    # Sample adequacy check (same formula as Tier 4)
    n_classes = len(np.unique(y))
    d_final = n_classes - 1          # LDA reduces to n_classes − 1 dims
    min_seed = d_final * (d_final + 3) // 2
    train_frac = 0.8
    n_min = int(np.ceil(2  * min_seed * n_classes / train_frac))
    n_rec = int(np.ceil(10 * min_seed * n_classes / train_frac))
    n_actual = len(X)
    status = (
        "OK"
        if n_actual >= n_rec
        else "LOW \u2014 RANSAC may underfit"
        if n_actual >= n_min
        else "CRITICAL \u2014 RANSAC will not run"
    )
    per_class_train = train_frac * n_actual / n_classes
    print(f"Sample check  : {n_actual} samples, {n_classes} classes, d={d_final} (LDA)")
    print(f"  min_seed={min_seed}  |  minimum={n_min}  recommended={n_rec}  [{status}]")
    print(f"  ~{per_class_train:.0f} training samples/class in final fit")

    return X, y


def run_cv_and_test_classification(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_splits: int,
    max_iterations: int | None = None,
    alpha: float = 2.0,
    pca_components: int = 128,
    consensus_threshold: float = 0.10,
    capture_threshold: float = 0.08,
    nudge_iterations: int = 20,
    nudge_learning_rate: float = 0.02,
    use_gpu: bool = False,
):
    """Cross-validated train/test pipeline for CIFAR-100 superclass classification.

    Mirrors the Tier 4 ``run_cv_and_test_classification`` but exposes all
    RANSAC and nudge hyperparameters as explicit arguments so they can be
    tuned for the larger 20-class / d=19 problem without modifying Tier 4.

    Pipeline per fold:
      1. Fit PCA(128, whiten) → LDA(19) → StandardScaler on fold-train only.
      2. Train one GEODE expert per class (additive RANSAC ellipsoids).
      3. Fit subtractive ellipsoids to carve inter-class overlap.
      4. Fit a Platt-scaling calibrator from per-class SDF scores.
      5. Evaluate on the held-out fold.

    Final model repeats steps 1–4 on the full training set, then tests on
    the held-out 20 % test split.
    """
    train_idx, test_idx = split_train_test_indices(len(X), test_fraction=0.2, seed=seed)
    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_test_raw,  y_test  = X[test_idx],  y[test_idx]
    class_ids = np.unique(y_train)

    fold_acc = []
    for fold_i, (cv_train_idx, cv_val_idx) in enumerate(
        k_fold_indices(len(X_train_raw), n_splits=n_splits, seed=seed),
        start=1,
    ):
        pca, lda, scaler = _build_transform(
            X_train_raw[cv_train_idx], y_train[cv_train_idx], pca_components, seed
        )
        X_cv_train = _apply_transform(X_train_raw[cv_train_idx], pca, lda, scaler)
        X_cv_val   = _apply_transform(X_train_raw[cv_val_idx],   pca, lda, scaler)

        models = fit_class_models(
            X=X_cv_train,
            y=y_train[cv_train_idx],
            class_ids=class_ids,
            consensus_threshold=consensus_threshold,
            capture_threshold=capture_threshold,
            alpha=alpha,
            max_iterations=max_iterations,
            nudge_iterations=nudge_iterations,
            nudge_learning_rate=nudge_learning_rate,
            use_gpu=use_gpu,
            seed=seed + fold_i * 1_000,
        )
        add_subtractive_ellipsoids(
            models=models,
            X=X_cv_train,
            y=y_train[cv_train_idx],
            class_ids=class_ids,
            capture_threshold=capture_threshold,
            alpha=alpha,
            use_gpu=use_gpu,
            seed=seed + fold_i * 1_000 + 500,
        )
        scales = compute_score_scales(models, X_cv_train, alpha=alpha, use_gpu=use_gpu)
        raw_scores = compute_raw_scores(
            models, X_cv_train, alpha=alpha, score_scales=scales, use_gpu=use_gpu
        )
        calibrator = _fit_calibrator(raw_scores, y_train[cv_train_idx])
        preds = predict_labels(
            models, X_cv_val, alpha=alpha,
            score_scales=scales, calibrator=calibrator,
            use_gpu=use_gpu,
        )
        fold_score = accuracy_score(y_train[cv_val_idx], preds)
        fold_acc.append(fold_score)
        print(f"  Fold {fold_i}/{n_splits} validation accuracy: {fold_score:.4f}")

    # Final model on the complete training set
    pca, lda, scaler = _build_transform(X_train_raw, y_train, pca_components, seed)
    X_train = _apply_transform(X_train_raw, pca, lda, scaler)
    X_test  = _apply_transform(X_test_raw,  pca, lda, scaler)

    final_models = fit_class_models(
        X=X_train,
        y=y_train,
        class_ids=class_ids,
        consensus_threshold=consensus_threshold,
        capture_threshold=capture_threshold,
        alpha=alpha,
        max_iterations=max_iterations,
        nudge_iterations=nudge_iterations,
        nudge_learning_rate=nudge_learning_rate,
        use_gpu=use_gpu,
        seed=seed + 100_000,
    )

    print("Fitting subtractive ellipsoids for boundary sharpening...")
    add_subtractive_ellipsoids(
        models=final_models,
        X=X_train,
        y=y_train,
        class_ids=class_ids,
        capture_threshold=capture_threshold,
        alpha=alpha,
        use_gpu=use_gpu,
        seed=seed + 100_500,
    )

    print("Active repair: targeting deeply misclassified training points...")
    _active_repair(
        final_models, X_train, y_train, class_ids,
        alpha=alpha, capture_threshold=capture_threshold, use_gpu=use_gpu,
        seed=seed + 100_600,
    )

    scales = compute_score_scales(final_models, X_train, alpha=alpha, use_gpu=use_gpu)
    raw_scores = compute_raw_scores(
        final_models, X_train, alpha=alpha, score_scales=scales, use_gpu=use_gpu
    )
    calibrator = _fit_calibrator(raw_scores, y_train)
    test_preds = predict_labels(
        final_models, X_test, alpha=alpha,
        score_scales=scales, calibrator=calibrator,
        use_gpu=use_gpu,
    )
    test_acc = accuracy_score(y_test, test_preds)

    return {
        "cv_mean_acc": float(np.mean(fold_acc)),
        "cv_std_acc":  float(np.std(fold_acc)),
        "test_acc":    test_acc,
        "class_count": len(class_ids),
        "n_experts":   sum(len(v) for v in final_models.values()),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Tier 5 CIFAR-100 superclass classification evaluation."
    )
    parser.add_argument(
        "--dataset-path",
        default="data/tier5/cifar100_superclass.npz",
        help="Path to the NPZ produced by dataset_utils.prepare_cifar100().",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=60000,
        help="Maximum images to use (default: all 60 000).",
    )
    parser.add_argument("--use-gpu", action="store_true", default=False)
    args = parser.parse_args()

    X, y = load_cifar100_npz(
        dataset_path=args.dataset_path,
        max_samples=args.max_samples,
        seed=42,
    )
    print(f"Loaded: {X.shape[0]} samples, {len(np.unique(y))} classes")

    result = run_cv_and_test_classification(
        X=X, y=y, seed=42, n_splits=5,
        pca_components=128, alpha=2.0,
        consensus_threshold=0.10, capture_threshold=0.08,
        use_gpu=args.use_gpu,
    )
    print(f"CV  Accuracy : {result['cv_mean_acc']*100:.2f}% +/- {result['cv_std_acc']*100:.2f}%")
    print(f"Test Accuracy: {result['test_acc']*100:.2f}%")
    print(f"Experts fitted: {result['n_experts']} across {result['class_count']} classes")


if __name__ == "__main__":
    main()
