import numpy as np


def symmetric_label_noise(
    labels: np.ndarray, rate: float, seed: int,
) -> tuple[np.ndarray, dict]:
    """Flip an exact fraction of labels uniformly to a different observed class."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("Symmetric label noise requires at least two classes.")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between zero and one.")
    rng = np.random.default_rng(seed)
    count = int(round(len(labels) * rate))
    indices = rng.choice(len(labels), count, replace=False)
    corrupted = labels.copy()
    for index in indices:
        alternatives = classes[classes != labels[index]]
        corrupted[index] = rng.choice(alternatives)
    return corrupted, {
        "kind": "symmetric_label_noise",
        "rate": float(rate),
        "indices": indices.tolist(),
    }


def class_conditional_label_noise(
    labels: np.ndarray,
    source_class,
    target_class,
    rate: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Flip an exact fraction of one source class to a specified target class."""
    labels = np.asarray(labels)
    if source_class == target_class:
        raise ValueError("source_class and target_class must differ.")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between zero and one.")
    candidates = np.flatnonzero(labels == source_class)
    if not len(candidates):
        raise ValueError(f"Source class {source_class!r} is absent.")
    count = int(round(len(candidates) * rate))
    indices = np.random.default_rng(seed).choice(candidates, count, replace=False)
    corrupted = labels.copy()
    corrupted[indices] = target_class
    return corrupted, {
        "kind": "class_conditional_label_noise",
        "rate": float(rate),
        "source_class": source_class.item() if hasattr(source_class, "item") else source_class,
        "target_class": target_class.item() if hasattr(target_class, "item") else target_class,
        "indices": indices.tolist(),
    }


def inject_feature_outliers(
    features: np.ndarray, rate: float, distance: float, seed: int,
) -> tuple[np.ndarray, dict]:
    """Replace selected rows with random directions at a scaled radial distance."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must have shape (samples, dimensions).")
    if not 0.0 <= rate <= 1.0 or distance <= 0.0:
        raise ValueError("rate must be in [0, 1] and distance must be positive.")
    rng = np.random.default_rng(seed)
    count = int(round(len(features) * rate))
    indices = rng.choice(len(features), count, replace=False)
    corrupted = features.copy()
    if count:
        center = np.mean(features, axis=0)
        scale = np.maximum(np.std(features, axis=0), 1e-12)
        directions = rng.normal(size=(count, features.shape[1]))
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        corrupted[indices] = center + directions * scale * distance
    return corrupted, {
        "kind": "feature_outliers",
        "rate": float(rate),
        "distance": float(distance),
        "indices": indices.tolist(),
    }


def mask_feature_dimensions(
    features: np.ndarray, fraction: float, seed: int,
) -> tuple[np.ndarray, dict]:
    """Set a deterministic subset of feature columns to zero for all samples."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not 0.0 <= fraction <= 1.0:
        raise ValueError("features must be 2D and fraction must be in [0, 1].")
    count = int(round(features.shape[1] * fraction))
    dimensions = np.random.default_rng(seed).choice(
        features.shape[1], count, replace=False,
    )
    corrupted = features.copy()
    corrupted[:, dimensions] = 0.0
    return corrupted, {
        "kind": "missing_dimensions",
        "fraction": float(fraction),
        "dimensions": dimensions.tolist(),
    }


def apply_covariance_shift(
    features: np.ndarray, strength: float, seed: int,
) -> tuple[np.ndarray, dict]:
    """Apply a seeded volume-preserving anisotropic linear shift around the mean."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or strength < 0.0:
        raise ValueError("features must be 2D and strength must be non-negative.")
    rng = np.random.default_rng(seed)
    orientation, _ = np.linalg.qr(rng.normal(size=(features.shape[1], features.shape[1])))
    exponents = np.linspace(-strength, strength, features.shape[1])
    transform = orientation @ np.diag(np.exp(exponents)) @ orientation.T
    center = np.mean(features, axis=0)
    shifted = (features - center) @ transform + center
    return shifted, {
        "kind": "covariance_shift",
        "strength": float(strength),
        "determinant": float(np.linalg.det(transform)),
    }