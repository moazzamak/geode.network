from collections.abc import Callable

import numpy as np

from src.greedy_constructor import GreedyConstructor
from src.inference_engine import InferenceEngine
from src.nudge_engine import NudgeEngine
from src.sdf_engine import EllipsoidExpert


def split_train_test_indices(num_samples: int, test_fraction: float = 0.2, seed: int = 42):
    if num_samples < 2:
        raise ValueError("At least 2 samples are required for train/test splitting.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1).")

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    test_size = int(round(num_samples * test_fraction))
    test_size = min(max(test_size, 1), num_samples - 1)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return train_idx, test_idx


def k_fold_indices(num_samples: int, n_splits: int = 5, seed: int = 42):
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if num_samples < n_splits:
        raise ValueError("num_samples must be >= n_splits.")

    rng = np.random.default_rng(seed)
    shuffled = np.arange(num_samples)
    rng.shuffle(shuffled)

    fold_sizes = np.full(n_splits, num_samples // n_splits, dtype=int)
    fold_sizes[: num_samples % n_splits] += 1

    current = 0
    for fold_size in fold_sizes:
        val_idx = shuffled[current : current + fold_size]
        train_idx = np.concatenate([shuffled[:current], shuffled[current + fold_size :]])
        current += fold_size
        yield train_idx, val_idx


def fit_experts(
    points: np.ndarray,
    consensus_threshold: float,
    capture_threshold: float,
    alpha: float,
    max_iterations: int | None,
    nudge_iterations: int,
    nudge_learning_rate: float,
    exclude_points: np.ndarray | None = None,
    use_gpu: bool = False,
    seed: int = 42,
    candidate_fitter: Callable[[np.ndarray, int], EllipsoidExpert] | None = None,
    candidate_seed_size: int | None = None,
    primitive_family: str = "sphere",
    gpu_candidate_fitting: bool = False,
):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must be shape (N, d).")

    constructor = GreedyConstructor(
        consensus_threshold=consensus_threshold,
        capture_threshold=capture_threshold,
        task_type="regression",
        alpha=alpha,
        max_iterations=max_iterations,
        use_gpu=use_gpu,
        seed=seed,
        candidate_fitter=candidate_fitter,
        candidate_seed_size=candidate_seed_size,
        primitive_family=primitive_family,
        gpu_candidate_fitting=gpu_candidate_fitting,
    )
    experts = constructor.build_model(points, exclude_points=exclude_points)
    if not experts:
        return []

    nudge_engine = NudgeEngine(
        learning_rate=nudge_learning_rate,
        iterations=nudge_iterations,
    )
    return nudge_engine.refine(experts, points)


def mean_abs_sdf_error(experts, points: np.ndarray, alpha: float):
    if not experts:
        return float("inf")
    inference_engine = InferenceEngine(experts, alpha=alpha)
    sdf = inference_engine.get_fused_sdf(points)
    return float(np.mean(np.abs(sdf)))


def run_cv_then_test(
    points: np.ndarray,
    test_fraction: float = 0.2,
    n_splits: int = 5,
    seed: int = 42,
    consensus_threshold: float = 0.8,
    capture_threshold: float = 0.1,
    alpha: float = 1.0,
    max_iterations: int = 500,
    nudge_iterations: int = 30,
    nudge_learning_rate: float = 0.01,
    use_gpu: bool = False,
):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must be shape (N, d).")

    train_idx, test_idx = split_train_test_indices(
        num_samples=len(points),
        test_fraction=test_fraction,
        seed=seed,
    )
    train_points = points[train_idx]
    test_points = points[test_idx]
    return run_cv_with_fixed_train_test(
        train_points=train_points,
        test_points=test_points,
        n_splits=n_splits,
        seed=seed,
        consensus_threshold=consensus_threshold,
        capture_threshold=capture_threshold,
        alpha=alpha,
        max_iterations=max_iterations,
        nudge_iterations=nudge_iterations,
        nudge_learning_rate=nudge_learning_rate,
        use_gpu=use_gpu,
    )


def run_cv_with_fixed_train_test(
    train_points: np.ndarray,
    test_points: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    consensus_threshold: float = 0.8,
    capture_threshold: float = 0.1,
    alpha: float = 1.0,
    max_iterations: int = 500,
    nudge_iterations: int = 30,
    nudge_learning_rate: float = 0.01,
    use_gpu: bool = False,
):
    train_points = np.asarray(train_points, dtype=np.float64)
    test_points = np.asarray(test_points, dtype=np.float64)
    if train_points.ndim != 2 or test_points.ndim != 2:
        raise ValueError("train_points and test_points must be shape (N, d).")
    if train_points.shape[1] != test_points.shape[1]:
        raise ValueError("train_points and test_points must have the same feature dimension.")

    fold_errors = []
    for fold_i, (cv_train_idx, cv_val_idx) in enumerate(
        k_fold_indices(len(train_points), n_splits=n_splits, seed=seed),
        start=1,
    ):
        fold_train = train_points[cv_train_idx]
        fold_val = train_points[cv_val_idx]
        fold_experts = fit_experts(
            points=fold_train,
            consensus_threshold=consensus_threshold,
            capture_threshold=capture_threshold,
            alpha=alpha,
            max_iterations=max_iterations,
            nudge_iterations=nudge_iterations,
            nudge_learning_rate=nudge_learning_rate,
            use_gpu=use_gpu,
            seed=seed + fold_i,
        )
        val_error = mean_abs_sdf_error(fold_experts, fold_val, alpha=alpha)
        fold_errors.append(val_error)
        print(f"  Fold {fold_i}/{n_splits} validation MAE(|SDF|): {val_error:.6f}")

    final_experts = fit_experts(
        points=train_points,
        consensus_threshold=consensus_threshold,
        capture_threshold=capture_threshold,
        alpha=alpha,
        max_iterations=max_iterations,
        nudge_iterations=nudge_iterations,
        nudge_learning_rate=nudge_learning_rate,
        use_gpu=use_gpu,
        seed=seed + 10_000,
    )
    test_error = mean_abs_sdf_error(final_experts, test_points, alpha=alpha)

    cv_mean = float(np.mean(fold_errors))
    cv_std = float(np.std(fold_errors)) if np.isfinite(cv_mean) else float('nan')

    return {
        "train_points": train_points,
        "test_points": test_points,
        "fold_errors": fold_errors,
        "cv_mean_error": cv_mean,
        "cv_std_error": cv_std,
        "test_error": test_error,
        "final_experts": final_experts,
    }
