import numpy as np
import os
import sys

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.common.moe_eval import run_cv_with_fixed_train_test


def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def ensure_train_test_files(base_name: str):
    train_path = f"data/tier1/{base_name}_train.npy"
    test_path = f"data/tier1/{base_name}_test.npy"
    legacy_path = f"data/tier1/{base_name}.npy"

    if os.path.exists(train_path) and os.path.exists(test_path):
        return np.load(train_path), np.load(test_path)

    if os.path.exists(legacy_path):
        print(
            f"Legacy dataset detected for '{base_name}'. "
            "Regenerate with experiments/tier1/data_gen.py for persisted train/test files."
        )
        all_points = np.load(legacy_path)
        train_points = all_points[:-300]
        test_points = all_points[-300:]
        return train_points, test_points

    raise FileNotFoundError(
        f"Missing dataset for '{base_name}'. Run experiments/tier1/data_gen.py first."
    )


def evaluate_dataset(
    name: str,
    train_points: np.ndarray,
    test_points: np.ndarray,
    gt_center: np.ndarray,
    gt_radii: np.ndarray,
):
    print(f"\nEvaluating {name} with train-only 5-fold CV + held-out test...")
    result = run_cv_with_fixed_train_test(
        train_points=train_points,
        test_points=test_points,
        n_splits=5,
        seed=42,
        consensus_threshold=0.8,
        capture_threshold=0.1,
        alpha=1.0,
        max_iterations=600,
        nudge_iterations=50,
        nudge_learning_rate=0.01,
    )

    print(f"CV mean MAE(|SDF|): {result['cv_mean_error']:.6f} +/- {result['cv_std_error']:.6f}")
    print(f"Final held-out test MAE(|SDF|): {result['test_error']:.6f}")

    if not result["final_experts"]:
        print("No experts found on final train split.")
        return None

    best_expert = result["final_experts"][0]
    center_rmse = calculate_rmse(gt_center, best_expert.center)
    radii_rmse = calculate_rmse(gt_radii, best_expert.radii)
    print(f"Ground Truth: center={gt_center}, radii={gt_radii}")
    print(f"Recovered Expert: center={best_expert.center}, radii={best_expert.radii}")
    print(f"Center RMSE: {center_rmse:.6f}")
    print(f"Radii RMSE: {radii_rmse:.6f}")
    return {
        "cv_mean_error": result["cv_mean_error"],
        "cv_std_error": result["cv_std_error"],
        "test_error": result["test_error"],
        "center_rmse": center_rmse,
        "radii_rmse": radii_rmse,
        "n_experts": len(result["final_experts"]),
    }


def main():
    sphere_train, sphere_test = ensure_train_test_files("sphere")
    ellipsoid_train, ellipsoid_test = ensure_train_test_files("ellipsoid")

    gt_sphere_center = np.array([0.0, 0.0, 0.0])
    gt_sphere_radii = np.array([1.0, 1.0, 1.0])
    gt_ellipsoid_center = np.array([3.0, 0.0, 0.0])
    gt_ellipsoid_radii = np.array([1.5, 0.8, 1.2])

    print("--- Tier 1: Geometry Regression Evaluation ---")
    evaluate_dataset("Sphere", sphere_train, sphere_test, gt_sphere_center, gt_sphere_radii)
    evaluate_dataset("Ellipsoid", ellipsoid_train, ellipsoid_test, gt_ellipsoid_center, gt_ellipsoid_radii)


if __name__ == "__main__":
    main()
