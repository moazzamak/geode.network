import argparse
import numpy as np
import os
import sys

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.common.moe_eval import run_cv_then_test


def load_modelnet_pointclouds(dataset_path: str, max_shapes: int) -> np.ndarray:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Missing dataset: {dataset_path}\n"
            "Please download a public point-cloud dataset (e.g. ModelNet10) and store it as "
            "data/tier2/modelnet10_pointclouds.npz with key 'pointclouds' shaped (N, P, 3)."
        )

    data = np.load(dataset_path)
    if "pointclouds" not in data:
        available = ", ".join(data.files)
        raise KeyError(
            f"Expected key 'pointclouds' in {dataset_path}. Found keys: {available}"
        )

    pointclouds = data["pointclouds"]
    if pointclouds.ndim != 3 or pointclouds.shape[-1] != 3:
        raise ValueError("Expected pointclouds shape (N, P, 3).")

    max_shapes = min(max_shapes, len(pointclouds))
    selected = pointclouds[:max_shapes]
    return selected.reshape(-1, 3)


def main():
    parser = argparse.ArgumentParser(description="Tier 2 point-cloud reconstruction evaluation.")
    parser.add_argument(
        "--dataset-path",
        default="data/tier2/modelnet10_pointclouds.npz",
        help="Path to .npz with key 'pointclouds' shaped (N, P, 3).",
    )
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=32,
        help="How many point clouds to aggregate into one reconstruction set.",
    )
    args = parser.parse_args()

    points = load_modelnet_pointclouds(args.dataset_path, args.max_shapes, use_gpu=False)
    print("--- Tier 2: Point-Cloud Reconstruction ---")
    print(f"Loaded points: {points.shape[0]} (dim={points.shape[1]})")

    result = run_cv_then_test(
        points=points,
        test_fraction=0.2,
        n_splits=5,
        seed=42,
        consensus_threshold=0.15,
        capture_threshold=0.08,
        alpha=1.0,
        max_iterations=300,
        nudge_iterations=30,
        nudge_learning_rate=0.02,
    )

    print(f"CV mean MAE(|SDF|): {result['cv_mean_error']:.6f} +/- {result['cv_std_error']:.6f}")
    print(f"Held-out test MAE(|SDF|): {result['test_error']:.6f}")
    print(f"Final expert count: {len(result['final_experts'])}")


if __name__ == "__main__":
    main()
