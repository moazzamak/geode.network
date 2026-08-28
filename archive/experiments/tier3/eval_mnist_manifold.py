import argparse
import numpy as np
import os
import sys

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.common.moe_eval import run_cv_then_test


def load_mnist_digit_subset(digit: int, limit: int, pca_components: int, random_seed: int) -> np.ndarray:
    try:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    except Exception as exc:
        raise RuntimeError(
            "Could not fetch MNIST from OpenML. Ensure internet access or provide a cached OpenML copy."
        ) from exc

    X = mnist.data.astype(np.float64) / 255.0
    y = mnist.target.astype(np.int32)

    chosen = np.where(y == digit)[0]
    if len(chosen) == 0:
        raise ValueError(f"No MNIST samples found for digit {digit}.")

    rng = np.random.default_rng(random_seed)
    rng.shuffle(chosen)
    subset_idx = chosen[: min(limit, len(chosen))]
    subset = X[subset_idx]

    pca = PCA(n_components=pca_components, random_state=random_seed)
    reduced = pca.fit_transform(subset)
    return reduced


def main():
    parser = argparse.ArgumentParser(description="Tier 3 MNIST manifold fitting.")
    parser.add_argument("--digit", type=int, default=0, help="Digit class to fit manifold for.")
    parser.add_argument(
        "--limit",
        type=int,
        default=4000,
        help="Maximum number of selected digit images before PCA.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=3,
        help="Reduced dimension for manifold fitting.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    points = load_mnist_digit_subset(
        digit=args.digit,
        limit=args.limit,
        pca_components=args.pca_components,
        random_seed=args.seed,
    )
    print("--- Tier 3: MNIST Manifold Fitting ---")
    print(f"Digit={args.digit}, samples={points.shape[0]}, dim={points.shape[1]}")

    result = run_cv_then_test(
        points=points,
        test_fraction=0.2,
        n_splits=5,
        seed=args.seed,
        consensus_threshold=0.12,
        capture_threshold=0.08,
        alpha=1.0,
        max_iterations=250,
        nudge_iterations=25,
        nudge_learning_rate=0.015,
    )

    print(f"CV mean MAE(|SDF|): {result['cv_mean_error']:.6f} +/- {result['cv_std_error']:.6f}")
    print(f"Held-out test MAE(|SDF|): {result['test_error']:.6f}")
    print(f"Final expert count: {len(result['final_experts'])}")


if __name__ == "__main__":
    main()
