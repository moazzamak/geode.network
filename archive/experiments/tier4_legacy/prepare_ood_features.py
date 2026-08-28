import argparse
from pathlib import Path

import numpy as np

from experiments.tier4.eval_complex_classification import _extract_cnn_features


def deterministic_sample_indices(
    population_size: int, sample_count: int, seed: int,
) -> np.ndarray:
    if sample_count < 1 or sample_count > population_size:
        raise ValueError("sample_count must be between one and population_size.")
    return np.random.default_rng(seed).permutation(population_size)[:sample_count]


def prepare_svhn_features(
    output_path: Path,
    *,
    raw_root: Path = Path("data/raw"),
    sample_count: int = 4_000,
    seed: int = 42,
) -> Path:
    """Cache MobileNetV2 features from a deterministic SVHN test subset."""
    if output_path.exists():
        print(f"SVHN feature cache already exists: {output_path}")
        return output_path

    from torchvision.datasets import SVHN

    dataset = SVHN(root=raw_root, split="test", download=True)
    indices = deterministic_sample_indices(len(dataset), sample_count, seed)
    images = np.transpose(dataset.data[indices], (0, 2, 3, 1)).astype(np.uint8)
    labels = np.asarray(dataset.labels, dtype=np.int32)[indices]
    features = _extract_cnn_features(images)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features,
        labels=labels,
        source_indices=indices,
        split=np.array("test"),
        seed=np.array(seed, dtype=np.int64),
    )
    print(f"Cached SVHN MobileNetV2 features: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real OOD feature caches")
    parser.add_argument("--dataset", choices=("svhn",), default="svhn")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tier7/svhn_test_mobilenetv2_4000_seed42.npz"),
    )
    parser.add_argument("--sample-count", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_svhn_features(
        args.output, sample_count=args.sample_count, seed=args.seed,
    )


if __name__ == "__main__":
    main()