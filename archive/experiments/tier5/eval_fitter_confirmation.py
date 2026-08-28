import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.common.experiment_manifest import (
    append_manifest,
    array_fingerprint,
    build_manifest,
)
from experiments.tier4.eval_fitter_screen import run_fitter_screen
from experiments.tier5.eval_cifar100_superclass import load_cifar100_npz


def _load_or_extract_features(config: dict) -> tuple[np.ndarray, np.ndarray]:
    cache_path = Path(config["feature_cache_path"])
    if cache_path.exists():
        with np.load(cache_path) as cached:
            print(f"Loaded cached Tier 5 features: {cache_path}")
            return cached["features"], cached["labels"]
    features, labels = load_cifar100_npz(
        config["dataset_path"],
        config["max_samples"],
        pca_components=config["pca_components"],
        seed=config["seed"],
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features, labels=labels)
    print(f"Cached Tier 5 features: {cache_path}")
    return features, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Confirm fitter candidates on Tier 5.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    features, labels = _load_or_extract_features(config)
    excluded = {
        "artifact_path", "dataset_path", "feature_cache_path", "max_samples",
    }
    experiment_config = {
        key: value for key, value in config.items() if key not in excluded
    }
    result = run_fitter_screen(features, labels, **experiment_config)
    manifest = build_manifest(
        config=config,
        seed=config["seed"],
        repo_root=Path(__file__).resolve().parents[2],
        dataset_fingerprint=array_fingerprint(labels),
        split_indices=np.array(list(result["split_hashes"].values()), dtype=str),
        features=features,
        device="mixed OpenCL/CPU" if config["use_gpu"] else "CPU",
    )
    manifest["metrics"] = result
    append_manifest(config["artifact_path"], manifest)
    print(f"Artifact: {config['artifact_path']}  id={manifest['experiment_id']}")


if __name__ == "__main__":
    main()