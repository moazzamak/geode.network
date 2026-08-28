"""Extract the frozen native DINOv2 cache for the spherical support study."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.tier4.prepare_v5_frozen_features import run_extraction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/configs/v5/m19_native_dinov2_sphere.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/v5/features/m19_native_dinov2_sphere",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_extraction(
        config_path=args.config,
        output_dir=args.output_dir,
    )
