import argparse
import json
from pathlib import Path

import numpy as np

from experiments.common.primitive_stability import evaluate_primitive_stability
from experiments.tier4.eval_complex_classification import fit_class_models
from experiments.tier5.eval_corruption_robustness import generate_multiclass_problem


def run_stability_experiment(config: dict) -> dict:
    problem = generate_multiclass_problem(
        seed=config["data_seed"],
        dimensions=config["dimensions"],
        class_count=config["class_count"],
        geometry_per_class=config["geometry_per_class"],
        calibration_per_class=config["calibration_per_class"],
        test_per_class=config["test_per_class"],
        center_radius=config["center_radius"],
        mode_offset=config["mode_offset"],
        noise_scale=config["noise_scale"],
    )
    class_ids = np.unique(problem["y_geometry"])
    models_by_seed = {
        seed: fit_class_models(
            problem["X_geometry"],
            problem["y_geometry"],
            class_ids,
            consensus_threshold=0.1,
            capture_threshold=0.1,
            alpha=2.0,
            max_iterations=config["max_iterations"],
            nudge_iterations=0,
            nudge_learning_rate=0.02,
            seed=seed,
        )
        for seed in config["construction_seeds"]
    }
    result = evaluate_primitive_stability(
        models_by_seed, problem["X_test"], alpha=2.0,
    )
    result["protocol"] = {
        "data_seed": config["data_seed"],
        "construction_seeds": config["construction_seeds"],
        "evaluation_count": len(problem["X_test"]),
        "evaluation_used_for_fitting": False,
        "geometry_variant": "additive",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Five-seed primitive stability study")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_stability_experiment(config)
    output_path = Path(config["artifact_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()