import argparse
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.common.ellipsoid_fitters import ELLIPSOID_FITTERS


def _sample_shell(
    rng: np.random.Generator,
    count: int,
    center: np.ndarray,
    radii: np.ndarray,
    orientation: np.ndarray,
    radial_range: tuple[float, float],
) -> np.ndarray:
    directions = rng.normal(size=(count, len(center)))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radial = rng.uniform(*radial_range, size=(count, 1))
    return (directions * radial * radii) @ orientation.T + center


def generate_problem(
    dimension: int,
    fit_samples: int,
    test_samples: int,
    anisotropy: float,
    outlier_rate: float,
    label_noise: float,
    overlap_fraction: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    orientation, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    radii = np.geomspace(anisotropy, 1.0, dimension)
    center = rng.normal(0.0, 0.5, size=dimension)
    fit_points = _sample_shell(
        rng, fit_samples, center, radii, orientation, (0.92, 1.0),
    )
    outlier_count = int(round(fit_samples * outlier_rate))
    noise_count = int(round(fit_samples * label_noise))
    if outlier_count + noise_count > fit_samples:
        raise ValueError("Combined outlier and label-noise rates cannot exceed one.")
    replace = rng.choice(fit_samples, outlier_count + noise_count, replace=False)
    if outlier_count:
        local_outliers = rng.uniform(-1.8, 1.8, size=(outlier_count, dimension))
        fit_points[replace[:outlier_count]] = (
            local_outliers * radii
        ) @ orientation.T + center
    if noise_count:
        mislabeled = _sample_shell(
            rng, noise_count, center, radii, orientation, (1.15, 1.8),
        )
        fit_points[replace[outlier_count:]] = mislabeled
    positives = _sample_shell(
        rng, test_samples, center, radii, orientation, (0.55, 0.98),
    )
    negatives = _sample_shell(
        rng, test_samples, center, radii, orientation, (1.08, 1.65),
    )
    overlap_count = int(round(test_samples * overlap_fraction))
    if overlap_count:
        overlapping_center = center + orientation[:, 0] * radii[0] * 1.25
        negatives[:overlap_count] = _sample_shell(
            rng,
            overlap_count,
            overlapping_center,
            radii,
            orientation,
            (0.55, 0.98),
        )
    return {
        "fit_points": fit_points,
        "test_points": np.vstack([positives, negatives]),
        "test_labels": np.concatenate([
            np.ones(test_samples, dtype=np.int64),
            np.zeros(test_samples, dtype=np.int64),
        ]),
        "center": center,
        "shape": orientation @ np.diag(1.0 / radii**2) @ orientation.T,
    }


def evaluate_fitter(name: str, problem: dict, seed: int) -> dict:
    tracemalloc.start()
    started = time.perf_counter()
    try:
        model = ELLIPSOID_FITTERS[name](problem["fit_points"], seed)
        elapsed = time.perf_counter() - started
        _, peak_memory = tracemalloc.get_traced_memory()
        predictions = (model.compute_sdf(problem["test_points"]) < 0.0).astype(int)
        fitted_shape = model.orientation @ np.diag(
            1.0 / model.radii**2
        ) @ model.orientation.T
        shape_error = np.linalg.norm(fitted_shape - problem["shape"], ord="fro")
        shape_error /= np.linalg.norm(problem["shape"], ord="fro")
        return {
            "fitter": name,
            "success": True,
            "balanced_accuracy": float(balanced_accuracy_score(
                problem["test_labels"], predictions,
            )),
            "center_error": float(np.linalg.norm(model.center - problem["center"])),
            "shape_relative_error": float(shape_error),
            "fit_seconds": float(elapsed),
            "peak_memory_bytes": int(peak_memory),
        }
    except (ValueError, np.linalg.LinAlgError) as error:
        elapsed = time.perf_counter() - started
        _, peak_memory = tracemalloc.get_traced_memory()
        return {
            "fitter": name,
            "success": False,
            "error": str(error),
            "fit_seconds": float(elapsed),
            "peak_memory_bytes": int(peak_memory),
        }
    finally:
        tracemalloc.stop()


def run_benchmark(config: dict) -> dict:
    records = []
    scenarios = config.get("scenarios")
    if scenarios is None:
        scenarios = [{
            "name": "default",
            "fit_samples": config["fit_samples"],
            "outlier_rate": config["outlier_rate"],
            "label_noise": config["label_noise"],
            "overlap_fraction": config.get("overlap_fraction", 0.0),
        }]
    for scenario in scenarios:
        for dimension in scenario.get("dimensions", config["dimensions"]):
            for seed in config["seeds"]:
                problem = generate_problem(
                    dimension=dimension,
                    fit_samples=scenario["fit_samples"],
                    test_samples=config["test_samples"],
                    anisotropy=scenario.get("anisotropy", config["anisotropy"]),
                    outlier_rate=scenario.get("outlier_rate", 0.0),
                    label_noise=scenario.get("label_noise", 0.0),
                    overlap_fraction=scenario.get("overlap_fraction", 0.0),
                    seed=seed,
                )
                for fitter in config["fitters"]:
                    record = evaluate_fitter(fitter, problem, seed)
                    record.update({
                        "scenario": scenario["name"],
                        "dimension": dimension,
                        "fit_samples": scenario["fit_samples"],
                        "seed": seed,
                    })
                    records.append(record)
    summary = []
    groups = sorted({
        (record["scenario"], record["dimension"], record["fit_samples"])
        for record in records
    })
    for scenario, dimension, fit_samples in groups:
        for fitter in config["fitters"]:
            group = [
                record for record in records
                if record["scenario"] == scenario
                and record["dimension"] == dimension
                and record["fitter"] == fitter
            ]
            successful = [record for record in group if record["success"]]
            row = {
                "scenario": scenario,
                "dimension": dimension,
                "fit_samples": fit_samples,
                "fitter": fitter,
                "success_rate": len(successful) / len(group),
            }
            for metric in (
                "balanced_accuracy", "center_error", "shape_relative_error",
                "fit_seconds", "peak_memory_bytes",
            ):
                values = [record[metric] for record in successful]
                row[f"mean_{metric}"] = float(np.mean(values)) if values else None
                row[f"std_{metric}"] = float(np.std(values)) if values else None
            summary.append(row)
    return {"config": config, "records": records, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark high-dimensional ellipsoid fitters.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = run_benchmark(config)
    output = Path(config["artifact_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()