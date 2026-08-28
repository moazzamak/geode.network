import argparse
import json
from pathlib import Path

import numpy as np

from experiments.common.ellipsoid_distance_reference import (
    numerical_closest_point,
    numerical_signed_distance,
)
from src.sdf_engine import EllipsoidExpert


def _unit_directions(dimensions: int, count: int, seed: int) -> np.ndarray:
    if dimensions == 2:
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, dimensions))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def evaluate_metric_distance(
    *,
    dimensions: tuple[int, ...] = (2, 3),
    eccentricities: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
    radial_factors: tuple[float, ...] = (0.5, 0.9, 1.1, 1.5, 2.0),
    direction_count: int = 64,
    seed: int = 42,
) -> dict:
    records = []
    for dimension in dimensions:
        directions = _unit_directions(dimension, direction_count, seed + dimension)
        for eccentricity in eccentricities:
            radii = np.ones(dimension)
            radii[0] = eccentricity
            ellipsoid = EllipsoidExpert(np.zeros(dimension), radii)
            surface_radii = 1.0 / np.sqrt(
                np.sum((directions / radii) ** 2, axis=1)
            )
            points = np.concatenate([
                directions * (factor * surface_radii)[:, None]
                for factor in radial_factors
            ])
            reference = numerical_signed_distance(ellipsoid, points)
            approximation = ellipsoid.compute_metric_sdf(points)
            absolute_error = np.abs(approximation - reference)
            relative_error = absolute_error / np.maximum(np.abs(reference), 1e-12)
            outside_mask = reference > 0.0
            inside_mask = reference < 0.0

            unsafe_steps = 0
            outside_count = 0
            for point, exact_distance, step_distance in zip(
                points, reference, approximation,
            ):
                if exact_distance <= 0.0:
                    continue
                outside_count += 1
                closest = numerical_closest_point(ellipsoid, point)
                direction = (closest - point) / exact_distance
                stepped = point + max(float(step_distance), 0.0) * direction
                if ellipsoid.compute_sdf(stepped[None, :])[0] < -1e-8:
                    unsafe_steps += 1

            records.append({
                "dimensions": dimension,
                "eccentricity": eccentricity,
                "query_count": len(points),
                "outside_query_count": outside_count,
                "mean_absolute_error": float(np.mean(absolute_error)),
                "max_absolute_error": float(np.max(absolute_error)),
                "mean_relative_error": float(np.mean(relative_error)),
                "max_relative_error": float(np.max(relative_error)),
                "outside_mean_absolute_error": float(np.mean(
                    absolute_error[outside_mask],
                )),
                "outside_max_absolute_error": float(np.max(
                    absolute_error[outside_mask],
                )),
                "inside_mean_absolute_error": float(np.mean(
                    absolute_error[inside_mask],
                )),
                "inside_max_absolute_error": float(np.max(
                    absolute_error[inside_mask],
                )),
                "outside_overestimate_fraction": float(np.mean(
                    approximation[outside_mask] > reference[outside_mask] + 1e-9,
                )),
                "unsafe_closest_ray_steps": unsafe_steps,
                "unsafe_closest_ray_step_fraction": (
                    unsafe_steps / outside_count if outside_count else 0.0
                ),
            })
    return {
        "method": "first_order_normalized_radial_correction",
        "reference": "multistart_slsqp_euclidean_closest_point",
        "safe_step_claim": False,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ellipsoid metric correction.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--directions", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = evaluate_metric_distance(
        direction_count=args.directions, seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()