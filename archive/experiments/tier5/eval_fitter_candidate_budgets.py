import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.common.ellipsoid_fitters import ELLIPSOID_FITTERS
from experiments.tier5.eval_high_dimensional_fitters import generate_problem


def _split_selection_test(problem: dict) -> tuple[np.ndarray, ...]:
    labels = problem["test_labels"]
    selection_indices = np.concatenate([
        np.flatnonzero(labels == class_id)[: len(np.flatnonzero(labels == class_id)) // 2]
        for class_id in (0, 1)
    ])
    test_indices = np.setdiff1d(np.arange(len(labels)), selection_indices)
    return (
        problem["test_points"][selection_indices],
        labels[selection_indices],
        problem["test_points"][test_indices],
        labels[test_indices],
    )


def _accuracy(model, points: np.ndarray, labels: np.ndarray) -> float:
    predictions = (model.compute_sdf(points) < 0.0).astype(np.int64)
    return float(balanced_accuracy_score(labels, predictions))


def _fit_best_candidate(
    fitter_name: str,
    fit_points: np.ndarray,
    selection_points: np.ndarray,
    selection_labels: np.ndarray,
    test_points: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
    candidate_limit: int | None = None,
    seconds_limit: float | None = None,
) -> dict:
    if (candidate_limit is None) == (seconds_limit is None):
        raise ValueError("Specify exactly one candidate or wall-clock limit.")
    dimension = fit_points.shape[1]
    required = dimension * (dimension + 3) // 2
    subset_size = min(len(fit_points), max(required, 2 * dimension + 1))
    rng = np.random.default_rng(seed)
    best_model = None
    best_selection_accuracy = -np.inf
    attempts = 0
    successes = 0
    started = time.perf_counter()
    while True:
        if candidate_limit is not None and attempts >= candidate_limit:
            break
        if seconds_limit is not None and attempts > 0:
            if time.perf_counter() - started >= seconds_limit:
                break
        attempts += 1
        indices = rng.choice(len(fit_points), subset_size, replace=False)
        try:
            model = ELLIPSOID_FITTERS[fitter_name](fit_points[indices], seed + attempts)
        except (ValueError, np.linalg.LinAlgError):
            continue
        successes += 1
        score = _accuracy(model, selection_points, selection_labels)
        if score > best_selection_accuracy:
            best_model = model
            best_selection_accuracy = score
    elapsed = time.perf_counter() - started
    return {
        "attempts": attempts,
        "successes": successes,
        "fit_seconds": float(elapsed),
        "selection_balanced_accuracy": (
            float(best_selection_accuracy) if best_model is not None else None
        ),
        "test_balanced_accuracy": (
            _accuracy(best_model, test_points, test_labels)
            if best_model is not None else None
        ),
        "test_used_for_selection": False,
    }


def run_benchmark(config: dict) -> dict:
    records = []
    for scenario in config["scenarios"]:
        for dimension in config["dimensions"]:
            for seed in config["seeds"]:
                problem = generate_problem(
                    dimension=dimension,
                    fit_samples=scenario["fit_samples"],
                    test_samples=config["test_samples"],
                    anisotropy=config["anisotropy"],
                    outlier_rate=scenario.get("outlier_rate", 0.0),
                    label_noise=scenario.get("label_noise", 0.0),
                    overlap_fraction=scenario.get("overlap_fraction", 0.0),
                    seed=seed,
                )
                slices = _split_selection_test(problem)
                for fitter in config["fitters"]:
                    for candidate_count in config["candidate_counts"]:
                        result = _fit_best_candidate(
                            fitter,
                            problem["fit_points"],
                            *slices,
                            seed=seed,
                            candidate_limit=candidate_count,
                        )
                        result.update({
                            "budget_type": "candidate_count",
                            "budget": candidate_count,
                            "scenario": scenario["name"],
                            "dimension": dimension,
                            "seed": seed,
                            "fitter": fitter,
                        })
                        records.append(result)
                    for seconds in config["wall_clock_seconds"]:
                        result = _fit_best_candidate(
                            fitter,
                            problem["fit_points"],
                            *slices,
                            seed=seed,
                            seconds_limit=seconds,
                        )
                        result.update({
                            "budget_type": "wall_clock",
                            "budget": seconds,
                            "scenario": scenario["name"],
                            "dimension": dimension,
                            "seed": seed,
                            "fitter": fitter,
                        })
                        records.append(result)
    summary = []
    keys = (
        "budget_type", "budget", "scenario", "dimension", "fitter",
    )
    groups = sorted({tuple(record[key] for key in keys) for record in records})
    for group_key in groups:
        group = [
            record for record in records
            if tuple(record[key] for key in keys) == group_key
        ]
        row = dict(zip(keys, group_key))
        for metric in (
            "attempts", "successes", "fit_seconds",
            "selection_balanced_accuracy", "test_balanced_accuracy",
        ):
            values = [record[metric] for record in group if record[metric] is not None]
            row[f"mean_{metric}"] = float(np.mean(values)) if values else None
            row[f"std_{metric}"] = float(np.std(values)) if values else None
        summary.append(row)
    return {"config": config, "records": records, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ellipsoid candidate-search budgets.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_benchmark(config)
    output = Path(config["artifact_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()