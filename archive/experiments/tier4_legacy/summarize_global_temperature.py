import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "accuracy",
    "negative_log_likelihood",
    "brier_score",
    "expected_calibration_error",
)


def summarize_runs(
    runs: list[dict],
    *,
    accuracy_tolerance: float = 0.0025,
    nll_tolerance: float = 0.01,
) -> dict:
    unique_runs = {}
    for run in runs:
        unique_runs.setdefault(run["experiment_id"], run)
    grouped = defaultdict(list)
    temperatures = []
    optimization_converged = True
    readouts_converged = True
    for run in unique_runs.values():
        seed = int(run["config"]["seed"])
        optimization = run["metrics"]["likelihood_optimization"]
        temperatures.append(float(optimization["covariance_temperature"]))
        optimization_converged &= bool(optimization["converged"])
        for record in run["metrics"]["records"]:
            readouts_converged &= bool(record["converged"])
            if record["split"] == "test":
                grouped[record["readout"]].append({
                    "seed": seed,
                    **{metric: float(record["metrics"][metric]) for metric in METRICS},
                })

    grouped_results = {}
    for name, rows in sorted(grouped.items()):
        grouped_results[name] = {
            "seeds": [row["seed"] for row in rows],
            **{
                f"{metric}_{statistic}": float(
                    np.std([row[metric] for row in rows], ddof=1)
                    if statistic == "std"
                    else np.mean([row[metric] for row in rows])
                )
                for metric in METRICS
                for statistic in ("mean", "std")
            },
        }

    paired = {}
    for tuned, baseline in (
        ("probabilistic_global_temperature", "probabilistic"),
        ("hybrid_global_temperature", "hybrid"),
    ):
        tuned_by_seed = {row["seed"]: row for row in grouped[tuned]}
        baseline_by_seed = {row["seed"]: row for row in grouped[baseline]}
        seeds = sorted(set(tuned_by_seed) & set(baseline_by_seed))
        paired[tuned] = {
            "baseline": baseline,
            "seeds": seeds,
            **{
                f"tuned_minus_{metric}": {
                    "mean": float(np.mean(differences)),
                    "per_seed": [float(value) for value in differences],
                }
                for metric in METRICS
                for differences in [[
                    tuned_by_seed[seed][metric] - baseline_by_seed[seed][metric]
                    for seed in seeds
                ]]
            },
        }

    tuned = grouped_results["hybrid_global_temperature"]
    baseline = grouped_results["hybrid"]
    accuracy_improved = tuned["accuracy_mean"] > baseline["accuracy_mean"]
    nll_improved = (
        tuned["negative_log_likelihood_mean"]
        < baseline["negative_log_likelihood_mean"]
    )
    accuracy_within_tolerance = (
        tuned["accuracy_mean"] >= baseline["accuracy_mean"] - accuracy_tolerance
    )
    nll_within_tolerance = (
        tuned["negative_log_likelihood_mean"]
        <= baseline["negative_log_likelihood_mean"] + nll_tolerance
    )
    gate_passed = optimization_converged and readouts_converged and (
        (nll_improved and accuracy_within_tolerance)
        or (accuracy_improved and nll_within_tolerance)
    )
    return {
        "schema_version": 1,
        "protocol": "tier4_global_covariance_temperature",
        "source_artifact": "logs/results/tier4_global_covariance_temperature_runs.jsonl",
        "source_rows": len(runs),
        "unique_experiments": len(unique_runs),
        "duplicate_retries": len(runs) - len(unique_runs),
        "seeds": sorted(int(run["config"]["seed"]) for run in unique_runs.values()),
        "selection_winners": dict(sorted(Counter(
            run["metrics"]["selected_score_input"] for run in unique_runs.values()
        ).items())),
        "covariance_temperature": {
            "mean": float(np.mean(temperatures)),
            "std": float(np.std(temperatures, ddof=1)),
            "minimum": float(np.min(temperatures)),
            "maximum": float(np.max(temperatures)),
            "per_seed": temperatures,
        },
        "all_optimizers_converged": optimization_converged,
        "all_readouts_converged": readouts_converged,
        "matched_frozen_models": True,
        "test_used_for_selection": False,
        "grouped_test_results": grouped_results,
        "paired_differences": paired,
        "advancement_gate": {
            "accuracy_tolerance": accuracy_tolerance,
            "nll_tolerance": nll_tolerance,
            "accuracy_improved": accuracy_improved,
            "negative_log_likelihood_improved": nll_improved,
            "accuracy_within_tolerance": accuracy_within_tolerance,
            "negative_log_likelihood_within_tolerance": nll_within_tolerance,
            "passed": gate_passed,
            "scope": "advance to per-class covariance temperature only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize M15.2a global temperature.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_path = Path(args.input)
    runs = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize_runs(runs)
    summary["source_artifact"] = input_path.as_posix()
    if "dplus2" in input_path.stem:
        summary["protocol"] += "_dplus2"
        summary["sphere_seed_rule"] = "d_plus_2"
    Path(args.output).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(
        f"Summarized {summary['unique_experiments']} experiments; "
        f"advancement gate passed={summary['advancement_gate']['passed']}."
    )


if __name__ == "__main__":
    main()