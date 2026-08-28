import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.tier4.summarize_global_temperature import METRICS


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
    optimization_converged = True
    readouts_converged = True
    temperature_rows = []
    calibration_improvements = []
    for run in unique_runs.values():
        seed = int(run["config"]["seed"])
        optimization = run["metrics"]["per_class_likelihood_optimization"]
        optimization_converged &= bool(optimization["converged"])
        temperatures = np.asarray(
            optimization["covariance_temperatures"], dtype=np.float64,
        )
        temperature_rows.append({
            "seed": seed,
            "minimum": float(np.min(temperatures)),
            "maximum": float(np.max(temperatures)),
            "mean": float(np.mean(temperatures)),
            "values": [float(value) for value in temperatures],
        })
        calibration_improvements.append(
            float(optimization["fitted_calibration_nll"])
            - float(optimization["baseline_calibration_nll"])
        )
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
        ("probabilistic_per_class_temperature", "probabilistic_global_temperature"),
        ("hybrid_per_class_temperature", "hybrid_global_temperature"),
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

    tuned = grouped_results["hybrid_per_class_temperature"]
    baseline = grouped_results["hybrid_global_temperature"]
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
        "protocol": "tier4_per_class_covariance_temperature",
        "source_artifact": "logs/results/tier4_per_class_covariance_temperature_runs.jsonl",
        "source_rows": len(runs),
        "unique_experiments": len(unique_runs),
        "duplicate_retries": len(runs) - len(unique_runs),
        "seeds": sorted(int(run["config"]["seed"]) for run in unique_runs.values()),
        "selection_winners": dict(sorted(Counter(
            run["metrics"]["selected_score_input"] for run in unique_runs.values()
        ).items())),
        "per_class_temperatures": temperature_rows,
        "calibration_nll_improvement": {
            "mean": float(np.mean(calibration_improvements)),
            "per_seed": calibration_improvements,
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
            "scope": "stop M15.2 if false; otherwise advance to mixture weights",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize M15.2b per-class temperature.")
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