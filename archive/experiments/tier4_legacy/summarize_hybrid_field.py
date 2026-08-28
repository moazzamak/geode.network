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
    converged = True
    for run in unique_runs.values():
        seed = int(run["config"]["seed"])
        for record in run["metrics"]["records"]:
            converged = converged and bool(record["converged"])
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
    hybrid_by_seed = {row["seed"]: row for row in grouped["hybrid"]}
    for baseline in ("geometric", "probabilistic"):
        baseline_by_seed = {row["seed"]: row for row in grouped[baseline]}
        seeds = sorted(set(hybrid_by_seed) & set(baseline_by_seed))
        paired[baseline] = {
            "seeds": seeds,
            **{
                f"hybrid_minus_{metric}": {
                    "mean": float(np.mean(differences)),
                    "per_seed": [float(value) for value in differences],
                }
                for metric in METRICS
                for differences in [[
                    hybrid_by_seed[seed][metric] - baseline_by_seed[seed][metric]
                    for seed in seeds
                ]]
            },
        }

    hybrid_nll = grouped_results["hybrid"]["negative_log_likelihood_mean"]
    hybrid_accuracy = grouped_results["hybrid"]["accuracy_mean"]
    nll_improves_both = all(
        hybrid_nll < grouped_results[baseline]["negative_log_likelihood_mean"]
        for baseline in ("geometric", "probabilistic")
    )
    accuracy_within_tolerance = all(
        hybrid_accuracy >= grouped_results[baseline]["accuracy_mean"] - accuracy_tolerance
        for baseline in ("geometric", "probabilistic")
    )
    accuracy_improves_both = all(
        hybrid_accuracy > grouped_results[baseline]["accuracy_mean"]
        for baseline in ("geometric", "probabilistic")
    )
    nll_within_tolerance = all(
        hybrid_nll <= grouped_results[baseline]["negative_log_likelihood_mean"] + nll_tolerance
        for baseline in ("geometric", "probabilistic")
    )
    gate_passed = converged and (
        (nll_improves_both and accuracy_within_tolerance)
        or (accuracy_improves_both and nll_within_tolerance)
    )
    return {
        "schema_version": 1,
        "protocol": "tier4_hybrid_field_ablation",
        "source_artifact": "logs/results/tier4_hybrid_field_ablation_runs.jsonl",
        "source_rows": len(runs),
        "unique_experiments": len(unique_runs),
        "duplicate_retries": len(runs) - len(unique_runs),
        "seeds": sorted(int(run["config"]["seed"]) for run in unique_runs.values()),
        "selection_winners": dict(sorted(Counter(
            run["metrics"]["selected_score_input"] for run in unique_runs.values()
        ).items())),
        "all_readouts_converged": converged,
        "matched_frozen_models": True,
        "model_fits_per_seed": 1,
        "readout_fits_per_seed": 4,
        "test_used_for_selection": False,
        "grouped_test_results": grouped_results,
        "paired_differences": paired,
        "advancement_gate": {
            "accuracy_tolerance": accuracy_tolerance,
            "nll_tolerance": nll_tolerance,
            "nll_improves_both": nll_improves_both,
            "accuracy_within_tolerance": accuracy_within_tolerance,
            "accuracy_improves_both": accuracy_improves_both,
            "nll_within_tolerance": nll_within_tolerance,
            "passed": gate_passed,
            "scope": "advance to frozen-geometry likelihood optimization only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the M15.1 hybrid field study.")
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