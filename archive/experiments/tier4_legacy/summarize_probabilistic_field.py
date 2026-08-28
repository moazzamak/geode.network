import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "accuracy",
    "negative_log_likelihood",
    "brier_score",
    "expected_calibration_error",
)


def summarize_runs(runs: list[dict]) -> dict:
    unique_runs = {}
    for run in runs:
        unique_runs.setdefault(run["experiment_id"], run)
    grouped = defaultdict(list)
    for run in unique_runs.values():
        seed = int(run["config"]["seed"])
        for record in run["metrics"]["records"]:
            if record["split"] != "test" or record["readout"] == "feature_logistic":
                continue
            semantics = record["score_semantics"]
            suffix = f"_{semantics}"
            family = record["geometry_variant"].removeprefix("fitter_").removesuffix(suffix)
            grouped[(family, semantics, record["readout"])].append({
                "seed": seed,
                **{metric: float(record["metrics"][metric]) for metric in METRICS},
            })

    grouped_results = {}
    for (family, semantics, readout), rows in sorted(grouped.items()):
        key = f"{family}:{semantics}:{readout}"
        grouped_results[key] = {
            "seeds": [row["seed"] for row in rows],
            **{
                f"{metric}_{statistic}": float(getattr(np, statistic)([
                    row[metric] for row in rows
                ], ddof=1) if statistic == "std" else getattr(np, statistic)([
                    row[metric] for row in rows
                ]))
                for metric in METRICS
                for statistic in ("mean", "std")
            },
        }

    paired_differences = {}
    families = sorted({key[0] for key in grouped})
    readouts = sorted({key[2] for key in grouped})
    for family in families:
        for readout in readouts:
            geometric = {
                row["seed"]: row for row in grouped[(family, "geometric", readout)]
            }
            probabilistic = {
                row["seed"]: row for row in grouped[(family, "probabilistic", readout)]
            }
            common_seeds = sorted(set(geometric) & set(probabilistic))
            if not common_seeds:
                continue
            paired_differences[f"{family}:{readout}"] = {
                "seeds": common_seeds,
                **{
                    f"probabilistic_minus_geometric_{metric}": {
                        "mean": float(np.mean(differences)),
                        "per_seed": [float(value) for value in differences],
                    }
                    for metric in METRICS
                    for differences in [[
                        probabilistic[seed][metric] - geometric[seed][metric]
                        for seed in common_seeds
                    ]]
                },
            }

    return {
        "schema_version": 1,
        "protocol": "tier4_probabilistic_field_ablation",
        "source_artifact": "logs/results/tier4_probabilistic_field_ablation_runs.jsonl",
        "source_rows": len(runs),
        "unique_experiments": len(unique_runs),
        "duplicate_retries": len(runs) - len(unique_runs),
        "seeds": sorted({int(run["config"]["seed"]) for run in unique_runs.values()}),
        "matched_frozen_models": True,
        "probabilistic_fitting_used": False,
        "subtractive_probability_supported": False,
        "test_used_for_selection": False,
        "grouped_test_results": grouped_results,
        "paired_differences": paired_differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the M14 field ablation.")
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
        f"Summarized {summary['unique_experiments']} unique experiments "
        f"to {args.output}."
    )


if __name__ == "__main__":
    main()