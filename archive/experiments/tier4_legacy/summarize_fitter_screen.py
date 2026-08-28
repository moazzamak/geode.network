import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.common.classification_metrics import paired_bootstrap_interval


def summarize_runs(runs: list[dict], bootstrap_resamples: int) -> dict:
    candidates = ("shrinkage_covariance", "minimum_covariance_determinant")
    comparisons = {}
    for candidate in candidates:
        comparisons[candidate] = {}
        for split in ("selection", "test"):
            differences = []
            targets = []
            candidate_predictions = []
            current_predictions = []
            for run in runs:
                records = [
                    record for record in run["metrics"]["records"]
                    if record["split"] == split and record["readout"] == "multinomial"
                ]
                current = next(
                    record for record in records
                    if record["geometry_variant"] == "fitter_current"
                )
                alternative = next(
                    record for record in records
                    if record["geometry_variant"] == f"fitter_{candidate}"
                )
                differences.append(
                    alternative["metrics"]["accuracy"]
                    - current["metrics"]["accuracy"]
                )
                targets.extend(alternative["targets"])
                candidate_predictions.extend(alternative["predictions"])
                current_predictions.extend(current["predictions"])
            differences = np.asarray(differences)
            comparisons[candidate][split] = {
                "mean_seed_difference": float(differences.mean()),
                "seed_standard_deviation": float(differences.std()),
                "wins": int(np.sum(differences > 0.0)),
                "ties": int(np.sum(differences == 0.0)),
                "pooled_paired_bootstrap": paired_bootstrap_interval(
                    np.asarray(targets),
                    np.asarray(candidate_predictions),
                    np.asarray(current_predictions),
                    n_resamples=bootstrap_resamples,
                    seed=42,
                ),
            }
    resources = {}
    for fitter in ("current", *candidates):
        records = []
        for run in runs:
            records.append(next(
                record for record in run["metrics"]["records"]
                if record["split"] == "selection"
                and record["readout"] == "multinomial"
                and record["geometry_variant"] == f"fitter_{fitter}"
            ))
        resources[fitter] = {
            "mean_fit_seconds": float(np.mean([
                record["performance"]["geometry_fit_seconds"] for record in records
            ])),
            "mean_experts": float(np.mean([
                record["model_stats"]["experts"] for record in records
            ])),
            "mean_additive_ellipsoids": float(np.mean([
                record["model_stats"]["additive_ellipsoids"] for record in records
            ])),
            "mean_empty_classes": float(np.mean([
                record["model_stats"]["empty_classes"] for record in records
            ])),
        }
    return {
        "seeds": [int(run["seed"]) for run in runs],
        "selection_metric": "multinomial_accuracy",
        "test_used_for_selection": False,
        "comparisons": comparisons,
        "resources": resources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Tier 4 fitter screens.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 53, 71])
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()
    runs_by_seed = {}
    with Path(args.input).open("r", encoding="utf-8") as stream:
        for line in stream:
            run = json.loads(line)
            if run["seed"] in args.seeds:
                runs_by_seed[int(run["seed"])] = run
    missing = set(args.seeds) - set(runs_by_seed)
    if missing:
        raise ValueError(f"Missing fitter-screen runs for seeds: {sorted(missing)}")
    summary = summarize_runs(
        [runs_by_seed[seed] for seed in args.seeds], args.bootstrap_resamples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()