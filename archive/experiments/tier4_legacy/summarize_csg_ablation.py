import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.common.classification_metrics import paired_bootstrap_interval


def _numeric_means(records: list[dict], field: str) -> dict[str, float]:
    keys = sorted({
        key for record in records for key, value in record[field].items()
        if isinstance(value, (int, float)) and value is not None
    })
    return {
        key: float(np.mean([
            record[field][key] for record in records
            if isinstance(record[field].get(key), (int, float))
            and record[field][key] is not None
        ]))
        for key in keys
    }


def summarize_runs(runs: list[dict], bootstrap_resamples: int = 5000) -> dict:
    readouts = ("raw", "temperature", "diagonal", "multinomial")
    comparisons = (("A1", "A0"), ("A2", "A1"))
    summaries = {}
    for readout in readouts:
        summaries[readout] = {}
        for first_variant, second_variant in comparisons:
            seed_differences = []
            pooled_targets = []
            pooled_first = []
            pooled_second = []
            for run in runs:
                records = [
                    record for record in run["metrics"]["records"]
                    if record["readout"] == readout
                ]
                first = next(
                    record for record in records
                    if record["geometry_variant"] == first_variant
                )
                second = next(
                    record for record in records
                    if record["geometry_variant"] == second_variant
                )
                seed_differences.append(
                    first["metrics"]["accuracy"] - second["metrics"]["accuracy"]
                )
                pooled_targets.extend(first["targets"])
                pooled_first.extend(first["predictions"])
                pooled_second.extend(second["predictions"])
            seed_differences = np.asarray(seed_differences, dtype=np.float64)
            interval = paired_bootstrap_interval(
                np.asarray(pooled_targets),
                np.asarray(pooled_first),
                np.asarray(pooled_second),
                n_resamples=bootstrap_resamples,
                seed=42,
            )
            summaries[readout][f"{first_variant}-{second_variant}"] = {
                "mean_seed_difference": float(seed_differences.mean()),
                "seed_standard_deviation": float(seed_differences.std()),
                "pooled_paired_bootstrap": interval,
            }

    audits = {}
    for variant in ("A1", "A2"):
        decisions = [
            decision
            for run in runs
            for decision in run["metrics"]["carve_audits"][variant]
        ]
        audits[variant] = {
            "decisions": len(decisions),
            "accepted": sum(decision["accepted"] for decision in decisions),
            "rejected": sum(not decision["accepted"] for decision in decisions),
            "recovered_false_positives": sum(
                decision["recovered_false_positives"]
                for decision in decisions if decision["accepted"]
            ),
            "damaged_true_positives": sum(
                decision["damaged_true_positives"]
                for decision in decisions if decision["accepted"]
            ),
        }

    grouped_records: dict[tuple[str, str, str], list[dict]] = {}
    for run in runs:
        for record in run["metrics"]["records"]:
            key = (
                record["method"],
                record["geometry_variant"],
                record["readout"],
            )
            grouped_records.setdefault(key, []).append(record)
    absolute_results = []
    for (method, geometry_variant, readout), records in sorted(grouped_records.items()):
        absolute_results.append({
            "method": method,
            "geometry_variant": geometry_variant,
            "readout": readout,
            "seed_count": len(records),
            "metrics_mean": _numeric_means(records, "metrics"),
            "performance_mean": _numeric_means(records, "performance"),
            "model_stats_mean": _numeric_means(records, "model_stats"),
        })
    return {
        "seeds": [int(run["seed"]) for run in runs],
        "run_count": len(runs),
        "absolute_results": absolute_results,
        "readout_deltas": summaries,
        "carve_audits": audits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize saved Tier 4 CSG runs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 53, 71])
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()

    runs_by_seed = {}
    with Path(args.input).open("r", encoding="utf-8") as stream:
        for line in stream:
            run = json.loads(line)
            if (
                run["config"].get("max_iterations") == args.max_iterations
                and run["seed"] in args.seeds
            ):
                runs_by_seed[int(run["seed"])] = run
    missing = set(args.seeds) - set(runs_by_seed)
    if missing:
        raise ValueError(f"Missing CSG runs for seeds: {sorted(missing)}")
    runs = [runs_by_seed[seed] for seed in args.seeds]
    summary = summarize_runs(runs, args.bootstrap_resamples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()