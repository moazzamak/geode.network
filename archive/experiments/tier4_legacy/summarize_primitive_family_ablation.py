"""Summarize matched primitive-family ablation runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import t


FAMILIES = (
    "full_covariance",
    "diagonal_covariance",
    "spherical_covariance",
)


def _t_interval(values: np.ndarray) -> list[float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return [mean, mean]
    half_width = float(
        t.ppf(0.975, len(values) - 1)
        * np.std(values, ddof=1)
        / np.sqrt(len(values))
    )
    return [mean - half_width, mean + half_width]


def summarize_runs(runs: list[dict]) -> dict:
    by_seed = {int(run["config"]["seed"]): run for run in runs}
    if len(by_seed) != len(runs):
        raise ValueError("Duplicate seeds are not allowed in a corrected rerun.")
    family_records: dict[str, list[dict]] = {family: [] for family in FAMILIES}
    for seed in sorted(by_seed):
        records = by_seed[seed]["metrics"]["records"]
        for family in FAMILIES:
            family_records[family].append(next(
                record
                for record in records
                if record["split"] == "test"
                and record["readout"] == "multinomial"
                and record["geometry_variant"] == f"fitter_{family}"
            ))

    families = {}
    for family, records in family_records.items():
        families[family] = {
            "accuracy_mean": float(np.mean([
                record["metrics"]["accuracy"] for record in records
            ])),
            "negative_log_likelihood_mean": float(np.mean([
                record["metrics"]["negative_log_likelihood"]
                for record in records
            ])),
            "fit_seconds_mean": float(np.mean([
                record["performance"]["geometry_fit_seconds"]
                for record in records
            ])),
            "primitives_mean": float(np.mean([
                record["model_stats"]["additive_ellipsoids"]
                for record in records
            ])),
        }

    sphere_accuracy = np.asarray([
        record["metrics"]["accuracy"]
        for record in family_records["spherical_covariance"]
    ])
    paired = {}
    for comparison in ("diagonal_covariance", "full_covariance"):
        differences = 100.0 * (
            sphere_accuracy
            - np.asarray([
                record["metrics"]["accuracy"]
                for record in family_records[comparison]
            ])
        )
        paired[f"sphere_minus_{comparison}"] = {
            "mean_pp": float(np.mean(differences)),
            "per_seed_pp": differences.tolist(),
            "t95_ci_pp": _t_interval(differences),
        }

    return {
        "schema_version": 1,
        "protocol": "tier4_primitive_family_ablation_dplus2",
        "comparative_result": True,
        "sphere_seed_rule": "d_plus_2",
        "seeds": sorted(by_seed),
        "selection_winners": dict(sorted(Counter(
            run["metrics"]["selected_fitter"] for run in by_seed.values()
        ).items())),
        "families": families,
        "paired_accuracy_differences": paired,
        "test_used_for_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runs = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize_runs(runs)
    output = Path(args.output)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
