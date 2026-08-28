"""Summarize matched primitive-family by subtractive-CSG ablation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


FAMILIES = (
    "full_covariance",
    "diagonal_covariance",
    "spherical_covariance",
)
VARIANTS = ("A0", "A1", "A2")


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


def _record(run: dict, variant: str) -> dict:
    expected = f"{run['metrics']['fitter']}_{variant}"
    return next(
        record
        for record in run["metrics"]["records"]
        if record["split"] == "test"
        and record["readout"] == "multinomial"
        and record["geometry_variant"] == expected
    )


def summarize_runs(runs: list[dict]) -> dict:
    indexed = {
        (run["metrics"]["fitter"], int(run["config"]["seed"])): run
        for run in runs
    }
    if len(indexed) != len(runs):
        raise ValueError("Duplicate fitter/seed runs are not allowed.")
    seeds = sorted({seed for _, seed in indexed})
    missing = [
        (family, seed)
        for family in FAMILIES
        for seed in seeds
        if (family, seed) not in indexed
    ]
    if missing:
        raise ValueError(f"Missing fitter/seed runs: {missing}")

    families = {}
    for family in FAMILIES:
        family_runs = [indexed[(family, seed)] for seed in seeds]
        variants = {}
        for variant in VARIANTS:
            records = [_record(run, variant) for run in family_runs]
            variants[variant] = {
                "accuracy_mean": float(np.mean([
                    record["metrics"]["accuracy"] for record in records
                ])),
                "negative_log_likelihood_mean": float(np.mean([
                    record["metrics"]["negative_log_likelihood"]
                    for record in records
                ])),
                "additive_primitives_mean": float(np.mean([
                    record["model_stats"]["additive_ellipsoids"]
                    for record in records
                ])),
                "subtractive_primitives_mean": float(np.mean([
                    record["model_stats"]["subtractive_ellipsoids"]
                    for record in records
                ])),
            }
        a0_accuracy = np.asarray([
            _record(run, "A0")["metrics"]["accuracy"] for run in family_runs
        ])
        for variant in ("A1", "A2"):
            records = [_record(run, variant) for run in family_runs]
            differences = 100.0 * (
                np.asarray([
                    record["metrics"]["accuracy"] for record in records
                ])
                - a0_accuracy
            )
            variants[f"{variant}_minus_A0"] = {
                "mean_pp": float(np.mean(differences)),
                "per_seed_pp": differences.tolist(),
                "t95_ci_pp": _t_interval(differences),
                "changed_predictions": int(sum(
                    np.count_nonzero(
                        np.asarray(_record(run, variant)["predictions"])
                        != np.asarray(_record(run, "A0")["predictions"])
                    )
                    for run in family_runs
                )),
                "accepted_carvings": int(sum(
                    decision["accepted"]
                    for run in family_runs
                    for phase in (
                        ("A1",) if variant == "A1" else ("A1", "A2")
                    )
                    for decision in run["metrics"]["carve_audits"][phase]
                )),
            }
        families[family] = variants

    sphere_a0 = np.asarray([
        _record(indexed[("spherical_covariance", seed)], "A0")
        ["metrics"]["accuracy"]
        for seed in seeds
    ])
    paired = {}
    for comparison in ("diagonal_covariance", "full_covariance"):
        differences = 100.0 * (
            sphere_a0
            - np.asarray([
                _record(indexed[(comparison, seed)], "A0")
                ["metrics"]["accuracy"]
                for seed in seeds
            ])
        )
        paired[f"sphere_minus_{comparison}"] = {
            "mean_pp": float(np.mean(differences)),
            "per_seed_pp": differences.tolist(),
            "t95_ci_pp": _t_interval(differences),
        }

    return {
        "schema_version": 1,
        "protocol": "tier4_primitive_csg_ablation_dplus2",
        "comparative_result": True,
        "sphere_seed_rule": "d_plus_2",
        "seeds": seeds,
        "run_count": len(runs),
        "families": families,
        "paired_A0_accuracy_differences": paired,
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
