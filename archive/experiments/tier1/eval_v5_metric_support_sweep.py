from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.classification_metrics import (
    balanced_accuracy,
    negative_log_likelihood,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    serialized_size,
    sha256_file,
    write_canonical_json,
)
from src.metric_parameterization import (
    METRIC_FAMILIES,
    fit_class_precision_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v5" / "metric_support_sweep.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v5" / "m18_metric_support"


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "milestone",
        "seeds",
        "eigenvalue_floor",
        "complexity_penalty",
        "evaluation_samples_per_class",
        "candidate_ranks",
        "families",
        "development",
        "test",
        "support_bins",
        "advancement",
    }
    if set(config) != required or config.get("schema_version") != 1:
        raise ValueError("Unsupported M18 sweep schema.")
    if config["milestone"] != "M18":
        raise ValueError("Metric support sweep must identify M18.")
    if tuple(config["seeds"]) != (11, 23, 37):
        raise ValueError("M18 development/test seeds are frozen to 11, 23, and 37.")
    if set(config["families"]) != set(METRIC_FAMILIES):
        raise ValueError("M18 must compare every registered metric family.")
    advancement = config["advancement"]
    if (
        set(advancement)
        != {"minimum_resource_reduction", "require_lower_aggregate_loss"}
        or isinstance(advancement["minimum_resource_reduction"], bool)
        or not isinstance(advancement["minimum_resource_reduction"], (int, float))
        or not 0.0 <= advancement["minimum_resource_reduction"] <= 1.0
        or not isinstance(advancement["require_lower_aggregate_loss"], bool)
    ):
        raise ValueError("Invalid M18 advancement configuration.")
    ranks = config["candidate_ranks"]
    if not ranks or ranks[0] != 0 or any(
        isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
        for rank in ranks
    ):
        raise ValueError("candidate_ranks must begin at zero and be nonnegative.")
    axis_fields = {
        "dimensions",
        "intrinsic_ranks",
        "samples_per_class",
        "condition_numbers",
        "contamination_fractions",
        "class_separations",
    }
    for split in ("development", "test"):
        axes = config[split]
        if set(axes) != axis_fields or any(not axes[field] for field in axis_fields):
            raise ValueError(f"{split} must define every non-empty sweep axis.")


def _candidate_keys(config: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    for family in config["families"]:
        if "low_rank" in family:
            keys.extend(
                f"{family}:rank={rank}" for rank in config["candidate_ranks"]
            )
        else:
            keys.append(family)
    return keys


def _parse_candidate(candidate: str) -> tuple[str, int]:
    if ":rank=" not in candidate:
        return candidate, 0
    family, rank = candidate.split(":rank=", 1)
    return family, int(rank)


def _orthogonal_matrix(dimension: int, rng: np.random.Generator) -> np.ndarray:
    matrix, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.sign(np.diag(matrix))
    signs[signs == 0.0] = 1.0
    return matrix * signs


def _covariance(
    dimension: int,
    intrinsic_rank: int,
    condition_number: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    rank = min(max(int(intrinsic_rank), 1), dimension)
    high = np.geomspace(1.0, 1.0 / condition_number, rank)
    eigenvalues = np.full(dimension, 1.0 / condition_number)
    eigenvalues[:rank] = high
    orientation = _orthogonal_matrix(dimension, rng)
    covariance = (orientation * eigenvalues) @ orientation.T
    return (covariance + covariance.T) * 0.5, orientation[:, 0]


def _generate_cell(
    *,
    dimension: int,
    intrinsic_rank: int,
    samples_per_class: int,
    condition_number: float,
    contamination_fraction: float,
    class_separation: float,
    evaluation_samples_per_class: int,
    seed: int,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    covariance, separation_axis = _covariance(
        dimension, intrinsic_rank, condition_number, rng
    )
    means = {
        0: -0.5 * class_separation * separation_axis,
        1: 0.5 * class_separation * separation_axis,
    }
    class_points = {
        class_id: rng.multivariate_normal(
            mean, covariance, size=samples_per_class
        )
        for class_id, mean in means.items()
    }
    contamination_count = int(round(contamination_fraction * samples_per_class))
    if contamination_count:
        for class_id in (0, 1):
            indices = rng.choice(
                samples_per_class, contamination_count, replace=False
            )
            class_points[class_id][indices] = rng.multivariate_normal(
                means[1 - class_id],
                covariance * 2.0,
                size=contamination_count,
            )
    evaluation = np.vstack(
        [
            rng.multivariate_normal(
                means[class_id],
                covariance,
                size=evaluation_samples_per_class,
            )
            for class_id in (0, 1)
        ]
    )
    labels = np.repeat([0, 1], evaluation_samples_per_class)
    return class_points, evaluation, labels


def _probabilities(
    fits: Mapping[int | str, Any],
    evaluation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    class_ids = np.asarray(sorted(fits, key=str))
    energies = []
    for class_id in class_ids:
        fit = fits[class_id]
        deltas = evaluation - fit.center
        energies.append(
            0.5
            * (
                fit.metric.quadratic_form(deltas)
                - fit.metric.log_determinant()
            )
        )
    energy_matrix = np.column_stack(energies)
    logits = -energy_matrix
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return class_ids, probabilities


def _evaluate_candidate(
    class_points: Mapping[int, np.ndarray],
    evaluation: np.ndarray,
    labels: np.ndarray,
    candidate: str,
    *,
    eigenvalue_floor: float,
) -> dict[str, Any]:
    family, rank = _parse_candidate(candidate)
    fits = fit_class_precision_metrics(
        class_points,
        family,
        rank=rank,
        eigenvalue_floor=eigenvalue_floor,
    )
    classes, probabilities = _probabilities(fits, evaluation)
    predictions = classes[probabilities.argmax(axis=1)]
    warnings = sorted(
        {
            warning
            for fit in fits.values()
            for warning in fit.warnings
        }
    )
    shared_count = max(
        (fit.metric.shared_parameter_count for fit in fits.values()),
        default=0,
    )
    local_count = sum(fit.metric.local_parameter_count for fit in fits.values())
    parameter_count = local_count + shared_count
    parameter_bytes = parameter_count * np.dtype(np.float64).itemsize
    dimension = next(iter(class_points.values())).shape[1]
    total_support = sum(len(points) for points in class_points.values())
    class_count = len(class_points)
    if family in {"spherical", "diagonal"}:
        fit_work_units = total_support * dimension
    elif family in {"full", "diagonal_low_rank"}:
        fit_work_units = (
            total_support * dimension**2 + class_count * dimension**3
        )
    else:
        fit_work_units = (
            total_support * dimension**2
            + dimension**3
            + total_support * dimension
        )
    serialized_bytes = serialized_size(
        {str(class_id): fit.metric.to_dict() for class_id, fit in fits.items()}
    )
    return {
        "candidate": candidate,
        "family": family,
        "rank": rank,
        "balanced_accuracy": balanced_accuracy(labels, predictions),
        "negative_log_likelihood": negative_log_likelihood(
            labels, probabilities, classes
        ),
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
        "serialized_bytes": serialized_bytes,
        "fit_work_units": fit_work_units,
        "warnings": warnings,
    }


def _axis_cells(axes: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    names = list(axes)
    normalized_names = {
        "dimensions": "dimension",
        "intrinsic_ranks": "intrinsic_rank",
        "samples_per_class": "samples_per_class",
        "condition_numbers": "condition_number",
        "contamination_fractions": "contamination_fraction",
        "class_separations": "class_separation",
    }
    return [
        {
            normalized_names[name]: value
            for name, value in zip(names, values)
        }
        for values in itertools.product(*(axes[name] for name in names))
    ]


def _support_bin(
    record: Mapping[str, Any],
    support_bins: Mapping[str, Any],
) -> str:
    ratio = record["samples_per_class"] / record["dimension"]
    low_edge, high_edge = support_bins["sample_dimension_edges"]
    if ratio <= low_edge:
        support = "low"
    elif ratio <= high_edge:
        support = "medium"
    else:
        support = "high"
    rank_fraction = record["intrinsic_rank"] / record["dimension"]
    rank_group = (
        "low_rank"
        if rank_fraction <= support_bins["rank_fraction_edge"]
        else "high_rank"
    )
    return f"{support}:{rank_group}"


def _record_loss(record: Mapping[str, Any], complexity_penalty: float) -> float:
    return float(
        record["negative_log_likelihood"]
        + 0.25 * (1.0 - record["balanced_accuracy"])
        + complexity_penalty * math.log1p(record["parameter_count"])
    )


def _select_policy(
    development_records: list[dict[str, Any]],
    candidate_keys: list[str],
    support_bins: Mapping[str, Any],
    complexity_penalty: float,
) -> dict[str, Any]:
    by_candidate = {
        candidate: [
            _record_loss(record, complexity_penalty)
            for record in development_records
            if record["candidate"] == candidate
        ]
        for candidate in candidate_keys
    }
    global_candidate = min(
        candidate_keys,
        key=lambda candidate: (float(np.mean(by_candidate[candidate])), candidate),
    )
    bins = sorted(
        {
            _support_bin(record, support_bins)
            for record in development_records
        }
    )
    selections: dict[str, str] = {}
    for bin_name in bins:
        selections[bin_name] = min(
            candidate_keys,
            key=lambda candidate: (
                float(
                    np.mean(
                        [
                            _record_loss(record, complexity_penalty)
                            for record in development_records
                            if record["candidate"] == candidate
                            and _support_bin(record, support_bins) == bin_name
                        ]
                    )
                ),
                candidate,
            ),
        )
    return {
        "schema_version": 1,
        "global_candidate": global_candidate,
        "support_bin_selections": selections,
        "support_bins": dict(support_bins),
        "complexity_penalty": complexity_penalty,
        "selected_from": "development_only",
    }


def _evaluate_policy(
    test_records: list[dict[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    selected_records = []
    global_records = []
    fallback_warnings = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in test_records:
        grouped.setdefault(record["cell_id"], []).append(record)
    for cell_id in sorted(grouped):
        records = grouped[cell_id]
        bin_name = _support_bin(records[0], policy["support_bins"])
        candidate = policy["support_bin_selections"].get(bin_name)
        if candidate is None:
            candidate = policy["global_candidate"]
            fallback_warnings.append(
                {
                    "cell_id": cell_id,
                    "warning": "unseen_support_bin_used_global_candidate",
                    "support_bin": bin_name,
                }
            )
        selected_records.append(
            next(record for record in records if record["candidate"] == candidate)
        )
        global_records.append(
            next(
                record
                for record in records
                if record["candidate"] == policy["global_candidate"]
            )
        )

    def summarize(records: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "mean_aggregate_loss": float(
                np.mean(
                    [
                        record["negative_log_likelihood"]
                        + 0.25 * (1.0 - record["balanced_accuracy"])
                        for record in records
                    ]
                )
            ),
            "mean_balanced_accuracy": float(
                np.mean([record["balanced_accuracy"] for record in records])
            ),
            "mean_negative_log_likelihood": float(
                np.mean([record["negative_log_likelihood"] for record in records])
            ),
            "median_parameter_count": float(
                np.median([record["parameter_count"] for record in records])
            ),
            "median_parameter_bytes": float(
                np.median([record["parameter_bytes"] for record in records])
            ),
            "median_serialized_bytes": float(
                np.median([record["serialized_bytes"] for record in records])
            ),
            "median_fit_work_units": float(
                np.median([record["fit_work_units"] for record in records])
            ),
        }

    policy_summary = summarize(selected_records)
    global_summary = summarize(global_records)
    parameter_byte_reduction = 1.0 - (
        policy_summary["median_parameter_bytes"]
        / global_summary["median_parameter_bytes"]
    )
    fit_work_reduction = 1.0 - (
        policy_summary["median_fit_work_units"]
        / global_summary["median_fit_work_units"]
    )
    return {
        "policy": policy_summary,
        "best_frozen_single_family": global_summary,
        "parameter_byte_reduction": parameter_byte_reduction,
        "fit_work_reduction": fit_work_reduction,
        "fallback_warnings": fallback_warnings,
        "selected_candidates": [
            {
                "cell_id": record["cell_id"],
                "candidate": record["candidate"],
                "support_bin": _support_bin(record, policy["support_bins"]),
            }
            for record in selected_records
        ],
    }


def run_sweep(config: Mapping[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    candidates = _candidate_keys(config)
    records: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "test": [],
    }
    for split in ("development", "test"):
        for axes in _axis_cells(config[split]):
            for seed in config["seeds"]:
                cell_payload = {"split": split, "seed": seed, **axes}
                cell_id = payload_hash(cell_payload)[:16]
                class_points, evaluation, labels = _generate_cell(
                    **axes,
                    evaluation_samples_per_class=config[
                        "evaluation_samples_per_class"
                    ],
                    seed=seed,
                )
                for candidate in candidates:
                    result = _evaluate_candidate(
                        class_points,
                        evaluation,
                        labels,
                        candidate,
                        eigenvalue_floor=config["eigenvalue_floor"],
                    )
                    records[split].append(
                        {
                            "cell_id": cell_id,
                            "split": split,
                            "seed": seed,
                            **axes,
                            **result,
                        }
                    )
    policy = _select_policy(
        records["development"],
        candidates,
        config["support_bins"],
        config["complexity_penalty"],
    )
    evaluation = _evaluate_policy(records["test"], policy)
    minimum_reduction = config["advancement"]["minimum_resource_reduction"]
    predictive_required = config["advancement"]["require_lower_aggregate_loss"]
    predictive_pass = (
        evaluation["policy"]["mean_aggregate_loss"]
        < evaluation["best_frozen_single_family"]["mean_aggregate_loss"]
    )
    resource_pass = max(
        evaluation["parameter_byte_reduction"],
        evaluation["fit_work_reduction"],
    ) >= minimum_reduction
    gate = {
        "predictive_pass": predictive_pass,
        "predictive_required": predictive_required,
        "resource_pass": resource_pass,
        "minimum_resource_reduction": minimum_reduction,
        "advancement_passed": (
            (predictive_pass or not predictive_required) and resource_pass
        ),
    }
    return {
        "schema_version": 1,
        "milestone": "M18",
        "config_hash": payload_hash(config),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "candidate_keys": candidates,
        "development_records": records["development"],
        "test_records": records["test"],
        "policy": policy,
        "evaluation": evaluation,
        "gate": gate,
    }


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    source_paths = [
        REPO_ROOT / "experiments" / "common" / "classification_metrics.py",
        REPO_ROOT / "experiments" / "common" / "v5_artifacts.py",
        REPO_ROOT / "experiments" / "tier1" / "eval_v5_metric_support_sweep.py",
        REPO_ROOT / "src" / "metric_parameterization.py",
        DEFAULT_CONFIG,
    ]
    source_hash = payload_hash(
        {
            path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
            for path in source_paths
        }
    )
    return {"commit": commit, "dirty": dirty, "source_hash": source_hash}


def write_sweep(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = run_sweep(config)
    result["source"] = _git_state()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "config_snapshot.json", config)
    write_canonical_json(output_dir / "support_sweep.json", result)
    write_canonical_json(output_dir / "frozen_metric_policy.json", result["policy"])
    summary = {
        "schema_version": 1,
        "milestone": "M18",
        "development_record_count": len(result["development_records"]),
        "test_record_count": len(result["test_records"]),
        "global_candidate": result["policy"]["global_candidate"],
        "support_bin_selections": result["policy"]["support_bin_selections"],
        "evaluation": result["evaluation"],
        "gate": result["gate"],
    }
    summary["artifact_count"] = 4
    write_canonical_json(output_dir / "summary.json", summary)
    index = build_artifact_index(output_dir)
    if len(index["artifacts"]) != summary["artifact_count"]:
        raise RuntimeError("M18 artifact count does not match the frozen summary.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M18 metric support sweep.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(write_sweep(args.config, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
