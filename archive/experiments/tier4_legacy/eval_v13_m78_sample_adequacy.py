"""M78 — sample-adequacy and basis-identifiability forensics (GEODE v13).

Registered hypothesis H78: the v12 M74 DomainNet transfer failure is driven by
rank-32 basis estimation from 60 samples per class, not by a transfer property
of the head.

Registration amendment R1: the M70 native DomainNet array holds exactly 100
observations per class, so `geometry_per_class` cannot exceed 60 under the M74
partition contract. The grid is therefore rank-first (Axis A) with a 3x sample
span (Axis B). See `analysis/RESEARCH_IMPLEMENTATION_PLAN_v13.md` Section 5.1.

Registration amendment R2: the first execution measured subspace stability by
fitting `initialize_projected_metric_fields` independently on each half, which
gives each half its own PCA frame and therefore measures projection variance
rather than basis identifiability. That run is void; it is retained at
`logs/results/v13/m78_sample_adequacy_void_r1/`. Stability is now measured
through a single shared projection by `experiments.common.v13_sample_adequacy`,
against a Monte-Carlo random-subspace reference.

Cells are independent and individually seeded, so they are executed in parallel
worker processes. Each worker trains single-threaded under
`torch.use_deterministic_algorithms(True)`, so results are identical to serial
execution.

Reproduce with::

    .\\.venv\\Scripts\\python.exe -m experiments.tier4.eval_v13_m78_sample_adequacy
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v13_sample_adequacy import basis_stability
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.tier4.eval_v12_m74_confirmation_transfer import (
    _control_outputs,
    _domainnet_partitions,
    _field_outputs,
    _fit_field,
    _load_domainnet,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v13" / "m78_sample_adequacy.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v13" / "m78_sample_adequacy"

_WORKER_STATE: dict[str, Any] = {}


def _resolve(path: str) -> Path:
    resolved = (REPO_ROOT / Path(path)).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError("M78 paths must remain inside the repository")
    return resolved


def _verify(specification: dict[str, str]) -> Path:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"M78 immutable artifact hash mismatch: {path}")
    return path


def _cell_config(
    base: dict[str, Any], *, geometry_per_class: int, rank: int
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["rank"] = int(rank)
    config["domainnet_transfer"]["geometry_per_class"] = int(geometry_per_class)
    return config


def _run_cell(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    base_config: dict[str, Any],
    geometry_per_class: int,
    rank: int,
    seed: int,
) -> dict[str, Any]:
    config = _cell_config(
        base_config, geometry_per_class=geometry_per_class, rank=rank
    )
    partitions = _domainnet_partitions(features, labels, config)
    fit_x, fit_y = partitions["geometry_fit"]
    calibration_x, calibration_y = partitions["score_calibration"]
    known_x, known_y = partitions["known_evaluation"]
    unknown_x, unknown_y = partitions["unknown_evaluation"]
    query_x = np.vstack([known_x, unknown_x])
    query_y = np.concatenate([known_y, unknown_y])
    known_classes = np.arange(
        int(config["domainnet_transfer"]["known_class_count"]), dtype=np.int64
    )

    initial, trained, history = _fit_field(fit_x, fit_y, config=config, seed=seed)
    field_metrics, _, _, _ = _field_outputs(
        trained,
        calibration_x,
        calibration_y,
        query_x,
        query_y,
        known_classes=known_classes,
        config=config,
    )
    controls, _ = _control_outputs(
        fit_x,
        fit_y,
        calibration_x,
        calibration_y,
        query_x,
        query_y,
        known_classes=known_classes,
        config=config,
        seed=seed,
    )
    stability_config = base_config["subspace_stability"]
    fitted_rank = int(initial.fields.rank)
    return {
        "geometry_per_class": int(geometry_per_class),
        "requested_rank": int(rank),
        "fitted_rank": fitted_rank,
        "seed": int(seed),
        "samples_per_fitted_dimension": float(
            geometry_per_class / max(fitted_rank, 1)
        ),
        "known_balanced_accuracy": float(field_metrics["known_balanced_accuracy"]),
        "unknown_recall": float(field_metrics["unknown_recall"]),
        "logistic_known_balanced_accuracy": float(
            controls["logistic"]["known_balanced_accuracy"]
        ),
        "logistic_unknown_recall": float(controls["logistic"]["unknown_recall"]),
        "gaussian_known_balanced_accuracy": float(
            controls["gaussian"]["known_balanced_accuracy"]
        ),
        "subspace_stability": basis_stability(
            fit_x,
            fit_y,
            output_dimension=int(config["projection_dimension"]),
            rank=rank,
            random_trials=int(stability_config["random_trials"]),
            random_seed=int(stability_config["random_seed"]),
        ),
        "final_probe_loss": float(history[-1]["probe"]),
        "state_hash": payload_hash(trained.to_dict()),
    }


def _worker_initializer(index_path: str, config_text: str) -> None:
    features, labels = _load_domainnet(Path(index_path))
    _WORKER_STATE["features"] = features
    _WORKER_STATE["labels"] = labels
    _WORKER_STATE["config"] = json.loads(config_text)


def _worker_run(task: tuple[int, int, int]) -> dict[str, Any]:
    geometry_per_class, rank, seed = task
    return _run_cell(
        _WORKER_STATE["features"],
        _WORKER_STATE["labels"],
        base_config=_WORKER_STATE["config"],
        geometry_per_class=geometry_per_class,
        rank=rank,
        seed=seed,
    )


def _aggregate(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(
            (cell["geometry_per_class"], cell["requested_rank"]), []
        ).append(cell)
    summary = []
    for (geometry_per_class, rank), items in sorted(grouped.items()):
        summary.append(
            {
                "geometry_per_class": geometry_per_class,
                "requested_rank": rank,
                "fitted_rank": items[0]["fitted_rank"],
                "samples_per_fitted_dimension": items[0][
                    "samples_per_fitted_dimension"
                ],
                "seed_count": len(items),
                "mean_known_balanced_accuracy": float(
                    np.mean([item["known_balanced_accuracy"] for item in items])
                ),
                "mean_unknown_recall": float(
                    np.mean([item["unknown_recall"] for item in items])
                ),
                "mean_logistic_known_balanced_accuracy": float(
                    np.mean(
                        [item["logistic_known_balanced_accuracy"] for item in items]
                    )
                ),
                "mean_logistic_unknown_recall": float(
                    np.mean([item["logistic_unknown_recall"] for item in items])
                ),
                "mean_subspace_principal_angle_degrees": float(
                    np.mean(
                        [
                            item["subspace_stability"][
                                "mean_principal_angle_degrees"
                            ]
                            for item in items
                        ]
                    )
                ),
                "random_subspace_angle_degrees": items[0]["subspace_stability"][
                    "random_subspace_angle_degrees"
                ],
                "mean_identifiability": float(
                    np.mean(
                        [
                            item["subspace_stability"]["identifiability"]
                            for item in items
                        ]
                    )
                ),
                "effective_stability_rank": items[0]["subspace_stability"][
                    "effective_rank"
                ],
            }
        )
    return summary


def _build_gate(
    summary: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    gate_config = config["gate"]
    axis_b = config["axis_b_sample_sensitivity"]
    threshold = float(gate_config["low_rank_improvement_points"])

    baseline = next(
        item
        for item in summary
        if item["geometry_per_class"] == 60 and item["requested_rank"] == 32
    )
    low_rank = [
        item
        for item in summary
        if item["geometry_per_class"] == 60 and item["requested_rank"] < 32
    ]
    best_accuracy_cell = max(
        low_rank, key=lambda item: item["mean_known_balanced_accuracy"]
    )
    accuracy_gain = 100.0 * (
        best_accuracy_cell["mean_known_balanced_accuracy"]
        - baseline["mean_known_balanced_accuracy"]
    )
    recall_gain = 100.0 * (
        max(item["mean_unknown_recall"] for item in low_rank)
        - baseline["mean_unknown_recall"]
    )

    sample_slope = {}
    for rank in axis_b["ranks"]:
        cells_for_rank = sorted(
            (
                item
                for item in summary
                if item["requested_rank"] == int(rank)
                and item["geometry_per_class"] in axis_b["geometry_per_class"]
            ),
            key=lambda item: item["geometry_per_class"],
        )
        if len(cells_for_rank) >= 2:
            sample_slope[str(rank)] = {
                "accuracy_points_20_to_60": 100.0
                * (
                    cells_for_rank[-1]["mean_known_balanced_accuracy"]
                    - cells_for_rank[0]["mean_known_balanced_accuracy"]
                ),
                "unknown_recall_points_20_to_60": 100.0
                * (
                    cells_for_rank[-1]["mean_unknown_recall"]
                    - cells_for_rank[0]["mean_unknown_recall"]
                ),
            }

    minimum_ratio = float(gate_config["minimum_samples_per_fitted_dimension"])
    void_cells = [
        {
            "geometry_per_class": item["geometry_per_class"],
            "requested_rank": item["requested_rank"],
            "samples_per_fitted_dimension": item["samples_per_fitted_dimension"],
        }
        for item in summary
        if item["samples_per_fitted_dimension"] < minimum_ratio
    ]
    identifiability_floor = float(gate_config["minimum_identifiability"])
    unidentified_cells = [
        {
            "geometry_per_class": item["geometry_per_class"],
            "requested_rank": item["requested_rank"],
            "identifiability": item["mean_identifiability"],
        }
        for item in summary
        if item["mean_identifiability"] < identifiability_floor
    ]

    parity_tolerance = float(gate_config["logistic_parity_tolerance_points"])
    parity_cells = [
        {
            "geometry_per_class": item["geometry_per_class"],
            "requested_rank": item["requested_rank"],
            "accuracy_gap_points": 100.0
            * (
                item["mean_known_balanced_accuracy"]
                - item["mean_logistic_known_balanced_accuracy"]
            ),
        }
        for item in summary
        if 100.0
        * (
            item["mean_logistic_known_balanced_accuracy"]
            - item["mean_known_balanced_accuracy"]
        )
        <= parity_tolerance
    ]

    gate = {
        "maximum_available_geometry_per_class": 60,
        "registration_amendments": ["R1", "R2"],
        "baseline_rank32_n60_accuracy": baseline["mean_known_balanced_accuracy"],
        "baseline_rank32_n60_unknown_recall": baseline["mean_unknown_recall"],
        "baseline_rank32_n60_identifiability": baseline["mean_identifiability"],
        "best_low_rank_cell": {
            "requested_rank": best_accuracy_cell["requested_rank"],
            "mean_known_balanced_accuracy": best_accuracy_cell[
                "mean_known_balanced_accuracy"
            ],
            "mean_identifiability": best_accuracy_cell["mean_identifiability"],
        },
        "best_low_rank_accuracy_gain_points": accuracy_gain,
        "best_low_rank_unknown_recall_gain_points": recall_gain,
        "low_rank_improvement_threshold_points": threshold,
        "w2_defect_confirmed": bool(
            accuracy_gain > threshold or recall_gain > threshold
        ),
        "sample_slope_20_to_60": sample_slope,
        "minimum_samples_per_fitted_dimension": minimum_ratio,
        "void_cells_below_minimum_ratio": void_cells,
        "minimum_identifiability": identifiability_floor,
        "unidentified_cells": unidentified_cells,
        "logistic_parity_cells": parity_cells,
        "m74_cell_is_void": bool(
            any(
                item["geometry_per_class"] == 60 and item["requested_rank"] == 32
                for item in void_cells
            )
        ),
        "unknown_recall_reopened": bool(recall_gain > threshold),
        "final_labels_opened": False,
    }
    gate["h78_confirmed"] = bool(
        gate["w2_defect_confirmed"] or gate["m74_cell_is_void"]
    )
    return gate


def run_m78(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    workers: int | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config_text = config_path.read_text(encoding="utf-8")
    config = json.loads(config_text)
    _verify(config["m74_config"])
    index_path = _verify(config["m70_native_index"])
    features, labels = _load_domainnet(index_path)

    seeds = [int(value) for value in config["seeds"]]
    axis_a = config["axis_a_rank_sensitivity"]
    axis_b = config["axis_b_sample_sensitivity"]
    grid: set[tuple[int, int]] = {
        (int(axis_a["geometry_per_class"]), int(rank)) for rank in axis_a["ranks"]
    }
    grid |= {
        (int(count), int(rank))
        for count in axis_b["geometry_per_class"]
        for rank in axis_b["ranks"]
    }
    tasks = [
        (geometry_per_class, rank, seed)
        for geometry_per_class, rank in sorted(grid)
        for seed in seeds
    ]

    worker_count = workers or min(len(tasks), max(1, (os.cpu_count() or 2) - 2))
    if worker_count > 1:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_worker_initializer,
            initargs=(str(index_path), config_text),
        ) as pool:
            cells = list(pool.map(_worker_run, tasks))
    else:
        cells = [
            _run_cell(
                features,
                labels,
                base_config=config,
                geometry_per_class=geometry_per_class,
                rank=rank,
                seed=seed,
            )
            for geometry_per_class, rank, seed in tasks
        ]

    summary = _aggregate(cells)
    evidence = {
        "schema_version": 2,
        "milestone": "M78",
        "program": "v13",
        "configuration_hash": sha256_file(config_path),
        "registration_amendments": ["R1", "R2"],
        "supersedes": "logs/results/v13/m78_sample_adequacy_void_r1",
        "corpus": {
            "name": "DomainNet (M70 native extraction)",
            "rows": int(len(features)),
            "dimension": int(features.shape[1]),
            "classes": int(len(np.unique(labels))),
            "observations_per_class": int(
                np.min(np.bincount(labels.astype(np.int64)))
            ),
        },
        "seeds": seeds,
        "worker_count": int(worker_count),
        "cells": cells,
        "summary": summary,
        "gate": _build_gate(summary, config),
        "final_labels_opened": False,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=None)
    arguments = parser.parse_args()
    evidence = run_m78(arguments.config, arguments.output, workers=arguments.workers)
    print(json.dumps(evidence["summary"], indent=2, sort_keys=True))
    print(json.dumps(evidence["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
