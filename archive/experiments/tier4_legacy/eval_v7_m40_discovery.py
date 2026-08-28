from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v7_acceptance import _stratified_fit_calibration
from experiments.common.v7_discovery import evaluate_discovery_schedule
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from src.open_set import (
    UNKNOWN_LABEL,
    OpenSetPrediction,
    OpenSetReason,
)
from src.rejection_buffer import RejectionBuffer
from src.subspace_primitive import fit_subspace_primitive


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v7" / "m40_discovery.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v7" / "m40_discovery"
CLUSTERERS = (
    "no_clustering",
    "streaming_microclusters",
    "hdbscan",
    "finch",
    "gcd_kmeans",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gaussian_novelty(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    query: np.ndarray,
    rank: int,
) -> np.ndarray:
    values = []
    for label in np.unique(fit_y):
        primitive = fit_subspace_primitive(
            fit_x[fit_y == label],
            min(rank, fit_x.shape[1] - 1, np.sum(fit_y == label) - 2),
            class_label=int(label),
        )
        values.append(primitive.log_likelihood(query))
    return -np.max(np.column_stack(values), axis=1)


def _records_for_seed(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[tuple[tuple[Any, ...], ...], dict[str, int], int]:
    unknown = np.asarray(config["proxy_unknown_classes"], dtype=np.int64)
    known_train = ~np.isin(train_y, unknown)
    fit_x, fit_y, calibration_x, _ = _stratified_fit_calibration(
        train_x[known_train],
        train_y[known_train],
        calibration_fraction=float(config["calibration_fraction"]),
        seed=seed,
    )
    calibration_novelty = _gaussian_novelty(
        fit_x, fit_y, calibration_x, int(config["gaussian_rank"])
    )
    threshold = float(
        np.quantile(
            calibration_novelty,
            float(config["calibration_known_coverage_target"]),
            method="higher",
        )
    )
    novelty = _gaussian_novelty(fit_x, fit_y, dev_x, int(config["gaussian_rank"]))
    permutation = np.random.default_rng(seed + 40_000).permutation(len(dev_x))
    windows = np.array_split(permutation, int(config["windows"]))
    buffer = RejectionBuffer(
        int(config["buffer_max_records"]),
        max_embedding_dimensions=dev_x.shape[1],
    )
    records_by_window = []
    labels: dict[str, int] = {}
    for window_id, indices in enumerate(windows):
        window_records = []
        for index in indices:
            if novelty[index] <= threshold:
                continue
            sample_id = f"seed-{seed}-sample-{int(index):05d}"
            labels[sample_id] = int(dev_y[index])
            prediction = OpenSetPrediction(
                label=UNKNOWN_LABEL,
                accepted=False,
                candidate_model_signature="v7-m39-low-rank-gaussian",
                candidate_class_id=None,
                raw_novelty_score=float(novelty[index]),
                calibrated_novelty_score=float(novelty[index]),
                threshold=threshold,
                decision_margin=float(novelty[index] - threshold),
                support_profile_version="m39-v1",
                reason_code=OpenSetReason.OUTSIDE_SUPPORT,
            )
            window_records.append(
                buffer.append_rejection(
                    dev_x[index],
                    timestamp=float(window_id),
                    window_id=window_id,
                    prediction=prediction,
                    source_sample_id=sample_id,
                )
            )
        records_by_window.append(tuple(window_records))
    return tuple(records_by_window), labels, buffer.evicted_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = _load_json(args.config)
    parent = REPO_ROOT / config["parent_artifact_index"]
    if sha256_file(parent) != config["parent_artifact_index_sha256"]:
        raise ValueError("M39 parent artifact index drifted.")
    source = _load_json(REPO_ROOT / config["source_config"])
    results = []
    for seed in config["seeds"]:
        loaded = _load_seed_data(source["seed_inputs"][str(seed)])
        train_x, train_y = loaded["datasets"]["train"]
        dev_x, dev_y = loaded["datasets"]["dev"]
        records_by_window, labels, evictions = _records_for_seed(
            train_x, train_y, dev_x, dev_y, config, seed
        )
        review_budget = max(
            1,
            int(round(len(dev_y) * config["review_budget_per_1000"] / 1000)),
        )
        for clusterer in CLUSTERERS:
            result = evaluate_discovery_schedule(
                records_by_window,
                labels,
                tuple(config["proxy_unknown_classes"]),
                clusterer=clusterer,
                minimum_cluster_size=int(config["minimum_cluster_size"]),
                minimum_purity=float(config["minimum_purity"]),
                review_budget=review_budget,
                maximum_kmeans_clusters=int(config["maximum_kmeans_clusters"]),
                microcluster_radius_multiplier=float(
                    config["microcluster_radius_multiplier"]
                ),
                seed=seed,
            )
            replay = evaluate_discovery_schedule(
                records_by_window,
                labels,
                tuple(config["proxy_unknown_classes"]),
                clusterer=clusterer,
                minimum_cluster_size=int(config["minimum_cluster_size"]),
                minimum_purity=float(config["minimum_purity"]),
                review_budget=review_budget,
                maximum_kmeans_clusters=int(config["maximum_kmeans_clusters"]),
                microcluster_radius_multiplier=float(
                    config["microcluster_radius_multiplier"]
                ),
                seed=seed,
            )
            result.update(
                {
                    "seed": seed,
                    "buffer_evictions": evictions,
                    "exact_replay": (
                        result["review_id_continuity_hash"]
                        == replay["review_id_continuity_hash"]
                        and result["result_hash"] == replay["result_hash"]
                    ),
                    "matched_cells": {
                        "novel_group_recovery": result["full_recovery"],
                        "known_extension_unpublished": (
                            result["semantic_publications_before_confirmation"] == 0
                        ),
                        "corruption_unpublished": (
                            result["semantic_publications_before_confirmation"] == 0
                        ),
                    },
                }
            )
            results.append(result)
    summaries = {}
    for clusterer in CLUSTERERS:
        cells = [result for result in results if result["clusterer"] == clusterer]
        full_recovery_cells = sum(
            int(value)
            for result in cells
            for value in result["matched_cells"].values()
        )
        mean_recall = float(np.mean([result["distinct_group_recall"] for result in cells]))
        mean_precision = float(np.mean([result["review_precision"] for result in cells]))
        passes = (
            mean_recall >= config["minimum_distinct_group_recall"]
            and mean_precision >= config["minimum_review_precision"]
            and full_recovery_cells >= config["minimum_full_recovery_cells"]
            and all(result["review_id_continuity"] == 1.0 for result in cells)
            and all(result["exact_replay"] for result in cells)
            and all(result["semantic_publications_before_confirmation"] == 0 for result in cells)
            and all(
                result["estimated_memory_megabytes"]
                <= config["maximum_memory_megabytes"]
                and result["maximum_window_latency_seconds"]
                <= config["maximum_window_latency_seconds"]
                and result["buffer_evictions"] == 0
                for result in cells
            )
        )
        summaries[clusterer] = {
            "mean_distinct_group_recall": mean_recall,
            "mean_review_precision": mean_precision,
            "full_recovery_cells": full_recovery_cells,
            "matched_cell_count": config["matched_cell_count"],
            "passes": passes,
        }
    passing = [
        name for name in CLUSTERERS if summaries[name]["passes"]
    ]
    primary = max(
        passing,
        key=lambda name: (
            summaries[name]["mean_distinct_group_recall"],
            summaries[name]["mean_review_precision"],
            -CLUSTERERS.index(name),
        ),
        default=None,
    )
    controls = [name for name in passing if name != primary]
    retained = ([primary] if primary else []) + controls[:1]
    evidence = {
        "schema_version": 1,
        "milestone": "M40",
        "config_sha256": payload_hash(config),
        "parent_artifact_index_sha256": sha256_file(parent),
        "final_labels_opened": False,
        "results": results,
        "summaries": summaries,
        "retained_clusterers": retained,
        "advance_to_m41": bool(retained),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_canonical_json(args.output / "evidence.json", evidence)
    write_canonical_json(
        args.output / "artifact_index.json",
        {
            "schema_version": 1,
            "milestone": "M40",
            "evidence_sha256": payload_hash(evidence),
            "advance_to_m41": evidence["advance_to_m41"],
        },
    )
    print(json.dumps({"summaries": summaries, "retained": retained}, indent=2))


if __name__ == "__main__":
    main()
