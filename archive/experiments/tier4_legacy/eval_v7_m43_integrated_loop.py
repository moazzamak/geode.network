from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v7_acceptance import _stratified_fit_calibration
from experiments.common.v7_adaptation import fit_gaussian_bundle
from experiments.common.v7_integrated import evaluate_integrated_cell
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.tier4.eval_v7_m40_discovery import _records_for_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v7" / "m43_integrated_loop.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v7" / "m43_integrated_loop"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parent(
    train_x: np.ndarray,
    train_y: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> Any:
    known = np.isin(train_y, config["known_classes"])
    fit_x, fit_y, calibration_x, _ = _stratified_fit_calibration(
        train_x[known],
        train_y[known],
        calibration_fraction=float(config["calibration_fraction"]),
        seed=seed,
    )
    provisional = fit_gaussian_bundle(
        fit_x, fit_y, rank=int(config["gaussian_rank"]), threshold=0.0
    )
    _, novelty = provisional.predict(calibration_x)
    threshold = float(
        np.quantile(
            novelty,
            float(config["calibration_known_coverage_target"]),
            method="higher",
        )
    )
    return fit_gaussian_bundle(
        fit_x, fit_y, rank=int(config["gaussian_rank"]), threshold=threshold
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = _load_json(args.config)
    parent_index = REPO_ROOT / config["parent_artifact_index"]
    if sha256_file(parent_index) != config["parent_artifact_index_sha256"]:
        raise ValueError("M42 parent artifact index drifted.")
    source = _load_json(REPO_ROOT / config["source_config"])
    results = []
    reject_everything_counts = {}
    for seed in config["seeds"]:
        loaded = _load_seed_data(source["seed_inputs"][str(seed)])
        train_x, train_y = loaded["datasets"]["train"]
        dev_x, dev_y = loaded["datasets"]["dev"]
        parent = _parent(train_x, train_y, config, seed)
        m40_config = {
            "proxy_unknown_classes": [
                config["confirmable_new_class"],
                config["remaining_unknown_class"],
            ],
            "known_classes": config["known_classes"],
            "calibration_fraction": config["calibration_fraction"],
            "calibration_known_coverage_target": config[
                "calibration_known_coverage_target"
            ],
            "gaussian_rank": config["gaussian_rank"],
            "windows": config["windows"],
            "buffer_max_records": 2000,
        }
        records_by_window, labels, _ = _records_for_seed(
            train_x, train_y, dev_x, dev_y, m40_config, seed
        )
        records = tuple(record for window in records_by_window for record in window)
        reject_everything_counts[str(seed)] = len(records)
        known_mask = np.isin(dev_y, config["known_classes"])
        unknown_mask = dev_y == config["remaining_unknown_class"]
        for discovery, review, adaptation in product(
            config["discovery_arms"],
            config["review_arms"],
            config["adaptation_arms"],
        ):
            results.append(
                evaluate_integrated_cell(
                    parent,
                    records,
                    labels,
                    dev_x,
                    dev_y,
                    known_mask,
                    unknown_mask,
                    discovery_arm=discovery,
                    review_arm=review,
                    adaptation_arm=adaptation,
                    target_label=int(config["confirmable_new_class"]),
                    review_budget=int(config["review_budget"]),
                    minimum_cluster_size=int(config["minimum_cluster_size"]),
                    affine_rank=int(config["affine_rank"]),
                    seed=seed,
                )
            )
    candidates = []
    for discovery in config["discovery_arms"]:
        cells = [
            result
            for result in results
            if result["discovery_arm"] == discovery
            and result["review_arm"] == "delayed_confirmation"
            and result["adaptation_arm"] == "rank16_affine_insertion"
        ]
        candidates.append(
            {
                "discovery_arm": discovery,
                "cells": cells,
                "mean_integration_fraction": float(
                    np.mean(
                        [
                            cell["integrated_confirmable_classes"]
                            / cell["confirmable_classes"]
                            for cell in cells
                        ]
                    )
                ),
                "mean_reviewed_samples": float(
                    np.mean([cell["reviewed_samples"] for cell in cells])
                ),
                "mean_known_accuracy": float(
                    np.mean(
                        [cell["adapted"]["known_balanced_accuracy"] for cell in cells]
                    )
                ),
                "mean_unknown_recall": float(
                    np.mean([cell["adapted"]["unknown_recall"] for cell in cells])
                ),
            }
        )
    eligible_candidates = [
        candidate
        for candidate in candidates
        if candidate["discovery_arm"] != "no_clustering"
    ]
    winner = max(
        eligible_candidates,
        key=lambda item: (
            item["mean_integration_fraction"],
            -item["mean_reviewed_samples"],
            item["mean_known_accuracy"],
        ),
    )
    winner_cells = winner["cells"]
    mean_reject_everything = float(np.mean(list(reject_everything_counts.values())))
    review_reduction = 1.0 - winner["mean_reviewed_samples"] / mean_reject_everything
    parent_known_accuracy = float(
        np.mean(
            [cell["baseline"]["known_balanced_accuracy"] for cell in winner_cells]
        )
    )
    known_drop = parent_known_accuracy - winner["mean_known_accuracy"]
    unknown_drop = (
        config["m39_unknown_recall_baseline"] - winner["mean_unknown_recall"]
    )
    contracts_pass = all(
        cell["unconfirmed_semantic_publications"] == 0
        and cell["unconfirmed_mutations"] == 0
        and cell["false_autonomous_class_creations"] == 0
        and cell["exact_replay"]
        and cell["exact_rollback"]
        and cell["graph_issues"] == 0
        and cell["fallback_contract"]
        and cell["audit_completeness"] == 1.0
        for cell in winner_cells
    )
    non_dominated = not any(
        candidate["mean_integration_fraction"] >= winner["mean_integration_fraction"]
        and candidate["mean_reviewed_samples"] <= winner["mean_reviewed_samples"]
        and candidate["mean_known_accuracy"] >= winner["mean_known_accuracy"]
        and (
            candidate["mean_integration_fraction"] > winner["mean_integration_fraction"]
            or candidate["mean_reviewed_samples"] < winner["mean_reviewed_samples"]
            or candidate["mean_known_accuracy"] > winner["mean_known_accuracy"]
        )
        for candidate in eligible_candidates
        if candidate["discovery_arm"] != winner["discovery_arm"]
    )
    gate = {
        "winner": winner["discovery_arm"],
        "mean_integration_fraction": winner["mean_integration_fraction"],
        "review_reduction": review_reduction,
        "known_accuracy_drop": known_drop,
        "mean_unknown_recall": winner["mean_unknown_recall"],
        "unknown_recall_drop_from_m39": unknown_drop,
        "contracts_pass": contracts_pass,
        "non_dominated": non_dominated,
    }
    gate["advance_to_m44"] = (
        gate["mean_integration_fraction"] >= config["minimum_confirmable_fraction"]
        and gate["review_reduction"] >= config["minimum_review_reduction"]
        and gate["known_accuracy_drop"] <= config["maximum_known_accuracy_drop"]
        and gate["unknown_recall_drop_from_m39"]
        <= config["maximum_unknown_recall_drop"]
        and gate["contracts_pass"]
        and gate["non_dominated"]
    )
    evidence = {
        "schema_version": 1,
        "milestone": "M43",
        "config_sha256": payload_hash(config),
        "parent_artifact_index_sha256": sha256_file(parent_index),
        "final_labels_opened": False,
        "factorial_cells": results,
        "reject_everything_counts": reject_everything_counts,
        "candidate_summaries": [
            {key: value for key, value in candidate.items() if key != "cells"}
            for candidate in candidates
        ],
        "gate": gate,
        "m44_status": "open" if gate["advance_to_m44"] else "blocked_by_m43",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_canonical_json(args.output / "evidence.json", evidence)
    write_canonical_json(
        args.output / "artifact_index.json",
        {
            "schema_version": 1,
            "milestone": "M43",
            "evidence_sha256": payload_hash(evidence),
            "advance_to_m44": gate["advance_to_m44"],
        },
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
