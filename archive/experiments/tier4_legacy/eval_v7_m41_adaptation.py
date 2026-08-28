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
from experiments.common.v7_adaptation import (
    GaussianAdaptationTransaction,
    bundle_metrics,
    fit_gaussian_bundle,
    new_class_migration,
)
from experiments.common.v7_protocol import ConfirmationEvent
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v7" / "m41_adaptation.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v7" / "m41_adaptation"
OPERATIONS = (
    "native_gaussian",
    "sdf_component",
    "full_class_local_refit",
    "full_model_retrain",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _calibrated_parent(
    train_x: np.ndarray,
    train_y: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray]:
    known = np.isin(train_y, config["known_classes"])
    fit_x, fit_y, calibration_x, _ = _stratified_fit_calibration(
        train_x[known],
        train_y[known],
        calibration_fraction=float(config["calibration_fraction"]),
        seed=seed,
    )
    provisional = fit_gaussian_bundle(
        fit_x,
        fit_y,
        rank=int(config["gaussian_rank"]),
        threshold=0.0,
    )
    _, novelty = provisional.predict(calibration_x)
    threshold = float(
        np.quantile(
            novelty,
            float(config["calibration_known_coverage_target"]),
            method="higher",
        )
    )
    return (
        fit_gaussian_bundle(
            fit_x,
            fit_y,
            rank=int(config["gaussian_rank"]),
            threshold=threshold,
        ),
        fit_x,
        fit_y,
    )


def _passes(
    baseline: dict[str, float],
    adapted: dict[str, float],
    config: dict[str, Any],
    *,
    exact_replay: bool,
    exact_rollback: bool,
    graph_issues: int,
    publication_without_confirmation_rejected: bool,
) -> bool:
    return (
        adapted["target_success"] - baseline["target_success"]
        >= config["minimum_target_improvement"]
        and baseline["known_balanced_accuracy"]
        - adapted["known_balanced_accuracy"]
        <= config["maximum_known_accuracy_drop"]
        and baseline["unknown_recall"] - adapted["unknown_recall"]
        <= config["maximum_unknown_recall_drop"]
        and adapted["known_nll"] - baseline["known_nll"]
        <= config["maximum_known_nll_regression"]
        and exact_replay
        and exact_rollback
        and graph_issues == 0
        and publication_without_confirmation_rejected
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = _load_json(args.config)
    parent_index = REPO_ROOT / config["parent_artifact_index"]
    if sha256_file(parent_index) != config["parent_artifact_index_sha256"]:
        raise ValueError("M40 parent artifact index drifted.")
    source = _load_json(REPO_ROOT / config["source_config"])
    results = []
    for seed in config["seeds"]:
        loaded = _load_seed_data(source["seed_inputs"][str(seed)])
        train_x, train_y = loaded["datasets"]["train"]
        dev_x, dev_y = loaded["datasets"]["dev"]
        parent, fit_x, fit_y = _calibrated_parent(train_x, train_y, config, seed)
        known_dev = np.isin(dev_y, config["known_classes"])
        new_class_indices = np.flatnonzero(dev_y == config["new_class"])
        rng = np.random.default_rng(seed + 41_000)
        new_class_indices = rng.permutation(new_class_indices)
        support_count = int(config["new_class_review_samples"])
        new_support_indices = new_class_indices[:support_count]
        new_target_indices = new_class_indices[support_count:]
        unknown_indices = np.flatnonzero(dev_y == config["remaining_unknown_class"])

        _, parent_novelty = parent.predict(dev_x)
        rejected_known = np.flatnonzero(known_dev & (parent_novelty > parent.threshold))
        counts = {
            int(label): int(np.sum(dev_y[rejected_known] == label))
            for label in config["known_classes"]
        }
        existing_label = max(counts, key=lambda label: (counts[label], -label))
        existing_indices = rejected_known[dev_y[rejected_known] == existing_label]
        split = max(1, len(existing_indices) // 2)
        existing_support_indices = existing_indices[:split]
        existing_target_indices = existing_indices[split:]
        if len(existing_target_indices) == 0:
            existing_target_indices = existing_support_indices

        scenarios = {
            "update_existing": {
                "label": existing_label,
                "support": dev_x[existing_support_indices],
                "target": dev_x[existing_target_indices],
                "confirmation": ConfirmationEvent(
                    review_id=f"review-m41-existing-{seed}",
                    response="existing_class",
                    confirmed_label=str(existing_label),
                    confirmed_window=5,
                ),
            },
            "create_new": {
                "label": int(config["new_class"]),
                "support": dev_x[new_support_indices],
                "target": dev_x[new_target_indices],
                "confirmation": ConfirmationEvent(
                    review_id=f"review-m41-new-{seed}",
                    response="new_class",
                    confirmed_label=str(config["new_class"]),
                    confirmed_window=5,
                ),
            },
        }
        for scenario, values in scenarios.items():
            label = int(values["label"])
            baseline = bundle_metrics(
                parent,
                dev_x[known_dev],
                dev_y[known_dev],
                values["target"],
                label,
                dev_x[unknown_indices],
            )
            for operation in OPERATIONS:
                rank = (
                    int(config["sdf_component_rank"])
                    if operation == "sdf_component"
                    else int(config["gaussian_rank"])
                )
                transaction = GaussianAdaptationTransaction(parent)
                unauthorised = GaussianAdaptationTransaction(parent)
                try:
                    unauthorised.apply(
                        confirmation=None,
                        label=label,
                        support=values["support"],
                        rank=rank,
                        operation=operation,
                        original_class_support=(
                            fit_x[fit_y == label]
                            if scenario == "update_existing"
                            else None
                        ),
                    )
                    publication_without_confirmation_rejected = False
                except PermissionError:
                    publication_without_confirmation_rejected = True
                child = transaction.apply(
                    confirmation=values["confirmation"],
                    label=label,
                    support=values["support"],
                    rank=rank,
                    operation=operation,
                    original_class_support=(
                        fit_x[fit_y == label]
                        if scenario == "update_existing"
                        else None
                    ),
                )
                adapted = bundle_metrics(
                    child,
                    dev_x[known_dev],
                    dev_y[known_dev],
                    values["target"],
                    label,
                    dev_x[unknown_indices],
                )
                replay_transaction = GaussianAdaptationTransaction(parent)
                replay = replay_transaction.apply(
                    confirmation=values["confirmation"],
                    label=label,
                    support=values["support"],
                    rank=rank,
                    operation=operation,
                    original_class_support=(
                        fit_x[fit_y == label]
                        if scenario == "update_existing"
                        else None
                    ),
                )
                prediction_before = parent.predict(dev_x)[0]
                rolled_back = transaction.rollback()
                prediction_after = rolled_back.predict(dev_x)[0]
                migration = (
                    new_class_migration(parent, child, values["confirmation"])
                    if scenario == "create_new"
                    else None
                )
                graph_issues = (
                    0
                    if migration is None
                    or (
                        migration.parent_class_order
                        == tuple(str(value) for value in parent.class_order)
                        and migration.child_class_order
                        == tuple(str(value) for value in child.class_order)
                    )
                    else 1
                )
                exact_replay = child.bundle_hash == replay.bundle_hash
                exact_rollback = (
                    rolled_back.bundle_hash == parent.bundle_hash
                    and np.array_equal(prediction_before, prediction_after)
                )
                results.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "operation": operation,
                        "confirmed_label": label,
                        "review_sample_count": len(values["support"]),
                        "baseline": baseline,
                        "adapted": adapted,
                        "exact_replay": exact_replay,
                        "exact_rollback": exact_rollback,
                        "graph_issues": graph_issues,
                        "publication_without_confirmation_rejected": (
                            publication_without_confirmation_rejected
                        ),
                        "passes": _passes(
                            baseline,
                            adapted,
                            config,
                            exact_replay=exact_replay,
                            exact_rollback=exact_rollback,
                            graph_issues=graph_issues,
                            publication_without_confirmation_rejected=(
                                publication_without_confirmation_rejected
                            ),
                        ),
                    }
                )
    summaries = {}
    for scenario in ("update_existing", "create_new"):
        arms = {}
        for operation in OPERATIONS:
            cells = [
                result
                for result in results
                if result["scenario"] == scenario
                and result["operation"] == operation
            ]
            arms[operation] = {
                "mean_target_improvement": float(
                    np.mean(
                        [
                            cell["adapted"]["target_success"]
                            - cell["baseline"]["target_success"]
                            for cell in cells
                        ]
                    )
                ),
                "all_seeds_pass": all(cell["passes"] for cell in cells),
            }
        summaries[scenario] = arms
    retained = {
        scenario: [
            operation
            for operation in OPERATIONS
            if summaries[scenario][operation]["all_seeds_pass"]
        ][:1]
        for scenario in summaries
    }
    evidence = {
        "schema_version": 1,
        "milestone": "M41",
        "config_sha256": payload_hash(config),
        "parent_artifact_index_sha256": sha256_file(parent_index),
        "final_labels_opened": False,
        "results": results,
        "summaries": summaries,
        "retained_operations": retained,
        "advance_to_m42": any(retained.values()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_canonical_json(args.output / "evidence.json", evidence)
    write_canonical_json(
        args.output / "artifact_index.json",
        {
            "schema_version": 1,
            "milestone": "M41",
            "evidence_sha256": payload_hash(evidence),
            "advance_to_m42": evidence["advance_to_m42"],
        },
    )
    print(json.dumps({"summaries": summaries, "retained": retained}, indent=2))


if __name__ == "__main__":
    main()
