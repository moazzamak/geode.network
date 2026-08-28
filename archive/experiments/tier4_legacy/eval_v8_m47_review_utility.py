from __future__ import annotations

import argparse
from dataclasses import replace
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.classification_metrics import balanced_accuracy
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v7_adaptation import (
    GaussianAdaptationTransaction,
    GaussianBundle,
    fit_gaussian_bundle,
)
from experiments.common.v7_protocol import ConfirmationEvent
from experiments.common.v8_diagnostics import (
    predictions_with_rejection,
    representativeness_metrics,
)
from experiments.common.v8_review_selection import (
    boundary_inclusive_indices,
    core_indices,
    kcenter_indices,
    paired_bootstrap_interval,
    random_stratified_indices,
)
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from src.runtime.schemas import ReviewSelectionEvidence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v8" / "m47_review_utility.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v8" / "m47_review_utility"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _partition_class(
    features: np.ndarray,
    *,
    seed: int,
    label: int,
    geometry_count: int,
    anchor_count: int,
    validation_count: int,
) -> dict[str, np.ndarray]:
    required = geometry_count + anchor_count + validation_count
    if len(features) < required:
        raise ValueError(f"class {label} has {len(features)} rows; {required} required")
    order = np.random.default_rng(seed * 1000 + label).permutation(len(features))
    geometry = order[:geometry_count]
    anchor = order[geometry_count : geometry_count + anchor_count]
    validation = order[
        geometry_count + anchor_count : geometry_count + anchor_count + validation_count
    ]
    if len(set(geometry) & set(anchor)) or len(set(geometry) & set(validation)):
        raise ValueError("episode class partitions overlap")
    return {
        "geometry": features[geometry],
        "anchor": features[anchor],
        "validation": features[validation],
        "geometry_indices": geometry,
        "validation_indices": validation,
    }


def _adapt(
    parent: GaussianBundle,
    *,
    label: int,
    support: np.ndarray,
    rank: int,
    review_id: str,
) -> tuple[GaussianBundle, bool]:
    confirmation = ConfirmationEvent(review_id, "new_class", str(label), 1)
    transaction = GaussianAdaptationTransaction(parent)
    child = transaction.apply(
        confirmation=confirmation,
        label=label,
        support=support,
        rank=rank,
        operation="sdf_component",
    )
    rolled_back = transaction.rollback()
    return child, rolled_back.bundle_hash == parent.bundle_hash


def _recalibrate(
    child: GaussianBundle, anchor_x: np.ndarray, coverage_target: float
) -> GaussianBundle:
    _, novelty = child.predict(anchor_x)
    threshold = float(np.quantile(novelty, coverage_target, method="higher"))
    return replace(child, threshold=threshold)


def _selector_indices(
    selector: str,
    features: np.ndarray,
    margin: np.ndarray,
    budget: int,
    seed: int,
) -> np.ndarray:
    if selector == "core_selected":
        return core_indices(features, budget)
    if selector == "boundary_inclusive":
        return boundary_inclusive_indices(features, margin, budget, seed)
    if selector == "coverage_selected":
        return kcenter_indices(features, budget)
    if selector.startswith("random"):
        suffix = selector.rsplit("_", maxsplit=1)[-1]
        offset = int(suffix) if suffix.isdigit() else 0
        return random_stratified_indices(features, budget, seed + offset * 100)
    raise ValueError(f"unsupported selector: {selector}")


def _episode_utility(
    parent: GaussianBundle,
    child: GaussianBundle,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    parent_predictions, _ = predictions_with_rejection(
        parent, x, parent.threshold, {}
    )
    child_predictions, _ = predictions_with_rejection(
        child, x, child.threshold, {}
    )
    parent_accuracy = balanced_accuracy(y, parent_predictions)
    child_accuracy = balanced_accuracy(y, child_predictions)
    return child_accuracy - parent_accuracy, parent_accuracy, child_accuracy


def _choose_utility_set(
    *,
    config: dict[str, Any],
    parent: GaussianBundle,
    candidate_x: np.ndarray,
    candidate_margin: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    next_anchor_x: np.ndarray,
    label: int,
    seed: int,
) -> tuple[np.ndarray, str, dict[str, float]]:
    scores = {}
    selections = {}
    for candidate_name in config["utility_candidates"]:
        selected = _selector_indices(
            candidate_name,
            candidate_x,
            candidate_margin,
            int(config["review_budget"]),
            seed,
        )
        proxy, _ = _adapt(
            parent,
            label=label,
            support=candidate_x[selected],
            rank=int(config["proxy_adaptation_rank"]),
            review_id=f"review-m47-proxy-{seed}-{label}-{candidate_name}",
        )
        proxy = _recalibrate(
            proxy, next_anchor_x, float(config["anchor_known_coverage_target"])
        )
        utility, _, _ = _episode_utility(
            parent, proxy, validation_x, validation_y
        )
        scores[candidate_name] = utility
        selections[candidate_name] = selected
    winner = max(scores, key=lambda name: (scores[name], name))
    return selections[winner], winner, scores


def _run_arm(
    config: dict[str, Any],
    seed: int,
    arm: str,
    partitions: dict[int, dict[str, np.ndarray]],
    dev_x: np.ndarray,
    dev_y: np.ndarray,
) -> list[dict[str, Any]]:
    initial = tuple(int(value) for value in config["initial_known_classes"])
    fit_x = np.concatenate([partitions[label]["geometry"] for label in initial])
    fit_y = np.concatenate(
        [
            np.full(len(partitions[label]["geometry"]), label, dtype=np.int64)
            for label in initial
        ]
    )
    anchor_x = np.concatenate([partitions[label]["anchor"] for label in initial])
    provisional = fit_gaussian_bundle(
        fit_x, fit_y, rank=int(config["gaussian_rank"]), threshold=0.0
    )
    _, novelty = provisional.predict(anchor_x)
    parent = replace(
        provisional,
        threshold=float(
            np.quantile(
                novelty,
                float(config["anchor_known_coverage_target"]),
                method="higher",
            )
        ),
    )
    rows = []
    for episode_index, label_value in enumerate(config["arrival_classes"]):
        label = int(label_value)
        candidate_x = partitions[label]["geometry"]
        candidate_ids = tuple(
            f"seed-{seed}-class-{label}-candidate-{int(index):04d}"
            for index in partitions[label]["geometry_indices"]
        )
        validation_ids = tuple(
            f"seed-{seed}-class-{label}-validation-{int(index):04d}"
            for index in partitions[label]["validation_indices"]
        )
        _, candidate_novelty = parent.predict(candidate_x)
        candidate_margin = candidate_novelty - parent.threshold
        next_anchor_x = np.concatenate((anchor_x, partitions[label]["anchor"]))
        proxy_x = np.concatenate(
            [
                partitions[class_id]["validation"]
                for class_id in (*parent.class_order, label)
            ]
        )
        proxy_y = np.concatenate(
            [
                np.full(
                    len(partitions[class_id]["validation"]),
                    class_id,
                    dtype=np.int64,
                )
                for class_id in (*parent.class_order, label)
            ]
        )
        expected_utility = None
        selector_provenance = arm
        proxy_scores: dict[str, float] = {}
        if arm == "review_everything":
            selected = np.arange(len(candidate_x), dtype=np.int64)
        elif arm == "utility_selected":
            selected, selector_provenance, proxy_scores = _choose_utility_set(
                config=config,
                parent=parent,
                candidate_x=candidate_x,
                candidate_margin=candidate_margin,
                validation_x=proxy_x,
                validation_y=proxy_y,
                next_anchor_x=next_anchor_x,
                label=label,
                seed=seed + episode_index * 10,
            )
            expected_utility = proxy_scores[selector_provenance]
        else:
            selected = _selector_indices(
                arm,
                candidate_x,
                candidate_margin,
                int(config["review_budget"]),
                seed + episode_index * 10,
            )
        support = candidate_x[selected]
        review_id = f"review-m47-{seed}-{label}-{arm}"
        child, rollback_exact = _adapt(
            parent,
            label=label,
            support=support,
            rank=int(config["production_adaptation_rank"]),
            review_id=review_id,
        )
        child = _recalibrate(
            child, next_anchor_x, float(config["anchor_known_coverage_target"])
        )
        evaluation_mask = np.isin(dev_y, child.class_order)
        utility, parent_accuracy, child_accuracy = _episode_utility(
            parent,
            child,
            dev_x[evaluation_mask],
            dev_y[evaluation_mask],
        )
        known_mask = np.isin(dev_y, parent.class_order)
        _, parent_known, child_known = _episode_utility(
            parent, child, dev_x[known_mask], dev_y[known_mask]
        )
        unknown_mask = dev_y > label
        _, parent_unknown_rejected = predictions_with_rejection(
            parent, dev_x[unknown_mask], parent.threshold, {}
        )
        _, child_unknown_rejected = predictions_with_rejection(
            child, dev_x[unknown_mask], child.threshold, {}
        )
        parent_unknown_recall = float(np.mean(parent_unknown_rejected))
        child_unknown_recall = float(np.mean(child_unknown_rejected))
        selected_ids = tuple(candidate_ids[int(index)] for index in selected)
        if arm != "review_everything":
            selection_evidence = ReviewSelectionEvidence(
                episode_id=f"seed-{seed}-arrival-{label}",
                selector=arm,
                candidate_ids=candidate_ids,
                selected_ids=selected_ids,
                validation_ids=validation_ids,
                review_budget=int(config["review_budget"]),
                selection_frozen_before_validation=True,
                expected_utility=expected_utility,
                realized_utility=utility,
            ).to_dict()
        else:
            selection_evidence = {
                "selector": arm,
                "selected_ids": list(selected_ids),
                "diagnostic_only": True,
            }
        representativeness = representativeness_metrics(candidate_x, support)
        rows.append(
            {
                "seed": seed,
                "episode_index": episode_index,
                "arrival_class": label,
                "arm": arm,
                "selector_provenance": selector_provenance,
                "proxy_scores": proxy_scores,
                "selection": selection_evidence,
                "reviewed_labels": len(selected),
                "duplicate_reviews": 0,
                "parent_balanced_accuracy": parent_accuracy,
                "child_balanced_accuracy": child_accuracy,
                "utility": utility,
                "known_regression": parent_known - child_known,
                "parent_remaining_unknown_recall": parent_unknown_recall,
                "remaining_unknown_recall": child_unknown_recall,
                "unknown_recall_drop": parent_unknown_recall - child_unknown_recall,
                "purity": 1.0,
                "persistence": 1.0,
                "boundary_margin_span": float(
                    np.max(candidate_margin[selected])
                    - np.min(candidate_margin[selected])
                ),
                "representativeness": representativeness,
                "update_support_count": len(selected),
                "component_count": sum(len(state.components) for state in child.classes),
                "latency_status": "not_measured_in_deterministic_replay",
                "confirmation_linked": child.confirmation_id is not None,
                "class_order_appended": child.class_order == (*parent.class_order, label),
                "exact_rollback": rollback_exact,
                "final_labels_opened": False,
            }
        )
        parent = child
        anchor_x = next_anchor_x
    return rows


def run_m47(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    for lock in config["parent_locks"]:
        if sha256_file(REPO_ROOT / lock["path"]) != lock["sha256"]:
            raise ValueError(f"M47 parent lock drifted: {lock['id']}")
    source = _load_json(REPO_ROOT / config["source_config"])
    cells = []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        loaded = _load_seed_data(source["seed_inputs"][str(seed)])
        train_x, train_y = loaded["datasets"]["train"]
        dev_x, dev_y = loaded["datasets"]["dev"]
        partitions = {}
        for label in range(10):
            partitions[label] = _partition_class(
                train_x[train_y == label],
                seed=seed,
                label=label,
                geometry_count=int(config["geometry_count_per_class"]),
                anchor_count=int(config["anchor_count_per_class"]),
                validation_count=int(config["proxy_validation_count_per_class"]),
            )
        for arm in config["arms"]:
            cells.extend(
                _run_arm(config, seed, arm, partitions, dev_x, dev_y)
            )
    utility = np.asarray(
        [cell["utility"] for cell in cells if cell["arm"] == "utility_selected"]
    )
    core = np.asarray(
        [cell["utility"] for cell in cells if cell["arm"] == "core_selected"]
    )
    interval = paired_bootstrap_interval(
        utility,
        core,
        confidence=float(config["bootstrap_confidence"]),
        n_resamples=int(config["bootstrap_resamples"]),
        seed=int(config["bootstrap_seed"]),
    )
    utility_cells = [cell for cell in cells if cell["arm"] == "utility_selected"]
    positive_cells = int(np.sum(utility - core > 0.0))
    safety = all(
        cell["reviewed_labels"] == int(config["review_budget"])
        and cell["known_regression"] <= float(config["maximum_known_accuracy_drop"])
        and cell["unknown_recall_drop"]
        <= float(config["maximum_unknown_recall_drop"])
        and cell["confirmation_linked"]
        and cell["class_order_appended"]
        and cell["exact_rollback"]
        and not cell["final_labels_opened"]
        for cell in utility_cells
    )
    gate = {
        "mean_utility_gain_over_core": interval["difference"],
        "paired_bootstrap": interval,
        "positive_cells": positive_cells,
        "total_cells": len(utility_cells),
        "safety_and_transactional_gates": safety,
    }
    gate["advance_to_m48"] = (
        float(interval["difference"]) >= float(config["minimum_mean_utility_gain"])
        and float(interval["lower"]) > 0.0
        and positive_cells >= int(config["minimum_positive_cells"])
        and safety
    )
    arm_summaries = {}
    for arm in config["arms"]:
        arm_cells = [cell for cell in cells if cell["arm"] == arm]
        arm_summaries[arm] = {
            "mean_utility": float(np.mean([cell["utility"] for cell in arm_cells])),
            "mean_known_regression": float(
                np.mean([cell["known_regression"] for cell in arm_cells])
            ),
            "mean_remaining_unknown_recall": float(
                np.mean([cell["remaining_unknown_recall"] for cell in arm_cells])
            ),
            "mean_reviewed_labels": float(
                np.mean([cell["reviewed_labels"] for cell in arm_cells])
            ),
            "all_transactions_pass": all(
                cell["confirmation_linked"]
                and cell["class_order_appended"]
                and cell["exact_rollback"]
                for cell in arm_cells
            ),
        }
    evidence = {
        "schema_version": 1,
        "milestone": "M47",
        "config_sha256": payload_hash(config),
        "cells": cells,
        "arm_summaries": arm_summaries,
        "gate": gate,
        "outcome": "continue_to_m48" if gate["advance_to_m48"] else "Outcome D",
        "m48_status": "open" if gate["advance_to_m48"] else "blocked_by_m47",
        "m50_status": "blocked_on_m48" if gate["advance_to_m48"] else "blocked_by_m47",
        "final_labels_opened": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    index = build_artifact_index(output_dir)
    return {
        "cell_count": len(cells),
        "eligible_cell_count": len(utility_cells),
        "mean_utility_gain_over_core": interval["difference"],
        "bootstrap_lower": interval["lower"],
        "positive_cells": positive_cells,
        "safety_gates": safety,
        "advance_to_m48": gate["advance_to_m48"],
        "outcome": evidence["outcome"],
        "final_labels_opened": False,
        "artifact_count": len(index["artifacts"]),
    }


def verify_m47(config_path: Path, output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "first"
        second = root / "second"
        first_summary = run_m47(config_path, first)
        second_summary = run_m47(config_path, second)
        first_files = {
            path.relative_to(first).as_posix(): path.read_bytes()
            for path in first.rglob("*")
            if path.is_file()
        }
        second_files = {
            path.relative_to(second).as_posix(): path.read_bytes()
            for path in second.rglob("*")
            if path.is_file()
        }
        if first_summary != second_summary or first_files != second_files:
            raise RuntimeError("M47 replay was not byte-identical")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(first, output_dir)
    summary = {**first_summary, "byte_identical_replay": True}
    write_canonical_json(output_dir / "verification.json", summary)
    build_artifact_index(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify_m47(args.config, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
