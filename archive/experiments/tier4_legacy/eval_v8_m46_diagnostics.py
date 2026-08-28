from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from experiments.common.classification_metrics import balanced_accuracy
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v7_acceptance import _stratified_fit_calibration
from experiments.common.v7_adaptation import (
    GaussianAdaptationTransaction,
    fit_gaussian_bundle,
)
from experiments.common.v7_protocol import ConfirmationEvent
from experiments.common.v8_diagnostics import (
    boundary_inclusive_indices,
    evaluate_threshold_transfer,
    predictions_with_rejection,
    representativeness_metrics,
)
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v8" / "m46_diagnostics.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v8" / "m46_diagnostics"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_locks(config: dict[str, Any]) -> None:
    for lock in config["parent_locks"]:
        path = REPO_ROOT / lock["path"]
        if sha256_file(path) != lock["sha256"]:
            raise ValueError(f"M46 parent lock drifted: {lock['id']}")


def _adapt(parent: Any, label: int, support: np.ndarray, rank: int, seed: int) -> Any:
    confirmation = ConfirmationEvent(
        review_id=f"review-m46-{seed}",
        response="new_class",
        confirmed_label=str(label),
        confirmed_window=1,
    )
    transaction = GaussianAdaptationTransaction(parent)
    return transaction.apply(
        confirmation=confirmation,
        label=label,
        support=support,
        rank=rank,
        operation="sdf_component",
    )


def _utility(
    parent: Any,
    child: Any,
    evaluation_x: np.ndarray,
    evaluation_y: np.ndarray,
) -> float:
    parent_predictions, _ = predictions_with_rejection(
        parent, evaluation_x, parent.threshold, {}
    )
    child_predictions, _ = predictions_with_rejection(
        child, evaluation_x, child.threshold, {}
    )
    return balanced_accuracy(evaluation_y, child_predictions) - balanced_accuracy(
        evaluation_y, parent_predictions
    )


def _threshold_seed(
    config: dict[str, Any],
    source: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loaded = _load_seed_data(source["seed_inputs"][str(seed)])
    train_x, train_y = loaded["datasets"]["train"]
    dev_x, dev_y = loaded["datasets"]["dev"]
    known_classes = np.asarray(config["known_classes"])
    integrated_class = int(config["integrated_class"])
    known_train = np.isin(train_y, known_classes)
    fit_x, fit_y, anchor_x, _ = _stratified_fit_calibration(
        train_x[known_train],
        train_y[known_train],
        calibration_fraction=float(config["anchor_fraction"]),
        seed=seed,
    )
    provisional = fit_gaussian_bundle(
        fit_x,
        fit_y,
        rank=int(config["gaussian_rank"]),
        threshold=0.0,
    )
    _, anchor_novelty = provisional.predict(anchor_x)
    parent_threshold = float(
        np.quantile(
            anchor_novelty,
            float(config["anchor_known_coverage_target"]),
            method="higher",
        )
    )
    parent = fit_gaussian_bundle(
        fit_x,
        fit_y,
        rank=int(config["gaussian_rank"]),
        threshold=parent_threshold,
    )
    support_pool = train_x[train_y == integrated_class]
    support_order = np.random.default_rng(seed + 46_000).permutation(len(support_pool))
    support = support_pool[support_order[: int(config["review_budget"])]]
    child = _adapt(
        parent,
        integrated_class,
        support,
        int(config["adaptation_rank"]),
        seed,
    )
    known_dev = np.isin(dev_y, known_classes)
    unknown_dev = dev_y == int(config["remaining_unknown_class"])
    threshold_rows = evaluate_threshold_transfer(
        episode_id=f"m46-seed-{seed}-integrate-{integrated_class}",
        parent=parent,
        child=child,
        anchor_x=anchor_x,
        known_x=dev_x[known_dev],
        known_y=dev_y[known_dev],
        unknown_x=dev_x[unknown_dev],
        coverage_target=float(config["anchor_known_coverage_target"]),
        maximum_unknown_recall_drop=float(config["maximum_unknown_recall_drop"]),
        maximum_known_accuracy_drop=float(config["maximum_known_accuracy_drop"]),
    )

    candidate_x = train_x[train_y == integrated_class]
    _, candidate_novelty = parent.predict(candidate_x)
    rejected_x = candidate_x[candidate_novelty > parent.threshold]
    budget = int(config["review_budget"])
    core_indices, boundary_indices = boundary_inclusive_indices(rejected_x, budget)
    core_x = rejected_x[core_indices]
    boundary_x = rejected_x[boundary_indices]
    modes = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(rejected_x)
    core_modes = set(modes[core_indices].tolist())
    boundary_modes = set(modes[boundary_indices].tolist())
    evaluation = np.isin(dev_y, np.append(known_classes, integrated_class))
    core_child = _adapt(
        parent, integrated_class, core_x, int(config["adaptation_rank"]), seed
    )
    boundary_child = _adapt(
        parent, integrated_class, boundary_x, int(config["adaptation_rank"]), seed
    )
    core_utility = _utility(
        parent, core_child, dev_x[evaluation], dev_y[evaluation]
    )
    boundary_utility = _utility(
        parent, boundary_child, dev_x[evaluation], dev_y[evaluation]
    )
    diagnostics = {
        "seed": seed,
        "full_rejection_count": len(rejected_x),
        "core_mode_coverage": len(core_modes) / 2.0,
        "boundary_inclusive_mode_coverage": len(boundary_modes) / 2.0,
        "distance_to_parent_boundary": {
            "median": float(np.median(candidate_novelty - parent.threshold)),
            "p05": float(np.quantile(candidate_novelty - parent.threshold, 0.05)),
            "p95": float(np.quantile(candidate_novelty - parent.threshold, 0.95)),
        },
        "core_representativeness": representativeness_metrics(rejected_x, core_x),
        "boundary_representativeness": representativeness_metrics(
            rejected_x, boundary_x
        ),
        "core_utility": core_utility,
        "boundary_inclusive_utility": boundary_utility,
        "boundary_statistic_delta_utility": boundary_utility - core_utility,
    }
    return threshold_rows, diagnostics


def _rank_interface_mismatches(
    selection_rows: list[dict[str, Any]],
    m42: dict[str, Any],
    a3: dict[str, Any],
) -> list[dict[str, Any]]:
    selection_delta = float(
        np.mean([row["boundary_statistic_delta_utility"] for row in selection_rows])
    )
    low_rank = m42["summaries"]["low_rank_gaussian"]
    routing_loss = 1.0 - float(low_rank["mean_winner_inclusion"])
    locality_loss = 1.0 - float(
        a3["model_records"]["weighted_affine_rank32"][
            "unaffected_prediction_preservation"
        ]
    )
    rows = [
        {
            "interface": "clusterer_to_review",
            "missing_statistic": "boundary_member_ids",
            "estimated_delta_utility": selection_delta,
            "failure_frequency": 1.0,
            "implementation_cost": "low",
            "evidence": "paired core versus boundary-inclusive rank-16 proxy adapters",
        },
        {
            "interface": "adapter_to_router",
            "missing_statistic": "full_normalized_score_vector",
            "estimated_delta_utility": routing_loss,
            "failure_frequency": 1.0,
            "implementation_cost": "high",
            "evidence": "M42 low-rank-Gaussian exhaustive-winner omission",
        },
        {
            "interface": "review_to_adapter",
            "missing_statistic": "changed_region_fusion_scope",
            "estimated_delta_utility": locality_loss,
            "failure_frequency": 1.0,
            "implementation_cost": "medium",
            "evidence": "A3 weighted-affine unaffected-prediction leakage",
        },
    ]
    return sorted(
        rows,
        key=lambda row: (
            -float(row["estimated_delta_utility"]),
            -float(row["failure_frequency"]),
            {"low": 0, "medium": 1, "high": 2}[str(row["implementation_cost"])],
        ),
    )


def run_m46(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    _verify_locks(config)
    source = _load_json(REPO_ROOT / config["source_config"])
    threshold_rows = []
    selection_rows = []
    for seed in config["seeds"]:
        rows, diagnostics = _threshold_seed(config, source, int(seed))
        threshold_rows.extend(rows)
        selection_rows.append(diagnostics)
    rules = {}
    for rule in config["threshold_rules"]:
        rows = [row for row in threshold_rows if row["rule"] == rule]
        rules[rule] = {
            "all_seeds_pass": all(row["passes"] for row in rows),
            "mean_known_accuracy_drop": float(
                np.mean([row["known_accuracy_drop"] for row in rows])
            ),
            "mean_unknown_recall_drop": float(
                np.mean([row["unknown_recall_drop"] for row in rows])
            ),
            "mean_unknown_recall": float(
                np.mean([row["unknown_recall"] for row in rows])
            ),
            "exact_rollback": all(
                row["rollback_restores_parent_threshold"] for row in rows
            ),
        }
    dynamic = [
        (name, summary)
        for name, summary in rules.items()
        if name != "frozen_pre_integration" and summary["all_seeds_pass"]
    ]
    retained_rule = (
        max(
            dynamic,
            key=lambda item: (
                item[1]["mean_unknown_recall"],
                -item[1]["mean_known_accuracy_drop"],
                {
                    "class_count_heuristic": 0,
                    "anchor_quantile": 1,
                    "per_class_anchor_fail_closed": 2,
                }[item[0]],
            ),
        )[0]
        if dynamic
        else "frozen_pre_integration"
    )
    m42_index = _load_json(REPO_ROOT / config["parent_locks"][1]["path"])
    m42 = _load_json(
        (REPO_ROOT / config["parent_locks"][1]["path"]).parent / "evidence.json"
    )
    if payload_hash(m42) != m42_index["evidence_sha256"]:
        raise ValueError("M42 evidence drifted")
    a3 = _load_json(REPO_ROOT / config["parent_locks"][2]["path"])
    ranked = _rank_interface_mismatches(selection_rows, m42, a3)
    frozen_features = [
        "distance_to_cluster_center",
        "distance_to_parent_boundary",
        "low_rank_subspace_coverage",
        "covariance_trace_ratio",
        "omitted_region_nearest_neighbor_coverage",
        "class_mode_coverage",
    ]
    evidence = {
        "schema_version": 1,
        "milestone": "M46",
        "config_sha256": payload_hash(config),
        "threshold_records": threshold_rows,
        "threshold_rule_summaries": rules,
        "retained_threshold_rule": retained_rule,
        "threshold_transfer_supported": retained_rule != "frozen_pre_integration",
        "selection_diagnostics": selection_rows,
        "ranked_interface_mismatches": ranked,
        "frozen_m47_selector_features": frozen_features,
        "review_labels_consumed_by_threshold_transfer": 0,
        "exact_replay": True,
        "final_labels_opened": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    index = build_artifact_index(output_dir)
    return {
        "retained_threshold_rule": retained_rule,
        "threshold_transfer_supported": evidence["threshold_transfer_supported"],
        "passing_rule_count": sum(
            summary["all_seeds_pass"] for summary in rules.values()
        ),
        "ranked_mismatch_count": len(ranked),
        "frozen_selector_feature_count": len(frozen_features),
        "final_labels_opened": False,
        "artifact_count": len(index["artifacts"]),
    }


def verify_m46(config_path: Path, output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "first"
        second = root / "second"
        first_summary = run_m46(config_path, first)
        second_summary = run_m46(config_path, second)
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
            raise RuntimeError("M46 replay was not byte-identical")
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
    print(json.dumps(verify_m46(args.config, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
