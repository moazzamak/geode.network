from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.common.classification_metrics import balanced_accuracy
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v7_adaptation import GaussianBundle, fit_gaussian_bundle
from experiments.common.v8_diagnostics import predictions_with_rejection
from experiments.common.v8_local_residual import (
    frozen_affected_region,
    residual_predictions,
)
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.tier4.eval_v8_m47_review_utility import (
    _adapt,
    _partition_class,
    _recalibrate,
)
from experiments.common.v8_review_selection import kcenter_indices
from src.runtime.schemas import LocalizedResidualScope


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v8" / "m49_local_residuals.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v8" / "m49_local_residuals"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _base_parent(
    config: dict[str, Any], partitions: dict[int, dict[str, np.ndarray]]
) -> tuple[GaussianBundle, np.ndarray]:
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
    return (
        _recalibrate(
            provisional,
            anchor_x,
            float(config["anchor_known_coverage_target"]),
        ),
        anchor_x,
    )


def _fit_temperature(
    child: GaussianBundle,
    parent: GaussianBundle,
    support: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    label: int,
    config: dict[str, Any],
) -> float:
    affected = frozen_affected_region(
        parent,
        support,
        validation_x,
        responsibility_threshold=float(config["responsibility_threshold"]),
        support_radius_multiplier=float(config["support_radius_multiplier"]),
    )
    candidates = (0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
    scored = []
    for temperature in candidates:
        predictions, _ = residual_predictions(
            child,
            validation_x,
            affected,
            target_label=label,
            target_temperature=temperature,
        )
        scored.append((balanced_accuracy(validation_y, predictions), temperature))
    return max(scored)[1]


def _fit_affine(
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    label: int,
    seed: int,
) -> Any:
    target = (validation_y == label).astype(np.int64)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=500,
            random_state=seed,
            solver="liblinear",
        ),
    )
    model.fit(validation_x, target)
    return model


def _run_attempt(
    *,
    attempt: str,
    config: dict[str, Any],
    seed: int,
    model_seed: int,
    parent: GaussianBundle,
    child: GaussianBundle,
    support: np.ndarray,
    proxy_x: np.ndarray,
    proxy_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    label: int,
) -> dict[str, Any]:
    evaluation_mask = np.isin(dev_y, child.class_order)
    known_mask = np.isin(dev_y, parent.class_order)
    unknown_mask = dev_y > label
    dev_affected = frozen_affected_region(
        parent,
        support,
        dev_x,
        responsibility_threshold=float(config["responsibility_threshold"]),
        support_radius_multiplier=float(config["support_radius_multiplier"]),
    )
    baseline_predictions, _ = predictions_with_rejection(
        child, dev_x, child.threshold, {}
    )
    if attempt == "local_target_temperature":
        parameter = _fit_temperature(
            child, parent, support, proxy_x, proxy_y, label, config
        )
        residual, _ = residual_predictions(
            child,
            dev_x,
            dev_affected,
            target_label=label,
            target_temperature=parameter,
        )
        residual_state = {"temperature": parameter}
    elif attempt == "bounded_class_local_affine":
        model = _fit_affine(proxy_x, proxy_y, label, model_seed)
        correction = np.clip(model.decision_function(dev_x), -3.0, 3.0)
        residual, _ = residual_predictions(
            child,
            dev_x,
            dev_affected,
            target_label=label,
            target_correction=correction,
        )
        residual_state = {
            "correction_bound": 3.0,
            "model_hash": payload_hash(
                {
                    "coefficient": model[-1].coef_.tolist(),
                    "intercept": model[-1].intercept_.tolist(),
                    "scale": model[0].scale_.tolist(),
                    "mean": model[0].mean_.tolist(),
                }
            ),
        }
    else:
        raise ValueError(f"unsupported M49 attempt: {attempt}")
    replay, _ = (
        residual_predictions(
            child,
            dev_x,
            dev_affected,
            target_label=label,
            target_temperature=float(residual_state["temperature"]),
        )
        if attempt == "local_target_temperature"
        else residual_predictions(
            child,
            dev_x,
            dev_affected,
            target_label=label,
            target_correction=np.clip(model.decision_function(dev_x), -3.0, 3.0),
        )
    )
    parent_predictions, _ = predictions_with_rejection(
        parent, dev_x, parent.threshold, {}
    )
    baseline_utility = balanced_accuracy(
        dev_y[evaluation_mask], baseline_predictions[evaluation_mask]
    ) - balanced_accuracy(dev_y[evaluation_mask], parent_predictions[evaluation_mask])
    residual_utility = balanced_accuracy(
        dev_y[evaluation_mask], residual[evaluation_mask]
    ) - balanced_accuracy(dev_y[evaluation_mask], parent_predictions[evaluation_mask])
    parent_known = balanced_accuracy(
        dev_y[known_mask], parent_predictions[known_mask]
    )
    residual_known = balanced_accuracy(dev_y[known_mask], residual[known_mask])
    _, parent_unknown_rejected = predictions_with_rejection(
        parent, dev_x[unknown_mask], parent.threshold, {}
    )
    residual_unknown_recall = float(np.mean(residual[unknown_mask] == -1))
    parent_unknown_recall = float(np.mean(parent_unknown_rejected))
    unaffected = ~dev_affected
    preservation = float(
        np.mean(residual[unaffected] == baseline_predictions[unaffected])
    )
    rollback_predictions, _ = predictions_with_rejection(
        child, dev_x, child.threshold, {}
    )
    scope = LocalizedResidualScope(
        episode_id=f"seed-{seed}-arrival-{label}-{attempt}",
        parent_bundle_hash=parent.bundle_hash,
        affected_sample_ids=tuple(
            f"seed-{seed}-dev-{index:04d}" for index in np.flatnonzero(dev_affected)
        ),
        unaffected_sample_ids=tuple(
            f"seed-{seed}-dev-{index:04d}" for index in np.flatnonzero(unaffected)
        ),
        activated_component_ids=tuple(
            f"parent-class-{class_id}" for class_id in parent.class_order
        ),
        responsibility_threshold=float(config["responsibility_threshold"]),
        minimum_unaffected_preservation=float(
            config["minimum_unaffected_prediction_preservation"]
        ),
    )
    return {
        "seed": seed,
        "arrival_class": label,
        "attempt": attempt,
        "scope": scope.to_dict(),
        "affected_fraction": float(np.mean(dev_affected)),
        "residual_state": residual_state,
        "baseline_utility": baseline_utility,
        "residual_utility": residual_utility,
        "utility_improvement": residual_utility - baseline_utility,
        "known_regression": parent_known - residual_known,
        "parent_remaining_unknown_recall": parent_unknown_recall,
        "remaining_unknown_recall": residual_unknown_recall,
        "unknown_recall_drop": parent_unknown_recall - residual_unknown_recall,
        "unaffected_prediction_preservation": preservation,
        "class_local_update": True,
        "region_local_update": True,
        "exact_replay": bool(np.array_equal(replay, residual)),
        "exact_rollback": bool(
            np.array_equal(rollback_predictions, baseline_predictions)
        ),
        "final_labels_opened": False,
    }


def run_m49(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    if len(config["attempts"]) != 2:
        raise ValueError("M49 permits exactly two preregistered attempts")
    for lock in config["parent_locks"]:
        if sha256_file(REPO_ROOT / lock["path"]) != lock["sha256"]:
            raise ValueError(f"M49 parent lock drifted: {lock['id']}")
    source = _load_json(REPO_ROOT / config["source_config"])
    rows = []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        loaded = _load_seed_data(source["seed_inputs"][str(seed)])
        train_x, train_y = loaded["datasets"]["train"]
        dev_x, dev_y = loaded["datasets"]["dev"]
        partitions = {
            label: _partition_class(
                train_x[train_y == label],
                seed=seed,
                label=label,
                geometry_count=int(config["geometry_count_per_class"]),
                anchor_count=int(config["anchor_count_per_class"]),
                validation_count=int(config["proxy_validation_count_per_class"]),
            )
            for label in range(10)
        }
        parent, anchor_x = _base_parent(config, partitions)
        for episode_index, label_value in enumerate(config["arrival_classes"]):
            label = int(label_value)
            candidate_x = partitions[label]["geometry"]
            selected = kcenter_indices(candidate_x, int(config["review_budget"]))
            support = candidate_x[selected]
            child, rollback_exact = _adapt(
                parent,
                label=label,
                support=support,
                rank=int(config["production_adaptation_rank"]),
                review_id=f"review-m49-{seed}-{label}",
            )
            if not rollback_exact:
                raise RuntimeError("M49 parent transaction did not roll back exactly")
            next_anchor = np.concatenate((anchor_x, partitions[label]["anchor"]))
            child = _recalibrate(
                child,
                next_anchor,
                float(config["anchor_known_coverage_target"]),
            )
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
            for attempt in config["attempts"]:
                rows.append(
                    _run_attempt(
                        attempt=attempt,
                        config=config,
                        seed=seed,
                        model_seed=seed + episode_index * 10,
                        parent=parent,
                        child=child,
                        support=support,
                        proxy_x=proxy_x,
                        proxy_y=proxy_y,
                        dev_x=dev_x,
                        dev_y=dev_y,
                        label=label,
                    )
                )
            parent = child
            anchor_x = next_anchor
    summaries = {}
    for attempt in config["attempts"]:
        attempt_rows = [row for row in rows if row["attempt"] == attempt]
        locality_pass = all(
            row["unaffected_prediction_preservation"]
            >= float(config["minimum_unaffected_prediction_preservation"])
            for row in attempt_rows
        )
        safety_pass = all(
            row["known_regression"] <= float(config["maximum_known_accuracy_drop"])
            and row["unknown_recall_drop"]
            <= float(config["maximum_unknown_recall_drop"])
            and row["class_local_update"]
            and row["region_local_update"]
            and row["exact_replay"]
            and row["exact_rollback"]
            for row in attempt_rows
        )
        mean_improvement = float(
            np.mean([row["utility_improvement"] for row in attempt_rows])
        )
        summaries[attempt] = {
            "mean_utility_improvement": mean_improvement,
            "minimum_unaffected_prediction_preservation": min(
                row["unaffected_prediction_preservation"] for row in attempt_rows
            ),
            "maximum_known_regression": max(
                row["known_regression"] for row in attempt_rows
            ),
            "maximum_unknown_recall_drop": max(
                row["unknown_recall_drop"] for row in attempt_rows
            ),
            "locality_pass": locality_pass,
            "safety_pass": safety_pass,
            "passes": (
                mean_improvement >= float(config["minimum_utility_improvement"])
                and locality_pass
                and safety_pass
            ),
        }
    both_fail_locality = all(
        not summary["locality_pass"] for summary in summaries.values()
    )
    retained = [
        attempt for attempt, summary in summaries.items() if summary["passes"]
    ]
    evidence = {
        "schema_version": 1,
        "milestone": "M49",
        "config_sha256": payload_hash(config),
        "attempt_rows": rows,
        "attempt_summaries": summaries,
        "retained_residual": retained[0] if retained else None,
        "outcome_e_locality_blocked": both_fail_locality,
        "final_main_program_outcome": "Outcome D",
        "final_labels_opened": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    index = build_artifact_index(output_dir)
    return {
        "attempt_count": len(summaries),
        "retained_residual": evidence["retained_residual"],
        "outcome_e_locality_blocked": both_fail_locality,
        "final_main_program_outcome": "Outcome D",
        "final_labels_opened": False,
        "artifact_count": len(index["artifacts"]),
    }


def verify_m49(config_path: Path, output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "first"
        second = root / "second"
        first_summary = run_m49(config_path, first)
        second_summary = run_m49(config_path, second)
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
            raise RuntimeError("M49 replay was not byte-identical")
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
    print(json.dumps(verify_m49(args.config, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
