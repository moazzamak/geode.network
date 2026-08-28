from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v6_factorial import primitive_field_matrix
from experiments.common.v61_weighted_readout import weighted_class_logits
from experiments.common.v7_acceptance import (
    AcceptanceOutput,
    _knn_head,
    _low_rank_gaussian_head,
    _rbf_head,
)
from experiments.common.v9_surface_support import (
    BoundedTubePrimitive,
    fit_bounded_tube,
)
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.tier4.eval_v9_m51_surface_diagnostics import _partition_seed
from src.subspace_primitive import SubspacePrimitive


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v9" / "m53_bounded_tube_s1.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v9" / "m53_bounded_tube_s1"


def _resolve(path: str) -> Path:
    return REPO_ROOT / Path(path)


def _verify(specification: dict[str, str]) -> Path:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"immutable artifact hash mismatch: {path}")
    return path


def _score_tubes(
    tubes: list[BoundedTubePrimitive],
    query: np.ndarray,
    *,
    bounded: bool,
) -> AcceptanceOutput:
    scores = np.column_stack(
        [
            tube.score(query) if bounded else tube.unbounded_score(query)
            for tube in tubes
        ]
    )
    classes = np.asarray([tube.class_label for tube in tubes], dtype=np.int64)
    state_hash = payload_hash(
        {
            "bounded": bounded,
            "tubes": [tube.to_dict() for tube in tubes],
        }
    )
    return AcceptanceOutput(
        predictions=classes[np.argmin(scores, axis=1)],
        novelty=np.min(scores, axis=1),
        state_hash=state_hash,
        size_bytes=sum(tube.parameter_count for tube in tubes) * 8,
    )


def _frozen_volume_output(
    student: dict[str, Any],
    query: np.ndarray,
    known_classes: np.ndarray,
) -> AcceptanceOutput:
    all_candidates = [
        SubspacePrimitive.from_dict(item["payload"])
        for item in student["selected_candidates"]
    ]
    all_labels = np.asarray(
        [int(candidate.class_label) for candidate in all_candidates], dtype=np.int64
    )
    eligible = np.isin(all_labels, known_classes)
    candidates = [
        candidate
        for candidate, keep in zip(all_candidates, eligible, strict=True)
        if keep
    ]
    labels = all_labels[eligible]
    weights = np.asarray(student["component_weights"], dtype=np.float64)[eligible]
    fields = primitive_field_matrix(
        candidates, query, primitive="subspace_r32", score="normalized_radial"
    )
    logits = weighted_class_logits(
        fields,
        labels,
        known_classes,
        weights,
        global_temperature=float(student["global_temperature"]),
    )
    probabilities = softmax(logits, axis=1)
    return AcceptanceOutput(
        predictions=known_classes[np.argmax(probabilities, axis=1)],
        novelty=-np.max(logits, axis=1),
        state_hash=payload_hash(
            {
                "parent_component_hash": student["parent_component_hash"],
                "known_classes": known_classes.tolist(),
                "weights": weights.tolist(),
                "temperature": student["global_temperature"],
            }
        ),
        size_bytes=sum(candidate.array_bytes for candidate in candidates)
        + weights.nbytes,
    )


def _metrics(
    output: AcceptanceOutput,
    *,
    calibration_count: int,
    evaluation_labels: np.ndarray,
    unknown_classes: np.ndarray,
    known_coverage_target: float,
) -> dict[str, Any]:
    calibration_novelty = output.novelty[:calibration_count]
    evaluation_novelty = output.novelty[calibration_count:]
    predictions = output.predictions[calibration_count:]
    unknown = np.isin(evaluation_labels, unknown_classes)
    known = ~unknown
    threshold = float(
        np.quantile(
            calibration_novelty, known_coverage_target, method="higher"
        )
    )
    accepted = evaluation_novelty <= threshold
    accepted_known = accepted & known
    accepted_accuracy = (
        float(
            balanced_accuracy_score(
                evaluation_labels[accepted_known], predictions[accepted_known]
            )
        )
        if len(np.unique(evaluation_labels[accepted_known])) > 1
        else 0.0
    )
    return {
        "threshold": threshold,
        "known_balanced_accuracy": float(
            balanced_accuracy_score(
                evaluation_labels[known], predictions[known]
            )
        ),
        "known_coverage": float(np.mean(accepted[known])),
        "unknown_recall": float(np.mean(~accepted[unknown])),
        "accepted_known_balanced_accuracy": accepted_accuracy,
        "auroc": float(
            roc_auc_score(unknown.astype(np.int64), evaluation_novelty)
        ),
        "prediction_class_count": int(len(np.unique(predictions[known]))),
        "collapsed_to_one_class": len(np.unique(predictions[known])) == 1,
        "state_hash": output.state_hash,
        "serialized_megabytes": float(output.size_bytes / (1024 * 1024)),
    }


def _stress_tangent_extrapolation(
    tubes: list[BoundedTubePrimitive],
    *,
    threshold: float,
) -> dict[str, Any]:
    multipliers = np.arange(1, 9, dtype=np.float64)
    component_scores: list[list[float]] = []
    system_acceptance: list[float] = []
    for multiplier in multipliers:
        probes = []
        own_scores = []
        for tube in tubes:
            for column in range(tube.rank):
                for sign in (-1.0, 1.0):
                    point = (
                        tube.center
                        + sign
                        * multiplier
                        * tube.tangent_extents[column]
                        * tube.basis[:, column]
                    )
                    probes.append(point)
                    own_scores.append(
                        float(tube.score(np.asarray([point]))[0])
                    )
        probe_array = np.asarray(probes, dtype=np.float64)
        output = _score_tubes(tubes, probe_array, bounded=True)
        component_scores.append(own_scores)
        system_acceptance.append(float(np.mean(output.novelty <= threshold)))
    means = [float(np.mean(values)) for values in component_scores]
    monotonic = all(
        later >= earlier for earlier, later in zip(means, means[1:])
    )
    return {
        "multipliers": multipliers.tolist(),
        "mean_own_component_scores": means,
        "system_acceptance_rates": system_acceptance,
        "component_score_monotonic": monotonic,
        "acceptance_at_8x": system_acceptance[-1],
    }


def run_evaluation(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    m51_config_path = _verify(config["m51_config"])
    _verify(config["m51_evidence"])
    parent_path = _verify(config["parent_student"])
    m51_config = json.loads(m51_config_path.read_text(encoding="utf-8"))
    source_path = _verify(m51_config["source_config"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    loaded = _load_seed_data(source["seed_inputs"][str(seed)])
    train_features, train_labels = loaded["datasets"]["train"]
    dev_features, dev_labels = loaded["datasets"]["dev"]
    known_classes = np.asarray(config["known_classes"], dtype=np.int64)
    unknown_classes = np.asarray(config["proxy_unknown_classes"], dtype=np.int64)
    partitions = _partition_seed(
        train_labels,
        dev_labels,
        seed=seed,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        geometry_fraction=float(m51_config["geometry_fraction"]),
    )
    geometry_x = train_features[partitions["geometry_fit"]]
    geometry_y = train_labels[partitions["geometry_fit"]]
    calibration_x = train_features[partitions["score_calibration"]]
    calibration_y = train_labels[partitions["score_calibration"]]
    evaluation_indices = np.concatenate(
        [partitions["development_eval"], partitions["unknown_eval"]]
    )
    evaluation_x = dev_features[evaluation_indices]
    evaluation_y = dev_labels[evaluation_indices]
    query = np.vstack([calibration_x, evaluation_x])
    calibration_count = len(calibration_x)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    factories: dict[str, Callable[[], AcceptanceOutput]] = {
        "signed_volume_a2": lambda: _frozen_volume_output(
            parent, query, known_classes
        ),
        "low_rank_gaussian": lambda: _low_rank_gaussian_head(
            geometry_x, geometry_y, query, rank=int(config["gaussian_rank"])
        ),
        "knn_support": lambda: _knn_head(
            geometry_x,
            geometry_y,
            query,
            per_class=int(config["knn_support_per_class"]),
            seed=seed,
        ),
        "rbf_svm": lambda: _rbf_head(
            geometry_x,
            geometry_y,
            query,
            seed=seed,
            c_value=float(config["rbf_svm"]["C"]),
            gamma=str(config["rbf_svm"]["gamma"]),
        ),
    }
    results: dict[str, Any] = {}
    for name, factory in factories.items():
        first = factory()
        second = factory()
        result = _metrics(
            first,
            calibration_count=calibration_count,
            evaluation_labels=evaluation_y,
            unknown_classes=unknown_classes,
            known_coverage_target=float(config["known_coverage_target"]),
        )
        result["exact_replay"] = bool(
            first.state_hash == second.state_hash
            and np.array_equal(first.predictions, second.predictions)
            and np.array_equal(first.novelty, second.novelty)
        )
        results[name] = result
    parent_parameter_count = sum(
        SubspacePrimitive.from_dict(item["payload"]).parameter_count
        for item in parent["selected_candidates"]
    )
    rank_results: dict[str, Any] = {}
    for rank in config["ranks"]:
        tubes = [
            fit_bounded_tube(
                geometry_x[geometry_y == class_label],
                calibration_x[calibration_y == class_label],
                rank=int(rank),
                extent_quantile=float(config["extent_quantile"]),
                scale_quantile=float(config["scale_quantile"]),
                penalty_weight=float(config["penalty_weight"]),
                class_label=int(class_label),
            )
            for class_label in known_classes
        ]
        bounded_output = _score_tubes(tubes, query, bounded=True)
        unbounded_output = _score_tubes(tubes, query, bounded=False)
        bounded = _metrics(
            bounded_output,
            calibration_count=calibration_count,
            evaluation_labels=evaluation_y,
            unknown_classes=unknown_classes,
            known_coverage_target=float(config["known_coverage_target"]),
        )
        unbounded = _metrics(
            unbounded_output,
            calibration_count=calibration_count,
            evaluation_labels=evaluation_y,
            unknown_classes=unknown_classes,
            known_coverage_target=float(config["known_coverage_target"]),
        )
        stress = _stress_tangent_extrapolation(
            tubes, threshold=float(bounded["threshold"])
        )
        parameter_count = sum(tube.parameter_count for tube in tubes)
        rank_results[str(rank)] = {
            "bounded_tube": bounded,
            "unbounded_residual": unbounded,
            "stress": stress,
            "parameter_count": parameter_count,
            "parent_parameter_count": parent_parameter_count,
            "parameter_budget_ratio": float(parameter_count / parent_parameter_count),
            "tube_state_hash": payload_hash([tube.to_dict() for tube in tubes]),
        }
    baseline = results["signed_volume_a2"]
    gate_config = config["screen_gate"]
    for rank, result in rank_results.items():
        bounded = result["bounded_tube"]
        operands = {
            "balanced_accuracy_gain": float(
                bounded["known_balanced_accuracy"]
                - baseline["known_balanced_accuracy"]
            ),
            "unknown_recall_loss": float(
                baseline["unknown_recall"] - bounded["unknown_recall"]
            ),
            "noncollapsed_predictions": not bounded["collapsed_to_one_class"],
            "component_score_monotonic": result["stress"][
                "component_score_monotonic"
            ],
            "tangent_acceptance_at_8x": result["stress"]["acceptance_at_8x"],
            "parameter_budget_ratio": result["parameter_budget_ratio"],
        }
        result["screen_operands"] = operands
        result["screen_passed"] = bool(
            operands["balanced_accuracy_gain"]
            >= float(gate_config["minimum_balanced_accuracy_gain"])
            and operands["unknown_recall_loss"]
            <= float(gate_config["maximum_unknown_recall_loss"])
            and operands["noncollapsed_predictions"]
            and operands["component_score_monotonic"]
            and operands["tangent_acceptance_at_8x"]
            <= float(gate_config["maximum_8x_tangent_acceptance"])
            and operands["parameter_budget_ratio"]
            <= float(gate_config["maximum_component_budget_ratio"])
        )
    retained = [
        int(rank)
        for rank, result in rank_results.items()
        if result["screen_passed"]
    ]
    evidence = {
        "schema_version": 1,
        "milestone": "M53-S1",
        "configuration_hash": sha256_file(config_path),
        "m51_shell_branch_closed": True,
        "final_labels_opened": False,
        "partition_hashes": {
            name: payload_hash(values.tolist())
            for name, values in partitions.items()
        },
        "controls": results,
        "rank_results": rank_results,
        "retained_ranks": retained,
        "advance_to_m53_s2": bool(retained),
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    write_canonical_json(
        output_dir / "verification.json",
        {
            "schema_version": 1,
            "milestone": "M53-S1",
            "evidence_sha256": sha256_file(output_dir / "evidence.json"),
            "all_controls_replay": all(
                result["exact_replay"] for result in results.values()
            ),
            "retained_ranks": retained,
            "advance_to_m53_s2": bool(retained),
        },
    )
    build_artifact_index(output_dir)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    evidence = run_evaluation(arguments.config, arguments.output)
    summary = {
        rank: {
            "screen_passed": result["screen_passed"],
            **result["screen_operands"],
        }
        for rank, result in evidence["rank_results"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
