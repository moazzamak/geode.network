import argparse
import json
from pathlib import Path

import numpy as np

from experiments.common.classification_metrics import accuracy
from experiments.common.ood_metrics import (
    ood_detection_metrics,
    ood_operating_point,
    select_ood_threshold,
    select_ood_threshold_at_known_coverage,
)
from experiments.common.ood_scores import (
    fit_feature_ood_scorers,
    maximum_probability_score,
    minimum_sdf_score,
    sdf_energy_score,
)
from experiments.common.score_readouts import fit_score_readout
from experiments.tier4.eval_complex_classification import (
    build_csg_variants,
    compute_raw_scores,
    compute_score_scales,
    fit_class_models,
    stratified_geometry_carve_calibration_split,
)
from experiments.tier5.eval_corruption_robustness import generate_multiclass_problem
from src.inference_engine import InferenceEngine


PRODUCTION_BINDABLE_SCORES = {
    "minimum_sdf",
    "sdf_energy",
    "maximum_probability",
}


def _validate_class_partition(
    known_classes: tuple[int, ...],
    proxy_unknown_classes: tuple[int, ...],
    final_unknown_classes: tuple[int, ...],
) -> None:
    groups = [set(known_classes), set(proxy_unknown_classes), set(final_unknown_classes)]
    if any(not group for group in groups):
        raise ValueError("Known, proxy-unknown, and final-unknown groups must be non-empty.")
    if any(groups[first] & groups[second] for first in range(3) for second in range(first)):
        raise ValueError("Leave-class-out groups must be disjoint.")
    if len(known_classes) < 2:
        raise ValueError("At least two known classes are required.")


def _mask(labels: np.ndarray, classes: tuple[int, ...]) -> np.ndarray:
    return np.isin(labels, np.asarray(classes))


def _all_scores(
    metric_scores: np.ndarray,
    raw_scores: np.ndarray,
    probabilities: np.ndarray,
    feature_scores: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "minimum_sdf": minimum_sdf_score(raw_scores),
        "minimum_metric_sdf": minimum_sdf_score(metric_scores),
        "sdf_energy": sdf_energy_score(raw_scores),
        "maximum_probability": maximum_probability_score(probabilities),
        "mahalanobis": feature_scores["mahalanobis"],
        "gmm_nll": feature_scores["gmm_nll"],
        "knn_distance": feature_scores["knn_distance"],
    }


def _metric_class_scores(
    models: dict,
    features: np.ndarray,
    class_ids: np.ndarray,
) -> np.ndarray:
    return np.column_stack([
        InferenceEngine(models[int(class_id)], alpha=2.0).get_metric_corrected_sdf(
            features,
        )
        if models[int(class_id)] else np.full(len(features), 10.0)
        for class_id in class_ids
    ])


def run_leave_class_out_episode(
    problem: dict[str, np.ndarray],
    *,
    known_classes: tuple[int, ...],
    proxy_unknown_classes: tuple[int, ...],
    final_unknown_classes: tuple[int, ...],
    seed: int = 42,
    max_iterations: int = 10,
    geometry_variant: str = "additive",
) -> dict:
    """Fit on known classes and reserve disjoint classes for selection and test."""
    _validate_class_partition(
        known_classes, proxy_unknown_classes, final_unknown_classes,
    )
    if geometry_variant not in {"additive", "validated_csg"}:
        raise ValueError("geometry_variant must be 'additive' or 'validated_csg'.")
    y_geometry = np.asarray(problem["y_geometry"])
    y_calibration = np.asarray(problem["y_calibration"])
    y_test = np.asarray(problem["y_test"])
    required = set(known_classes) | set(proxy_unknown_classes) | set(final_unknown_classes)
    observed = set(np.unique(np.concatenate([y_geometry, y_calibration, y_test])).tolist())
    if not required <= observed:
        raise ValueError(f"Problem is missing declared classes: {sorted(required - observed)}")

    geometry_mask = _mask(y_geometry, known_classes)
    X_geometry = problem["X_geometry"][geometry_mask]
    y_known_geometry = y_geometry[geometry_mask]
    class_ids = np.asarray(known_classes, dtype=np.int32)

    calibration_known_indices = np.flatnonzero(_mask(y_calibration, known_classes))
    id_validation_indices, carve_indices, readout_indices = (
        stratified_geometry_carve_calibration_split(
        calibration_known_indices,
        y_calibration[calibration_known_indices],
        carve_fraction=0.2,
        calibration_fraction=0.25,
        seed=seed,
        )
    )
    proxy_indices = np.flatnonzero(_mask(y_calibration, proxy_unknown_classes))
    id_test_indices = np.flatnonzero(_mask(y_test, known_classes))
    final_unknown_indices = np.flatnonzero(_mask(y_test, final_unknown_classes))
    if not all(map(len, (
        readout_indices,
        id_validation_indices,
        proxy_indices,
        id_test_indices,
        final_unknown_indices,
    ))):
        raise ValueError("Every episode split must contain samples.")

    models = fit_class_models(
        X_geometry,
        y_known_geometry,
        class_ids,
        consensus_threshold=0.1,
        capture_threshold=0.1,
        alpha=2.0,
        max_iterations=max_iterations,
        nudge_iterations=0,
        nudge_learning_rate=0.02,
        seed=seed,
    )
    carve_audit = []
    if geometry_variant == "validated_csg":
        variants, audits = build_csg_variants(
            models,
            X_geometry,
            y_known_geometry,
            problem["X_calibration"][carve_indices],
            y_calibration[carve_indices],
            class_ids,
            capture_threshold=0.1,
            alpha=2.0,
            max_iterations=max_iterations,
            seed=seed,
            mdl_penalty_weight=0.01,
            min_penalized_gain=0.0,
        )
        models = variants["A1"]
        carve_audit = audits["A1"]
    scales = compute_score_scales(
        models, X_geometry, alpha=2.0, class_labels=y_known_geometry,
    )
    readout_scores = compute_raw_scores(
        models, problem["X_calibration"][readout_indices], 2.0, scales,
    )
    readout = fit_score_readout(
        "multinomial",
        readout_scores,
        y_calibration[readout_indices],
        class_ids,
        seed=seed,
    )
    density_scorers = fit_feature_ood_scorers(
        X_geometry,
        gmm_components=len(class_ids),
        knn_k=min(5, len(X_geometry)),
        seed=seed,
    )

    def evaluate(features: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        raw = compute_raw_scores(models, features, 2.0, scales)
        probabilities = readout.predict_proba(raw)
        return probabilities, _all_scores(
            _metric_class_scores(models, features, class_ids),
            raw,
            probabilities,
            density_scorers.score(features),
        )

    id_validation_probabilities, id_validation_scores = evaluate(
        problem["X_calibration"][id_validation_indices],
    )
    proxy_probabilities, proxy_scores = evaluate(
        problem["X_calibration"][proxy_indices],
    )
    id_test_probabilities, id_test_scores = evaluate(problem["X_test"][id_test_indices])
    final_unknown_probabilities, final_unknown_scores = evaluate(
        problem["X_test"][final_unknown_indices],
    )

    validation = {}
    frozen_thresholds = {}
    frozen_class_thresholds = {}
    proxy_predictions = class_ids[proxy_probabilities.argmax(axis=1)]
    id_validation_predictions = class_ids[
        id_validation_probabilities.argmax(axis=1)
    ]
    for score_name in id_validation_scores:
        threshold = select_ood_threshold(proxy_scores[score_name])
        frozen_thresholds[score_name] = threshold
        validation[score_name] = {
            "detection": ood_detection_metrics(
                id_validation_scores[score_name], proxy_scores[score_name],
            ),
            "selected_threshold": threshold,
        }
        class_thresholds = {
            int(class_id): (
                select_ood_threshold(
                    proxy_scores[score_name][proxy_predictions == class_id],
                )
                if np.any(proxy_predictions == class_id) else threshold
            )
            for class_id in class_ids
        }
        frozen_class_thresholds[score_name] = class_thresholds
        validation[f"{score_name}_per_class"] = {
            "detection": validation[score_name]["detection"],
            "selected_thresholds": class_thresholds,
            "fallback_classes": [
                int(class_id) for class_id in class_ids
                if not np.any(proxy_predictions == class_id)
            ],
        }
        coverage_threshold = select_ood_threshold_at_known_coverage(
            id_validation_scores[score_name],
        )
        validation[f"{score_name}_coverage90"] = {
            "detection": validation[score_name]["detection"],
            "selected_threshold": coverage_threshold,
        }
        coverage_class_thresholds = {
            int(class_id): (
                select_ood_threshold_at_known_coverage(
                    id_validation_scores[score_name][
                        id_validation_predictions == class_id
                    ],
                )
                if np.any(id_validation_predictions == class_id)
                else coverage_threshold
            )
            for class_id in class_ids
        }
        validation[f"{score_name}_coverage90_per_class"] = {
            "detection": validation[score_name]["detection"],
            "selected_thresholds": coverage_class_thresholds,
            "fallback_classes": [
                int(class_id) for class_id in class_ids
                if not np.any(id_validation_predictions == class_id)
            ],
        }
    selected_score = max(
        frozen_thresholds,
        key=lambda name: (
            validation[name]["detection"]["auroc"],
            -validation[name]["detection"]["fpr95"],
            name,
        ),
    )

    predictions = class_ids[id_test_probabilities.argmax(axis=1)]
    final_unknown_predictions = class_ids[
        final_unknown_probabilities.argmax(axis=1)
    ]
    final = {}
    for score_name, threshold in frozen_thresholds.items():
        known_accept = id_test_scores[score_name] < threshold
        unknown_reject = final_unknown_scores[score_name] >= threshold
        known_correct = predictions == y_test[id_test_indices]
        final[score_name] = {
            "detection": ood_detection_metrics(
                id_test_scores[score_name], final_unknown_scores[score_name],
            ),
            "operating_point": ood_operating_point(
                id_test_scores[score_name], final_unknown_scores[score_name], threshold,
            ),
            "known_coverage": float(np.mean(known_accept)),
            "known_accuracy": accuracy(y_test[id_test_indices], predictions),
            "accepted_known_accuracy": (
                float(np.mean(known_correct[known_accept]))
                if np.any(known_accept) else None
            ),
            "open_set_accuracy": float((
                np.count_nonzero(known_accept & known_correct)
                + np.count_nonzero(unknown_reject)
            ) / (len(known_accept) + len(unknown_reject))),
        }
        class_thresholds = frozen_class_thresholds[score_name]
        known_class_thresholds = np.asarray([
            class_thresholds[int(class_id)] for class_id in predictions
        ])
        unknown_class_thresholds = np.asarray([
            class_thresholds[int(class_id)]
            for class_id in final_unknown_predictions
        ])
        known_accept = id_test_scores[score_name] < known_class_thresholds
        unknown_reject = (
            final_unknown_scores[score_name] >= unknown_class_thresholds
        )
        final[f"{score_name}_per_class"] = {
            "detection": final[score_name]["detection"],
            "operating_point": {
                "known_coverage": float(np.mean(known_accept)),
                "true_positive_rate": float(np.mean(unknown_reject)),
            },
            "known_coverage": float(np.mean(known_accept)),
            "known_accuracy": final[score_name]["known_accuracy"],
            "accepted_known_accuracy": (
                float(np.mean(known_correct[known_accept]))
                if np.any(known_accept) else None
            ),
            "open_set_accuracy": float((
                np.count_nonzero(known_accept & known_correct)
                + np.count_nonzero(unknown_reject)
            ) / (len(known_accept) + len(unknown_reject))),
        }
        for suffix, thresholds in (
            (
                "coverage90",
                validation[f"{score_name}_coverage90"]["selected_threshold"],
            ),
            (
                "coverage90_per_class",
                validation[f"{score_name}_coverage90_per_class"][
                    "selected_thresholds"
                ],
            ),
        ):
            if isinstance(thresholds, dict):
                known_thresholds = np.asarray([
                    thresholds[int(class_id)] for class_id in predictions
                ])
                unknown_thresholds = np.asarray([
                    thresholds[int(class_id)]
                    for class_id in final_unknown_predictions
                ])
            else:
                known_thresholds = thresholds
                unknown_thresholds = thresholds
            known_accept = id_test_scores[score_name] < known_thresholds
            unknown_reject = final_unknown_scores[score_name] >= unknown_thresholds
            final[f"{score_name}_{suffix}"] = {
                "detection": final[score_name]["detection"],
                "operating_point": {
                    "known_coverage": float(np.mean(known_accept)),
                    "true_positive_rate": float(np.mean(unknown_reject)),
                },
                "known_coverage": float(np.mean(known_accept)),
                "known_accuracy": final[score_name]["known_accuracy"],
                "accepted_known_accuracy": (
                    float(np.mean(known_correct[known_accept]))
                    if np.any(known_accept) else None
                ),
                "open_set_accuracy": float((
                    np.count_nonzero(known_accept & known_correct)
                    + np.count_nonzero(unknown_reject)
                ) / (len(known_accept) + len(unknown_reject))),
            }

    return {
        "protocol": {
            "seed": seed,
            "known_classes": list(known_classes),
            "proxy_unknown_classes": list(proxy_unknown_classes),
            "final_unknown_classes": list(final_unknown_classes),
            "geometry_classes": sorted(np.unique(y_known_geometry).tolist()),
            "geometry_variant": geometry_variant,
            "readout_classes": sorted(np.unique(y_calibration[readout_indices]).tolist()),
            "proxy_unknown_used_for_selection": True,
            "final_unknown_used_for_selection": False,
            "final_test_used_for_selection": False,
            "readout_calibration_count": len(readout_indices),
            "known_carve_count": len(carve_indices),
            "id_validation_count": len(id_validation_indices),
            "proxy_unknown_validation_count": len(proxy_indices),
            "id_test_count": len(id_test_indices),
            "final_unknown_test_count": len(final_unknown_indices),
        },
        "selection": {
            "score": selected_score,
            "threshold": frozen_thresholds[selected_score],
            "production_binding_supported": (
                selected_score in PRODUCTION_BINDABLE_SCORES
            ),
        },
        "carve_audit": {
            "candidate_count": len(carve_audit),
            "accepted_count": sum(
                bool(record.get("accepted")) for record in carve_audit
            ),
        },
        "validation": validation,
        "final_test": final,
    }


def run_leave_class_out_study(
    problem: dict[str, np.ndarray],
    episodes: list[dict],
    *,
    seed: int = 42,
    max_iterations: int = 10,
    minimum_known_coverage: float = 0.9,
    minimum_unknown_recall: float = 0.5,
    geometry_variant: str = "additive",
) -> dict:
    """Run predeclared episodes with globally disjoint proxy/final class pools."""
    if len(episodes) < 2:
        raise ValueError("A transfer study requires at least two episodes.")
    if not 0.0 <= minimum_known_coverage <= 1.0:
        raise ValueError("minimum_known_coverage must lie in [0, 1].")
    if not 0.0 <= minimum_unknown_recall <= 1.0:
        raise ValueError("minimum_unknown_recall must lie in [0, 1].")
    proxy_pool = {
        class_id
        for episode in episodes
        for class_id in episode["proxy_unknown_classes"]
    }
    final_pool = {
        class_id
        for episode in episodes
        for class_id in episode["final_unknown_classes"]
    }
    overlap = proxy_pool & final_pool
    if overlap:
        raise ValueError(
            f"Proxy and final unknown pools must be globally disjoint: {sorted(overlap)}",
        )

    results = []
    for position, episode in enumerate(episodes):
        results.append(run_leave_class_out_episode(
            problem,
            known_classes=tuple(episode["known_classes"]),
            proxy_unknown_classes=tuple(episode["proxy_unknown_classes"]),
            final_unknown_classes=tuple(episode["final_unknown_classes"]),
            seed=seed + position,
            max_iterations=max_iterations,
            geometry_variant=geometry_variant,
        ))

    score_names = tuple(results[0]["final_test"])
    summary = {}
    for score_name in score_names:
        coverage = np.array([
            result["final_test"][score_name]["known_coverage"]
            for result in results
        ])
        unknown_recall = np.array([
            result["final_test"][score_name]["operating_point"]["true_positive_rate"]
            for result in results
        ])
        open_set_accuracy = np.array([
            result["final_test"][score_name]["open_set_accuracy"]
            for result in results
        ])
        gate_by_episode = (
            (coverage >= minimum_known_coverage)
            & (unknown_recall >= minimum_unknown_recall)
        )
        summary[score_name] = {
            "known_coverage_mean": float(np.mean(coverage)),
            "known_coverage_minimum": float(np.min(coverage)),
            "unknown_recall_mean": float(np.mean(unknown_recall)),
            "unknown_recall_minimum": float(np.min(unknown_recall)),
            "open_set_accuracy_mean": float(np.mean(open_set_accuracy)),
            "episodes_passing_gate": int(np.count_nonzero(gate_by_episode)),
            "all_episodes_pass_gate": bool(np.all(gate_by_episode)),
        }

    return {
        "protocol": {
            "seed": seed,
            "episode_count": len(episodes),
            "proxy_unknown_pool": sorted(proxy_pool),
            "final_unknown_pool": sorted(final_pool),
            "proxy_and_final_pools_disjoint": True,
            "minimum_known_coverage": minimum_known_coverage,
            "minimum_unknown_recall": minimum_unknown_recall,
            "geometry_variant": geometry_variant,
            "final_unknown_used_for_selection": False,
        },
        "episodes": results,
        "summary": summary,
        "scores_passing_all_episodes": [
            name for name, record in summary.items()
            if record["all_episodes_pass_gate"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy leave-class-out OOD episode")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/results/tier4_leave_class_out_smoke.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.config is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        problem = generate_multiclass_problem(**config["problem"])
        result = run_leave_class_out_study(
            problem,
            config["episodes"],
            seed=config["seed"],
            max_iterations=config["max_iterations"],
            minimum_known_coverage=config["minimum_known_coverage"],
            minimum_unknown_recall=config["minimum_unknown_recall"],
            geometry_variant=config.get("geometry_variant", "additive"),
        )
        output_path = Path(config["artifact_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    problem = generate_multiclass_problem(
        seed=args.seed,
        dimensions=4,
        class_count=6,
        geometry_per_class=30,
        calibration_per_class=20,
        test_per_class=40,
        center_radius=3.0,
        mode_offset=0.7,
        noise_scale=0.7,
    )
    result = run_leave_class_out_episode(
        problem,
        known_classes=(0, 1, 2),
        proxy_unknown_classes=(3,),
        final_unknown_classes=(4, 5),
        seed=args.seed,
        max_iterations=10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()