from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiments.tier4.eval_calibrated_graph_migration import (
    _build_model,
    _fit_calibrator,
)
from experiments.tier4.eval_complex_classification import _extract_hog_features
from src.model_fingerprint import InputSpec
from src.replay_constrained_fitter import fit_replay_constrained_expert
from src.candidate_usefulness import (
    CandidateUsefulnessPolicy,
    evaluate_candidate_usefulness,
)


def _take(indices: np.ndarray, start: int, count: int) -> np.ndarray:
    return indices[start:start + count]


def _apply_mode_shift(
    images: np.ndarray,
    mode_shift: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if mode_shift == "horizontal_flip":
        return np.flip(images, axis=2).copy()
    if mode_shift == "brightness":
        return np.clip(images.astype(np.float64) * 0.55, 0, 255).astype(np.uint8)
    if mode_shift == "gaussian_noise":
        noise = rng.normal(0.0, 25.0, size=images.shape)
        return np.clip(images.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    if mode_shift == "center_occlusion":
        shifted = images.copy()
        shifted[:, 10:22, 10:22, :] = 0
        return shifted
    raise ValueError(f"Unknown mode shift: {mode_shift}.")


def run_real_feature_mode_update(
    *,
    dataset_path: str,
    seed: int,
    samples_per_slice: int = 40,
    pca_components: int = 8,
    minimum_proposal_gain: float = 0.5,
    minimum_geometric_coverage: float = 0.5,
    mode_shift: str = "horizontal_flip",
    target_class: int = 0,
    other_class: int = 1,
) -> dict:
    data = np.load(dataset_path)
    images = data["images"]
    labels = data["labels"].astype(np.int64)
    rng = np.random.default_rng(seed)
    if target_class == other_class:
        raise ValueError("target_class and other_class must differ.")
    target_indices = np.flatnonzero(labels == target_class)
    other_indices = np.flatnonzero(labels == other_class)
    ood_classes = tuple(
        int(class_id)
        for class_id in np.unique(labels)
        if class_id not in {target_class, other_class}
    )
    rng.shuffle(target_indices)
    rng.shuffle(other_indices)
    ood_per_class = max(2, samples_per_slice // 8)
    ood_ids = np.concatenate([
        rng.choice(
            np.flatnonzero(labels == class_id),
            size=ood_per_class,
            replace=False,
        )
        for class_id in ood_classes
    ])
    required_target = samples_per_slice * 9
    required_other = samples_per_slice * 7
    if len(target_indices) < required_target or len(other_indices) < required_other:
        raise ValueError("Dataset does not contain enough samples for the protocol.")

    live_target_ids = _take(target_indices, 0, samples_per_slice * 3)
    proposal_ids = _take(target_indices, samples_per_slice * 3, samples_per_slice * 2)
    calibration_old_ids = _take(
        target_indices, samples_per_slice * 5, samples_per_slice,
    )
    calibration_new_ids = _take(
        target_indices, samples_per_slice * 6, samples_per_slice,
    )
    test_old_ids = _take(target_indices, samples_per_slice * 7, samples_per_slice)
    test_new_ids = _take(target_indices, samples_per_slice * 8, samples_per_slice)
    live_other_ids = _take(other_indices, 0, samples_per_slice * 3)
    calibration_other_ids = _take(
        other_indices, samples_per_slice * 3, samples_per_slice * 2,
    )
    test_other_ids = _take(other_indices, samples_per_slice * 5, samples_per_slice * 2)

    proposal_images = _apply_mode_shift(images[proposal_ids], mode_shift, rng)
    calibration_new_images = _apply_mode_shift(
        images[calibration_new_ids], mode_shift, rng,
    )
    test_new_images = _apply_mode_shift(images[test_new_ids], mode_shift, rng)
    image_groups = (
        images[live_target_ids],
        images[live_other_ids],
        proposal_images,
        images[calibration_old_ids],
        calibration_new_images,
        images[calibration_other_ids],
        images[test_old_ids],
        test_new_images,
        images[test_other_ids],
        images[ood_ids],
    )
    group_sizes = [len(group) for group in image_groups]
    all_features = _extract_hog_features(np.concatenate(image_groups))
    offsets = np.cumsum([0, *group_sizes])
    groups = tuple(
        all_features[offsets[index]:offsets[index + 1]]
        for index in range(len(group_sizes))
    )
    (
        live_target,
        live_other,
        proposal,
        calibration_old,
        calibration_new,
        calibration_other,
        test_old,
        test_new,
        test_other,
        ood_features,
    ) = groups

    live_X = np.vstack([live_target, live_other])
    live_y = np.concatenate([
        np.zeros(len(live_target), dtype=np.int64),
        np.ones(len(live_other), dtype=np.int64),
    ])
    pca = PCA(n_components=pca_components, whiten=True, random_state=seed)
    live_pca = pca.fit_transform(live_X)
    scaler = StandardScaler().fit(live_pca)
    live_geometry = scaler.transform(live_pca)
    live_model = _build_model(
        "cifar_hog_mode_update",
        InputSpec("raw_hog", dim=all_features.shape[1]),
        live_geometry,
        live_y,
        (0, 1),
    )
    live_model.pca = pca
    live_model.scaler = scaler

    live_calibration_X = np.vstack([
        calibration_old, calibration_other[:samples_per_slice],
    ])
    live_calibration_y = np.concatenate([
        np.zeros(len(calibration_old), dtype=np.int64),
        np.ones(samples_per_slice, dtype=np.int64),
    ])
    live_model.calibrator = _fit_calibrator(
        live_model, live_calibration_X, live_calibration_y, seed,
    )
    test_X = np.vstack([test_old, test_new, test_other])
    test_y = np.concatenate([
        np.zeros(len(test_old) + len(test_new), dtype=np.int64),
        np.ones(len(test_other), dtype=np.int64),
    ])
    live_predictions_before = live_model.predict(test_X)
    live_ood_scores = live_model.sdf_scores(ood_features)
    live_ood_unknown_recall = float(np.mean(live_ood_scores.min(axis=1) >= 0.0))

    proposal_features = live_model._to_feature_space(proposal)
    exclusion_features = live_model._to_feature_space(live_other)
    constrained_fit = fit_replay_constrained_expert(
        proposal_features,
        exclusion_features,
        exclusion_margin=0.1,
    )
    baseline_proposal_success = float(np.mean(live_model.predict(proposal) == 0))
    usefulness = evaluate_candidate_usefulness(
        baseline_success=baseline_proposal_success,
        geometric_coverage=constrained_fit.positive_coverage,
        policy=CandidateUsefulnessPolicy(
            minimum_proposal_gain,
            minimum_geometric_coverage,
        ),
    )
    transaction_attempted = False
    transaction = {"accepted": False}
    candidate_predictions = live_model.predict(test_X)
    old_slice = slice(0, len(test_old))
    new_slice = slice(len(test_old), len(test_old) + len(test_new))
    other_slice = slice(len(test_old) + len(test_new), len(test_X))
    live_predictions_after = live_model.predict(test_X)

    return {
        "protocol": {
            "seed": seed,
            "dataset_path": dataset_path,
            "representation": "cifar10_hog",
            "mode_shift": mode_shift,
            "target_class": target_class,
            "other_class": other_class,
            "ood_control_classes": list(ood_classes),
            "ood_controls_used_for_fitting": False,
            "source_images_disjoint_across_slices": True,
            "test_used_for_fitting_or_calibration": False,
            "mutation_published": False,
            "safety_validation_available": False,
            "benchmark_scope": "pre_transaction_usefulness_screen",
        },
        "update": {
            "usefulness_eligible": usefulness.eligible,
            "usefulness_failed_criteria": list(usefulness.failed_criteria),
            "baseline_proposal_success": usefulness.baseline_success,
            "maximum_possible_gain": usefulness.maximum_possible_gain,
            "transaction_attempted": transaction_attempted,
            "transaction_accepted": transaction["accepted"],
            "radius_scale": constrained_fit.radius_scale,
            "fit_positive_coverage": constrained_fit.positive_coverage,
            "exclusion_violations": constrained_fit.exclusion_violations,
            "live_model_unchanged": np.array_equal(
                live_predictions_before, live_predictions_after,
            ),
        },
        "observational_test": {
            "live_accuracy": float(np.mean(live_predictions_before == test_y)),
            "candidate_accuracy": float(np.mean(candidate_predictions == test_y)),
            "old_mode_accuracy_before": float(np.mean(
                live_predictions_before[old_slice] == 0,
            )),
            "old_mode_accuracy_after": float(np.mean(
                candidate_predictions[old_slice] == 0,
            )),
            "new_mode_accuracy_before": float(np.mean(
                live_predictions_before[new_slice] == 0,
            )),
            "new_mode_accuracy_after": float(np.mean(
                candidate_predictions[new_slice] == 0,
            )),
            "other_class_accuracy_before": float(np.mean(
                live_predictions_before[other_slice] == 1,
            )),
            "other_class_accuracy_after": float(np.mean(
                candidate_predictions[other_slice] == 1,
            )),
            "live_ood_unknown_recall": live_ood_unknown_recall,
        },
    }


def run_multiseed(**config) -> dict:
    seeds = tuple(config.pop("seeds"))
    mode_shifts = tuple(config.pop("mode_shifts", ("horizontal_flip",)))
    class_pairs = tuple(
        tuple(pair) for pair in config.pop("class_pairs", ((0, 1),))
    )
    runs = [
        run_real_feature_mode_update(
            seed=seed,
            mode_shift=mode_shift,
            target_class=target_class,
            other_class=other_class,
            **config,
        )
        for target_class, other_class in class_pairs
        for mode_shift in mode_shifts
        for seed in seeds
    ]
    family_summary = {}
    for target_class, other_class in class_pairs:
        for mode_shift in mode_shifts:
            key = f"{target_class}_vs_{other_class}:{mode_shift}"
            family_runs = [
                run for run in runs
                if run["protocol"]["mode_shift"] == mode_shift
                and run["protocol"]["target_class"] == target_class
                and run["protocol"]["other_class"] == other_class
            ]
            family_summary[key] = {
                "usefulness_eligible": sum(
                    run["update"]["usefulness_eligible"] for run in family_runs
                ),
                "baseline_new_mode_accuracy_mean": float(np.mean([
                    run["observational_test"]["new_mode_accuracy_before"]
                    for run in family_runs
                ])),
                "fit_positive_coverage_mean": float(np.mean([
                    run["update"]["fit_positive_coverage"] for run in family_runs
                ])),
                "live_ood_unknown_recall_mean": float(np.mean([
                    run["observational_test"]["live_ood_unknown_recall"]
                    for run in family_runs
                ])),
            }
    return {
        "protocol": {
            "seeds": list(seeds),
            "mode_shifts": list(mode_shifts),
            "class_pairs": [list(pair) for pair in class_pairs],
            "benchmark_scope": "pre_transaction_usefulness_screen",
            "mutation_published": False,
        },
        "summary": {
            "usefulness_eligible": sum(
                run["update"]["usefulness_eligible"] for run in runs
            ),
            "transactions_attempted": sum(
                run["update"]["transaction_attempted"] for run in runs
            ),
            "accepted_transactions": sum(
                run["update"]["transaction_accepted"] for run in runs
            ),
            "exclusion_violations": sum(
                run["update"]["exclusion_violations"] for run in runs
            ),
            "usefulness_failed_criterion_counts": {
                criterion: sum(
                    criterion in run["update"]["usefulness_failed_criteria"]
                    for run in runs
                )
                for criterion in (
                    "insufficient_gain_headroom",
                    "insufficient_geometric_coverage",
                )
            },
            "fit_positive_coverage_mean": float(np.mean([
                run["update"]["fit_positive_coverage"] for run in runs
            ])),
            "fit_positive_coverage_maximum": float(np.max([
                run["update"]["fit_positive_coverage"] for run in runs
            ])),
            "radius_scale_mean": float(np.mean([
                run["update"]["radius_scale"] for run in runs
            ])),
            "candidate_accuracy_mean": float(np.mean([
                run["observational_test"]["candidate_accuracy"] for run in runs
            ])),
            "new_mode_accuracy_before_mean": float(np.mean([
                run["observational_test"]["new_mode_accuracy_before"]
                for run in runs
            ])),
            "old_mode_accuracy_drop_mean": float(np.mean([
                run["observational_test"]["old_mode_accuracy_before"]
                - run["observational_test"]["old_mode_accuracy_after"]
                for run in runs
            ])),
            "new_mode_accuracy_gain_mean": float(np.mean([
                run["observational_test"]["new_mode_accuracy_after"]
                - run["observational_test"]["new_mode_accuracy_before"]
                for run in runs
            ])),
            "other_class_accuracy_drop_mean": float(np.mean([
                run["observational_test"]["other_class_accuracy_before"]
                - run["observational_test"]["other_class_accuracy_after"]
                for run in runs
            ])),
            "live_ood_unknown_recall_mean": float(np.mean([
                run["observational_test"]["live_ood_unknown_recall"]
                for run in runs
            ])),
        },
        "families": family_summary,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-feature mode update study")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact_path = Path(config.pop("artifact_path"))
    result = run_multiseed(**config)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()