from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiments.common.representation_metrics import (
    compute_representation_diagnostics,
)
from experiments.common.ood_metrics import (
    conformal_prediction_sets,
    conformal_probability_threshold,
    conformal_set_metrics,
    ood_detection_metrics,
    ood_operating_point,
    select_ood_threshold_at_known_coverage,
)
from experiments.common.ood_scores import (
    fit_class_conditional_ood_scorers,
    fit_feature_ood_scorers,
    maximum_probability_score,
    minimum_sdf_score,
    sdf_energy_score,
)
from experiments.tier4.eval_calibrated_graph_migration import (
    _build_model,
    _fit_calibrator,
)
from experiments.tier4.eval_complex_classification import (
    _extract_cnn_features,
    _extract_hog_features,
    _extract_resnet18_features,
)
from src.inference_engine import InferenceEngine
from src.model_fingerprint import InputSpec


def _validate_partition(
    known_classes: tuple[int, ...],
    proxy_unknown_classes: tuple[int, ...],
    final_unknown_classes: tuple[int, ...],
) -> None:
    groups = tuple(map(set, (
        known_classes, proxy_unknown_classes, final_unknown_classes,
    )))
    if len(known_classes) < 2 or any(not group for group in groups):
        raise ValueError("At least two known and one proxy/final class are required.")
    if any(
        groups[first] & groups[second]
        for first in range(len(groups))
        for second in range(first)
    ):
        raise ValueError("Known, proxy, and final class groups must be disjoint.")


def _episode_source_slices(
    labels: np.ndarray,
    *,
    known_classes: tuple[int, ...],
    proxy_unknown_classes: tuple[int, ...],
    final_unknown_classes: tuple[int, ...],
    samples_per_slice: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    slices: dict[str, list[np.ndarray]] = {
        "geometry": [],
        "calibration": [],
        "id_validation": [],
        "id_test": [],
        "proxy_unknown": [],
        "final_unknown": [],
    }
    for class_id in known_classes:
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        required = 6 * samples_per_slice
        if len(indices) < required:
            raise ValueError(f"Class {class_id} has fewer than {required} samples.")
        slices["geometry"].append(indices[:3 * samples_per_slice])
        slices["calibration"].append(indices[3 * samples_per_slice:4 * samples_per_slice])
        slices["id_validation"].append(indices[4 * samples_per_slice:5 * samples_per_slice])
        slices["id_test"].append(indices[5 * samples_per_slice:required])
    for name, classes in (
        ("proxy_unknown", proxy_unknown_classes),
        ("final_unknown", final_unknown_classes),
    ):
        for class_id in classes:
            indices = np.flatnonzero(labels == class_id)
            rng.shuffle(indices)
            if len(indices) < samples_per_slice:
                raise ValueError(
                    f"Class {class_id} has fewer than {samples_per_slice} samples.",
                )
            slices[name].append(indices[:samples_per_slice])
    return {name: np.concatenate(parts) for name, parts in slices.items()}


def _metric_sdf_scores(model, raw_features: np.ndarray) -> np.ndarray:
    features = model._to_feature_space(raw_features)
    return np.column_stack([
        InferenceEngine(model.class_models[class_id], alpha=model.alpha)
        .get_metric_corrected_sdf(features) / model.score_scales[class_id]
        for class_id in model.class_ids
    ])


def _extract_representation(
    images: np.ndarray, representation: str,
) -> np.ndarray:
    if representation == "hog":
        return _extract_hog_features(images)
    if representation == "mobilenetv2":
        return _extract_cnn_features(images)
    if representation == "resnet18_imagenet":
        return _extract_resnet18_features(images)
    if representation == "pooled_rgb_8x8":
        pooled = images.reshape(len(images), 8, 4, 8, 4, 3).mean(axis=(2, 4))
        return pooled.reshape(len(images), -1).astype(np.float64) / 255.0
    raise ValueError(f"Unknown representation: {representation}.")


def _score_features(model, raw_features: np.ndarray, density_scorers, class_scorers):
    raw_scores = model.sdf_scores(raw_features)
    probabilities = model.calibrator.predict_proba(raw_scores)
    transformed = model._to_feature_space(raw_features)
    return probabilities, {
        "minimum_raw_sdf": minimum_sdf_score(raw_scores),
        "minimum_metric_sdf": minimum_sdf_score(
            _metric_sdf_scores(model, raw_features),
        ),
        "sdf_energy": sdf_energy_score(raw_scores),
        "maximum_probability": maximum_probability_score(probabilities),
        **density_scorers.score(transformed),
        **class_scorers.score(transformed),
    }


def _run_episode(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    known_classes: tuple[int, ...],
    proxy_unknown_classes: tuple[int, ...],
    final_unknown_classes: tuple[int, ...],
    seed: int,
    samples_per_slice: int,
    pca_components: int,
    minimum_known_coverage: float,
    representation: str,
    include_score_payload: bool = False,
) -> dict:
    _validate_partition(
        known_classes, proxy_unknown_classes, final_unknown_classes,
    )
    source_slices = _episode_source_slices(
        labels,
        known_classes=known_classes,
        proxy_unknown_classes=proxy_unknown_classes,
        final_unknown_classes=final_unknown_classes,
        samples_per_slice=samples_per_slice,
        seed=seed,
    )
    slice_names = tuple(source_slices)
    source_ids = np.concatenate([source_slices[name] for name in slice_names])
    if len(np.unique(source_ids)) != len(source_ids):
        raise RuntimeError("Episode source slices are not disjoint.")
    all_features = _extract_representation(images[source_ids], representation)
    offsets = np.cumsum([0, *[len(source_slices[name]) for name in slice_names]])
    raw_features = {
        name: all_features[offsets[position]:offsets[position + 1]]
        for position, name in enumerate(slice_names)
    }
    slice_labels = {name: labels[source_slices[name]] for name in slice_names}

    component_count = min(
        pca_components,
        raw_features["geometry"].shape[1],
        len(raw_features["geometry"]) - 1,
    )
    pca = PCA(n_components=component_count, whiten=True, random_state=seed)
    geometry_pca = pca.fit_transform(raw_features["geometry"])
    scaler = StandardScaler().fit(geometry_pca)
    geometry = scaler.transform(geometry_pca)
    model = _build_model(
        "cifar_hog_ood_transfer",
        InputSpec(representation, dim=all_features.shape[1]),
        geometry,
        slice_labels["geometry"],
        known_classes,
    )
    model.pca = pca
    model.scaler = scaler
    model.calibrator = _fit_calibrator(
        model,
        raw_features["calibration"],
        slice_labels["calibration"],
        seed,
    )
    density_scorers = fit_feature_ood_scorers(
        geometry,
        gmm_components=len(known_classes),
        knn_k=min(5, len(geometry)),
        seed=seed,
    )
    class_scorers = fit_class_conditional_ood_scorers(
        geometry, slice_labels["geometry"],
    )

    calibration_probabilities, _ = _score_features(
        model, raw_features["calibration"], density_scorers, class_scorers,
    )
    id_validation_probabilities, id_validation_scores = _score_features(
        model, raw_features["id_validation"], density_scorers, class_scorers,
    )
    proxy_probabilities, proxy_scores = _score_features(
        model, raw_features["proxy_unknown"], density_scorers, class_scorers,
    )
    id_test_probabilities, id_test_scores = _score_features(
        model, raw_features["id_test"], density_scorers, class_scorers,
    )
    final_unknown_probabilities, final_unknown_scores = _score_features(
        model, raw_features["final_unknown"], density_scorers, class_scorers,
    )

    validation = {}
    final_test = {}
    for score_name in id_validation_scores:
        threshold = select_ood_threshold_at_known_coverage(
            id_validation_scores[score_name], minimum_known_coverage,
        )
        validation[score_name] = {
            "detection": ood_detection_metrics(
                id_validation_scores[score_name], proxy_scores[score_name],
            ),
            "operating_point": ood_operating_point(
                id_validation_scores[score_name], proxy_scores[score_name], threshold,
            ),
            "threshold": threshold,
        }
        final_test[score_name] = {
            "detection": ood_detection_metrics(
                id_test_scores[score_name], final_unknown_scores[score_name],
            ),
            "operating_point": ood_operating_point(
                id_test_scores[score_name], final_unknown_scores[score_name], threshold,
            ),
        }

    classes = np.asarray(known_classes)
    conformal_threshold = conformal_probability_threshold(
        slice_labels["calibration"], calibration_probabilities, classes,
    )
    id_sets = conformal_prediction_sets(id_test_probabilities, conformal_threshold)
    final_sets = conformal_prediction_sets(
        final_unknown_probabilities, conformal_threshold,
    )
    id_predictions = classes[id_test_probabilities.argmax(axis=1)]
    result = {
        "protocol": {
            "seed": seed,
            "representation": representation,
            "known_classes": list(known_classes),
            "proxy_unknown_classes": list(proxy_unknown_classes),
            "final_unknown_classes": list(final_unknown_classes),
            "samples_per_slice": samples_per_slice,
            "pca_components": component_count,
            "source_images_disjoint_across_slices": True,
            "transform_fit_classes": list(known_classes),
            "proxy_unknown_used_for_score_selection": True,
            "final_unknown_used_for_score_selection": False,
            "mutation_published": False,
        },
        "representation_diagnostics": compute_representation_diagnostics(
            geometry, slice_labels["geometry"],
        ),
        "closed_set": {
            "id_test_accuracy": float(np.mean(
                id_predictions == slice_labels["id_test"],
            )),
        },
        "conformal_control": {
            "threshold": conformal_threshold,
            **conformal_set_metrics(
                slice_labels["id_test"], id_sets, classes,
            ),
            "final_unknown_empty_set_recall": float(np.mean(
                np.sum(final_sets, axis=1) == 0,
            )),
            "final_unknown_average_set_size": float(np.mean(
                np.sum(final_sets, axis=1),
            )),
        },
        "validation": validation,
        "final_test": final_test,
    }
    if include_score_payload:
        score_names = tuple(id_validation_scores)
        result["score_payload"] = {
            "score_names": list(score_names),
            "id_validation": np.column_stack([
                id_validation_scores[name] for name in score_names
            ]).tolist(),
            "proxy_unknown": np.column_stack([
                proxy_scores[name] for name in score_names
            ]).tolist(),
            "id_test": np.column_stack([
                id_test_scores[name] for name in score_names
            ]).tolist(),
            "final_unknown": np.column_stack([
                final_unknown_scores[name] for name in score_names
            ]).tolist(),
            "id_validation_predicted_classes": classes[
                id_validation_probabilities.argmax(axis=1)
            ].tolist(),
            "proxy_unknown_predicted_classes": classes[
                proxy_probabilities.argmax(axis=1)
            ].tolist(),
            "id_test_predicted_classes": id_predictions.tolist(),
            "final_unknown_predicted_classes": classes[
                final_unknown_probabilities.argmax(axis=1)
            ].tolist(),
            "id_validation_embeddings": model._to_feature_space(
                raw_features["id_validation"],
            ).tolist(),
            "proxy_unknown_embeddings": model._to_feature_space(
                raw_features["proxy_unknown"],
            ).tolist(),
            "id_test_embeddings": model._to_feature_space(
                raw_features["id_test"],
            ).tolist(),
            "final_unknown_embeddings": model._to_feature_space(
                raw_features["final_unknown"],
            ).tolist(),
            "id_validation_representation_embeddings": raw_features[
                "id_validation"
            ].tolist(),
            "proxy_unknown_representation_embeddings": raw_features[
                "proxy_unknown"
            ].tolist(),
            "id_test_representation_embeddings": raw_features["id_test"].tolist(),
            "final_unknown_representation_embeddings": raw_features[
                "final_unknown"
            ].tolist(),
            "id_validation_labels": slice_labels["id_validation"].tolist(),
            "proxy_unknown_labels": slice_labels["proxy_unknown"].tolist(),
            "id_test_labels": slice_labels["id_test"].tolist(),
            "final_unknown_labels": slice_labels["final_unknown"].tolist(),
        }
    return result


def run_real_feature_ood_episode(
    *,
    dataset_path: str,
    known_classes: tuple[int, ...],
    proxy_unknown_classes: tuple[int, ...],
    final_unknown_classes: tuple[int, ...],
    seed: int = 42,
    samples_per_slice: int = 40,
    pca_components: int = 8,
    minimum_known_coverage: float = 0.9,
    representation: str = "hog",
    include_score_payload: bool = False,
) -> dict:
    with np.load(dataset_path) as data:
        return _run_episode(
            data["images"], data["labels"].astype(np.int64),
            known_classes=known_classes,
            proxy_unknown_classes=proxy_unknown_classes,
            final_unknown_classes=final_unknown_classes,
            seed=seed,
            samples_per_slice=samples_per_slice,
            pca_components=pca_components,
            minimum_known_coverage=minimum_known_coverage,
            representation=representation,
            include_score_payload=include_score_payload,
        )


def run_real_feature_ood_transfer(
    *,
    dataset_path: str,
    episodes: list[dict],
    seeds: list[int],
    samples_per_slice: int = 40,
    pca_components: int = 8,
    minimum_known_coverage: float = 0.9,
    minimum_unknown_recall: float = 0.5,
    representation: str = "hog",
    include_score_payload: bool = False,
) -> dict:
    proxy_pool = {
        class_id for episode in episodes
        for class_id in episode["proxy_unknown_classes"]
    }
    final_pool = {
        class_id for episode in episodes
        for class_id in episode["final_unknown_classes"]
    }
    if proxy_pool & final_pool:
        raise ValueError("Proxy and final unknown pools must be globally disjoint.")
    with np.load(dataset_path) as data:
        images = data["images"]
        labels = data["labels"].astype(np.int64)
        runs = [
            _run_episode(
                images,
                labels,
                known_classes=tuple(episode["known_classes"]),
                proxy_unknown_classes=tuple(episode["proxy_unknown_classes"]),
                final_unknown_classes=tuple(episode["final_unknown_classes"]),
                seed=seed,
                samples_per_slice=samples_per_slice,
                pca_components=pca_components,
                minimum_known_coverage=minimum_known_coverage,
                representation=representation,
                include_score_payload=include_score_payload,
            )
            for seed in seeds
            for episode in episodes
        ]

    score_names = tuple(runs[0]["validation"])
    development_summary = {
        score_name: {
            "known_coverage_mean": float(np.mean([
                1.0 - run["validation"][score_name]["operating_point"][
                    "false_positive_rate"
                ]
                for run in runs
            ])),
            "proxy_unknown_recall_mean": float(np.mean([
                run["validation"][score_name]["operating_point"][
                    "true_positive_rate"
                ]
                for run in runs
            ])),
            "auroc_mean": float(np.mean([
                run["validation"][score_name]["detection"]["auroc"]
                for run in runs
            ])),
            "fpr95_mean": float(np.mean([
                run["validation"][score_name]["detection"]["fpr95"]
                for run in runs
            ])),
        }
        for score_name in score_names
    }
    selected_score = max(
        score_names,
        key=lambda name: (
            development_summary[name]["proxy_unknown_recall_mean"],
            development_summary[name]["auroc_mean"],
            -development_summary[name]["fpr95_mean"],
            name,
        ),
    )
    final_summary = {}
    for score_name in score_names:
        coverage = np.asarray([
            1.0 - run["final_test"][score_name]["operating_point"][
                "false_positive_rate"
            ]
            for run in runs
        ])
        recall = np.asarray([
            run["final_test"][score_name]["operating_point"]["true_positive_rate"]
            for run in runs
        ])
        passes = (
            (coverage >= minimum_known_coverage)
            & (recall >= minimum_unknown_recall)
        )
        final_summary[score_name] = {
            "known_coverage_mean": float(np.mean(coverage)),
            "known_coverage_minimum": float(np.min(coverage)),
            "unknown_recall_mean": float(np.mean(recall)),
            "unknown_recall_minimum": float(np.min(recall)),
            "cells_passing_gate": int(np.count_nonzero(passes)),
            "all_cells_pass_gate": bool(np.all(passes)),
        }
    return {
        "protocol": {
            "dataset_path": dataset_path,
            "seeds": list(seeds),
            "episode_count": len(episodes),
            "cell_count": len(runs),
            "proxy_unknown_pool": sorted(proxy_pool),
            "final_unknown_pool": sorted(final_pool),
            "proxy_and_final_pools_disjoint": True,
            "minimum_known_coverage": minimum_known_coverage,
            "minimum_unknown_recall": minimum_unknown_recall,
            "representation": representation,
            "score_family_selected_on_development_only": True,
            "final_unknown_used_for_selection": False,
            "mutation_published": False,
        },
        "selection": {
            "score": selected_score,
            "development": development_summary[selected_score],
        },
        "development_summary": development_summary,
        "final_summary": final_summary,
        "production_gate_passed": final_summary[selected_score][
            "all_cells_pass_gate"
        ],
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-feature leave-class-out OOD transfer study",
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact_path = Path(config.pop("artifact_path"))
    result = run_real_feature_ood_transfer(**config)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "selection": result["selection"],
        "selected_final": result["final_summary"][result["selection"]["score"]],
        "production_gate_passed": result["production_gate_passed"],
    }, indent=2))


if __name__ == "__main__":
    main()