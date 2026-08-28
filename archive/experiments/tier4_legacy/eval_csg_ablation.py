import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.common.ellipsoid_fitters import (
    ELLIPSOID_FITTERS,
    FITTER_PRIMITIVE_FAMILIES,
    GPU_CANDIDATE_FITTERS,
)
from experiments.common.experiment_manifest import (
    append_manifest,
    array_fingerprint,
    build_manifest,
)
from experiments.common.model_stats import model_structure_stats
from experiments.common.moe_eval import split_train_test_indices
from experiments.tier4.eval_complex_classification import (
    _apply_transform,
    _build_transform,
    build_csg_variants,
    compute_raw_scores,
    compute_score_scales,
    evaluate_classical_baselines,
    evaluate_score_readouts,
    fit_class_models,
    load_cifar_npz,
    stratified_geometry_carve_calibration_split,
)


def run_csg_ablation_experiment(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    pca_components: int = 128,
    alpha: float = 2.0,
    consensus_threshold: float = 0.12,
    capture_threshold: float = 0.08,
    max_iterations: int | None = 10,
    nudge_iterations: int = 20,
    nudge_learning_rate: float = 0.02,
    carve_fraction: float = 0.15,
    calibration_fraction: float = 0.2,
    mdl_penalty_weight: float = 0.01,
    min_penalized_gain: float = 0.0,
    bootstrap_resamples: int = 500,
    baseline_rbf_sample_limit: int = 10_000,
    use_gpu: bool = False,
    fitter: str = "current",
) -> dict:
    if fitter != "current" and fitter not in ELLIPSOID_FITTERS:
        raise ValueError(f"unknown primitive fitter: {fitter}")
    candidate_fitter = None if fitter == "current" else ELLIPSOID_FITTERS[fitter]
    primitive_family = FITTER_PRIMITIVE_FAMILIES.get(fitter, "sphere")
    gpu_candidate_fitting = use_gpu and fitter in GPU_CANDIDATE_FITTERS
    print(
        f"Primitive family: {primitive_family.replace('_', ' ')} "
        f"({'OpenCL' if use_gpu else 'CPU'})"
    )
    train_idx, test_idx = split_train_test_indices(
        len(X), test_fraction=0.2, seed=seed,
    )
    geometry_idx, carve_idx, calibration_idx = (
        stratified_geometry_carve_calibration_split(
            train_idx,
            y[train_idx],
            carve_fraction=carve_fraction,
            calibration_fraction=calibration_fraction,
            seed=seed,
        )
    )
    class_ids = np.unique(y[geometry_idx])
    pca, lda, scaler = _build_transform(
        X[geometry_idx], y[geometry_idx], pca_components, seed,
    )
    X_geometry = _apply_transform(X[geometry_idx], pca, lda, scaler)
    X_carve = _apply_transform(X[carve_idx], pca, lda, scaler)
    X_calibration = _apply_transform(X[calibration_idx], pca, lda, scaler)
    X_test = _apply_transform(X[test_idx], pca, lda, scaler)

    fit_started = time.perf_counter()
    additive_models = fit_class_models(
        X_geometry,
        y[geometry_idx],
        class_ids,
        consensus_threshold=consensus_threshold,
        capture_threshold=capture_threshold,
        alpha=alpha,
        max_iterations=max_iterations,
        nudge_iterations=nudge_iterations,
        nudge_learning_rate=nudge_learning_rate,
        use_gpu=use_gpu,
        seed=seed,
        candidate_fitter=candidate_fitter,
        primitive_family=primitive_family,
        gpu_candidate_fitting=gpu_candidate_fitting,
    )
    variants, audits = build_csg_variants(
        additive_models,
        X_geometry,
        y[geometry_idx],
        X_carve,
        y[carve_idx],
        class_ids,
        capture_threshold=capture_threshold,
        alpha=alpha,
        max_iterations=max_iterations,
        seed=seed,
        use_gpu=use_gpu,
        mdl_penalty_weight=mdl_penalty_weight,
        min_penalized_gain=min_penalized_gain,
        candidate_fitter=candidate_fitter,
        primitive_family=primitive_family,
        gpu_candidate_fitting=gpu_candidate_fitting,
    )
    geometry_fit_seconds = time.perf_counter() - fit_started

    records = []
    for variant_name, models in variants.items():
        scales = compute_score_scales(
            models, X_geometry, alpha=alpha, use_gpu=use_gpu,
        )
        calibration_scores = compute_raw_scores(
            models,
            X_calibration,
            alpha=alpha,
            score_scales=scales,
            use_gpu=use_gpu,
        )
        test_scores = compute_raw_scores(
            models,
            X_test,
            alpha=alpha,
            score_scales=scales,
            use_gpu=use_gpu,
        )
        variant_records = evaluate_score_readouts(
            calibration_scores=calibration_scores,
            calibration_labels=y[calibration_idx],
            calibration_features=X_calibration,
            evaluation_scores=test_scores,
            evaluation_labels=y[test_idx],
            evaluation_features=X_test,
            class_ids=class_ids,
            dataset="cifar10",
            split="test",
            representation="mobilenetv2",
            geometry_variant=(
                variant_name if fitter == "current"
                else f"{fitter}_{variant_name}"
            ),
            model_stats=model_structure_stats(models),
            geometry_sample_count=len(geometry_idx),
            geometry_fit_seconds=geometry_fit_seconds,
            seed=seed,
            evaluation_indices=test_idx,
            bootstrap_resamples=bootstrap_resamples,
            include_predictions=True,
        )
        for mode, record in variant_records.items():
            if mode != "feature_logistic" or variant_name == "A0":
                records.append(record)

    baseline_records = evaluate_classical_baselines(
        geometry_features=X_geometry,
        geometry_labels=y[geometry_idx],
        evaluation_features=X_test,
        evaluation_labels=y[test_idx],
        evaluation_indices=test_idx,
        class_ids=class_ids,
        geode_models=variants["A0"],
        dataset="cifar10",
        split="test",
        representation="mobilenetv2",
        seed=seed,
        rbf_sample_limit=baseline_rbf_sample_limit,
        bootstrap_resamples=bootstrap_resamples,
    )
    records.extend(baseline_records.values())

    return {
        "seed": int(seed),
        "fitter": fitter,
        "primitive_family": primitive_family,
        "gpu_candidate_fitting": gpu_candidate_fitting,
        "records": records,
        "carve_audits": audits,
        "split_counts": {
            "geometry": int(len(geometry_idx)),
            "carve_acceptance": int(len(carve_idx)),
            "score_calibration": int(len(calibration_idx)),
            "test": int(len(test_idx)),
        },
        "split_hashes": {
            "geometry": array_fingerprint(geometry_idx),
            "carve_acceptance": array_fingerprint(carve_idx),
            "score_calibration": array_fingerprint(calibration_idx),
            "test": array_fingerprint(test_idx),
        },
    }


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", required=True)
    config_args, _ = config_parser.parse_known_args()
    with Path(config_args.config).open("r", encoding="utf-8") as stream:
        defaults = json.load(stream)

    parser = argparse.ArgumentParser(description="Tier 4 A0/A1/A2 CSG ablation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-path")
    parser.add_argument("--artifact-path")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--pca-components", type=int)
    parser.add_argument("--feature-extractor")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--consensus-threshold", type=float)
    parser.add_argument("--capture-threshold", type=float)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--nudge-iterations", type=int)
    parser.add_argument("--nudge-learning-rate", type=float)
    parser.add_argument("--carve-fraction", type=float)
    parser.add_argument("--calibration-fraction", type=float)
    parser.add_argument("--mdl-penalty-weight", type=float)
    parser.add_argument("--min-penalized-gain", type=float)
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--baseline-rbf-sample-limit", type=int)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction)
    parser.add_argument("--fitter")
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    X, y = load_cifar_npz(
        args.dataset_path,
        args.max_samples,
        pca_components=args.pca_components,
        seed=args.seed,
        feature_extractor=args.feature_extractor,
    )
    config = {
        key: value for key, value in vars(args).items()
        if key not in {"config", "artifact_path"}
    }
    result = run_csg_ablation_experiment(
        X,
        y,
        **{
            key: value for key, value in config.items()
            if key not in {"dataset_path", "max_samples", "feature_extractor"}
        },
    )
    manifest = build_manifest(
        config=config,
        seed=args.seed,
        repo_root=Path(__file__).resolve().parents[2],
        dataset_fingerprint=array_fingerprint(
            np.frombuffer(Path(args.dataset_path).read_bytes(), dtype=np.uint8),
        ),
        split_indices=np.array(list(result["split_hashes"].values()), dtype=str),
        features=X,
        device="OpenCL" if args.use_gpu else "CPU",
    )
    manifest["metrics"] = result
    append_manifest(args.artifact_path, manifest)
    print(f"Artifact: {args.artifact_path}  id={manifest['experiment_id']}")


if __name__ == "__main__":
    main()