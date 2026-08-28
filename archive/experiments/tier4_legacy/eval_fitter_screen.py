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
    compute_raw_scores,
    compute_score_scales,
    evaluate_score_readouts,
    fit_class_models,
    load_cifar_npz,
    stratified_geometry_carve_calibration_split,
)


def run_fitter_screen(
    X: np.ndarray,
    y: np.ndarray,
    *,
    fitters: list[str],
    seed: int = 42,
    pca_components: int = 128,
    alpha: float = 2.0,
    consensus_threshold: float = 0.12,
    capture_threshold: float = 0.08,
    max_iterations: int | None = 10,
    nudge_iterations: int = 0,
    nudge_learning_rate: float = 0.02,
    selection_fraction: float = 0.15,
    calibration_fraction: float = 0.2,
    bootstrap_resamples: int = 100,
    use_gpu: bool = True,
    dataset: str = "cifar10",
) -> dict:
    train_idx, test_idx = split_train_test_indices(
        len(X), test_fraction=0.2, seed=seed,
    )
    geometry_idx, selection_idx, calibration_idx = (
        stratified_geometry_carve_calibration_split(
            train_idx,
            y[train_idx],
            carve_fraction=selection_fraction,
            calibration_fraction=calibration_fraction,
            seed=seed,
        )
    )
    class_ids = np.unique(y[geometry_idx])
    pca, lda, scaler = _build_transform(
        X[geometry_idx], y[geometry_idx], pca_components, seed,
    )
    transformed = {
        "geometry": _apply_transform(X[geometry_idx], pca, lda, scaler),
        "selection": _apply_transform(X[selection_idx], pca, lda, scaler),
        "calibration": _apply_transform(X[calibration_idx], pca, lda, scaler),
        "test": _apply_transform(X[test_idx], pca, lda, scaler),
    }
    indices = {"selection": selection_idx, "test": test_idx}
    labels = {"selection": y[selection_idx], "test": y[test_idx]}
    records = []
    selection_scores = {}
    for fitter_name in fitters:
        candidate_fitter = (
            None if fitter_name == "current" else ELLIPSOID_FITTERS[fitter_name]
        )
        primitive_family = FITTER_PRIMITIVE_FAMILIES.get(fitter_name, "ellipsoid")
        gpu_candidate_fitting = use_gpu and fitter_name in GPU_CANDIDATE_FITTERS
        fitter_uses_gpu = use_gpu
        backend = "OpenCL" if fitter_uses_gpu else "CPU"
        print(f"Primitive family: {primitive_family.replace('_', ' ')} ({backend})")
        started = time.perf_counter()
        models = fit_class_models(
            transformed["geometry"],
            y[geometry_idx],
            class_ids,
            consensus_threshold=consensus_threshold,
            capture_threshold=capture_threshold,
            alpha=alpha,
            max_iterations=max_iterations,
            nudge_iterations=nudge_iterations,
            nudge_learning_rate=nudge_learning_rate,
            use_gpu=fitter_uses_gpu,
            seed=seed,
            candidate_fitter=candidate_fitter,
            primitive_family=primitive_family,
            gpu_candidate_fitting=gpu_candidate_fitting,
        )
        fit_seconds = time.perf_counter() - started
        structure = model_structure_stats(models)
        scales = compute_score_scales(
            models,
            transformed["geometry"],
            alpha=alpha,
            use_gpu=fitter_uses_gpu,
        )
        calibration_scores = compute_raw_scores(
            models,
            transformed["calibration"],
            alpha=alpha,
            score_scales=scales,
            use_gpu=fitter_uses_gpu,
        )
        for split in ("selection", "test"):
            evaluation_scores = compute_raw_scores(
                models,
                transformed[split],
                alpha=alpha,
                score_scales=scales,
                use_gpu=fitter_uses_gpu,
            )
            split_records = evaluate_score_readouts(
                calibration_scores=calibration_scores,
                calibration_labels=y[calibration_idx],
                calibration_features=transformed["calibration"],
                evaluation_scores=evaluation_scores,
                evaluation_labels=labels[split],
                evaluation_features=transformed[split],
                class_ids=class_ids,
                dataset=dataset,
                split=split,
                representation="mobilenetv2",
                geometry_variant=f"fitter_{fitter_name}",
                model_stats=structure,
                geometry_sample_count=len(geometry_idx),
                geometry_fit_seconds=fit_seconds,
                seed=seed,
                evaluation_indices=indices[split],
                bootstrap_resamples=bootstrap_resamples,
                include_predictions=True,
            )
            records.extend(split_records.values())
            if split == "selection":
                selection_scores[fitter_name] = split_records["multinomial"][
                    "metrics"
                ]["accuracy"]
    selected_fitter = max(selection_scores, key=selection_scores.get)
    return {
        "records": records,
        "selection_metric": "multinomial_accuracy",
        "selection_scores": selection_scores,
        "selected_fitter": selected_fitter,
        "test_used_for_selection": False,
        "split_counts": {
            "geometry": len(geometry_idx),
            "selection": len(selection_idx),
            "calibration": len(calibration_idx),
            "test": len(test_idx),
        },
        "split_hashes": {
            "geometry": array_fingerprint(geometry_idx),
            "selection": array_fingerprint(selection_idx),
            "calibration": array_fingerprint(calibration_idx),
            "test": array_fingerprint(test_idx),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen candidate fitters on Tier 4.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    defaults = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.seed is not None:
        defaults["seed"] = args.seed
    X, y = load_cifar_npz(
        defaults["dataset_path"],
        defaults["max_samples"],
        pca_components=defaults["pca_components"],
        seed=defaults["seed"],
        feature_extractor=defaults["feature_extractor"],
    )
    experiment_config = {
        key: value for key, value in defaults.items()
        if key not in {"artifact_path", "dataset_path", "max_samples", "feature_extractor"}
    }
    result = run_fitter_screen(X, y, **experiment_config)
    manifest = build_manifest(
        config=defaults,
        seed=defaults["seed"],
        repo_root=Path(__file__).resolve().parents[2],
        dataset_fingerprint=array_fingerprint(
            np.frombuffer(Path(defaults["dataset_path"]).read_bytes(), dtype=np.uint8),
        ),
        split_indices=np.array(list(result["split_hashes"].values()), dtype=str),
        features=X,
        device="mixed OpenCL/CPU" if defaults["use_gpu"] else "CPU",
    )
    manifest["metrics"] = result
    append_manifest(defaults["artifact_path"], manifest)
    print(f"Artifact: {defaults['artifact_path']}  id={manifest['experiment_id']}")


if __name__ == "__main__":
    main()