import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

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
from experiments.common.result_records import classification_result_record
from experiments.common.score_readouts import fit_score_readout
from experiments.tier4.eval_complex_classification import (
    _apply_transform,
    _build_transform,
    compute_raw_scores,
    compute_score_scales,
    fit_class_models,
    load_cifar_npz,
    stratified_geometry_carve_calibration_split,
)
from experiments.tier4.eval_probabilistic_field import _probabilistic_scores
from src.gpu_engine import GPUInferenceEngine
from src.probabilistic_engine import ProbabilisticInferenceEngine


def _fit_global_covariance_temperature(
    models: dict,
    class_ids: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    use_gpu: bool,
) -> tuple[float, dict]:
    if use_gpu:
        engine = GPUInferenceEngine(
            [models[int(class_id)] for class_id in class_ids], alpha=1.0,
        )
    else:
        engine = ProbabilisticInferenceEngine(models)
    class_lookup = {int(class_id): column for column, class_id in enumerate(class_ids)}
    target_columns = np.asarray([class_lookup[int(label)] for label in labels])

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        scores = engine.class_nlls(
            points, covariance_temperature=temperature,
        )
        logits = -scores
        selected = logits[np.arange(len(labels)), target_columns]
        return float(np.mean(logsumexp(logits, axis=1) - selected))

    baseline_nll = objective(0.0)
    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    temperature = float(np.exp(result.x))
    return temperature, {
        "converged": bool(result.success),
        "function_evaluations": int(result.nfev),
        "baseline_calibration_nll": baseline_nll,
        "fitted_calibration_nll": float(result.fun),
        "log_temperature": float(result.x),
    }


def _fit_per_class_covariance_temperature(
    models: dict,
    class_ids: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    use_gpu: bool,
    initial_temperature: float,
) -> tuple[np.ndarray, dict]:
    if use_gpu:
        engine = GPUInferenceEngine(
            [models[int(class_id)] for class_id in class_ids], alpha=1.0,
        )
    else:
        engine = ProbabilisticInferenceEngine(models)
    class_lookup = {int(class_id): column for column, class_id in enumerate(class_ids)}
    target_columns = np.asarray([class_lookup[int(label)] for label in labels])

    def objective(log_temperatures: np.ndarray) -> float:
        scores = engine.class_nlls(
            points, covariance_temperature=np.exp(log_temperatures),
        )
        logits = -scores
        selected = logits[np.arange(len(labels)), target_columns]
        return float(np.mean(logsumexp(logits, axis=1) - selected))

    initial = np.full(len(class_ids), np.log(initial_temperature))
    baseline_nll = objective(initial)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(-4.0, 4.0)] * len(class_ids),
    )
    temperatures = np.exp(result.x)
    return temperatures, {
        "converged": bool(result.success),
        "function_evaluations": int(result.nfev),
        "iterations": int(result.nit),
        "baseline_calibration_nll": baseline_nll,
        "fitted_calibration_nll": float(result.fun),
        "log_temperatures": [float(value) for value in result.x],
        "covariance_temperatures": [float(value) for value in temperatures],
    }


def _evaluate_input(
    *,
    name: str,
    evaluation_input: np.ndarray,
    evaluation_labels: np.ndarray,
    evaluation_features: np.ndarray,
    readout,
    readout_fit_seconds: float,
    readout_input_width: int,
    calibration_sample_count: int,
    minimum_calibration_class_count: int,
    class_ids: np.ndarray,
    dataset: str,
    split: str,
    fitter_name: str,
    model_stats: dict,
    geometry_sample_count: int,
    geometry_fit_seconds: float,
    seed: int,
    evaluation_indices: np.ndarray,
    bootstrap_resamples: int,
) -> dict:
    mode = "feature_logistic" if name == "feature_control" else "multinomial"
    inference_started = time.perf_counter()
    probabilities = readout.predict_proba(
        evaluation_input,
        evaluation_features if mode == "feature_logistic" else None,
    )
    inference_seconds = time.perf_counter() - inference_started
    record = classification_result_record(
        dataset=dataset,
        split=split,
        seed=seed,
        method="logistic_regression" if name == "feature_control" else "geode",
        representation="mobilenetv2",
        geometry_variant=f"fitter_{fitter_name}_hybrid_ablation",
        readout=name,
        y_true=evaluation_labels,
        probabilities=probabilities,
        classes=class_ids,
        model_stats=model_stats,
        performance={
            "geometry_fit_seconds": float(geometry_fit_seconds),
            "readout_fit_seconds": float(readout_fit_seconds),
            "readout_fit_iterations": int(readout.fit_iterations),
            "readout_iteration_limit": readout.iteration_limit,
            "readout_input_standardized": bool(readout.classifier_mean is not None),
            "readout_input_width": int(readout_input_width),
            "inference_seconds": float(inference_seconds),
        },
        adequacy={
            "geometry_samples": int(geometry_sample_count),
            "calibration_samples": int(calibration_sample_count),
            "minimum_calibration_class_count": int(minimum_calibration_class_count),
        },
        warnings=list(readout.fit_warnings),
        converged=readout.converged,
        bootstrap_resamples=bootstrap_resamples,
        split_hash=array_fingerprint(np.asarray(evaluation_indices, dtype=np.int64)),
        feature_hash=array_fingerprint(evaluation_features),
    )
    record["score_semantics"] = name
    return record


def run_hybrid_field_ablation(
    X: np.ndarray,
    y: np.ndarray,
    *,
    fitter: str = "spherical_covariance",
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
    optimize_global_temperature: bool = False,
    optimize_per_class_temperature: bool = False,
) -> dict:
    if fitter not in FITTER_PRIMITIVE_FAMILIES:
        raise ValueError(f"hybrid field requires a covariance fitter, got {fitter}")
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
    primitive_family = FITTER_PRIMITIVE_FAMILIES[fitter]
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
        use_gpu=use_gpu,
        seed=seed,
        candidate_fitter=ELLIPSOID_FITTERS[fitter],
        primitive_family=primitive_family,
        gpu_candidate_fitting=use_gpu and fitter in GPU_CANDIDATE_FITTERS,
    )
    fit_seconds = time.perf_counter() - started
    structure = model_structure_stats(models)
    scales = compute_score_scales(
        models, transformed["geometry"], alpha=alpha, use_gpu=use_gpu,
    )
    geometric = {
        split: compute_raw_scores(
            models, values, alpha=alpha, score_scales=scales, use_gpu=use_gpu,
        )
        for split, values in transformed.items()
    }
    probabilistic = {
        split: _probabilistic_scores(models, class_ids, values, use_gpu)
        for split, values in transformed.items()
    }
    score_inputs = {
        "geometric": geometric,
        "probabilistic": probabilistic,
        "hybrid": {
            split: np.column_stack([geometric[split], probabilistic[split]])
            for split in transformed
        },
        "feature_control": transformed,
    }
    likelihood_optimization = None
    per_class_likelihood_optimization = None
    global_temperature = 1.0
    if optimize_global_temperature:
        global_temperature, likelihood_optimization = _fit_global_covariance_temperature(
            models,
            class_ids,
            transformed["calibration"],
            y[calibration_idx],
            use_gpu,
        )
        likelihood_optimization["covariance_temperature"] = global_temperature
        tuned_probabilistic = {
            split: _probabilistic_scores(
                models,
                class_ids,
                values,
                use_gpu,
                covariance_temperature=global_temperature,
            )
            for split, values in transformed.items()
        }
        score_inputs["probabilistic_global_temperature"] = tuned_probabilistic
        score_inputs["hybrid_global_temperature"] = {
            split: np.column_stack([geometric[split], tuned_probabilistic[split]])
            for split in transformed
        }
    if optimize_per_class_temperature:
        temperatures, per_class_likelihood_optimization = (
            _fit_per_class_covariance_temperature(
                models,
                class_ids,
                transformed["calibration"],
                y[calibration_idx],
                use_gpu,
                initial_temperature=global_temperature,
            )
        )
        tuned_probabilistic = {
            split: _probabilistic_scores(
                models,
                class_ids,
                values,
                use_gpu,
                covariance_temperature=temperatures,
            )
            for split, values in transformed.items()
        }
        score_inputs["probabilistic_per_class_temperature"] = tuned_probabilistic
        score_inputs["hybrid_per_class_temperature"] = {
            split: np.column_stack([geometric[split], tuned_probabilistic[split]])
            for split in transformed
        }
    records = []
    minimum_calibration_class_count = int(min(
        np.sum(y[calibration_idx] == class_id) for class_id in class_ids
    ))
    for name, inputs in score_inputs.items():
        mode = "feature_logistic" if name == "feature_control" else "multinomial"
        readout_started = time.perf_counter()
        readout = fit_score_readout(
            mode,
            inputs["calibration"],
            y[calibration_idx],
            class_ids,
            calibration_features=transformed["calibration"],
            seed=seed,
        )
        readout_fit_seconds = time.perf_counter() - readout_started
        for split, split_idx in (("selection", selection_idx), ("test", test_idx)):
            records.append(_evaluate_input(
                name=name,
                evaluation_input=inputs[split],
                evaluation_labels=y[split_idx],
                evaluation_features=transformed[split],
                readout=readout,
                readout_fit_seconds=readout_fit_seconds,
                readout_input_width=inputs["calibration"].shape[1],
                calibration_sample_count=len(calibration_idx),
                minimum_calibration_class_count=minimum_calibration_class_count,
                class_ids=class_ids,
                dataset=dataset,
                split=split,
                fitter_name=fitter,
                model_stats=structure,
                geometry_sample_count=len(geometry_idx),
                geometry_fit_seconds=fit_seconds,
                seed=seed,
                evaluation_indices=split_idx,
                bootstrap_resamples=bootstrap_resamples,
            ))
    selection_records = {
        record["readout"]: record for record in records if record["split"] == "selection"
    }
    selection_candidates = tuple(
        name for name in score_inputs if name != "feature_control"
    )
    selected = min(
        selection_candidates,
        key=lambda name: selection_records[name]["metrics"]["negative_log_likelihood"],
    )
    return {
        "records": records,
        "selected_score_input": selected,
        "selection_metric": "negative_log_likelihood",
        "selection_used_for_model_choice": True,
        "test_used_for_selection": False,
        "model_fit_count": 1,
        "readout_fit_count": len(score_inputs),
        "likelihood_optimization": likelihood_optimization,
        "per_class_likelihood_optimization": per_class_likelihood_optimization,
        "probabilistic_fitting_used": False,
        "subtractive_probability_supported": False,
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
    parser = argparse.ArgumentParser(description="Compare matched hybrid field readouts.")
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
    excluded = {"artifact_path", "dataset_path", "max_samples", "feature_extractor"}
    result = run_hybrid_field_ablation(
        X, y, **{key: value for key, value in defaults.items() if key not in excluded}
    )
    manifest = build_manifest(
        config=defaults,
        seed=defaults["seed"],
        repo_root=Path(__file__).resolve().parents[2],
        dataset_fingerprint=array_fingerprint(
            np.frombuffer(Path(defaults["dataset_path"]).read_bytes(), dtype=np.uint8),
        ),
        split_indices=np.array(list(result["split_hashes"].values()), dtype=str),
        features=X,
        device="OpenCL" if defaults["use_gpu"] else "CPU",
    )
    manifest["metrics"] = result
    append_manifest(defaults["artifact_path"], manifest)
    print(f"Artifact: {defaults['artifact_path']}  id={manifest['experiment_id']}")


if __name__ == "__main__":
    main()