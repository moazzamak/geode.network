from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.common.ood_metrics import (
    ood_detection_metrics,
    ood_operating_point,
    select_ood_threshold_at_known_coverage,
)
from experiments.tier4.eval_real_feature_ood_transfer import (
    run_real_feature_ood_transfer,
)


@dataclass
class SupportCalibrator:
    global_model: object
    class_models: dict[int, object]

    def score(
        self, features: np.ndarray, predicted_classes: np.ndarray,
    ) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        predicted_classes = np.asarray(predicted_classes, dtype=np.int64)
        scores = self.global_model.predict_proba(features)[:, 1]
        for class_id, model in self.class_models.items():
            mask = predicted_classes == class_id
            if np.any(mask):
                scores[mask] = model.predict_proba(features[mask])[:, 1]
        return scores


def _make_logistic(C: float, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
        ),
    )


def fit_support_calibrator(
    id_scores: np.ndarray,
    proxy_scores: np.ndarray,
    id_predicted_classes: np.ndarray,
    proxy_predicted_classes: np.ndarray,
    *,
    model_form: str,
    C: float,
    seed: int,
) -> SupportCalibrator:
    if model_form not in {"global", "class_conditional"}:
        raise ValueError("model_form must be 'global' or 'class_conditional'.")
    if C <= 0.0:
        raise ValueError("C must be positive.")
    id_scores = np.asarray(id_scores, dtype=np.float64)
    proxy_scores = np.asarray(proxy_scores, dtype=np.float64)
    features = np.vstack([id_scores, proxy_scores])
    labels = np.concatenate([
        np.zeros(len(id_scores), dtype=np.int64),
        np.ones(len(proxy_scores), dtype=np.int64),
    ])
    predicted_classes = np.concatenate([
        np.asarray(id_predicted_classes, dtype=np.int64),
        np.asarray(proxy_predicted_classes, dtype=np.int64),
    ])
    if features.ndim != 2 or not len(id_scores) or not len(proxy_scores):
        raise ValueError("ID and proxy score matrices must be non-empty.")
    if len(predicted_classes) != len(features):
        raise ValueError("Predicted classes must align with score rows.")
    global_model = _make_logistic(C, seed).fit(features, labels)
    class_models = {}
    if model_form == "class_conditional":
        for class_id in np.unique(predicted_classes):
            mask = predicted_classes == class_id
            class_labels = labels[mask]
            counts = np.bincount(class_labels, minlength=2)
            if np.all(counts >= 2):
                class_models[int(class_id)] = _make_logistic(C, seed).fit(
                    features[mask], class_labels,
                )
    return SupportCalibrator(global_model, class_models)


def _development_split(length: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if length < 4:
        raise ValueError("Development splits require at least four samples.")
    order = np.random.default_rng(seed).permutation(length)
    midpoint = length // 2
    return order[:midpoint], order[midpoint:]


def _payload_arrays(run: dict) -> dict[str, np.ndarray]:
    payload = run["score_payload"]
    return {
        name: np.asarray(values)
        for name, values in payload.items()
        if name != "score_names"
    }


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    repeats: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("Bootstrap values must be a non-empty vector.")
    if repeats < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap configuration.")
    rng = np.random.default_rng(seed)
    sampled_means = np.mean(
        values[rng.integers(0, len(values), size=(repeats, len(values)))],
        axis=1,
    )
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(sampled_means, tail)),
        "upper": float(np.quantile(sampled_means, 1.0 - tail)),
        "confidence": confidence,
        "resampling_unit": "episode_seed_cell",
    }


def _fit_and_score(
    arrays: dict[str, np.ndarray],
    *,
    model_form: str,
    C: float,
    seed: int,
    fit_id_indices: np.ndarray,
    fit_proxy_indices: np.ndarray,
    evaluation_id: str,
    evaluation_unknown: str,
) -> tuple[np.ndarray, np.ndarray]:
    calibrator = fit_support_calibrator(
        arrays["id_validation"][fit_id_indices],
        arrays["proxy_unknown"][fit_proxy_indices],
        arrays["id_validation_predicted_classes"][fit_id_indices],
        arrays["proxy_unknown_predicted_classes"][fit_proxy_indices],
        model_form=model_form,
        C=C,
        seed=seed,
    )
    return (
        calibrator.score(
            arrays[evaluation_id],
            arrays[f"{evaluation_id}_predicted_classes"],
        ),
        calibrator.score(
            arrays[evaluation_unknown],
            arrays[f"{evaluation_unknown}_predicted_classes"],
        ),
    )


def run_support_objective_study(
    *,
    regularization_grid: list[float],
    model_forms: list[str],
    baseline_score: str = "maximum_probability",
    bootstrap_repeats: int = 2000,
    **config,
) -> dict:
    if not regularization_grid or any(C <= 0.0 for C in regularization_grid):
        raise ValueError("regularization_grid must contain positive values.")
    if not model_forms or not set(model_forms) <= {"global", "class_conditional"}:
        raise ValueError("Unknown or empty model_forms.")
    config = dict(config)
    config["representation"] = "mobilenetv2"
    config["include_score_payload"] = True
    base = run_real_feature_ood_transfer(**config)
    if baseline_score not in base["final_summary"]:
        raise ValueError(f"Unknown frozen baseline score: {baseline_score}.")
    minimum_known_coverage = base["protocol"]["minimum_known_coverage"]
    minimum_unknown_recall = base["protocol"]["minimum_unknown_recall"]

    development = {}
    for model_form in model_forms:
        for C in regularization_grid:
            key = f"{model_form}:C={C:g}"
            cells = []
            for position, run in enumerate(base["runs"]):
                arrays = _payload_arrays(run)
                fit_id, select_id = _development_split(
                    len(arrays["id_validation"]),
                    run["protocol"]["seed"] + 1009 + position,
                )
                fit_proxy, select_proxy = _development_split(
                    len(arrays["proxy_unknown"]),
                    run["protocol"]["seed"] + 2003 + position,
                )
                id_scores, unknown_scores = _fit_and_score(
                    arrays,
                    model_form=model_form,
                    C=C,
                    seed=run["protocol"]["seed"],
                    fit_id_indices=fit_id,
                    fit_proxy_indices=fit_proxy,
                    evaluation_id="id_validation",
                    evaluation_unknown="proxy_unknown",
                )
                id_scores = id_scores[select_id]
                unknown_scores = unknown_scores[select_proxy]
                threshold = select_ood_threshold_at_known_coverage(
                    id_scores, minimum_known_coverage,
                )
                point = ood_operating_point(id_scores, unknown_scores, threshold)
                cells.append({
                    "known_coverage": 1.0 - point["false_positive_rate"],
                    "unknown_recall": point["true_positive_rate"],
                    "detection": ood_detection_metrics(id_scores, unknown_scores),
                })
            passes = np.asarray([
                cell["known_coverage"] >= minimum_known_coverage
                and cell["unknown_recall"] >= minimum_unknown_recall
                for cell in cells
            ])
            development[key] = {
                "model_form": model_form,
                "C": C,
                "known_coverage_mean": float(np.mean([
                    cell["known_coverage"] for cell in cells
                ])),
                "proxy_unknown_recall_mean": float(np.mean([
                    cell["unknown_recall"] for cell in cells
                ])),
                "proxy_unknown_recall_minimum": float(np.min([
                    cell["unknown_recall"] for cell in cells
                ])),
                "auroc_mean": float(np.mean([
                    cell["detection"]["auroc"] for cell in cells
                ])),
                "cells_passing_gate": int(np.count_nonzero(passes)),
            }
    selected_key = max(
        development,
        key=lambda key: (
            development[key]["cells_passing_gate"],
            development[key]["proxy_unknown_recall_minimum"],
            development[key]["proxy_unknown_recall_mean"],
            development[key]["auroc_mean"],
            development[key]["model_form"] == "global",
            -development[key]["C"],
        ),
    )
    selected = development[selected_key]

    final_cells = []
    for run in base["runs"]:
        arrays = _payload_arrays(run)
        calibrator = fit_support_calibrator(
            arrays["id_validation"],
            arrays["proxy_unknown"],
            arrays["id_validation_predicted_classes"],
            arrays["proxy_unknown_predicted_classes"],
            model_form=selected["model_form"],
            C=selected["C"],
            seed=run["protocol"]["seed"],
        )
        validation_scores = calibrator.score(
            arrays["id_validation"], arrays["id_validation_predicted_classes"],
        )
        threshold = select_ood_threshold_at_known_coverage(
            validation_scores, minimum_known_coverage,
        )
        id_test_scores = calibrator.score(
            arrays["id_test"], arrays["id_test_predicted_classes"],
        )
        final_unknown_scores = calibrator.score(
            arrays["final_unknown"], arrays["final_unknown_predicted_classes"],
        )
        point = ood_operating_point(
            id_test_scores, final_unknown_scores, threshold,
        )
        known_coverage = 1.0 - point["false_positive_rate"]
        unknown_recall = point["true_positive_rate"]
        final_cells.append({
            "seed": run["protocol"]["seed"],
            "known_classes": run["protocol"]["known_classes"],
            "known_coverage": known_coverage,
            "unknown_recall": unknown_recall,
            "threshold": threshold,
            "id_score_mean": float(np.mean(id_test_scores)),
            "unknown_score_mean": float(np.mean(final_unknown_scores)),
            "passed_gate": (
                known_coverage >= minimum_known_coverage
                and unknown_recall >= minimum_unknown_recall
            ),
            "detection": ood_detection_metrics(
                id_test_scores, final_unknown_scores,
            ),
        })
    final_summary = {
        "known_coverage_mean": float(np.mean([
            cell["known_coverage"] for cell in final_cells
        ])),
        "known_coverage_minimum": float(np.min([
            cell["known_coverage"] for cell in final_cells
        ])),
        "unknown_recall_mean": float(np.mean([
            cell["unknown_recall"] for cell in final_cells
        ])),
        "unknown_recall_minimum": float(np.min([
            cell["unknown_recall"] for cell in final_cells
        ])),
        "cells_passing_gate": int(sum(
            cell["passed_gate"] for cell in final_cells
        )),
        "all_cells_pass_gate": all(
            cell["passed_gate"] for cell in final_cells
        ),
        "known_coverage_interval": _bootstrap_mean_interval(
            np.asarray([cell["known_coverage"] for cell in final_cells]),
            seed=3101,
            repeats=bootstrap_repeats,
        ),
        "unknown_recall_interval": _bootstrap_mean_interval(
            np.asarray([cell["unknown_recall"] for cell in final_cells]),
            seed=3102,
            repeats=bootstrap_repeats,
        ),
    }
    baseline = dict(base["final_summary"][baseline_score])
    baseline_coverage = np.asarray([
        1.0 - run["final_test"][baseline_score]["operating_point"][
            "false_positive_rate"
        ]
        for run in base["runs"]
    ])
    baseline_recall = np.asarray([
        run["final_test"][baseline_score]["operating_point"]["true_positive_rate"]
        for run in base["runs"]
    ])
    baseline["known_coverage_interval"] = _bootstrap_mean_interval(
        baseline_coverage, seed=3201, repeats=bootstrap_repeats,
    )
    baseline["unknown_recall_interval"] = _bootstrap_mean_interval(
        baseline_recall, seed=3202, repeats=bootstrap_repeats,
    )
    for run in base["runs"]:
        run.pop("score_payload", None)
    return {
        "protocol": {
            **base["protocol"],
            "representation": "mobilenetv2",
            "development_fit_selection_disjoint": True,
            "model_form_selected_on_development_only": True,
            "final_unknown_used_for_selection": False,
            "mutation_published": False,
        },
        "baseline": {
            "score": baseline_score,
            **baseline,
        },
        "development_candidates": development,
        "selection": {
            "key": selected_key,
            **selected,
        },
        "final_summary": final_summary,
        "production_gate_passed": final_summary["all_cells_pass_gate"],
        "final_cells": final_cells,
        "base_runs": base["runs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Development-only real-feature support objective study",
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact_path = Path(config.pop("artifact_path"))
    result = run_support_objective_study(**config)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "baseline": result["baseline"],
        "selection": result["selection"],
        "final_summary": result["final_summary"],
        "production_gate_passed": result["production_gate_passed"],
    }, indent=2))


if __name__ == "__main__":
    main()