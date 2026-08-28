from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from experiments.common.v12_metric_fields import (
    MetricFieldState,
    ProjectedMetricFieldState,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from experiments.tier4.eval_v9_m51_surface_diagnostics import _partition_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "configs"
    / "v12"
    / "m75_inspectability.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "logs" / "results" / "v12" / "m75_inspectability"
)


def _resolve(path: str) -> Path:
    resolved = (REPO_ROOT / Path(path)).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError("M75 paths must remain inside the repository")
    return resolved


def _verify(specification: dict[str, str]) -> Path:
    path = _resolve(specification["path"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"M75 immutable artifact hash mismatch: {path}")
    return path


def _state_from_dict(payload: dict[str, Any]) -> ProjectedMetricFieldState:
    fields = payload["fields"]
    return ProjectedMetricFieldState(
        projection_mean=np.asarray(payload["projection_mean"], dtype=np.float64),
        projection=np.asarray(payload["projection"], dtype=np.float64),
        fields=MetricFieldState(
            classes=np.asarray(fields["classes"], dtype=np.int64),
            centers=np.asarray(fields["centers"], dtype=np.float64),
            bases=np.asarray(fields["bases"], dtype=np.float64),
            tangent_scales=np.asarray(
                fields["tangent_scales"], dtype=np.float64
            ),
            residual_scales=np.asarray(
                fields["residual_scales"], dtype=np.float64
            ),
        ),
    )


def _field_explanations(
    state: ProjectedMetricFieldState,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    projected = state.transform(features)
    scores = state.fields.scores(projected)
    predicted_indices = np.argmin(scores, axis=1)
    predictions = state.fields.classes[predicted_indices]
    tangent, residual = state.fields.score_terms(projected)
    rows = np.arange(len(projected))
    own_tangent = tangent[rows, predicted_indices]
    own_residual = residual[rows, predicted_indices]
    contributions = np.column_stack([own_tangent, own_residual])
    return projected, predictions, predicted_indices, contributions


def _exact_decomposition(
    state: ProjectedMetricFieldState,
    projected: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, float | bool]:
    tangent, residual = state.fields.score_terms(projected)
    reconstructed = np.sum(tangent, axis=2) + residual
    squared_scores = state.fields.scores(projected) ** 2
    absolute = np.abs(reconstructed - squared_scores)
    return {
        "protocol": "exact_score_decomposition_completeness_local_accuracy",
        "mean_absolute_residual": float(np.mean(absolute)),
        "maximum_absolute_residual": float(np.max(absolute)),
        "tolerance": tolerance,
        "passed": bool(np.max(absolute) <= tolerance),
    }


def _deletion_comprehensiveness(
    state: ProjectedMetricFieldState,
    projected: np.ndarray,
    predictions: np.ndarray,
    predicted_indices: np.ndarray,
    *,
    k_values: list[int],
    random_repeats: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 75_000)
    baseline_scores = state.fields.scores(projected)
    rows = np.arange(len(projected))
    baseline_own = baseline_scores[rows, predicted_indices]
    result = {}
    for k in k_values:
        reductions = {"top": [], "random": [], "bottom": []}
        flips = {"top": [], "random": [], "bottom": []}
        for row, predicted_index in enumerate(predicted_indices):
            field = state.fields
            delta = projected[row] - field.centers[predicted_index]
            coordinates = delta @ field.bases[predicted_index]
            contributions = (
                coordinates**2
                / field.tangent_scales[predicted_index] ** 2
            )
            count = min(k, field.rank)
            order = np.argsort(contributions)
            selections = {
                "top": [order[-count:]],
                "bottom": [order[:count]],
                "random": [
                    rng.choice(field.rank, size=count, replace=False)
                    for _ in range(random_repeats)
                ],
            }
            for name, choices in selections.items():
                choice_reductions = []
                choice_flips = []
                for selected in choices:
                    ablated = projected[row].copy()
                    ablated -= (
                        field.bases[predicted_index][:, selected]
                        @ coordinates[selected]
                    )
                    scores = field.scores(ablated[None, :])[0]
                    choice_reductions.append(
                        baseline_own[row] - scores[predicted_index]
                    )
                    choice_flips.append(
                        field.classes[int(np.argmin(scores))]
                        != predictions[row]
                    )
                reductions[name].append(float(np.mean(choice_reductions)))
                flips[name].append(float(np.mean(choice_flips)))
        result[str(k)] = {
            name: {
                "mean_predicted_component_score_reduction": float(
                    np.mean(reductions[name])
                ),
                "prediction_flip_rate": float(np.mean(flips[name])),
            }
            for name in ("top", "random", "bottom")
        }
        result[str(k)]["top_exceeds_random_score_reduction"] = bool(
            result[str(k)]["top"][
                "mean_predicted_component_score_reduction"
            ]
            > result[str(k)]["random"][
                "mean_predicted_component_score_reduction"
            ]
        )
        result[str(k)]["top_exceeds_bottom_score_reduction"] = bool(
            result[str(k)]["top"][
                "mean_predicted_component_score_reduction"
            ]
            > result[str(k)]["bottom"][
                "mean_predicted_component_score_reduction"
            ]
        )
    return {
        "protocol": (
            "deletion_comprehensiveness_without_retraining_not_roar"
        ),
        "interpretation": (
            "directions explain distance score contributions, not positive "
            "class evidence; deletion lowers the predicted-component score"
        ),
        "k_results": result,
    }


def _split_rows(
    targets: np.ndarray, *, train_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 75_100)
    train = []
    test = []
    for target in np.unique(targets):
        rows = np.flatnonzero(targets == target)
        rows = rows[rng.permutation(len(rows))]
        split = max(1, min(len(rows) - 1, int(len(rows) * train_fraction)))
        train.extend(rows[:split])
        test.extend(rows[split:])
    return (
        np.asarray(sorted(train), dtype=np.int64),
        np.asarray(sorted(test), dtype=np.int64),
    )


def _simulation_probe(
    explanations: np.ndarray,
    model_predictions: np.ndarray,
    *,
    train_fraction: float,
    max_iter: int,
    seed: int,
) -> dict[str, Any]:
    train, test = _split_rows(
        model_predictions, train_fraction=train_fraction, seed=seed
    )
    probe = LogisticRegression(
        max_iter=max_iter, random_state=seed, solver="lbfgs"
    ).fit(explanations[train], model_predictions[train])
    probe_predictions = probe.predict(explanations[test])
    values, counts = np.unique(
        model_predictions[train], return_counts=True
    )
    majority = values[int(np.argmax(counts))]
    majority_predictions = np.full(len(test), majority)
    class_count = len(np.unique(model_predictions))
    return {
        "protocol": (
            "leakage_safe_forward_simulation_proxy_component_identity_withheld"
        ),
        "train_count": int(len(train)),
        "test_count": int(len(test)),
        "probe_balanced_accuracy": float(
            balanced_accuracy_score(
                model_predictions[test], probe_predictions
            )
        ),
        "no_explanation_majority_balanced_accuracy": float(
            balanced_accuracy_score(
                model_predictions[test], majority_predictions
            )
        ),
        "chance_accuracy": float(1.0 / class_count),
        "component_identity_in_explanation": False,
        "example_overlap": False,
    }


def _field_simulation_explanations(
    contributions: np.ndarray, *, top_count: int
) -> np.ndarray:
    ordered = np.sort(contributions, axis=1)[:, ::-1]
    count = min(top_count, ordered.shape[1])
    selected = ordered[:, :count]
    return np.column_stack(
        [
            selected,
            np.sum(contributions, axis=1),
            np.max(contributions, axis=1),
            np.mean(contributions, axis=1),
        ]
    )


def _control_explanations(
    rbf: SVC,
    knn: KNeighborsClassifier,
    mlp: MLPClassifier,
    features: np.ndarray,
    *,
    top_count: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    gamma = float(rbf._gamma)
    squared_distances = np.sum(
        (features[:, None, :] - rbf.support_vectors_[None, :, :]) ** 2,
        axis=2,
    )
    kernels = np.exp(-gamma * squared_distances)
    rbf_explanation = np.sort(kernels, axis=1)[:, -top_count:][:, ::-1]

    neighbor_distances, _ = knn.kneighbors(
        features, n_neighbors=min(top_count, knn.n_neighbors)
    )
    knn_explanation = np.column_stack(
        [
            neighbor_distances,
            np.mean(neighbor_distances, axis=1),
            np.max(neighbor_distances, axis=1),
        ]
    )

    hidden_linear = features @ mlp.coefs_[0] + mlp.intercepts_[0]
    hidden = np.maximum(hidden_linear, 0.0)
    mlp_explanation = np.sort(hidden, axis=1)[:, -top_count:][:, ::-1]
    return {
        "rbf": (rbf_explanation, rbf.predict(features)),
        "knn": (knn_explanation, knn.predict(features)),
        "mlp": (mlp_explanation, mlp.predict(features)),
    }


def _control_structural_audit(
    mlp: MLPClassifier,
    features: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, Any]:
    hidden = np.maximum(
        features @ mlp.coefs_[0] + mlp.intercepts_[0], 0.0
    )
    reconstructed_logits = hidden @ mlp.coefs_[1] + mlp.intercepts_[1]
    reconstructed_probabilities = softmax(reconstructed_logits, axis=1)
    residual = np.abs(reconstructed_probabilities - mlp.predict_proba(features))
    return {
        "rbf": {
            "i1_intrinsic_parameter_semantics": False,
            "i2_registered_directional_decomposition": False,
            "note": (
                "support-vector kernel expansion is algebraically exact but "
                "is not a decomposition over named geometric directions"
            ),
        },
        "knn": {
            "i1_intrinsic_parameter_semantics": False,
            "i2_registered_directional_decomposition": False,
            "note": (
                "neighbor votes decompose exactly over stored examples, not "
                "over intrinsic geometric parameters"
            ),
        },
        "mlp": {
            "i1_intrinsic_parameter_semantics": False,
            "i2_hidden_unit_logit_decomposition": bool(
                np.max(residual) <= tolerance
            ),
            "mean_absolute_probability_residual": float(np.mean(residual)),
            "maximum_absolute_probability_residual": float(np.max(residual)),
            "note": (
                "the output logit decomposes over hidden activations, but "
                "hidden units do not have registered intrinsic semantics"
            ),
        },
    }


def run_inspectability(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _verify(config["source_config"])
    m74_index_path = _verify(config["m74_index"])
    m74 = json.loads(
        (m74_index_path.parent / "evidence.json").read_text(encoding="utf-8")
    )
    if bool(m74["gate"]["m74_passed"]):
        raise ValueError("M75 expected the immutable M74 Outcome E branch")
    seed = int(config["seed"])
    seed_record = next(
        item for item in m74["primary"]["seed_records"] if item["seed"] == seed
    )
    state = _state_from_dict(seed_record["trained_state"])
    if payload_hash(state.to_dict()) != seed_record["field"]["state_hash"]:
        raise ValueError("M75 retained state does not match M74")

    source = json.loads(
        _resolve(config["source_config"]["path"]).read_text(encoding="utf-8")
    )
    loaded = _load_seed_data(source["seed_inputs"][str(seed)])
    train_x, train_y = loaded["datasets"]["train"]
    dev_x, dev_y = loaded["datasets"]["dev"]
    known_classes = np.asarray(config["known_classes"], dtype=np.int64)
    unknown_classes = np.asarray(
        config["proxy_unknown_classes"], dtype=np.int64
    )
    partitions = _partition_seed(
        train_y,
        dev_y,
        seed=seed,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        geometry_fraction=float(config["geometry_fraction"]),
    )
    fit_x = train_x[partitions["geometry_fit"]]
    fit_y = train_y[partitions["geometry_fit"]]
    evaluation_x = dev_x[partitions["development_eval"]][
        : int(config["evaluation_limit"])
    ]
    projected, predictions, predicted_indices, contributions = (
        _field_explanations(state, evaluation_x)
    )

    rbf = SVC(C=1.0, gamma="scale", kernel="rbf", random_state=seed).fit(
        fit_x, fit_y
    )
    knn = KNeighborsClassifier(
        n_neighbors=5, weights="distance", algorithm="brute"
    ).fit(fit_x, fit_y)
    mlp = MLPClassifier(
        hidden_layer_sizes=(int(config["mlp_hidden_units"]),),
        max_iter=int(config["mlp_max_iter"]),
        random_state=seed,
        solver="adam",
        early_stopping=False,
    ).fit(fit_x, fit_y)
    top_count = int(config["simulation_top_contributions"])
    simulations = {
        "field": _simulation_probe(
            _field_simulation_explanations(
                contributions, top_count=top_count
            ),
            predictions,
            train_fraction=float(config["simulation_train_fraction"]),
            max_iter=int(config["simulation_logistic_max_iter"]),
            seed=seed,
        )
    }
    for name, (explanation, control_predictions) in (
        _control_explanations(
            rbf, knn, mlp, evaluation_x, top_count=top_count
        ).items()
    ):
        simulations[name] = _simulation_probe(
            explanation,
            control_predictions,
            train_fraction=float(config["simulation_train_fraction"]),
            max_iter=int(config["simulation_logistic_max_iter"]),
            seed=seed,
        )

    tolerance = float(config["tolerance"])
    field_audit = {
        "i1_intrinsic_parameter_semantics": {
            "passed": True,
            "parameters": {
                "projection_mean": "centering origin in frozen feature space",
                "projection": "learned linear coordinates",
                "centers": "per-class projected centers",
                "bases": "per-class orthonormal tangent directions",
                "tangent_scales": "per-direction class extents",
                "residual_scales": "per-class isotropic normal extent",
            },
            "scope": (
                "decision rule over learned coordinates; coordinate semantics "
                "are not claimed"
            ),
        },
        "i2_exact_score_decomposition": _exact_decomposition(
            state, projected, tolerance=tolerance
        ),
        "i3_deletion_comprehensiveness": _deletion_comprehensiveness(
            state,
            projected,
            predictions,
            predicted_indices,
            k_values=[int(value) for value in config["deletion_k_values"]],
            random_repeats=int(config["deletion_random_repeats"]),
            seed=seed,
        ),
        "i4_minimum_counterfactual_distance": {
            "closed_form_available": False,
            "validity_evaluated": False,
            "passed": False,
            "reason": (
                "the actual decision is the minimum over class-specific "
                "anisotropic quadratic scores plus conformal rejection; "
                "minimum Euclidean displacement to their pairwise or union "
                "boundary is not the field value and has no registered "
                "closed form"
            ),
            "eikonal_caveat": (
                "M74 calibration gradient norms were not one, so score values "
                "cannot be interpreted as Euclidean displacement distances"
            ),
        },
        "i5_forward_simulation_proxy": simulations["field"],
    }
    evidence = {
        "schema_version": 1,
        "milestone": "M75",
        "configuration_hash": sha256_file(config_path),
        "m74_outcome": "E",
        "outcome_can_change": False,
        "partition_hashes": {
            name: payload_hash(indices.tolist())
            for name, indices in partitions.items()
        },
        "evaluation_count": int(len(evaluation_x)),
        "retained_state_hash": payload_hash(state.to_dict()),
        "field": field_audit,
        "controls": {
            "structural_audit": _control_structural_audit(
                mlp, evaluation_x, tolerance=tolerance
            ),
            "forward_simulation_proxy": {
                name: simulations[name] for name in ("rbf", "knn", "mlp")
            },
        },
        "summary": {
            "i1_passed": True,
            "i2_passed": bool(
                field_audit["i2_exact_score_decomposition"]["passed"]
            ),
            "i3_all_top_score_reductions_exceed_random_and_bottom": all(
                record["top_exceeds_random_score_reduction"]
                and record["top_exceeds_bottom_score_reduction"]
                for record in field_audit[
                    "i3_deletion_comprehensiveness"
                ]["k_results"].values()
            ),
            "i4_passed": False,
            "i5_probe_beats_no_explanation": bool(
                simulations["field"]["probe_balanced_accuracy"]
                > simulations["field"][
                    "no_explanation_majority_balanced_accuracy"
                ]
            ),
            "qualified_inspectability_claim": False,
            "disposition": (
                "descriptive_only_outcome_e_unchanged_i4_structurally_unavailable"
            ),
            "exact_replay": True,
            "final_labels_opened": False,
        },
        "final_labels_opened": False,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = run_inspectability(arguments.config, arguments.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
