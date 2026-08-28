"""Focused tests for M19 frozen-space head controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.common.classification_baselines import WeightedKNNClassifier
from experiments.common.representation_metrics import (
    compute_representation_diagnostics,
)
from experiments.tier4.eval_v5_frozen_space_heads import (
    compute_geode_component_efficiency,
)
from experiments.tier4.eval_v5_flowers_heads import (
    _blocked_head,
    _minimum_class_support,
)
from src.sdf_engine import EllipsoidExpert, Expert


def test_weighted_knn_predicts_separated_classes() -> None:
    features = np.array(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]],
        dtype=np.float64,
    )
    labels = np.array([10, 10, 20, 20], dtype=np.int64)
    model = WeightedKNNClassifier(n_neighbors=2, temperature=0.07).fit(
        features, labels
    )

    predictions = model.predict(np.array([[0.8, 0.2], [-0.8, -0.2]]))
    probabilities = model.predict_proba(np.array([[0.8, 0.2], [-0.8, -0.2]]))

    assert predictions.tolist() == [10, 20]
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_weighted_knn_is_deterministic_under_ties() -> None:
    features = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
    labels = np.array([2, 1, 3], dtype=np.int64)
    query = np.array([[1.0, 0.0]], dtype=np.float64)
    model = WeightedKNNClassifier(n_neighbors=1).fit(features, labels)

    first = model.predict(query)
    second = model.predict(query)

    assert first.tolist() == [2]
    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    ("n_neighbors", "temperature"),
    [(0, 0.07), (1, 0.0), (1, float("nan"))],
)
def test_weighted_knn_rejects_invalid_configuration(
    n_neighbors: int, temperature: float
) -> None:
    with pytest.raises(ValueError):
        WeightedKNNClassifier(n_neighbors=n_neighbors, temperature=temperature)


def test_weighted_knn_batches_queries_without_changing_results() -> None:
    features = np.array(
        [[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0], [-0.8, -0.2]],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    queries = np.array([[0.9, 0.1], [-0.9, -0.1], [0.7, 0.3]])
    batched = WeightedKNNClassifier(
        n_neighbors=2, query_batch_size=1
    ).fit(features, labels)
    unbatched = WeightedKNNClassifier(
        n_neighbors=2, query_batch_size=10
    ).fit(features, labels)

    assert np.array_equal(batched.predict(queries), unbatched.predict(queries))
    assert np.allclose(
        batched.predict_proba(queries), unbatched.predict_proba(queries)
    )


def test_weighted_knn_rejects_nonfinite_features() -> None:
    model = WeightedKNNClassifier(n_neighbors=1)
    with pytest.raises(ValueError, match="finite"):
        model.fit(np.array([[np.nan, 0.0]]), np.array([0]))


def test_representation_diagnostics_separate_compact_classes() -> None:
    features = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 0, 1, 1, 1])

    metrics = compute_representation_diagnostics(
        features,
        labels,
        n_neighbors=2,
    )

    assert metrics["neighborhood_purity"] == 1.0
    assert metrics["minimum_centroid_separation"] > 14.0
    assert metrics["within_class_radius"] < 0.1
    assert metrics["compactness_ratio"] < 0.01
    assert metrics["local_intrinsic_dimension"] >= 0.0


def test_representation_diagnostics_are_deterministic() -> None:
    features = np.arange(48, dtype=np.float64).reshape(12, 4)
    labels = np.repeat([0, 1, 2], 4)

    first = compute_representation_diagnostics(features, labels, n_neighbors=3)
    second = compute_representation_diagnostics(features, labels, n_neighbors=3)

    assert first == second


@pytest.mark.parametrize(
    ("features", "labels", "message"),
    [
        (np.ones((3, 2)), np.array([0, 0]), "sample count"),
        (np.array([[0.0], [1.0], [np.nan]]), np.array([0, 0, 1]), "finite"),
        (np.ones((3, 2)), np.array([0, 0, 0]), "two classes"),
    ],
)
def test_representation_diagnostics_reject_invalid_inputs(
    features: np.ndarray,
    labels: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_representation_diagnostics(features, labels)


def test_geode_component_efficiency_counts_greedy_prefix() -> None:
    features = np.array(
        [[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0]],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1])

    class_models = []
    for centers in (((0.0, 0.0), (2.0, 0.0)), ((10.0, 0.0), (12.0, 0.0))):
        expert = Expert(alpha=100.0)
        for center in centers:
            expert.add_ellipsoid(
                EllipsoidExpert(np.array(center), np.array([0.25, 0.25]))
            )
        class_models.append({"model": [expert]})
    head = {
        "name": "current_geode",
        "status": "fitted",
        "classes": np.array([0, 1]),
        "model": class_models,
    }

    result = compute_geode_component_efficiency(
        head,
        features,
        labels,
        target_coverage=1.0,
    )

    assert result["status"] == "evaluated"
    assert result["mean_components_required"] == 2.0
    assert result["max_components_required"] == 2
    assert all(
        record["achieved_coverage"] == 1.0 for record in result["per_class"]
    )


def test_geode_component_efficiency_reports_unmet_target() -> None:
    features = np.array([[0.0], [2.0], [10.0], [12.0]], dtype=np.float64)
    labels = np.array([0, 0, 1, 1])
    empty_head = {
        "name": "current_geode",
        "status": "fitted",
        "classes": np.array([0, 1]),
        "model": [{"model": []}, {"model": []}],
    }

    result = compute_geode_component_efficiency(
        empty_head,
        features,
        labels,
    )

    assert result["status"] == "target_unmet"
    assert result["classes_reaching_target"] == 0
    assert result["mean_components_required"] is None


def test_geode_component_efficiency_preserves_blocked_reason() -> None:
    result = compute_geode_component_efficiency(
        {"status": "blocked", "reason": "dimension limit"},
        np.ones((4, 2)),
        np.array([0, 0, 1, 1]),
    )

    assert result == {
        "status": "blocked",
        "reason": "dimension limit",
        "target_coverage": 0.95,
    }


def test_flowers_support_and_blocked_head_are_explicit() -> None:
    labels = np.repeat(np.arange(3), [5, 7, 6])
    support = _minimum_class_support(labels)
    assert support == 5
    assert min(10, support - 1) == 4
    blocked = _blocked_head("current_geode", "needs d+2 points")
    assert blocked["status"] == "blocked"
    assert blocked["accuracy"] is None
    assert blocked["reason"] == "needs d+2 points"


def test_native_dinov2_config_meets_sphere_seed_exactly() -> None:
    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "v5"
            / "m19_native_dinov2_sphere.json"
        ).read_text(encoding="utf-8")
    )
    dimension = config["backbones"][0]["output_dimension"]
    support = config["split_protocol"]["total_per_class_train"]
    assert config["geode_config"]["primitive_family"] == "sphere"
    assert config["geode_config"]["minimum_seed_rule"] == "d_plus_2"
    assert support == dimension + 2
    assert config["geode_config"]["max_iterations"] == 1


def test_native_dinov2_support_pilot_permits_second_sphere_seed() -> None:
    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "v5"
            / "m19_native_dinov2_sphere_support_pilot.json"
        ).read_text(encoding="utf-8")
    )
    dimension = config["backbones"][0]["output_dimension"]
    support = config["split_protocol"]["total_per_class_train"]
    observed_max_first_capture = 0.580310880829015
    conservative_residual = support * (1.0 - observed_max_first_capture)

    assert config["seed"] == 11
    assert config["split_protocol"]["selection_seed"] == config["seed"]
    assert config["geode_config"]["minimum_seed_size"] == dimension + 2
    assert conservative_residual >= dimension + 2
    assert config["geode_config"]["max_iterations"] == 1


@pytest.mark.parametrize("seed", [23, 37])
def test_native_dinov2_support_s2_configs_are_frozen(seed: int) -> None:
    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "v5"
            / f"m19_native_dinov2_sphere_support_s2_seed{seed}.json"
        ).read_text(encoding="utf-8")
    )

    assert config["stage"] == "S2-Native-Sphere-Support"
    assert config["seed"] == seed
    assert config["split_protocol"]["selection_seed"] == seed
    assert config["split_protocol"]["total_per_class_train"] == 1000
    assert config["geode_config"]["minimum_seed_size"] == 386
    assert config["geode_config"]["max_iterations"] == 1
    assert config["limits"]["test_used_for_selection"] is False
