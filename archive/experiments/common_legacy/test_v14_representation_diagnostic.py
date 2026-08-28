"""Tests for M89's diagnostic instrument.

The run is non-gating, so the risk is not a wrong verdict but a quantity that
looks meaningful and is not. Each test pins one instrument against a case whose
answer is known by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.common.v13_boundary import (
    apply_offsets,
    fit_geometry,
    matched_coverage_offsets,
)
from experiments.tier4.eval_v14_m89_representation_diagnostic import (
    acceptance_multiplicity,
    corpus_composition,
    separation_report,
)


def _two_far_classes(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    a = rng.normal(0.0, 0.1, size=(256, 8))
    b = rng.normal(0.0, 0.1, size=(256, 8)) + 50.0
    return np.concatenate([a, b]), np.concatenate([np.zeros(256), np.ones(256)]).astype(
        np.int64
    )


def _matched(
    features: np.ndarray, labels: np.ndarray, geometry, *, class_count: int
) -> np.ndarray:
    """Radii inflated to the 90% own-class coverage M89 reads acceptance at.

    At raw fitted radii a score of one is one Mahalanobis sigma, which accepts
    almost nothing in any dimension of interest, so multiplicity measured there
    would be near zero for every corpus regardless of how classes overlap.
    """
    log_beta = np.concatenate(
        [np.log(geometry.tangent_scales), np.log(geometry.residual_scales)[:, None]],
        axis=1,
    )
    offsets = matched_coverage_offsets(
        features, labels, geometry, log_beta, coverage=0.9, class_count=class_count
    )
    return apply_offsets(log_beta, offsets)


def test_separated_classes_are_accepted_by_exactly_one_region() -> None:
    rng = np.random.default_rng(89001)
    features, labels = _two_far_classes(rng)
    geometry = fit_geometry(features, labels, rank=2, class_count=2)
    matched = _matched(features, labels, geometry, class_count=2)
    report = acceptance_multiplicity(
        features, geometry, matched, class_count=2, labels=labels
    )
    assert report["accepted_by_more_than_one"] == 0.0
    assert report["accepted_by_exactly_one"] >= 0.85
    assert report["argmin_score_accuracy"] == 1.0


def test_coincident_classes_are_accepted_by_both_regions() -> None:
    rng = np.random.default_rng(89002)
    features = rng.normal(0.0, 1.0, size=(512, 8))
    labels = np.tile([0, 1], 256).astype(np.int64)
    geometry = fit_geometry(features, labels, rank=2, class_count=2)
    matched = _matched(features, labels, geometry, class_count=2)
    report = acceptance_multiplicity(
        features, geometry, matched, class_count=2, labels=labels
    )
    assert report["accepted_by_more_than_one"] > 0.5
    assert report["argmin_score_accuracy"] < 0.6


def test_spread_over_separation_is_small_when_classes_are_far_apart() -> None:
    rng = np.random.default_rng(89003)
    features, labels = _two_far_classes(rng)
    domains = np.zeros(len(features), dtype=np.int64)
    report = separation_report(
        features,
        labels,
        domains,
        class_count=2,
        domain_count=1,
        minimum_cell_support=100,
        floor_samples_per_dimension=10,
    )
    assert report["class_level"]["spread_over_separation"] < 0.05


def test_domain_dominance_is_detected_when_domains_are_displaced() -> None:
    """Two classes 4 apart, two domains 40 apart: domain must dominate."""
    rng = np.random.default_rng(89004)
    blocks, labels, domains = [], [], []
    for k in range(2):
        for d in range(2):
            block = rng.normal(0.0, 0.5, size=(128, 8))
            block[:, 0] += 4.0 * k
            block[:, 1] += 40.0 * d
            blocks.append(block)
            labels.append(np.full(128, k))
            domains.append(np.full(128, d))
    report = separation_report(
        np.concatenate(blocks),
        np.concatenate(labels).astype(np.int64),
        np.concatenate(domains).astype(np.int64),
        class_count=2,
        domain_count=2,
        minimum_cell_support=100,
        floor_samples_per_dimension=10,
    )
    dominance = report["domain_dominates_class"]
    assert dominance["sibling_exceeds_foreign"] is True
    assert report["cell_level"]["spread_over_separation"] < (
        report["class_level"]["spread_over_separation"]
    )


def test_domain_dominance_is_not_reported_when_classes_are_the_larger_axis() -> None:
    rng = np.random.default_rng(89005)
    blocks, labels, domains = [], [], []
    for k in range(2):
        for d in range(2):
            block = rng.normal(0.0, 0.5, size=(128, 8))
            block[:, 0] += 40.0 * k
            block[:, 1] += 4.0 * d
            blocks.append(block)
            labels.append(np.full(128, k))
            domains.append(np.full(128, d))
    report = separation_report(
        np.concatenate(blocks),
        np.concatenate(labels).astype(np.int64),
        np.concatenate(domains).astype(np.int64),
        class_count=2,
        domain_count=2,
        minimum_cell_support=100,
        floor_samples_per_dimension=10,
    )
    assert report["domain_dominates_class"]["sibling_exceeds_foreign"] is False


def test_permitted_rank_respects_the_sample_floor() -> None:
    rng = np.random.default_rng(89006)
    blocks, labels, domains = [], [], []
    supports = {0: 300, 1: 120}
    for k in range(2):
        for d in range(2):
            n = supports[d]
            blocks.append(rng.normal(0.0, 1.0, size=(n, 8)) + 10.0 * k)
            labels.append(np.full(n, k))
            domains.append(np.full(n, d))
    report = separation_report(
        np.concatenate(blocks),
        np.concatenate(labels).astype(np.int64),
        np.concatenate(domains).astype(np.int64),
        class_count=2,
        domain_count=2,
        minimum_cell_support=100,
        floor_samples_per_dimension=10,
    )
    mixture = report["mixture_probe"]
    assert mixture["cells_retained"] == 4
    assert mixture["smallest_retained_support"] == 120
    assert mixture["permitted_common_rank"] == 12


def test_cells_below_minimum_support_do_not_set_the_rank() -> None:
    rng = np.random.default_rng(89007)
    blocks, labels, domains = [], [], []
    supports = {0: 300, 1: 20}
    for k in range(2):
        for d in range(2):
            n = supports[d]
            blocks.append(rng.normal(0.0, 1.0, size=(n, 8)) + 10.0 * k)
            labels.append(np.full(n, k))
            domains.append(np.full(n, d))
    report = separation_report(
        np.concatenate(blocks),
        np.concatenate(labels).astype(np.int64),
        np.concatenate(domains).astype(np.int64),
        class_count=2,
        domain_count=2,
        minimum_cell_support=100,
        floor_samples_per_dimension=10,
    )
    mixture = report["mixture_probe"]
    assert mixture["cells_retained"] == 2
    assert mixture["permitted_common_rank"] == 30


def test_composition_records_an_absent_domain_as_zero() -> None:
    domains = np.array([0, 0, 1, 3, 3, 3], dtype=np.int64)
    report = corpus_composition(domains, domain_count=6, label="unit")
    assert report["row_count"] == 6
    assert report["by_domain"]["sketch"]["row_count"] == 0
    assert report["by_domain"]["sketch"]["fraction"] == 0.0
    assert report["by_domain"]["quickdraw"]["row_count"] == 3
    assert report["by_domain"]["quickdraw"]["fraction"] == pytest.approx(0.5)
    total = sum(entry["fraction"] for entry in report["by_domain"].values())
    assert total == pytest.approx(1.0)


def test_acceptance_multiplicity_is_deterministic() -> None:
    rng = np.random.default_rng(89008)
    features, labels = _two_far_classes(rng)
    geometry = fit_geometry(features, labels, rank=2, class_count=2)
    matched = _matched(features, labels, geometry, class_count=2)
    torch.set_num_threads(1)
    first = acceptance_multiplicity(
        features, geometry, matched, class_count=2, labels=labels
    )
    second = acceptance_multiplicity(
        features, geometry, matched, class_count=2, labels=labels, chunk_rows=64
    )
    assert first == second
