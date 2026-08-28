"""Tests for M90's arms, nulls and gate.

The risk in M90 is not a crash but a silently unfair comparison: a null that
does not match its arm's structure, a coverage rule that fixes each component's
own coverage instead of the class's, or a gate that reads a figure from an arm
whose instrument failed. Each is pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.tier4.eval_v14_m90_representation_remedies import (
    Assignment,
    _clears,
    _l2_normalise,
    _random_groups,
    _verdict,
    class_assignment,
    class_scores,
    domain_cell_assignment,
    fit_components,
    random_cell_assignment,
    union_coverage_offsets,
)


def _labelled(class_count: int, per_class: int) -> np.ndarray:
    return np.repeat(np.arange(class_count), per_class).astype(np.int64)


def test_class_assignment_owns_one_component_per_class() -> None:
    labels = _labelled(4, 10)
    assignment = class_assignment(labels, 4)
    assert assignment.count == 4
    assert list(assignment.owner) == [0, 1, 2, 3]
    assert np.array_equal(assignment.starts, np.array([0, 1, 2, 3]))


def test_domain_cells_below_support_are_not_given_a_component() -> None:
    labels = _labelled(2, 300)
    domains = np.concatenate(
        [
            np.repeat([0, 1], [250, 50]),
            np.repeat([0, 1], [250, 50]),
        ]
    ).astype(np.int64)
    assignment, sizes = domain_cell_assignment(
        labels, domains, class_count=2, domain_count=2, minimum_support=100
    )
    # One class-level component each, plus only the 250-row cell.
    assert assignment.count == 4
    assert sizes == [[250], [250]]
    assert list(assignment.owner) == [0, 0, 1, 1]


def test_every_row_is_covered_by_its_class_level_component() -> None:
    labels = _labelled(3, 200)
    domains = np.tile(np.repeat([0, 1], 100), 3).astype(np.int64)
    assignment, _ = domain_cell_assignment(
        labels, domains, class_count=3, domain_count=2, minimum_support=100
    )
    for k in range(3):
        owned = [
            rows
            for rows, owner in zip(assignment.memberships, assignment.owner)
            if owner == k
        ]
        union = np.unique(np.concatenate(owned))
        assert np.array_equal(union, np.flatnonzero(labels == k))


def test_null_matches_component_count_and_sizes_exactly() -> None:
    labels = _labelled(3, 300)
    domains = np.tile(np.repeat([0, 1, 2], 100), 3).astype(np.int64)
    arm, sizes = domain_cell_assignment(
        labels, domains, class_count=3, domain_count=3, minimum_support=100
    )
    null = random_cell_assignment(labels, sizes, class_count=3, seed=7)
    assert null.count == arm.count
    assert list(null.owner) == list(arm.owner)
    assert [len(m) for m in null.memberships] == [len(m) for m in arm.memberships]


def test_null_cells_are_disjoint_and_stay_inside_their_class() -> None:
    labels = _labelled(2, 400)
    domains = np.tile(np.repeat([0, 1], 200), 2).astype(np.int64)
    _, sizes = domain_cell_assignment(
        labels, domains, class_count=2, domain_count=2, minimum_support=100
    )
    null = random_cell_assignment(labels, sizes, class_count=2, seed=11)
    for k in range(2):
        cells = [
            rows
            for rows, owner in zip(null.memberships, null.owner)
            if owner == k and len(rows) < 400
        ]
        stacked = np.concatenate(cells)
        assert len(stacked) == len(np.unique(stacked))
        assert set(np.unique(labels[stacked])) == {k}


def test_random_groups_preserve_the_group_size_profile() -> None:
    sizes = np.array([600, 200, 200], dtype=np.int64)
    groups = _random_groups(1000, sizes, seed=3)
    counts = np.bincount(groups, minlength=3)
    assert counts.sum() == 1000
    assert list(counts) == [600, 200, 200]


def test_l2_normalise_puts_every_row_on_the_unit_sphere() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(0.0, 10.0, size=(64, 8))
    normalised = _l2_normalise(features)
    assert np.allclose(np.linalg.norm(normalised, axis=1), 1.0)


def test_union_coverage_matches_the_class_not_each_component() -> None:
    """Coverage is a property of the class's union, so it holds with a mixture.

    Two well-separated domain modes per class: matching each component
    separately would cover far more than 90%, because each mode would be
    inflated to hold 90% of the whole class.
    """
    rng = np.random.default_rng(13)
    blocks, labels, domains = [], [], []
    for k in range(2):
        for d in range(2):
            block = rng.normal(0.0, 0.5, size=(150, 6))
            block[:, 0] += 20.0 * k
            block[:, 1] += 30.0 * d
            blocks.append(block)
            labels.append(np.full(150, k))
            domains.append(np.full(150, d))
    features = np.concatenate(blocks)
    labels = np.concatenate(labels).astype(np.int64)
    domains = np.concatenate(domains).astype(np.int64)

    assignment, _ = domain_cell_assignment(
        labels, domains, class_count=2, domain_count=2, minimum_support=100
    )
    geometry = fit_components(features, assignment, rank=3)
    log_beta = np.concatenate(
        [np.log(geometry.tangent_scales), np.log(geometry.residual_scales)[:, None]],
        axis=1,
    )
    offsets = union_coverage_offsets(
        features, labels, geometry, log_beta, assignment, coverage=0.9, class_count=2
    )
    matched = log_beta + offsets[assignment.owner][:, None]
    scores = class_scores(features, geometry, matched, assignment)
    covered = float(np.mean(scores[np.arange(len(labels)), labels] <= 1.0))
    assert covered == pytest.approx(0.9, abs=0.05)


def test_class_scores_take_the_minimum_over_a_class_components() -> None:
    rng = np.random.default_rng(17)
    labels = _labelled(2, 300)
    domains = np.tile(np.repeat([0, 1], 150), 2).astype(np.int64)
    features = rng.normal(0.0, 1.0, size=(600, 6))
    features[labels == 1, 0] += 25.0
    assignment, _ = domain_cell_assignment(
        labels, domains, class_count=2, domain_count=2, minimum_support=100
    )
    geometry = fit_components(features, assignment, rank=3)
    log_beta = np.concatenate(
        [np.log(geometry.tangent_scales), np.log(geometry.residual_scales)[:, None]],
        axis=1,
    )
    reduced = class_scores(features, geometry, log_beta, assignment)
    assert reduced.shape == (600, 2)
    assert np.all(reduced.argmin(axis=1) == labels)


def test_fit_components_refuses_a_component_below_the_rank_contract() -> None:
    rng = np.random.default_rng(19)
    features = rng.normal(size=(6, 8))
    assignment = Assignment((np.arange(6),), np.array([0], dtype=np.int64))
    with pytest.raises(ValueError, match="needs"):
        fit_components(features, assignment, rank=10)


def _arm(*, multiplicity: float, recall: float, auroc: float, valid: bool = True):
    return {
        "controls": {"valid": valid},
        "acceptance_multiplicity": {"mean": multiplicity},
        "rejection_recall": {"rejection_recall": recall},
        "auroc": {"auroc": auroc},
    }


GATE = {
    "multiplicity_bar": 40.0,
    "v13_rejection_recall": 0.11875,
    "v13_geometry_auroc": 0.585085105895996,
    "auroc_margin": 0.02,
}


def test_gate_requires_all_three_bars() -> None:
    assert _clears(_arm(multiplicity=10.0, recall=0.5, auroc=0.8), GATE)["clears"]
    assert not _clears(_arm(multiplicity=50.0, recall=0.5, auroc=0.8), GATE)["clears"]
    assert not _clears(_arm(multiplicity=10.0, recall=0.05, auroc=0.8), GATE)["clears"]
    assert not _clears(_arm(multiplicity=10.0, recall=0.5, auroc=0.59), GATE)["clears"]


def test_an_invalid_instrument_cannot_clear_the_gate() -> None:
    result = _clears(
        _arm(multiplicity=1.0, recall=0.99, auroc=0.99, valid=False), GATE
    )
    assert result["clears"] is False
    assert result["reason"] == "instrument_invalid"


def test_an_arm_that_does_not_beat_its_null_does_not_rescue_its_hypothesis() -> None:
    strong = _arm(multiplicity=5.0, recall=0.9, auroc=0.9)
    arms = {
        "cosine": _arm(multiplicity=90.0, recall=0.0, auroc=0.5),
        "domain_centred": dict(strong),
        "domain_centred_null": dict(strong),
        "domain_mixture": dict(strong),
        "domain_mixture_null": dict(strong),
    }
    for name, arm in arms.items():
        arm["gate"] = _clears(arm, GATE)
    verdict = _verdict(arms, GATE)
    assert verdict["h90_misspecification"]["verdict"] == "refuted"
    assert verdict["h91_coordinates"]["verdict"] == "refuted"
    assert verdict["m91_opens"] is True


def test_an_invalid_instrument_leaves_its_hypothesis_undetermined() -> None:
    """A void arm is not a negative one, and it must not open the next stage."""
    arms = {
        "cosine": _arm(multiplicity=90.0, recall=0.0, auroc=0.5),
        "domain_centred": _arm(multiplicity=90.0, recall=0.0, auroc=0.5),
        "domain_centred_null": _arm(multiplicity=90.0, recall=0.0, auroc=0.5),
        "domain_mixture": _arm(
            multiplicity=5.0, recall=0.9, auroc=0.9, valid=False
        ),
        "domain_mixture_null": _arm(multiplicity=90.0, recall=0.0, auroc=0.5),
    }
    for arm in arms.values():
        arm["gate"] = _clears(arm, GATE)
    verdict = _verdict(arms, GATE)
    assert verdict["h90_misspecification"]["verdict"] == "undetermined"
    assert verdict["h91_coordinates"]["verdict"] == "refuted"
    assert verdict["m91_opens"] is False


def test_an_arm_that_clears_and_beats_its_null_survives() -> None:
    arms = {
        "cosine": _arm(multiplicity=90.0, recall=0.0, auroc=0.5),
        "domain_centred": _arm(multiplicity=90.0, recall=0.0, auroc=0.5),
        "domain_centred_null": _arm(multiplicity=90.0, recall=0.0, auroc=0.5),
        "domain_mixture": _arm(multiplicity=5.0, recall=0.9, auroc=0.9),
        "domain_mixture_null": _arm(multiplicity=8.0, recall=0.4, auroc=0.7),
    }
    for arm in arms.values():
        arm["gate"] = _clears(arm, GATE)
    verdict = _verdict(arms, GATE)
    assert verdict["h90_misspecification"]["verdict"] == "survives"
    assert verdict["m91_opens"] is False
