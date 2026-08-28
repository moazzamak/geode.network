"""Tests for M83's absolute-scale boundary supervision.

The load-bearing tests here are the ones that pin the instrument rather than
the result: the v12-form arm must read as degenerate and the absolute arm must
not. If both read the same way the degeneracy report is not measuring what M77
measured, and M83 is void by its own registration.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from experiments.common.v13_boundary import (
    HELD_OUT_FAMILIES,
    Geometry,
    absolute_unit,
    acceptance_rate,
    apply_offsets,
    average_ranks,
    boundary_displacement,
    boundary_scores,
    build_probe_spec,
    data_scale_unit,
    degeneracy_report,
    domain_auroc,
    domain_matched_partition,
    domain_stratified_halves,
    exposure_owners,
    exposure_term,
    exposure_validity,
    far_field_points,
    fit_geometry,
    global_scale_unit,
    known_term,
    matched_coverage_offsets,
    minimum_scores_numpy,
    moment_matched_negatives,
    owner_agreement,
    owner_scores_numpy,
    probe_points_numpy,
    probe_rejection,
    probe_term,
    probe_validity,
    rejection_recall,
    sample_exposure,
    score_auroc,
    shuffled_owners,
    tangent_anisotropy,
    train_boundary,
    train_exposure_boundary,
)

CLASS_COUNT = 6
DIMENSION = 24
RANK = 4
PER_CLASS = 60
TRAIN_FAMILIES = ("axis_tangent", "normal")
RELATIVE_ONLY = ("axis_tangent",)
#: Probes land just outside the fitted boundaries, where the hinge is active.
MULTIPLIERS = (1.0, 2.0)
#: Far enough out that every boundary already rejects them and the hinge is
#: clamped. Used only to prove saturation is not read as degeneracy.
SATURATED_MULTIPLIERS = (40.0, 80.0)
MARGIN = 0.25


@pytest.fixture(scope="module")
def corpus() -> tuple[np.ndarray, np.ndarray]:
    """Well-separated anisotropic Gaussian classes.

    Anisotropy matters: an isotropic corpus would make the boundary's shape
    term degenerate for reasons unrelated to the objective, and the shape
    displacement operand would read zero for every arm.
    """
    generator = np.random.default_rng(83831)
    centers = generator.normal(scale=6.0, size=(CLASS_COUNT, DIMENSION))
    scales = generator.uniform(0.3, 1.6, size=(CLASS_COUNT, DIMENSION))
    features = np.concatenate(
        [
            centers[label]
            + scales[label] * generator.normal(size=(PER_CLASS, DIMENSION))
            for label in range(CLASS_COUNT)
        ]
    )
    labels = np.repeat(np.arange(CLASS_COUNT), PER_CLASS)
    return features.astype(np.float64), labels.astype(np.int64)


@pytest.fixture(scope="module")
def geometry(corpus: tuple[np.ndarray, np.ndarray]) -> Geometry:
    features, labels = corpus
    return fit_geometry(features, labels, rank=RANK, class_count=CLASS_COUNT)


@pytest.fixture(scope="module")
def unit(geometry: Geometry) -> float:
    return global_scale_unit(geometry)


@pytest.fixture(scope="module")
def initial_log_beta(geometry: Geometry) -> np.ndarray:
    return np.concatenate(
        [
            np.log(geometry.tangent_scales),
            np.log(geometry.residual_scales)[:, None],
        ],
        axis=1,
    )


# ---------------------------------------------------------------------------
# Geometry and the absolute unit
# ---------------------------------------------------------------------------


def test_fitted_geometry_has_the_registered_shapes(geometry: Geometry) -> None:
    assert geometry.centers.shape == (CLASS_COUNT, DIMENSION)
    assert geometry.bases.shape == (CLASS_COUNT, DIMENSION, RANK)
    assert geometry.tangent_scales.shape == (CLASS_COUNT, RANK)
    assert geometry.residual_scales.shape == (CLASS_COUNT,)
    assert np.all(geometry.tangent_scales > 0.0)


def test_bases_are_orthonormal(geometry: Geometry) -> None:
    for label in range(CLASS_COUNT):
        basis = geometry.bases[label]
        assert np.allclose(basis.T @ basis, np.eye(RANK), atol=1e-8)


def test_absolute_unit_is_independent_of_fitted_extents(geometry: Geometry) -> None:
    """The reference length must not be readable off any class's own extent.

    Inflating every fitted extent tenfold must not move it by a hair; only
    moving the centroids may.
    """
    inflated = Geometry(
        geometry.centers,
        geometry.bases,
        geometry.tangent_scales * 10.0,
        geometry.residual_scales * 10.0,
    )
    assert absolute_unit(inflated.centers) == absolute_unit(geometry.centers)
    assert absolute_unit(geometry.centers * 3.0) == pytest.approx(
        3.0 * absolute_unit(geometry.centers)
    )


def test_the_placement_unit_is_one_scalar_for_every_class(
    geometry: Geometry,
) -> None:
    """The property that actually blocks the v12 cancellation.

    A placement length cannot cancel against a class's own radius unless it
    carries that radius. Perturbing a single class's extents must leave the
    unit that every *other* class is probed at unchanged — and because the
    unit is a median over all classes and axes, one class cannot move it far
    even for itself.
    """
    perturbed = Geometry(
        geometry.centers,
        geometry.bases,
        geometry.tangent_scales.copy(),
        geometry.residual_scales,
    )
    perturbed.tangent_scales[0] *= 50.0
    assert global_scale_unit(perturbed) == pytest.approx(
        global_scale_unit(geometry), rel=0.2
    )
    assert np.isscalar(global_scale_unit(geometry)) or isinstance(
        global_scale_unit(geometry), float
    )


# ---------------------------------------------------------------------------
# The defect M83 corrects, reproduced directly
# ---------------------------------------------------------------------------


def test_v12_relative_probes_score_exactly_the_multiplier(
    geometry: Geometry, initial_log_beta: np.ndarray, unit: float
) -> None:
    """v12's placement is algebraically blind to the extent it supervises.

    A probe placed at a multiple of a class's own extent scores that multiple,
    whatever the extent is, for every class. This is the arithmetic behind
    M77's zero-gradient finding, asserted here so the arm cannot silently stop
    reproducing it.
    """
    spec = build_probe_spec(
        geometry, families=TRAIN_FAMILIES, multipliers=(4.0,), seed=3
    )
    points = probe_points_numpy(
        geometry, initial_log_beta, spec, placement="relative", unit=unit
    )
    scores = owner_scores_numpy(points, spec.owners, geometry, initial_log_beta)
    assert scores == pytest.approx(np.full(len(scores), 4.0), rel=1e-8)


def test_relative_probe_scores_do_not_move_when_the_radii_do(
    geometry: Geometry, initial_log_beta: np.ndarray, unit: float
) -> None:
    """The cancellation itself, stated as a property rather than a number."""
    spec = build_probe_spec(
        geometry, families=TRAIN_FAMILIES, multipliers=MULTIPLIERS, seed=3
    )
    scores = []
    for shift in (-1.0, 0.0, 1.5):
        beta = initial_log_beta + shift
        points = probe_points_numpy(
            geometry, beta, spec, placement="relative", unit=unit
        )
        scores.append(owner_scores_numpy(points, spec.owners, geometry, beta))
    assert scores[0] == pytest.approx(scores[1], rel=1e-9)
    assert scores[1] == pytest.approx(scores[2], rel=1e-9)


def test_absolute_probe_scores_move_when_the_radii_do(
    geometry: Geometry, initial_log_beta: np.ndarray, unit: float
) -> None:
    """The correction. Halving every radius must double every probe score."""
    spec = build_probe_spec(
        geometry, families=TRAIN_FAMILIES, multipliers=MULTIPLIERS, seed=3
    )
    points = probe_points_numpy(
        geometry, initial_log_beta, spec, placement="absolute", unit=unit
    )
    base = owner_scores_numpy(points, spec.owners, geometry, initial_log_beta)
    halved = initial_log_beta + np.log(0.5)
    shrunk = owner_scores_numpy(points, spec.owners, geometry, halved)
    assert shrunk == pytest.approx(2.0 * base, rel=1e-9)
    assert float(np.std(base)) > 0.0


def test_absolute_probes_travel_the_unit_whatever_the_class(
    geometry: Geometry, initial_log_beta: np.ndarray, unit: float
) -> None:
    """Displacement is a global length, identical for every class."""
    spec = build_probe_spec(
        geometry, families=TRAIN_FAMILIES, multipliers=(1.0,), seed=3
    )
    points = probe_points_numpy(
        geometry, initial_log_beta, spec, placement="absolute", unit=unit
    )
    travelled = np.linalg.norm(points - geometry.centers[spec.owners], axis=1)
    assert travelled == pytest.approx(np.full(len(travelled), unit), rel=1e-9)


def test_the_two_placements_share_their_directions(
    geometry: Geometry, initial_log_beta: np.ndarray, unit: float
) -> None:
    """The arms must differ in distance alone, or the comparison is confounded."""
    spec = build_probe_spec(
        geometry, families=TRAIN_FAMILIES, multipliers=(1.0,), seed=11
    )
    absolute = probe_points_numpy(
        geometry, initial_log_beta, spec, placement="absolute", unit=unit
    )
    relative = probe_points_numpy(
        geometry, initial_log_beta, spec, placement="relative", unit=unit
    )
    for index, owner in enumerate(spec.owners):
        first = absolute[index] - geometry.centers[owner]
        second = relative[index] - geometry.centers[owner]
        cosine = float(
            first @ second / (np.linalg.norm(first) * np.linalg.norm(second))
        )
        assert cosine == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# The gating degeneracy contract
# ---------------------------------------------------------------------------


def test_degeneracy_report_finds_the_v12_form_degenerate(
    geometry: Geometry, initial_log_beta: np.ndarray, unit: float
) -> None:
    """M77's finding, generalised into a standing contract."""
    spec = build_probe_spec(
        geometry, families=TRAIN_FAMILIES, multipliers=(0.5,), seed=5
    )
    report = degeneracy_report(
        geometry,
        initial_log_beta,
        spec,
        placement="relative",
        unit=unit,
        margin=MARGIN,
    )
    assert report["gradient_norm_log_beta"] < 1e-12
    assert report["rescale_spread"] < 1e-12


def test_degeneracy_report_finds_the_absolute_form_live(
    geometry: Geometry, initial_log_beta: np.ndarray, unit: float
) -> None:
    """The positive end of the same instrument.

    If this failed while the previous test passed, the report would be
    rejecting every objective rather than discriminating between them, and a
    negative M83 result would carry no information.
    """
    spec = build_probe_spec(
        geometry, families=TRAIN_FAMILIES, multipliers=MULTIPLIERS, seed=5
    )
    report = degeneracy_report(
        geometry,
        initial_log_beta,
        spec,
        placement="absolute",
        unit=unit,
        margin=MARGIN,
    )
    assert report["gradient_norm_log_beta"] > 1e-6
    assert report["rescale_spread"] > 1e-6


def test_a_saturated_hinge_is_not_mistaken_for_degeneracy(
    geometry: Geometry, initial_log_beta: np.ndarray, unit: float
) -> None:
    """Why the contract is read without the hinge.

    Probes placed far beyond every boundary are already rejected, so every
    relu is clamped and the objective's own gradient is exactly zero — the
    healthy arm looking identical to the broken one. The scale-blindness
    reading must survive that, and the saturation must be visible in the
    evidence rather than silently changing the verdict.
    """
    spec = build_probe_spec(
        geometry, families=TRAIN_FAMILIES, multipliers=SATURATED_MULTIPLIERS, seed=5
    )
    report = degeneracy_report(
        geometry,
        initial_log_beta,
        spec,
        placement="absolute",
        unit=unit,
        margin=MARGIN,
    )
    assert report["active_fraction"] == 0.0
    assert report["objective_gradient_norm_log_beta"] == 0.0
    assert report["gradient_norm_log_beta"] > 1e-6


def test_degeneracy_report_reads_the_probe_term_alone(
    geometry: Geometry,
    initial_log_beta: np.ndarray,
    unit: float,
    corpus: tuple[np.ndarray, np.ndarray],
) -> None:
    """Regression: including the known-class term would pass every arm.

    Real data points do not move when the radii do, so their term has a
    non-zero gradient under *any* placement rule. A degeneracy test run on the
    total loss would therefore certify v12's objective as live, which is the
    exact error M77 exists to prevent.
    """
    features, labels = corpus
    spec = build_probe_spec(
        geometry, families=RELATIVE_ONLY, multipliers=(0.5,), seed=5
    )
    parameter = torch.tensor(initial_log_beta, requires_grad=True)
    total = probe_term(
        geometry,
        parameter,
        spec,
        placement="relative",
        unit=unit,
        margin=MARGIN,
    ) + known_term(
        torch.as_tensor(features),
        torch.as_tensor(labels),
        geometry,
        parameter,
        margin=MARGIN,
    )
    (gradient,) = torch.autograd.grad(total, [parameter])
    assert float(torch.linalg.vector_norm(gradient)) > 1e-6

    report = degeneracy_report(
        geometry,
        initial_log_beta,
        spec,
        placement="relative",
        unit=unit,
        margin=MARGIN,
    )
    assert report["gradient_norm_log_beta"] < 1e-12


def test_degeneracy_needs_the_relative_probes_rebuilt_in_the_graph(
    geometry: Geometry, initial_log_beta: np.ndarray, unit: float
) -> None:
    """Regression on the subtlety that nearly produced a false positive.

    v12 regenerated its negatives from the live parameters every batch, so the
    extent in the displacement and the extent in the score were one tensor and
    cancelled. Freezing those probes as constants breaks the cancellation and
    makes the objective look healthy: the numerator stops moving while the
    denominator keeps moving, and the gradient becomes non-zero for a reason
    that has nothing to do with the supervision working. The relative arm must
    therefore rebuild its points inside the graph.
    """
    spec = build_probe_spec(
        geometry, families=TRAIN_FAMILIES, multipliers=(0.5,), seed=5
    )
    frozen = torch.as_tensor(
        probe_points_numpy(
            geometry, initial_log_beta, spec, placement="relative", unit=unit
        )
    )
    parameter = torch.tensor(initial_log_beta, requires_grad=True)
    scores = boundary_scores(
        frozen, geometry, parameter, classes=torch.as_tensor(spec.owners)
    )
    frozen_loss = torch.mean(torch.relu(1.0 + MARGIN - scores))
    (frozen_gradient,) = torch.autograd.grad(frozen_loss, [parameter])
    assert float(torch.linalg.vector_norm(frozen_gradient)) > 1e-6

    live = degeneracy_report(
        geometry,
        initial_log_beta,
        spec,
        placement="relative",
        unit=unit,
        margin=MARGIN,
    )
    assert live["gradient_norm_log_beta"] < 1e-12


# ---------------------------------------------------------------------------
# Phase B
# ---------------------------------------------------------------------------


def _train(
    geometry: Geometry,
    corpus: tuple[np.ndarray, np.ndarray],
    unit: float,
    *,
    placement: str,
    shuffle: bool = False,
    epochs: int = 6,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    features, labels = corpus
    spec = build_probe_spec(
        geometry, families=TRAIN_FAMILIES, multipliers=MULTIPLIERS, seed=7
    )
    if shuffle:
        spec = spec.with_owners(
            shuffled_owners(spec.owners, class_count=CLASS_COUNT, seed=99)
        )
    return train_boundary(
        features,
        labels,
        geometry,
        spec,
        placement=placement,
        unit=unit,
        epochs=epochs,
        batch_size=64,
        learning_rate=0.05,
        margin=MARGIN,
        probe_weight=1.0,
        seed=7,
    )


def test_phase_b_leaves_the_geometry_untouched(
    geometry: Geometry, corpus: tuple[np.ndarray, np.ndarray], unit: float
) -> None:
    """The curriculum's whole point: only the boundary may move."""
    before = (
        geometry.centers.copy(),
        geometry.bases.copy(),
        geometry.tangent_scales.copy(),
        geometry.residual_scales.copy(),
    )
    _train(geometry, corpus, unit, placement="absolute", epochs=2)
    assert np.array_equal(geometry.centers, before[0])
    assert np.array_equal(geometry.bases, before[1])
    assert np.array_equal(geometry.tangent_scales, before[2])
    assert np.array_equal(geometry.residual_scales, before[3])


def test_absolute_training_moves_the_boundary_shape(
    geometry: Geometry,
    corpus: tuple[np.ndarray, np.ndarray],
    unit: float,
    initial_log_beta: np.ndarray,
) -> None:
    final, history = _train(geometry, corpus, unit, placement="absolute")
    displacement = boundary_displacement(initial_log_beta, final)
    assert displacement["shape"] > 1e-4
    assert history[-1]["total"] <= history[0]["total"]


def test_the_v12_form_probe_term_cannot_change_the_outcome(
    geometry: Geometry, corpus: tuple[np.ndarray, np.ndarray], unit: float
) -> None:
    """The defect stated at its strongest: the term is not merely weak.

    Training the relative arm with the probe term fully weighted lands within
    floating-point noise of deleting the term outright, while the absolute arm
    moves the boundary by order one. Whatever v12's probe supervision cost to
    compute, it bought nothing.

    The residual is not exactly zero because Adam divides by the running root
    mean square of the gradient, so even a 1e-17 numerical remnant is rescaled
    into a visible step. That is worth knowing on its own: under an adaptive
    optimiser a dead term still produces movement, which is exactly why N83.1
    forbids treating movement as evidence.
    """
    features, labels = corpus
    spec = build_probe_spec(
        geometry, families=TRAIN_FAMILIES, multipliers=MULTIPLIERS, seed=7
    )

    def effect(placement: str) -> float:
        common = dict(
            placement=placement,
            unit=unit,
            epochs=4,
            batch_size=64,
            learning_rate=0.05,
            margin=MARGIN,
            seed=7,
        )
        weighted, _ = train_boundary(
            features, labels, geometry, spec, probe_weight=1.0, **common
        )
        deleted, _ = train_boundary(
            features, labels, geometry, spec, probe_weight=0.0, **common
        )
        return float(np.max(np.abs(weighted - deleted)))

    assert effect("relative") < 1e-12
    assert effect("absolute") > 1e-2


def test_training_is_deterministic_under_a_fixed_seed(
    geometry: Geometry, corpus: tuple[np.ndarray, np.ndarray], unit: float
) -> None:
    first, _ = _train(geometry, corpus, unit, placement="absolute", epochs=3)
    second, _ = _train(geometry, corpus, unit, placement="absolute", epochs=3)
    assert np.array_equal(first, second)


def test_displacement_separates_radius_from_shape() -> None:
    """A uniform inflation is pure radius and must survive as zero shape.

    Coverage matching removes exactly the radius component, so a shape term
    that leaked radius would let an arm claim movement the measurement later
    cancels.
    """
    initial = np.zeros((3, 5))
    inflated = initial + np.array([0.4, -0.2, 1.1])[:, None]
    displacement = boundary_displacement(initial, inflated)
    assert displacement["shape"] == pytest.approx(0.0, abs=1e-12)
    assert displacement["radius"] > 0.0

    reshaped = np.zeros((3, 5))
    reshaped[0] = np.array([1.0, -1.0, 1.0, -1.0, 0.0])
    displacement = boundary_displacement(initial, reshaped)
    assert displacement["shape"] > 0.0
    assert displacement["radius"] == pytest.approx(0.0, abs=1e-12)

    tilted = np.zeros((3, 5))
    tilted[0] = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    displacement = boundary_displacement(initial, tilted)
    assert displacement["radius"] == pytest.approx(
        float(np.linalg.norm([0.2, 0.0, 0.0]) / np.sqrt(3)), rel=1e-9
    )


# ---------------------------------------------------------------------------
# Matched coverage and the rejection operands
# ---------------------------------------------------------------------------


def test_matched_coverage_hits_the_requested_rate(
    geometry: Geometry,
    corpus: tuple[np.ndarray, np.ndarray],
    initial_log_beta: np.ndarray,
) -> None:
    """N83.3: no recall figure may be read off an unmatched operating point."""
    features, labels = corpus
    offsets = matched_coverage_offsets(
        features,
        labels,
        geometry,
        initial_log_beta,
        coverage=0.9,
        class_count=CLASS_COUNT,
    )
    matched = apply_offsets(initial_log_beta, offsets)
    assert acceptance_rate(features, labels, geometry, matched) == pytest.approx(
        0.9, abs=0.05
    )


def test_matched_coverage_holds_on_rows_it_did_not_see(
    geometry: Geometry,
    corpus: tuple[np.ndarray, np.ndarray],
    initial_log_beta: np.ndarray,
) -> None:
    """The conformal order statistic exists for the split, not for the fit.

    Reading coverage on the calibration rows themselves says nothing: any
    threshold can be made to hit a rate on the sample that chose it. The
    property N83.3 actually needs is that a *fresh* row is accepted at about
    the requested rate, which the empirical quantile undershoots at this sample
    size and the conformal statistic does not.

    A single split cannot show this. With 30 calibration rows the threshold is
    the 28th order statistic and its coverage is Beta(28, 3) distributed, so
    one split scatters by about five points for reasons that have nothing to do
    with the estimator. The guarantee is a statement about the expectation, so
    the expectation is what the test reads.
    """
    features, labels = corpus
    rates: list[float] = []
    for repeat in range(24):
        generator = np.random.default_rng(83900 + repeat)
        calibration: list[np.ndarray] = []
        held_out: list[np.ndarray] = []
        for label in range(CLASS_COUNT):
            shuffled = generator.permutation(np.flatnonzero(labels == label))
            calibration.append(shuffled[: PER_CLASS // 2])
            held_out.append(shuffled[PER_CLASS // 2 :])
        rows = np.concatenate(calibration)
        fresh = np.concatenate(held_out)
        offsets = matched_coverage_offsets(
            features[rows],
            labels[rows],
            geometry,
            initial_log_beta,
            coverage=0.9,
            class_count=CLASS_COUNT,
        )
        matched = apply_offsets(initial_log_beta, offsets)
        rates.append(
            acceptance_rate(features[fresh], labels[fresh], geometry, matched)
        )
    assert float(np.mean(rates)) == pytest.approx(0.9, abs=0.03)


# ---------------------------------------------------------------------------
# The domain-matched partition
# ---------------------------------------------------------------------------


def _skewed_corpus() -> tuple[np.ndarray, np.ndarray]:
    """A corpus with the defect the real one has: one domain stored first.

    Every class holds 40 rows of domain 3 followed by a handful of each other
    domain, so a positional split hands the evaluation set nothing but the
    tail domains.
    """
    labels: list[int] = []
    domains: list[int] = []
    for label in range(CLASS_COUNT):
        owned = [3] * 40 + [0] * 8 + [1] * 6 + [2] * 4 + [4] * 2
        labels.extend([label] * len(owned))
        domains.extend(owned)
    return np.asarray(labels, dtype=np.int64), np.asarray(domains, dtype=np.int64)


def test_a_positional_split_would_have_missed_the_dominant_domain() -> None:
    """The defect this partition exists to remove, stated as a test."""
    labels, domains = _skewed_corpus()
    tail = np.concatenate(
        [np.flatnonzero(labels == label)[50:] for label in range(CLASS_COUNT)]
    )
    assert np.count_nonzero(domains[tail] == 3) == 0


def test_the_partition_matches_the_requested_domain_profile() -> None:
    labels, domains = _skewed_corpus()
    quota = (2, 1, 1, 5, 1, 0)
    fit_rows, evaluation_rows, report = domain_matched_partition(
        labels, domains, quota=quota, fit_per_class=50
    )
    assert report["unmet_by_domain"] == {}
    assert report["maximum_profile_deviation"] == pytest.approx(0.0, abs=1e-12)
    counts = [
        int(np.count_nonzero(domains[evaluation_rows] == domain)) for domain in range(6)
    ]
    assert counts == [value * CLASS_COUNT for value in quota]
    assert len(fit_rows) == 50 * CLASS_COUNT
    assert not set(fit_rows.tolist()) & set(evaluation_rows.tolist())


def test_the_partition_borrows_across_classes_when_a_class_is_short() -> None:
    """A class that cannot supply a domain is covered by one that can.

    Class 0 holds no paintings at all. Rather than leave the painting total
    short, another class supplies two, and class 0 spends its freed slot on a
    domain it does have. The per-class mixture drifts; the aggregate profile,
    which is the thing N83.2 is about, comes out exact.
    """
    labels: list[int] = []
    domains: list[int] = []
    for label in range(CLASS_COUNT):
        owned = [3] * 40 + [0] * 8 + [1] * 6 + [4] * 6
        if label != 0:
            owned += [2] * 6
        labels.extend([label] * len(owned))
        domains.extend(owned)
    label_array = np.asarray(labels, dtype=np.int64)
    domain_array = np.asarray(domains, dtype=np.int64)

    _, evaluation_rows, report = domain_matched_partition(
        label_array, domain_array, quota=(2, 1, 1, 5, 3, 0), fit_per_class=48
    )
    assert report["unmet_by_domain"] == {}
    assert report["maximum_profile_deviation"] == pytest.approx(0.0, abs=1e-12)
    assert int(np.count_nonzero(domain_array[evaluation_rows] == 2)) == CLASS_COUNT
    # Class 0 supplied no paintings and spent the slot on a domain it has;
    # a class with paintings to spare supplied two.
    owned = evaluation_rows[label_array[evaluation_rows] == 0]
    assert int(np.count_nonzero(domain_array[owned] == 2)) == 0
    assert len(owned) == 12
    assert (
        max(
            int(
                np.count_nonzero(
                    domain_array[
                        evaluation_rows[label_array[evaluation_rows] == label]
                    ]
                    == 2
                )
            )
            for label in range(1, CLASS_COUNT)
        )
        == 2
    )


def test_the_partition_reports_what_it_could_not_fill() -> None:
    """A shortfall is recorded, never quietly papered over."""
    labels, domains = _skewed_corpus()
    _, evaluation_rows, report = domain_matched_partition(
        labels, domains, quota=(2, 1, 1, 5, 3, 0), fit_per_class=48
    )
    assert report["unmet_by_domain"] == {"4": CLASS_COUNT}
    assert int(np.count_nonzero(domains[evaluation_rows] == 4)) == 2 * CLASS_COUNT
    assert report["maximum_profile_deviation"] > 0.0


def test_the_halves_keep_the_domain_profile_the_partition_bought() -> None:
    """Both halves must carry the mixture, not just the union of them.

    The quota here is deliberately odd in four of the five domains, which is
    the shape the real corpus has and the one a naive floor gets wrong.
    """
    labels, domains = _skewed_corpus()
    _, evaluation_rows, report = domain_matched_partition(
        labels, domains, quota=(3, 1, 1, 5, 2, 0), fit_per_class=48
    )
    assert report["unmet_by_domain"] == {}
    calibration, report_rows = domain_stratified_halves(
        labels, domains, evaluation_rows
    )
    assert not set(calibration.tolist()) & set(report_rows.tolist())
    assert len(calibration) == len(report_rows) == len(evaluation_rows) // 2
    for domain in range(6):
        counts = [
            int(np.count_nonzero(domains[half] == domain))
            for half in (calibration, report_rows)
        ]
        assert counts[0] == counts[1]


def test_matched_coverage_cancels_a_pure_radius_change(
    geometry: Geometry,
    corpus: tuple[np.ndarray, np.ndarray],
    initial_log_beta: np.ndarray,
) -> None:
    """Why shape is the honest operand.

    Two boundaries differing only in radius are the *same* boundary once
    coverage is matched, so any claim resting on radius movement is a claim the
    measurement erases.
    """
    features, labels = corpus
    inflated = initial_log_beta + 0.7
    first = apply_offsets(
        initial_log_beta,
        matched_coverage_offsets(
            features,
            labels,
            geometry,
            initial_log_beta,
            coverage=0.9,
            class_count=CLASS_COUNT,
        ),
    )
    second = apply_offsets(
        inflated,
        matched_coverage_offsets(
            features,
            labels,
            geometry,
            inflated,
            coverage=0.9,
            class_count=CLASS_COUNT,
        ),
    )
    assert first == pytest.approx(second, abs=1e-9)


def test_rejection_recall_counts_only_rows_no_class_accepts(
    geometry: Geometry, initial_log_beta: np.ndarray
) -> None:
    """A row any class accepts is not rejected, however far it is from most."""
    near = geometry.centers[2][None, :]
    far = (geometry.centers.mean(axis=0) + 500.0)[None, :]
    generous = initial_log_beta + 2.0
    assert rejection_recall(near, geometry, generous)["rejection_recall"] == 0.0
    assert rejection_recall(far, geometry, initial_log_beta)["rejection_recall"] == 1.0

    _, scores = minimum_scores_numpy(
        np.concatenate([near, far]), geometry, initial_log_beta
    )
    assert scores[0] < scores[1]


def test_rejection_recall_reports_every_domain(
    geometry: Geometry, initial_log_beta: np.ndarray
) -> None:
    """N83.2 forbids quoting the aggregate alone."""
    rows = np.repeat(geometry.centers, 2, axis=0)
    domains = np.tile(np.array([0, 3]), CLASS_COUNT)
    report = rejection_recall(rows, geometry, initial_log_beta, domains=domains)
    assert set(report["per_domain"]) == {"0", "1", "2", "3", "4", "5"}
    assert report["per_domain"]["0"]["row_count"] == CLASS_COUNT
    assert report["per_domain"]["1"]["rejection_recall"] is None


def test_empty_inputs_report_none_so_evidence_can_serialise(
    geometry: Geometry, initial_log_beta: np.ndarray
) -> None:
    """Regression from M82: nan cannot survive canonical JSON.

    Sketch contributed 158 of 73,728 corpus rows and no atom was dominated by
    it, which produced a nan that killed M82's payload hash twelve minutes into
    a run. Empty groups must report None.
    """
    report = rejection_recall(
        np.zeros((0, DIMENSION)),
        geometry,
        initial_log_beta,
        domains=np.zeros(0, dtype=np.int64),
    )
    assert report["rejection_recall"] is None
    json.dumps(report, allow_nan=False)


def test_held_out_families_are_disjoint_from_the_training_families() -> None:
    """The generalisation operand is only meaningful if the probes are unseen."""
    assert not set(HELD_OUT_FAMILIES) & set(TRAIN_FAMILIES)


def test_probe_rejection_reads_the_owner_not_the_nearest_class(
    geometry: Geometry, initial_log_beta: np.ndarray, unit: float
) -> None:
    """A probe is a negative *for its owner*, whoever else might accept it."""
    spec = build_probe_spec(
        geometry, families=HELD_OUT_FAMILIES, multipliers=(4.0,), seed=13
    )
    tight = initial_log_beta - 3.0
    assert (
        probe_rejection(geometry, tight, spec, placement="absolute", unit=unit) == 1.0
    )
    generous = initial_log_beta + 6.0
    assert (
        probe_rejection(geometry, generous, spec, placement="absolute", unit=unit)
        == 0.0
    )


def test_shuffled_owners_preserve_count_and_change_assignment(
    geometry: Geometry,
) -> None:
    """N83.1's null must differ only in the correspondence it destroys."""
    spec = build_probe_spec(
        geometry, families=TRAIN_FAMILIES, multipliers=MULTIPLIERS, seed=17
    )
    shuffled = shuffled_owners(spec.owners, class_count=CLASS_COUNT, seed=17)
    assert len(shuffled) == len(spec.owners)
    assert shuffled.min() >= 0 and shuffled.max() < CLASS_COUNT
    assert not np.array_equal(shuffled, spec.owners)
    reassigned = spec.with_owners(shuffled)
    assert np.array_equal(reassigned.directions, spec.directions)
    assert np.array_equal(reassigned.multipliers, spec.multipliers)


# ---------------------------------------------------------------------------
# N83.8 — the probe ladder has to be capable of being a negative
# ---------------------------------------------------------------------------


def test_the_tangent_unit_places_every_probe_inside_the_known_cloud(
    corpus: tuple[np.ndarray, np.ndarray],
    geometry: Geometry,
    initial_log_beta: np.ndarray,
) -> None:
    """N83.8. The defect that voided M83.1, stated as a test.

    ``global_scale_unit`` is the median fitted tangent scale, a per-direction
    spread inside a rank-4 subspace of 24 dimensions. The distance from a row
    to its own centroid is dominated by the 20 dimensions that subspace does
    not describe, so the unit is several times smaller than the cloud it is
    meant to reach past, and a ladder built on it never leaves the data.
    """
    features, labels = corpus
    spec = build_probe_spec(
        geometry, families=TRAIN_FAMILIES, multipliers=MULTIPLIERS, seed=17
    )
    report = probe_validity(
        features,
        labels,
        geometry,
        initial_log_beta,
        spec,
        unit=global_scale_unit(geometry),
    )
    assert report["reaches_past_known_cloud"] is False
    assert report["fraction_beyond_known_median"] == 0.0
    # Not marginal: the farthest probe is inside the nearest decile of the data.
    assert (
        report["probe_distance_maximum"] < report["known_distance_tenth_percentile"]
    )


def test_the_data_unit_reaches_past_the_known_cloud(
    corpus: tuple[np.ndarray, np.ndarray],
    geometry: Geometry,
    initial_log_beta: np.ndarray,
) -> None:
    """N83.8. The correction, held to the requirement it was chosen for."""
    features, labels = corpus
    spec = build_probe_spec(
        geometry, families=TRAIN_FAMILIES, multipliers=MULTIPLIERS, seed=17
    )
    report = probe_validity(
        features,
        labels,
        geometry,
        initial_log_beta,
        spec,
        unit=data_scale_unit(features, labels, geometry),
    )
    assert report["reaches_past_known_cloud"] is True
    assert report["fraction_beyond_known_median"] > 0.5
    assert (
        report["probe_distance_maximum"] > report["known_distance_median"]
    )


def test_the_data_unit_does_not_depend_on_the_learnable_radii(
    corpus: tuple[np.ndarray, np.ndarray], geometry: Geometry
) -> None:
    """N83.8. The one property the old unit had that must not be lost.

    v12's defect was a placement length that cancelled against the radius it
    was supervising. The replacement is a statistic of frozen data, so it is
    invariant to the radii by construction — checked rather than asserted in
    prose, because this is the property the whole milestone turns on.
    """
    features, labels = corpus
    unit = data_scale_unit(features, labels, geometry)
    inflated = Geometry(
        geometry.centers,
        geometry.bases,
        geometry.tangent_scales * 100.0,
        geometry.residual_scales * 100.0,
    )
    assert data_scale_unit(features, labels, inflated) == pytest.approx(unit)


def test_a_probe_ladder_inside_the_cloud_cannot_be_rejected(
    corpus: tuple[np.ndarray, np.ndarray],
    geometry: Geometry,
    initial_log_beta: np.ndarray,
) -> None:
    """N83.8. The geometric defect tied to the symptom it produced.

    M83.1 read zero held-out probe rejection from every arm, including the
    untrained one, and the gate could not see why. This is why: at any boundary
    matched to 90 percent known coverage, a ladder that stops short of the data
    is entirely interior, so its rejection rate is zero no matter what the
    boundary learned. The same ladder in the data unit is rejectable, which is
    what makes the figure an operand rather than a constant.
    """
    features, labels = corpus
    spec = build_probe_spec(
        geometry, families=TRAIN_FAMILIES, multipliers=MULTIPLIERS, seed=17
    )
    offsets = matched_coverage_offsets(
        features,
        labels,
        geometry,
        initial_log_beta,
        coverage=0.9,
        class_count=CLASS_COUNT,
    )
    matched = apply_offsets(initial_log_beta, offsets)

    inside = probe_rejection(
        geometry,
        matched,
        spec,
        placement="absolute",
        unit=global_scale_unit(geometry),
    )
    outside = probe_rejection(
        geometry,
        matched,
        spec,
        placement="absolute",
        unit=data_scale_unit(features, labels, geometry),
    )
    assert inside == 0.0
    assert outside > 0.2


def test_the_gate_voids_a_run_whose_probes_are_interior() -> None:
    """N83.8. The backstop, tested where M83.1 needed it.

    The module-level checks above cannot stop a runner from sealing evidence
    with a mis-scaled unit, which is precisely what happened. This asserts the
    gate short-circuits on the ladder before it reads any recall, and that it
    says void rather than negative — the distinction the whole correction turns
    on.
    """
    from experiments.tier4.eval_v13_m83_boundary import _gate

    config = {
        "gate": {
            "maximum_gradient_norm_for_degeneracy": 1e-9,
            "maximum_rescale_spread_for_degeneracy": 1e-9,
            "minimum_gradient_norm_for_live": 1e-6,
        },
        "arms": {},
    }
    interior = {
        "probe_validity": {
            "probe_distance_maximum": 8.61,
            "known_distance_tenth_percentile": 24.36,
            "known_distance_median": 30.19,
            "fraction_beyond_known_median": 0.0,
            "reaches_past_known_cloud": False,
        }
    }
    verdict = _gate([interior, interior], config)
    assert verdict["verdict"] == "probe_ladder_interior"
    assert verdict["probe_ladder"]["passes"] is False
    # Nothing below the ladder may be reported, the way N83.1 voids everything
    # below a failed degeneracy report.
    assert "rejection_recall" not in verdict
    assert "degeneracy" not in verdict


# ---------------------------------------------------------------------------
# M84 - real out-group exposure
# ---------------------------------------------------------------------------


OUT_GROUP_CLASSES = 8


@pytest.fixture(scope="module")
def out_group(corpus: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Classes the geometry never saw, sitting where the known ones sit.

    Placed as displaced neighbours of the corpus classes rather than drawn
    independently, because independent draws in this many dimensions land far
    outside every boundary and are rejected for free. M83 established that on
    the real corpus the out-of-set is *not* radially separable, so an out-group
    fixture that is trivially rejected would make every test below pass without
    exercising anything.
    """
    features, labels = corpus
    generator = np.random.default_rng(84841)
    centres = np.stack(
        [features[labels == label].mean(axis=0) for label in range(CLASS_COUNT)]
    )
    displacement = generator.normal(scale=0.25, size=(OUT_GROUP_CLASSES, DIMENSION))
    out_centres = centres[np.arange(OUT_GROUP_CLASSES) % CLASS_COUNT] + displacement
    scales = generator.uniform(0.3, 1.6, size=(OUT_GROUP_CLASSES, DIMENSION))
    out_features = np.concatenate(
        [
            out_centres[label]
            + scales[label] * generator.normal(size=(PER_CLASS, DIMENSION))
            for label in range(OUT_GROUP_CLASSES)
        ]
    )
    out_labels = np.repeat(np.arange(OUT_GROUP_CLASSES), PER_CLASS)
    return out_features.astype(np.float64), out_labels.astype(np.int64)


@pytest.fixture(scope="module")
def matched_log_beta(
    corpus: tuple[np.ndarray, np.ndarray],
    geometry: Geometry,
    initial_log_beta: np.ndarray,
) -> np.ndarray:
    """N84.6's initialisation: the fitted radii inflated to 90% coverage."""
    features, labels = corpus
    offsets = matched_coverage_offsets(
        features,
        labels,
        geometry,
        initial_log_beta,
        coverage=0.9,
        class_count=CLASS_COUNT,
    )
    return apply_offsets(initial_log_beta, offsets)


def test_the_exposure_term_is_inert_at_the_fitted_radii_and_live_at_coverage(
    out_group, geometry, initial_log_beta, matched_log_beta
):
    """N84.6. The defect that would have produced a false flat ladder."""
    features, _ = out_group
    raw = exposure_validity(features, geometry, initial_log_beta, margin=MARGIN)
    matched = exposure_validity(features, geometry, matched_log_beta, margin=MARGIN)
    assert raw["active_fraction"] == 0.0
    assert raw["term_is_live"] is False
    assert raw["already_rejected_fraction"] == 1.0
    assert matched["active_fraction"] > 0.5
    assert matched["term_is_live"] is True


def test_an_inert_exposure_term_has_exactly_no_gradient(
    out_group, geometry, initial_log_beta, matched_log_beta
):
    """The mechanism behind N84.6, read through autograd rather than a count.

    This is the test that would have caught a ladder trained at the raw fit.
    A hinge every negative already satisfies does not merely learn slowly, it
    receives identically zero gradient, so the run would have been flat and
    indistinguishable from a genuine negative.
    """
    features, _ = out_group

    def gradient_norm(log_beta: np.ndarray) -> float:
        parameter = torch.tensor(log_beta, dtype=torch.float64, requires_grad=True)
        owners = torch.as_tensor(
            exposure_owners(features, geometry, log_beta), dtype=torch.long
        )
        loss = exposure_term(
            torch.as_tensor(features, dtype=torch.float64),
            geometry,
            parameter,
            owners=owners,
            margin=MARGIN,
        )
        (grad,) = torch.autograd.grad(loss, [parameter], allow_unused=True)
        return 0.0 if grad is None else float(torch.linalg.vector_norm(grad))

    assert gradient_norm(initial_log_beta) == 0.0
    assert gradient_norm(matched_log_beta) > 1e-6


def test_exposure_owners_are_the_exact_argmin(out_group, geometry, matched_log_beta):
    """N84.7's contract: the owner is the boundary that would accept the row.

    Stated as exactness rather than as the 0.3065 nearest-centroid agreement
    measured on the real corpus. That figure is a property of DomainNet's
    anisotropy, and this fixture's classes are separated cleanly enough that
    the two rules coincide; contriving the fixture to disagree would test the
    fixture. What must hold everywhere is that the owner is the argmin.
    """
    features, _ = out_group
    owners = exposure_owners(features, geometry, matched_log_beta)
    best_class, _ = minimum_scores_numpy(features, geometry, matched_log_beta)
    assert np.array_equal(owners, best_class)
    scores = boundary_scores(
        torch.as_tensor(features, dtype=torch.float64),
        geometry,
        torch.as_tensor(matched_log_beta, dtype=torch.float64),
    ).numpy()
    assert np.array_equal(owners, np.argmin(scores, axis=1))


def test_owner_agreement_is_exact_when_nothing_moves(
    out_group, geometry, matched_log_beta
):
    features, _ = out_group
    owners = exposure_owners(features, geometry, matched_log_beta)
    assert owner_agreement(features, geometry, matched_log_beta, owners) == 1.0


def test_the_ladder_sampler_honours_both_axes(out_group):
    """A cell is a statement about count and diversity jointly."""
    _, labels = out_group
    rows = sample_exposure(labels, count=40, diversity=4, seed=11)
    assert len(rows) == 40
    assert len(np.unique(rows)) == 40
    drawn, counts = np.unique(labels[rows], return_counts=True)
    assert len(drawn) == 4
    assert set(counts.tolist()) == {10}
    assert len(sample_exposure(labels, count=0, diversity=1, seed=11)) == 0
    with pytest.raises(ValueError):
        sample_exposure(labels, count=10, diversity=4, seed=11)
    with pytest.raises(ValueError):
        sample_exposure(labels, count=40, diversity=OUT_GROUP_CLASSES + 1, seed=11)


def test_the_ladder_sampler_is_a_function_of_its_seed(out_group):
    _, labels = out_group
    first = sample_exposure(labels, count=40, diversity=4, seed=11)
    assert np.array_equal(first, sample_exposure(labels, count=40, diversity=4, seed=11))
    assert not np.array_equal(
        first, sample_exposure(labels, count=40, diversity=4, seed=23)
    )


def test_the_null_carries_the_moments_and_not_the_content(out_group):
    """N84.3. Same first and second moments, none of the rows."""
    features, _ = out_group
    null = moment_matched_negatives(features, count=8000, seed=11)
    assert null.shape == (8000, DIMENSION)
    # Read relative to the sample's own spread; an absolute tolerance would be
    # a statement about this fixture's scale rather than about the null.
    spread = np.linalg.norm(np.std(features, axis=0))
    mean_error = np.linalg.norm(np.mean(null, axis=0) - np.mean(features, axis=0))
    assert mean_error / spread < 0.05
    real_covariance = np.cov(features, rowvar=False)
    null_covariance = np.cov(null, rowvar=False)
    relative = np.linalg.norm(null_covariance - real_covariance) / np.linalg.norm(
        real_covariance
    )
    assert relative < 0.2
    # None of the null rows is one of the real ones.
    nearest = np.min(
        np.linalg.norm(null[:200, None, :] - features[None, :, :], axis=2), axis=1
    )
    assert np.min(nearest) > 1e-6


def test_exposure_training_rejects_an_out_group_the_geometry_admits(
    corpus, out_group, geometry, matched_log_beta
):
    """The instrument control M83 lacked until it was too late.

    Exposure supervision must be able to work where the geometry permits it.
    If this fails, a flat ladder on the real corpus would say nothing about
    exposure and everything about the objective being broken. The exposure
    classes and the evaluation classes are disjoint, as N84.1 requires.
    """
    features, labels = corpus
    out_features, out_labels = out_group
    expose = out_labels < OUT_GROUP_CLASSES // 2
    evaluate = ~expose
    negatives = out_features[expose]
    owners = exposure_owners(negatives, geometry, matched_log_beta)

    trained, history = train_exposure_boundary(
        features,
        labels,
        geometry,
        matched_log_beta,
        negatives=negatives,
        owners=owners,
        epochs=30,
        batch_size=64,
        exposure_batch_size=64,
        learning_rate=0.05,
        margin=MARGIN,
        exposure_weight=1.0,
        seed=11,
    )
    assert history[-1]["exposure"] < history[0]["exposure"]

    def recall(log_beta: np.ndarray, rows: np.ndarray) -> float:
        offsets = matched_coverage_offsets(
            features, labels, geometry, log_beta, coverage=0.9, class_count=CLASS_COUNT
        )
        return float(
            rejection_recall(out_features[rows], geometry, apply_offsets(log_beta, offsets))[
                "rejection_recall"
            ]
        )

    # Two separate claims, because they are separately falsifiable and only the
    # first is about the objective being wired correctly. On the classes it was
    # shown, the supervision must move the boundary substantially; on classes it
    # has never seen, it need only transfer in the right direction. Requiring
    # one threshold for both would have made this a test of how far a toy
    # corpus happens to generalise.
    #
    # The gains are modest by construction and that is the point: coverage
    # matching removes any uniform inflation afterwards, so only shape survives
    # to be measured. The in-sample bar sits at roughly half the effect this
    # fixture produces, so the test fails on the objective breaking rather than
    # on optimiser noise.
    assert recall(trained, expose) > recall(matched_log_beta, expose) + 0.05
    assert recall(trained, evaluate) > recall(matched_log_beta, evaluate)


def test_training_without_negatives_leaves_the_out_group_where_it_was(
    corpus, out_group, geometry, matched_log_beta
):
    """The known_only arm. N84.4 separates "no negatives" from "no training"."""
    features, labels = corpus
    out_features, _ = out_group
    trained, history = train_exposure_boundary(
        features,
        labels,
        geometry,
        matched_log_beta,
        negatives=np.empty((0, DIMENSION)),
        owners=np.empty(0, dtype=np.int64),
        epochs=30,
        batch_size=64,
        exposure_batch_size=64,
        learning_rate=0.05,
        margin=MARGIN,
        exposure_weight=1.0,
        seed=11,
    )
    assert all(entry["exposure"] == 0.0 for entry in history)

    def recall(log_beta: np.ndarray) -> float:
        offsets = matched_coverage_offsets(
            features, labels, geometry, log_beta, coverage=0.9, class_count=CLASS_COUNT
        )
        return float(
            rejection_recall(out_features, geometry, apply_offsets(log_beta, offsets))[
                "rejection_recall"
            ]
        )

    assert abs(recall(trained) - recall(matched_log_beta)) < 0.10


def test_anisotropy_reads_zero_on_equal_radii_and_rises_with_spread():
    """N84.5's descriptive operand, and its insensitivity to uniform radius."""
    equal = np.zeros((CLASS_COUNT, RANK + 1))
    assert tangent_anisotropy(equal) == 0.0
    assert tangent_anisotropy(equal + 5.0) == 0.0
    spread = equal.copy()
    spread[:, 0] = 2.0
    assert tangent_anisotropy(spread) > 0.5


def test_average_ranks_averages_ties() -> None:
    ranks = average_ranks(np.array([1.0, 2.0, 2.0, 5.0]))
    assert list(ranks) == [1.0, 2.5, 2.5, 4.0]


def test_score_auroc_is_one_when_separated() -> None:
    assert score_auroc(np.array([0.1, 0.2]), np.array([0.8, 0.9])) == 1.0


def test_score_auroc_is_zero_when_inverted() -> None:
    assert score_auroc(np.array([0.8, 0.9]), np.array([0.1, 0.2])) == 0.0


def test_score_auroc_is_half_when_tied() -> None:
    assert score_auroc(np.full(8, 3.0), np.full(5, 3.0)) == 0.5


def test_score_auroc_counts_ties_as_half() -> None:
    # One known below, one tied: 0.5 of the pair mass separates.
    assert score_auroc(np.array([0.0, 1.0]), np.array([1.0])) == 0.75


def test_score_auroc_matches_sklearn() -> None:
    from sklearn.metrics import roc_auc_score

    generator = np.random.default_rng(4)
    known = generator.normal(0.0, 1.0, size=200)
    unseen = generator.normal(0.6, 1.0, size=150)
    truth = np.concatenate([np.zeros(len(known)), np.ones(len(unseen))])
    assert score_auroc(known, unseen) == pytest.approx(
        roc_auc_score(truth, np.concatenate([known, unseen]))
    )


def test_score_auroc_requires_both_sides() -> None:
    with pytest.raises(ValueError):
        score_auroc(np.array([1.0]), np.array([]))


def test_domain_auroc_reports_absent_domains_as_none() -> None:
    report = domain_auroc(
        np.array([0.0, 0.0]),
        np.array([0, 0]),
        np.array([1.0, 1.0]),
        np.array([0, 0]),
        domain_count=3,
    )
    assert report["auroc"] == 1.0
    assert report["per_domain"]["0"]["auroc"] == 1.0
    assert report["per_domain"]["1"]["auroc"] is None


def test_domain_auroc_exceeds_pooled_under_a_domain_shift() -> None:
    # Domain 1 scores high on both sides. Pooling ranks its knowns above
    # domain 0's unseen rows, so the pooled figure falls below both parts.
    known = np.array([0.0, 0.1, 10.0, 10.1])
    unseen = np.array([0.2, 0.3, 10.2, 10.3])
    domains = np.array([0, 0, 1, 1])
    report = domain_auroc(known, domains, unseen, domains, domain_count=2)
    assert report["per_domain"]["0"]["auroc"] == 1.0
    assert report["per_domain"]["1"]["auroc"] == 1.0
    assert report["within_domain_auroc"] == 1.0
    assert report["auroc"] < report["within_domain_auroc"]


def test_far_field_points_sit_outside_the_data() -> None:
    generator = np.random.default_rng(7)
    features = generator.normal(size=(400, 16))
    points = far_field_points(features, count=32, multiplier=5.0, seed=1)
    centre = features.mean(axis=0)
    inner = np.median(np.linalg.norm(features - centre, axis=1))
    outer = np.linalg.norm(points - centre, axis=1)
    assert outer.min() > 4.0 * inner


def test_far_field_points_must_be_outside() -> None:
    with pytest.raises(ValueError):
        far_field_points(np.zeros((4, 3)), count=2, multiplier=1.0, seed=1)
