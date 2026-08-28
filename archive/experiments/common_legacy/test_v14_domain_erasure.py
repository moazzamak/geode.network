"""Tests for M90.2's erasers, probe and geometric instrument.

The failure mode here is an erasure that looks applied but is not, which would
read as "domain removed, nothing improved" and quietly refute a hypothesis on a
no-op. So the erasers are tested against their defining property -- no linear
classifier can recover the concept -- rather than against a shape check.
"""

from __future__ import annotations

import numpy as np

from experiments.tier4.eval_v14_m90_2_domain_erasure import (
    _h94,
    dominance_ratio,
    domain_probe,
    erasure_certificate,
    leace_eraser,
    random_direction_remover,
    top_direction_remover,
    zca_whitener,
)


def _domain_shifted(
    *, groups: int = 3, per_group: int = 400, dimension: int = 12, seed: int = 0
):
    """Features whose dominant axis is the group, with a weaker second signal."""
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0.0, 8.0, (groups, dimension))
    assignments = np.repeat(np.arange(groups), per_group).astype(np.int64)
    features = rng.normal(0.0, 1.0, (groups * per_group, dimension))
    return features + offsets[assignments], assignments


def test_leace_removes_all_linear_predictability_of_the_concept() -> None:
    features, groups = _domain_shifted()
    eraser, removed = leace_eraser(features, groups, group_count=3, floor=1e-10, singular_tolerance=1e-10)
    erased = eraser(features)
    centred = erased - erased.mean(axis=0)
    concept = np.zeros((len(groups), 3))
    concept[np.arange(len(groups)), groups] = 1.0
    concept -= concept.mean(axis=0)
    cross = centred.T @ concept / len(features)
    assert removed == 2  # a centred one-hot over 3 groups has rank 2
    assert np.abs(cross).max() < 1e-8


def test_leace_leaves_the_group_means_coincident() -> None:
    features, groups = _domain_shifted()
    eraser, _ = leace_eraser(features, groups, group_count=3, floor=1e-10, singular_tolerance=1e-10)
    erased = eraser(features)
    means = np.stack([erased[groups == g].mean(axis=0) for g in range(3)])
    assert np.abs(means - means.mean(axis=0)).max() < 1e-8


def test_leace_is_an_affine_map_not_a_per_row_recipe() -> None:
    """It must apply to rows whose group label is unknown, which is the point."""
    features, groups = _domain_shifted()
    eraser, _ = leace_eraser(features, groups, group_count=3, floor=1e-10, singular_tolerance=1e-10)
    fresh = np.random.default_rng(99).normal(0.0, 1.0, (5, features.shape[1]))
    combined = eraser(np.vstack([features, fresh]))
    assert np.allclose(combined[: len(features)], eraser(features))
    assert np.allclose(combined[len(features) :], eraser(fresh))


def test_leace_preserves_what_is_orthogonal_to_the_concept() -> None:
    """Minimal edit: a direction uncorrelated with the concept must survive."""
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(2), 500).astype(np.int64)
    features = rng.normal(0.0, 1.0, (1000, 6))
    features[:, 0] += np.where(groups == 0, -5.0, 5.0)
    eraser, _ = leace_eraser(features, groups, group_count=2, floor=1e-10, singular_tolerance=1e-10)
    erased = eraser(features)
    kept = np.corrcoef(features[:, 3], erased[:, 3])[0, 1]
    assert kept > 0.99


def test_top_direction_removal_drops_the_dominant_axes() -> None:
    features, _ = _domain_shifted()
    remover, count = top_direction_remover(features, count=2)
    reduced = remover(features)
    original_rank_energy = np.linalg.svd(features - features.mean(axis=0))[1]
    reduced_energy = np.linalg.svd(reduced - reduced.mean(axis=0))[1]
    assert count == 2
    assert reduced_energy[0] < original_rank_energy[2] + 1e-8


def test_random_direction_removal_matches_the_count_but_not_the_choice() -> None:
    features, _ = _domain_shifted()
    top, _ = top_direction_remover(features, count=2)
    null, _ = random_direction_remover(features, count=2, seed=5)
    assert np.linalg.matrix_rank(top.matrix) == np.linalg.matrix_rank(null.matrix)
    assert not np.allclose(top.matrix, null.matrix)


def test_random_direction_removal_is_deterministic_for_a_seed() -> None:
    features, _ = _domain_shifted()
    first, _ = random_direction_remover(features, count=2, seed=5)
    second, _ = random_direction_remover(features, count=2, seed=5)
    other, _ = random_direction_remover(features, count=2, seed=6)
    assert np.allclose(first.matrix, second.matrix)
    assert not np.allclose(first.matrix, other.matrix)


def test_whitening_makes_the_pooled_covariance_the_identity() -> None:
    features, _ = _domain_shifted()
    whitener, _ = zca_whitener(features, floor=1e-10)
    whitened = whitener(features)
    covariance = np.cov(whitened.T, bias=True)
    assert np.abs(covariance - np.eye(features.shape[1])).max() < 1e-6


def test_the_probe_reads_domain_before_erasure_and_not_after() -> None:
    features, groups = _domain_shifted(per_group=300, seed=11)
    split = np.arange(len(groups)) % 2 == 0
    before = domain_probe(
        features[split],
        groups[split],
        features[~split],
        groups[~split],
        max_iterations=200,
        regularisation=1.0,
        seed=1,
    )
    eraser, _ = leace_eraser(features[split], groups[split], group_count=3, floor=1e-10, singular_tolerance=1e-10)
    after = domain_probe(
        eraser(features[split]),
        groups[split],
        eraser(features[~split]),
        groups[~split],
        max_iterations=200,
        regularisation=1.0,
        seed=1,
    )
    assert before["balanced_accuracy"] > 0.9
    assert after["balanced_accuracy"] < 0.5


def test_dominance_ratio_is_none_when_no_class_spans_two_domains() -> None:
    separation = {
        "domain_dominates_class": {
            "own_class_sibling_cell_gap_median": None,
            "nearest_foreign_class_cell_gap_median": 19.0,
        }
    }
    assert dominance_ratio(separation) is None


def test_dominance_ratio_divides_sibling_by_foreign() -> None:
    separation = {
        "domain_dominates_class": {
            "own_class_sibling_cell_gap_median": 32.295,
            "nearest_foreign_class_cell_gap_median": 19.153,
        }
    }
    assert abs(dominance_ratio(separation) - 32.295 / 19.153) < 1e-12


def _arm(ratio: float | None, probe: float) -> dict:
    return {"dominance_ratio": ratio, "probe": {"balanced_accuracy": probe}}


GATE = {"dominance_ratio_bar": 1.0}
PROBE_BAR = 0.25


def test_the_null_removes_the_same_budget_as_the_arm_it_controls() -> None:
    """N90.2.17. A random grouping has no signal, so a tolerance cannot size it."""
    features, groups = _domain_shifted(groups=6, per_group=300, seed=17)
    rng = np.random.default_rng(3)
    scrambled = rng.permutation(groups)
    _, real = leace_eraser(
        features, groups, group_count=6, floor=1e-10, singular_tolerance=1e-10
    )
    _, null = leace_eraser(
        features, scrambled, group_count=6, floor=1e-10, singular_tolerance=1e-10
    )
    assert real == 5
    assert null == real


def test_the_erasure_survives_a_float32_corpus() -> None:
    """N90.2.16. Fitting in float32 degraded LEACE's exact guarantee 200-fold."""
    features, groups = _domain_shifted(dimension=64, per_group=600, seed=13)
    single = features.astype(np.float32)
    eraser, _ = leace_eraser(
        single, groups, group_count=3, floor=1e-10, singular_tolerance=1e-10
    )
    erased = eraser(single)
    assert erased.dtype == np.float32  # downstream must see the corpus dtype
    certificate = erasure_certificate(single, erased, groups, group_count=3)
    assert certificate["first_moment_erased"] is True
    assert certificate["first_moment_residual_fraction"] < 1e-5


def test_the_certificate_separates_a_real_erasure_from_a_no_op() -> None:
    """A refuted H94 must not be readable as a misapplied eraser."""
    features, groups = _domain_shifted()
    eraser, _ = leace_eraser(
        features, groups, group_count=3, floor=1e-10, singular_tolerance=1e-10
    )
    erased = erasure_certificate(features, eraser(features), groups, group_count=3)
    untouched = erasure_certificate(features, features, groups, group_count=3)
    assert erased["first_moment_erased"] is True
    assert untouched["first_moment_erased"] is False
    assert (
        erased["max_pairwise_domain_mean_gap_after"]
        < erased["max_pairwise_domain_mean_gap_before"]
    )


def test_the_certificate_reports_surviving_second_moment_structure() -> None:
    """Equal means with unequal spreads is the case LEACE cannot reach."""
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(2), 500).astype(np.int64)
    features = rng.normal(0.0, 1.0, (1000, 5))
    features[groups == 1] *= 4.0  # same mean, four times the spread
    certificate = erasure_certificate(features, features, groups, group_count=2)
    assert certificate["second_moment_variance_ratio"] > 5.0


def test_h94_needs_both_bars_and_both_nulls() -> None:
    assert _h94(_arm(0.8, 0.18), _arm(1.4, 0.9), GATE, probe_bar=PROBE_BAR)[
        "removes_domain_dominance"
    ]
    # ratio bar cleared, probe bar not
    assert not _h94(_arm(0.8, 0.40), _arm(1.4, 0.9), GATE, probe_bar=PROBE_BAR)[
        "removes_domain_dominance"
    ]
    # both bars cleared but the null does just as well, so the arm proved nothing
    assert not _h94(_arm(0.8, 0.18), _arm(0.7, 0.15), GATE, probe_bar=PROBE_BAR)[
        "removes_domain_dominance"
    ]


def test_h94_is_not_awarded_on_a_missing_ratio() -> None:
    result = _h94(_arm(None, 0.18), _arm(1.4, 0.9), GATE, probe_bar=PROBE_BAR)
    assert result["removes_domain_dominance"] is False
    assert result["dominance_ratio_below_bar"] is False


def test_the_probe_reports_convergence() -> None:
    """An unconverged probe understates domain signal, so it must be visible."""
    features, groups = _domain_shifted(per_group=200, seed=21)
    split = np.arange(len(groups)) % 2 == 0
    starved = domain_probe(
        features[split],
        groups[split],
        features[~split],
        groups[~split],
        max_iterations=1,
        regularisation=1.0,
        seed=1,
    )
    ample = domain_probe(
        features[split],
        groups[split],
        features[~split],
        groups[~split],
        max_iterations=2000,
        regularisation=1.0,
        seed=1,
    )
    assert starved["converged"] is False
    assert ample["converged"] is True


def test_chance_follows_the_domains_actually_present() -> None:
    """The report rows carry no sketch, so chance is 1/5 rather than 1/6."""
    features, groups = _domain_shifted(groups=6, per_group=200, seed=31)
    present = groups != 5
    result = domain_probe(
        features,
        groups,
        features[present],
        groups[present],
        max_iterations=2000,
        regularisation=1.0,
        seed=1,
    )
    assert result["domains_present"] == 5
    assert abs(result["chance_balanced_accuracy"] - 0.2) < 1e-12
