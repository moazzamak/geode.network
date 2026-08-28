"""Tests for the M82 naming channel.

The positive controls matter more than the unit checks here. A naming channel
that silently returns something plausible for every atom would pass a suite of
shape assertions, so each behaviour that M82's conclusions depend on is pinned
by a test that fails if the behaviour is faked: a planted name must be
recovered, a shuffled null must collapse, an unnameable atom must report itself
as unnameable rather than take a default term, and the gating purity control
must refuse to produce a number when nothing is eligible.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.common.v13_naming import (
    atom_class_purity,
    atom_exemplars,
    domain_breakdown,
    far_field_rate,
    grouped_explanation,
    matched_random_grouping,
    name_atoms,
    names_to_groups,
    naming_agreement,
    purity_positive_control,
    shuffled_exemplars,
    sparse_atom_exemplars,
    split_exemplars,
)


@pytest.fixture
def codes() -> np.ndarray:
    """Sparse codes where atom a fires hardest on rows of class a."""
    generator = np.random.default_rng(31337)
    rows, atoms = 400, 8
    values = np.zeros((rows, atoms), dtype=np.float32)
    for row in range(rows):
        atom = row % atoms
        values[row, atom] = 1.0 + generator.random()
        values[row, (atom + 1) % atoms] = 0.25 * generator.random()
    return values


@pytest.fixture
def labels() -> np.ndarray:
    return np.array([row % 8 for row in range(400)], dtype=np.int64)


def test_exemplars_are_the_strongest_activations(codes: np.ndarray) -> None:
    exemplars = atom_exemplars(codes, top_count=10)
    for atom in range(codes.shape[1]):
        rows = exemplars.rows[atom]
        assert len(rows) == 10
        taken = codes[rows, atom]
        untaken = np.delete(codes[:, atom], rows)
        assert taken.min() >= untaken[untaken > 0].max()


def test_dead_atom_yields_no_exemplars() -> None:
    values = np.zeros((16, 3), dtype=np.float32)
    values[:, 0] = 1.0
    exemplars = atom_exemplars(values, top_count=4)
    assert len(exemplars.rows[0]) == 4
    assert len(exemplars.rows[1]) == 0
    assert exemplars.live().tolist() == [True, False, False]


def test_exemplar_halves_are_disjoint_and_balanced(codes: np.ndarray) -> None:
    exemplars = atom_exemplars(codes, top_count=10)
    first, second = split_exemplars(exemplars, seed=7)
    for atom in range(codes.shape[1]):
        assert len(first[atom]) == len(second[atom]) == 5
        assert not set(first[atom].tolist()) & set(second[atom].tolist())
        assert set(first[atom].tolist()) <= set(exemplars.rows[atom].tolist())


def test_shuffled_null_preserves_set_sizes(codes: np.ndarray) -> None:
    exemplars = atom_exemplars(codes, top_count=10)
    shuffled = shuffled_exemplars(exemplars, seed=11)
    for atom in range(codes.shape[1]):
        assert len(shuffled[atom]) == len(exemplars.rows[atom])


def test_naming_recovers_a_planted_name(codes: np.ndarray, labels: np.ndarray) -> None:
    """Positive control: if atom a's exemplars sit on term a, it must be named a."""
    generator = np.random.default_rng(5)
    terms = generator.normal(size=(8, 32)).astype(np.float32)
    terms /= np.linalg.norm(terms, axis=1, keepdims=True)
    images = terms[labels] + 0.01 * generator.normal(size=(400, 32)).astype(np.float32)
    images /= np.linalg.norm(images, axis=1, keepdims=True)

    exemplars = atom_exemplars(codes, top_count=10)
    names, scores = name_atoms(exemplars.rows, images, terms)
    assert names.tolist() == list(range(8))
    assert np.all(scores > 0.9)


def test_unnameable_atom_reports_itself_rather_than_defaulting() -> None:
    values = np.zeros((16, 2), dtype=np.float32)
    values[:, 0] = 1.0
    exemplars = atom_exemplars(values, top_count=4)
    terms = np.eye(3, dtype=np.float32)
    images = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (16, 1))
    names, scores = name_atoms(exemplars.rows, images, terms)
    assert names[0] == 0
    assert names[1] == -1
    assert np.isnan(scores[1])


def test_agreement_is_one_for_identical_and_low_for_unrelated() -> None:
    first = np.arange(50, dtype=np.int64)
    assert naming_agreement(first, first.copy())["agreement"] == 1.0
    generator = np.random.default_rng(3)
    unrelated = generator.permutation(50).astype(np.int64)
    assert naming_agreement(first, unrelated)["agreement"] < 0.2


def test_agreement_ignores_atoms_neither_could_name() -> None:
    first = np.array([1, -1, 2, -1], dtype=np.int64)
    second = np.array([1, 3, 2, -1], dtype=np.int64)
    result = naming_agreement(first, second)
    assert result["comparable_atoms"] == 2
    assert result["agreement"] == 1.0


def test_far_field_rate_counts_absent_terms() -> None:
    names = np.array([0, 5, 130, 200, -1], dtype=np.int64)
    result = far_field_rate(names, in_corpus_terms=128)
    assert result["named_atoms"] == 4
    assert result["false_naming_rate"] == pytest.approx(0.5)


def test_far_field_rate_separates_style_names_from_absent_objects() -> None:
    """Regression. The first M82 run reported an 85.86 percent false-naming
    rate, but 82 percent of atoms were named by a style term, which sits above
    the in-corpus object terms by index and was therefore counted as an absent
    object. An atom named after a rendering style the corpus does contain has
    not been misnamed after something absent, and the two must be reported
    apart. The raw rate is kept so the earlier figure stays reconstructible."""
    names = np.array([0, 5, 200, 346, 349], dtype=np.int64)
    result = far_field_rate(names, in_corpus_terms=128, style_terms_start=345)
    assert result["false_naming_rate"] == pytest.approx(0.6)
    assert result["style_named_atoms"] == 2
    assert result["object_named_atoms"] == 3
    assert result["object_false_naming_rate"] == pytest.approx(1 / 3)


def test_purity_control_scores_a_pure_atom(codes: np.ndarray, labels: np.ndarray) -> None:
    exemplars = atom_exemplars(codes, top_count=10)
    dominant, purity = atom_class_purity(exemplars.rows, labels, class_count=8)
    assert purity.min() == 1.0
    control = purity_positive_control(
        dominant, exemplars.rows, labels, class_count=8, purity_threshold=0.75
    )
    assert control["eligible_atoms"] == 8
    assert control["accuracy"] == 1.0


def test_purity_control_refuses_to_score_when_nothing_is_eligible(
    codes: np.ndarray, labels: np.ndarray
) -> None:
    """A control with no eligible atoms must report nan, not a flattering number."""
    exemplars = atom_exemplars(codes, top_count=10)
    names = np.full(8, -1, dtype=np.int64)
    control = purity_positive_control(
        names, exemplars.rows, labels, class_count=8, purity_threshold=0.75
    )
    assert control["eligible_atoms"] == 0
    assert control["accuracy"] is None


def test_purity_control_detects_a_wrong_name(
    codes: np.ndarray, labels: np.ndarray
) -> None:
    exemplars = atom_exemplars(codes, top_count=10)
    dominant, _ = atom_class_purity(exemplars.rows, labels, class_count=8)
    shifted = (dominant + 1) % 8
    control = purity_positive_control(
        shifted, exemplars.rows, labels, class_count=8, purity_threshold=0.75
    )
    assert control["accuracy"] == 0.0


def test_domain_breakdown_attributes_atoms_to_their_domain(
    codes: np.ndarray, labels: np.ndarray
) -> None:
    domains = np.where(labels < 4, 0, 3).astype(np.int64)
    exemplars = atom_exemplars(codes, top_count=10)
    dominant, _ = atom_class_purity(exemplars.rows, labels, class_count=8)
    breakdown = domain_breakdown(
        dominant,
        exemplars.rows,
        labels,
        domains,
        class_count=8,
        domain_count=6,
        purity_threshold=0.75,
    )
    assert breakdown["0"]["atoms_in_domain"] == 4
    assert breakdown["3"]["atoms_in_domain"] == 4
    assert breakdown["0"]["accuracy"] == 1.0
    assert breakdown["1"]["eligible_atoms"] == 0


def test_empty_results_are_none_so_the_evidence_hash_can_serialise_them(
    codes: np.ndarray, labels: np.ndarray
) -> None:
    """Regression. The first M82 run computed for twelve minutes and then died
    in ``payload_hash``, because DomainNet's sketch domain contributes 158 of
    73,728 images and no atom is dominated by it, so its per-domain accuracy
    was ``nan`` and canonical JSON refuses non-finite floats. Empty results are
    ``None``, which is also the more honest encoding: nothing was measurable,
    which is not the same as measuring zero."""
    exemplars = atom_exemplars(codes, top_count=10)
    unnamed = np.full(8, -1, dtype=np.int64)
    payload = {
        "per_domain": domain_breakdown(
            unnamed,
            exemplars.rows,
            labels,
            np.zeros(len(labels), dtype=np.int64),
            class_count=8,
            domain_count=6,
            purity_threshold=0.75,
        ),
        "purity_control": purity_positive_control(
            unnamed, exemplars.rows, labels, class_count=8, purity_threshold=0.75
        ),
        "far_field": far_field_rate(unnamed, in_corpus_terms=8),
        "agreement": naming_agreement(unnamed, unnamed),
    }
    json.dumps(payload, allow_nan=False)


def _sparsify(codes: np.ndarray, slots: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-codes, axis=1, kind="stable")[:, :slots]
    return order.astype(np.int64), np.take_along_axis(codes, order, axis=1)


def test_sparse_exemplars_match_the_dense_form(codes: np.ndarray) -> None:
    """The sparse path exists only to avoid a 1.2 GB dense matrix; it must not
    change a single exemplar."""
    indices, values = _sparsify(codes, 2)
    dense = atom_exemplars(codes, top_count=10)
    sparse = sparse_atom_exemplars(
        indices, values, dictionary_size=codes.shape[1], top_count=10
    )
    for atom in range(codes.shape[1]):
        assert sparse.rows[atom].tolist() == dense.rows[atom].tolist()
        assert np.allclose(sparse.activations[atom], dense.activations[atom])


def test_grouped_explanation_pools_contributions_by_group() -> None:
    indices = np.array([[0, 1, 2]], dtype=np.int64)
    contributions = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    groups = np.array([0, 0, 1], dtype=np.int64)
    built = grouped_explanation(indices, contributions, groups, group_count=2)
    assert built[0, 0] == pytest.approx(3.0)
    assert built[0, 1] == pytest.approx(4.0)
    assert built[0, 2] == pytest.approx(7.0)


def test_grouped_explanation_drops_ungrouped_atoms() -> None:
    """An atom the channel could not name must inform the probe of nothing."""
    indices = np.array([[0, 1]], dtype=np.int64)
    contributions = np.array([[1.0, 9.0]], dtype=np.float32)
    groups = np.array([0, -1], dtype=np.int64)
    built = grouped_explanation(indices, contributions, groups, group_count=1)
    assert built[0, 0] == pytest.approx(1.0)


def test_matched_random_grouping_preserves_group_sizes() -> None:
    groups = np.array([0, 0, 0, 1, 1, -1, 2, -1], dtype=np.int64)
    permuted = matched_random_grouping(groups, seed=4)
    original = np.unique(groups, return_counts=True)
    shuffled = np.unique(permuted, return_counts=True)
    assert original[0].tolist() == shuffled[0].tolist()
    assert original[1].tolist() == shuffled[1].tolist()
    assert permuted.tolist() != groups.tolist()


def test_names_to_groups_compacts_only_used_terms() -> None:
    names = np.array([300, -1, 7, 300], dtype=np.int64)
    groups, used = names_to_groups(names)
    assert used == [7, 300]
    assert groups.tolist() == [1, -1, 0, 1]
