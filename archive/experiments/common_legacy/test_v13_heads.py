"""Tests for the M81 heads, explanations and the I5 protocol.

Every arm here is a comparative instrument, so the suite is weighted towards
controls that must *fail*: a null that does not collapse, or a positive control
that does not saturate, means the instrument cannot support the comparison the
milestone is built on.

Four tests are regressions on defects that were measured during development and
are registered as notes N81.3 to N81.5 and the metric-field variance floor.
Each of those defects produced a plausible-looking number, which is why they
are pinned rather than merely fixed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.common.v13_heads import (
    _scatter_dense,
    cited_atoms_per_decision,
    expected_gradients,
    fit_decision_list,
    fit_metric_field,
    fit_mlp,
    fit_sparse_linear,
    integrated_gradients,
    withheld_explanation,
)
from experiments.common.v13_i5 import (
    forward_simulation,
    shuffled_explanation_control,
)
from experiments.common.v13_linear_probe import balanced_accuracy
from experiments.common.v13_sparse_dictionary import (
    SparseCodes,
    fit_sparse_dictionary,
)

CLASS_COUNT = 8
ACTIVE_ATOMS = 8
DICTIONARY_SIZE = 128


@pytest.fixture(scope="module")
def corpus() -> tuple[np.ndarray, np.ndarray]:
    """Eight well-separated Gaussian clusters, so every head has a reachable
    answer and weakness is attributable to the head rather than the data."""
    generator = np.random.default_rng(4242)
    centers = generator.normal(scale=4.0, size=(CLASS_COUNT, 32))
    labels = np.repeat(np.arange(CLASS_COUNT), 128).astype(np.int64)
    features = centers[labels] + generator.normal(scale=0.5, size=(len(labels), 32))
    return features.astype(np.float32), labels


@pytest.fixture(scope="module")
def codes(corpus) -> SparseCodes:
    features, _ = corpus
    dictionary, _ = fit_sparse_dictionary(
        features,
        dictionary_size=DICTIONARY_SIZE,
        active_atoms=ACTIVE_ATOMS,
        epochs=20,
        batch_size=128,
        learning_rate=1e-3,
        seed=11,
    )
    return dictionary.codes(features)


@pytest.fixture(scope="module")
def linear_head(codes, corpus):
    _, labels = corpus
    return fit_sparse_linear(
        codes,
        labels,
        class_count=CLASS_COUNT,
        l1_penalty=0.0,
        epochs=20,
        batch_size=128,
        learning_rate=1e-2,
        seed=11,
    )


# --------------------------------------------------------------------------
# Sparse scoring must agree with the dense definition it stands in for
# --------------------------------------------------------------------------


def test_sparse_linear_scores_match_dense_reference(linear_head, codes):
    """The gathered score path exists only to avoid a 2.1 GB dense matrix. It
    has to compute the same thing the readable definition does."""
    rows = np.arange(64)
    dense = _scatter_dense(codes, rows)
    expected = (dense @ linear_head.weight.T + linear_head.bias).numpy()
    actual = linear_head.scores(codes)[rows]
    assert np.allclose(actual, expected, atol=1e-4)


def test_metric_field_scores_match_dense_reference(codes, corpus):
    """The field's sparse path folds inactive atoms into a per-class constant.
    That algebra is easy to get subtly wrong, so it is checked against the
    literal negative weighted squared distance."""
    _, labels = corpus
    head = fit_metric_field(codes, labels, class_count=CLASS_COUNT)
    rows = np.arange(64)
    dense = _scatter_dense(codes, rows)
    difference = dense.unsqueeze(1) - head.centers.unsqueeze(0)
    expected = -(head.precisions.unsqueeze(0) * difference.pow(2)).sum(dim=2)
    actual = head.scores(codes)[rows]
    assert np.allclose(actual, expected.numpy(), rtol=1e-3, atol=1e-2)


def test_decision_list_prediction_follows_first_firing_rule(codes, corpus):
    _, labels = corpus
    head = fit_decision_list(codes, labels, class_count=CLASS_COUNT, max_rules=32)
    fired = head._fired(codes)
    predicted = head.predict(codes)
    matched = fired >= 0
    assert np.array_equal(predicted[matched], head.classes[fired[matched]])
    assert np.all(predicted[~matched] == head.default_class)


# --------------------------------------------------------------------------
# Positive controls on the heads themselves
# --------------------------------------------------------------------------


def test_every_head_separates_separable_clusters(codes, corpus, linear_head):
    """If any arm cannot solve eight well-separated clusters, a weak result on
    the real corpus is uninformative about sparse bases."""
    _, labels = corpus
    field = fit_metric_field(codes, labels, class_count=CLASS_COUNT)
    rules = fit_decision_list(
        codes, labels, class_count=CLASS_COUNT, max_rules=64
    )
    chance = 1.0 / CLASS_COUNT
    assert balanced_accuracy(linear_head.scores(codes).argmax(1), labels) > 0.80
    assert balanced_accuracy(field.scores(codes).argmax(1), labels) > 0.80
    assert balanced_accuracy(rules.predict(codes), labels) > 3 * chance


def test_metric_field_variance_floor_is_relative_to_data_scale(codes, corpus):
    """Regression. An absolute floor of 1e-6 assigns precision 1e6 to every
    atom a class never activates, so one unexpected atom outvotes the whole
    explanation and the head collapses to exact chance. Precisions must stay
    within a few orders of magnitude of each other."""
    _, labels = corpus
    head = fit_metric_field(codes, labels, class_count=CLASS_COUNT)
    assert head.precisions.max().item() / head.precisions.min().item() < 1e4
    assert balanced_accuracy(head.scores(codes).argmax(1), labels) > 0.80


def test_decision_list_selects_class_specific_atoms(codes, corpus):
    """Regression. Selecting rules on raw coverage picks the most frequently
    active atoms, which are the least class-specific ones. Every rule kept must
    win more rows than it loses."""
    _, labels = corpus
    head = fit_decision_list(codes, labels, class_count=CLASS_COUNT, max_rules=64)
    assert len(head.atoms) > 0
    for atom, klass in zip(head.atoms, head.classes):
        fires = np.any(codes.indices == atom, axis=1)
        hit = int(np.sum(labels[fires] == klass))
        assert 2 * hit > int(np.sum(fires))


# --------------------------------------------------------------------------
# Explanation length: the deployment budget is the quantity under test
# --------------------------------------------------------------------------


def test_proximal_l1_produces_exact_zeros_and_shortens_explanations(codes, corpus):
    """Regression on N81.4. Adding `l1 * |w|` to the loss and letting Adam
    differentiate it shrinks every coefficient uniformly without ever reaching
    zero: accuracy falls while the number of atoms cited per decision does not.
    The proximal step must actually shorten the explanation."""
    _, labels = corpus
    fitted = {}
    heads = {}
    for penalty in (0.0, 0.3):
        head = fit_sparse_linear(
            codes,
            labels,
            class_count=CLASS_COUNT,
            l1_penalty=penalty,
            epochs=20,
            batch_size=128,
            learning_rate=1e-2,
            seed=11,
        )
        heads[penalty] = head
        predicted = head.scores(codes).argmax(1)
        fitted[penalty] = cited_atoms_per_decision(
            head.contributions(codes, predicted), budget=4
        )
    assert (heads[0.3].weight == 0.0).any()
    assert not (heads[0.0].weight == 0.0).all()
    assert fitted[0.3]["mean_active_atoms"] < fitted[0.0]["mean_active_atoms"]


def test_atom_budget_keeps_atoms_that_actually_fire(codes, corpus):
    """Regression on N81.5. Ranking the budget by coefficient magnitude keeps
    rare atoms, which carry large weights precisely because they seldom fire.
    The head then cites almost nothing and collapses. The support must be
    chosen by contribution mass, so a budgeted head still cites atoms."""
    _, labels = corpus
    budget = 16
    head = fit_sparse_linear(
        codes,
        labels,
        class_count=CLASS_COUNT,
        l1_penalty=0.0,
        epochs=20,
        batch_size=128,
        learning_rate=1e-2,
        seed=11,
        atom_budget=budget,
    )
    assert int((head.weight.abs() > 0).sum().item()) <= budget * CLASS_COUNT
    predicted = head.scores(codes).argmax(1)
    cited = cited_atoms_per_decision(head.contributions(codes, predicted), budget=10)
    assert cited["mean_active_atoms"] > 1.0
    assert balanced_accuracy(predicted, labels) > 2.0 / CLASS_COUNT


def test_withheld_explanation_hides_atom_identity(linear_head, codes):
    """The probe must not be able to read which atom fired, only how much."""
    predicted = linear_head.scores(codes).argmax(1)
    contributions = linear_head.contributions(codes, predicted)
    explanation = withheld_explanation(contributions, top_count=4)
    assert explanation.shape == (codes.rows, 7)
    permuted = withheld_explanation(
        contributions[:, ::-1].copy(), top_count=4
    )
    assert np.allclose(explanation, permuted)


# --------------------------------------------------------------------------
# Gradient attributions
# --------------------------------------------------------------------------


def test_integrated_gradients_satisfies_completeness(corpus):
    """The completeness axiom is what makes IG an attribution rather than a
    heuristic: attributions must sum to the change in the explained logit."""
    features, labels = corpus
    model = fit_mlp(
        features, labels, class_count=CLASS_COUNT, hidden=32,
        epochs=10, batch_size=128, learning_rate=1e-3, seed=11,
    )
    rows = np.arange(32)
    baseline = features.mean(axis=0)
    with torch.no_grad():
        logits = model(torch.from_numpy(features[rows]))
        base_logits = model(torch.from_numpy(baseline).unsqueeze(0))
    predicted = logits.argmax(1).numpy()
    attributions = integrated_gradients(
        model, features[rows], predicted, baseline=baseline, steps=256
    )
    target = (
        logits.gather(1, torch.from_numpy(predicted).unsqueeze(1)).squeeze(1)
        - base_logits[0, predicted]
    ).numpy()
    assert np.allclose(attributions.sum(axis=1), target, rtol=0.05, atol=0.05)


def test_expected_gradients_is_reproducible_under_seed(corpus):
    features, labels = corpus
    model = fit_mlp(
        features, labels, class_count=CLASS_COUNT, hidden=32,
        epochs=5, batch_size=128, learning_rate=1e-3, seed=11,
    )
    rows = np.arange(32)
    predicted = model(torch.from_numpy(features[rows])).argmax(1).detach().numpy()
    kwargs = dict(reference=features[:64], samples=16, seed=8102)
    first = expected_gradients(model, features[rows], predicted, **kwargs)
    second = expected_gradients(model, features[rows], predicted, **kwargs)
    assert np.array_equal(first, second)


# --------------------------------------------------------------------------
# The I5 instrument: it must saturate when it should and collapse when it must
# --------------------------------------------------------------------------


def test_i5_recovers_a_prediction_encoded_in_the_explanation():
    """Positive control. If the explanation trivially determines the decision,
    I5 must approach 1. An instrument that cannot detect a perfect explanation
    cannot be trusted when it reports a weak one."""
    generator = np.random.default_rng(11)
    predictions = generator.integers(0, CLASS_COUNT, size=512)
    explanations = np.zeros((512, 3), dtype=np.float32)
    explanations[:, 0] = predictions * 10.0
    result = forward_simulation(
        explanations, predictions, class_count=CLASS_COUNT,
        train_fraction=0.7, max_iter=2000, seed=11,
    )
    assert result["probe_balanced_accuracy"] > 0.90
    assert result["component_identity_in_explanation"] is False


def test_shuffled_explanation_null_collapses_i5():
    """The R5 null. Permuting explanations against the decisions they explain
    must destroy I5, otherwise the probe is reading the marginal class
    distribution and every I5 number in the milestone is uninterpretable."""
    generator = np.random.default_rng(11)
    predictions = generator.integers(0, CLASS_COUNT, size=512)
    explanations = np.zeros((512, 3), dtype=np.float32)
    explanations[:, 0] = predictions * 10.0
    informative = forward_simulation(
        explanations, predictions, class_count=CLASS_COUNT,
        train_fraction=0.7, max_iter=2000, seed=11,
    )
    null = shuffled_explanation_control(
        explanations, predictions, class_count=CLASS_COUNT,
        train_fraction=0.7, max_iter=2000, seed=11,
    )
    assert null["probe_balanced_accuracy"] < 0.30
    assert informative["probe_balanced_accuracy"] > 3 * null["probe_balanced_accuracy"]


def test_i5_reports_degeneracy_rather_than_a_flattering_number():
    """A head that predicts one class would otherwise score a perfect I5."""
    predictions = np.zeros(256, dtype=np.int64)
    explanations = np.random.default_rng(11).normal(size=(256, 3)).astype(np.float32)
    result = forward_simulation(
        explanations, predictions, class_count=CLASS_COUNT,
        train_fraction=0.7, max_iter=2000, seed=11,
    )
    assert result["degenerate_single_prediction"] is True
    assert result["probe_balanced_accuracy"] is None


def test_i5_split_is_disjoint_and_stratified():
    """Example overlap between probe train and test would inflate I5 for every
    arm at once, which is the failure mode hardest to notice."""
    from experiments.common.v13_i5 import _stratified_split

    targets = np.repeat(np.arange(CLASS_COUNT), 64)
    train, test = _stratified_split(targets, train_fraction=0.7, seed=11)
    assert len(np.intersect1d(train, test)) == 0
    assert len(train) + len(test) == len(targets)
    assert set(np.unique(targets[train])) == set(np.unique(targets))
    assert set(np.unique(targets[test])) == set(np.unique(targets))
