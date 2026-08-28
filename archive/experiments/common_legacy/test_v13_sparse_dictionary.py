"""Focused tests for the M80 sparse concept dictionary.

Two of these are positive controls in the M78 sense: they assert that a broken
operand would actually be detected. A test suite that only confirms the happy
path cannot distinguish a working measurement from a vacuous one, which is the
failure M77 found in the v12 probe objective.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.common.v13_linear_probe import (
    balanced_accuracy,
    dense_probe_accuracy,
    sparse_probe_accuracy,
)
from experiments.common.v13_sparse_dictionary import (
    atom_label_entropy,
    fit_sparse_dictionary,
    random_dictionary,
    reconstruction_r2,
)


@pytest.fixture(scope="module")
def clustered_corpus() -> tuple[np.ndarray, np.ndarray]:
    """Eight well-separated Gaussian clusters in 32 dimensions."""
    generator = np.random.default_rng(4242)
    centers = generator.normal(size=(8, 32)).astype(np.float32) * 6.0
    labels = np.repeat(np.arange(8), 128).astype(np.int64)
    features = centers[labels] + generator.normal(size=(1024, 32)).astype(np.float32)
    return features.astype(np.float32), labels


@pytest.fixture(scope="module")
def fitted(clustered_corpus):
    features, _ = clustered_corpus
    dictionary, diagnostics = fit_sparse_dictionary(
        features,
        dictionary_size=64,
        active_atoms=4,
        epochs=12,
        batch_size=256,
        learning_rate=1e-2,
        seed=11,
    )
    return dictionary, diagnostics


def test_codes_are_exactly_k_sparse_and_non_negative(clustered_corpus, fitted):
    features, _ = clustered_corpus
    dictionary, _ = fitted
    codes = dictionary.codes(features)

    assert codes.indices.shape == (len(features), 4)
    assert np.all(codes.values >= 0.0)
    assert np.all(codes.active_atom_count() <= 4)


def test_decoder_atoms_stay_unit_norm(fitted):
    dictionary, _ = fitted
    norms = dictionary.decoder_weight.norm(dim=0).numpy()
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_training_reduces_reconstruction_loss(fitted):
    _, diagnostics = fitted
    trace = diagnostics["epoch_loss_trace"]
    assert diagnostics["loss_decreased"]
    assert trace[-1] < trace[0]


def test_trained_dictionary_beats_random_control(clustered_corpus, fitted):
    """Positive control for the reconstruction operand.

    If a random dictionary reconstructed as well as a trained one, R^2 would be
    measuring sparse random projection rather than anything learned.
    """
    features, _ = clustered_corpus
    dictionary, _ = fitted
    control = random_dictionary(
        features, dictionary_size=64, active_atoms=4, seed=8001
    )

    trained_r2 = reconstruction_r2(features, dictionary.reconstruct(features))
    control_r2 = reconstruction_r2(features, control.reconstruct(features))
    assert trained_r2 > control_r2 + 0.05


def test_shuffled_labels_raise_atom_entropy(clustered_corpus, fitted):
    """Positive control for the purity operand.

    Under shuffled labels the atoms cannot be monosemantic, so entropy must rise
    towards the uniform bound of log2(8) = 3 bits. A purity measure that did not
    move here would be reporting a constant.
    """
    features, labels = clustered_corpus
    dictionary, _ = fitted
    codes = dictionary.codes(features)

    true_entropy = atom_label_entropy(codes, labels, class_count=8)["mean_bits"]
    shuffled = np.random.default_rng(8002).permutation(labels)
    shuffled_entropy = atom_label_entropy(codes, shuffled, class_count=8)["mean_bits"]

    assert shuffled_entropy > true_entropy
    assert shuffled_entropy == pytest.approx(3.0, abs=0.5)


def test_dead_atom_fraction_is_reported_not_repaired(clustered_corpus):
    """An 8-cluster corpus cannot use 512 atoms, so some must be dead."""
    features, _ = clustered_corpus
    dictionary, _ = fit_sparse_dictionary(
        features,
        dictionary_size=512,
        active_atoms=4,
        epochs=4,
        batch_size=256,
        learning_rate=1e-2,
        seed=11,
    )
    usage = dictionary.codes(features).atom_usage()
    assert np.mean(usage == 0) > 0.0


def test_probes_recover_separable_classes(clustered_corpus, fitted):
    features, labels = clustered_corpus
    dictionary, _ = fitted
    codes = dictionary.codes(features)

    dense = dense_probe_accuracy(
        features, labels, features, labels,
        class_count=8, epochs=20, batch_size=256, learning_rate=1e-2, seed=11,
    )
    sparse = sparse_probe_accuracy(
        codes, labels, codes, labels,
        class_count=8, epochs=20, batch_size=256, learning_rate=1e-2, seed=11,
    )
    assert dense > 0.95
    assert sparse > 0.80


def test_balanced_accuracy_is_not_dominated_by_a_large_class():
    """Balanced accuracy must penalise ignoring a rare class."""
    actual = np.array([0] * 90 + [1] * 10)
    predicted = np.zeros(100, dtype=np.int64)
    assert balanced_accuracy(predicted, actual) == pytest.approx(0.5)


def test_fit_is_reproducible_single_threaded(clustered_corpus):
    """The threading contract the milestone depends on."""
    features, _ = clustered_corpus
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        first, _ = fit_sparse_dictionary(
            features, dictionary_size=64, active_atoms=4,
            epochs=3, batch_size=256, learning_rate=1e-2, seed=11,
        )
        second, _ = fit_sparse_dictionary(
            features, dictionary_size=64, active_atoms=4,
            epochs=3, batch_size=256, learning_rate=1e-2, seed=11,
        )
    finally:
        torch.set_num_threads(previous)

    assert torch.equal(first.decoder_weight, second.decoder_weight)
    assert torch.equal(first.encoder_weight, second.encoder_weight)
