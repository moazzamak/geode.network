"""Linear probes for M80 — the accuracy operand of the sparse dictionary gate.

The gate compares a probe on sparse codes against a probe on raw features. That
comparison is only meaningful if both probes are trained identically, so both use
the same optimiser, schedule, epoch budget, and seed. The only difference is the
input representation, which is the variable under test.

The code probe never materialises a dense ``rows x dictionary_size`` matrix. At
8192 atoms and 65,536 rows that would be 2.1 GB per cell, and the codes are
exactly k-sparse, so the logits are computed by gathering the active atom rows of
the weight matrix instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from experiments.common.v13_sparse_dictionary import SparseCodes


def balanced_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean per-class recall, the program's standing accuracy operand."""
    recalls = [
        float(np.mean(predicted[actual == label] == label))
        for label in np.unique(actual)
    ]
    return float(np.mean(recalls))


def _train(
    forward: Any,
    parameters: list[torch.Tensor],
    row_count: int,
    labels: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    generator = torch.Generator().manual_seed(int(seed))
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
    objective = torch.nn.CrossEntropyLoss()
    for _ in range(int(epochs)):
        order = torch.randperm(row_count, generator=generator)
        for start in range(0, row_count, batch_size):
            rows = order[start : start + batch_size]
            loss = objective(forward(rows), labels[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def dense_probe_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    class_count: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> float:
    """Multinomial linear probe on raw features. This is the bar the gate uses."""
    generator = torch.Generator().manual_seed(int(seed))
    dimension = int(train_features.shape[1])
    weight = (
        torch.randn(dimension, class_count, generator=generator, dtype=torch.float32)
        * 0.01
    ).requires_grad_(True)
    bias = torch.zeros(class_count, dtype=torch.float32).requires_grad_(True)

    features = torch.from_numpy(np.ascontiguousarray(train_features, np.float32))
    targets = torch.from_numpy(train_labels.astype(np.int64))

    _train(
        lambda rows: features[rows] @ weight + bias,
        [weight, bias],
        len(train_features),
        targets,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )

    with torch.no_grad():
        logits = torch.from_numpy(
            np.ascontiguousarray(test_features, np.float32)
        ) @ weight + bias
        predicted = logits.argmax(dim=1).numpy()
    return balanced_accuracy(predicted, test_labels)


def sparse_probe_accuracy(
    train_codes: SparseCodes,
    train_labels: np.ndarray,
    test_codes: SparseCodes,
    test_labels: np.ndarray,
    *,
    class_count: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> float:
    """Multinomial linear probe over sparse codes, gathered rather than densified."""
    generator = torch.Generator().manual_seed(int(seed))
    weight = (
        torch.randn(
            train_codes.dictionary_size,
            class_count,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.01
    ).requires_grad_(True)
    bias = torch.zeros(class_count, dtype=torch.float32).requires_grad_(True)

    indices = torch.from_numpy(train_codes.indices)
    values = torch.from_numpy(train_codes.values)
    targets = torch.from_numpy(train_labels.astype(np.int64))

    def forward(rows: torch.Tensor) -> torch.Tensor:
        gathered = weight[indices[rows]]  # (rows, active_atoms, class_count)
        return (gathered * values[rows].unsqueeze(-1)).sum(dim=1) + bias

    _train(
        forward,
        [weight, bias],
        train_codes.rows,
        targets,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )

    test_indices = torch.from_numpy(test_codes.indices)
    test_values = torch.from_numpy(test_codes.values)
    with torch.no_grad():
        gathered = weight[test_indices]
        logits = (gathered * test_values.unsqueeze(-1)).sum(dim=1) + bias
        predicted = logits.argmax(dim=1).numpy()
    return balanced_accuracy(predicted, test_labels)
