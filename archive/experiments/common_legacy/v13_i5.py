"""The I5 forward-simulation protocol for M81.

Reproduces the v12 M75 protocol exactly, because Amendment R4 registers the
v12 record (GEODE 17.737%, RBF 22.772%, kNN 25.246%) as the I5-8 reference and
a comparison against a differently-measured number would be worthless:

* the probe predicts the **model's own prediction**, not the ground-truth
  label, so it measures simulatability rather than accuracy;
* explanations withhold component identity;
* the probe train/test split is stratified on the model's predictions and
  disjoint by row.

One deviation, and it is a tightening: explanations are computed on rows the
head never saw during fitting. v12 drew probe train and test from the same
pool the head was fitted on.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


def _stratified_split(
    targets: np.ndarray, *, train_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed + 81_100)
    train: list[int] = []
    test: list[int] = []
    for target in np.unique(targets):
        rows = np.flatnonzero(targets == target)
        rows = rows[generator.permutation(len(rows))]
        split = max(1, min(len(rows) - 1, int(len(rows) * train_fraction)))
        train.extend(rows[:split].tolist())
        test.extend(rows[split:].tolist())
    return (
        np.asarray(sorted(train), dtype=np.int64),
        np.asarray(sorted(test), dtype=np.int64),
    )


def forward_simulation(
    explanations: np.ndarray,
    model_predictions: np.ndarray,
    *,
    class_count: int,
    train_fraction: float,
    max_iter: int,
    seed: int,
) -> dict[str, Any]:
    """I5. Balanced accuracy of a probe predicting the model's own output."""
    distinct = np.unique(model_predictions)
    if len(distinct) < 2:
        # A head that emits one class would otherwise score a perfect I5. It is
        # recorded as unmeasurable rather than as a number, and `None` rather
        # than NaN because the evidence hash refuses non-finite floats.
        return {
            "probe_balanced_accuracy": None,
            "majority_balanced_accuracy": None,
            "chance_accuracy": 1.0 / class_count,
            "degenerate_single_prediction": True,
            "distinct_predictions": 1,
            "train_count": 0,
            "test_count": 0,
            "component_identity_in_explanation": False,
            "example_overlap": False,
        }

    train, test = _stratified_split(
        model_predictions, train_fraction=train_fraction, seed=seed
    )
    # Standardise on the probe-training rows only. Explanation columns mix
    # per-atom contributions with sums, whose scales differ by orders of
    # magnitude, and unscaled lbfgs does not converge. Without this the probe
    # measures optimiser failure rather than how much the explanation carries.
    # Applied identically to every arm and every control; see note N81.3.
    center = explanations[train].mean(axis=0)
    spread = explanations[train].std(axis=0)
    spread[spread < 1e-12] = 1.0
    scaled = (explanations - center) / spread

    probe = LogisticRegression(
        max_iter=max_iter, random_state=seed, solver="lbfgs"
    ).fit(scaled[train], model_predictions[train])
    predicted = probe.predict(scaled[test])

    values, counts = np.unique(model_predictions[train], return_counts=True)
    majority = np.full(len(test), values[int(np.argmax(counts))])

    return {
        "probe_balanced_accuracy": float(
            balanced_accuracy_score(model_predictions[test], predicted)
        ),
        "majority_balanced_accuracy": float(
            balanced_accuracy_score(model_predictions[test], majority)
        ),
        "chance_accuracy": 1.0 / class_count,
        "degenerate_single_prediction": False,
        "distinct_predictions": int(len(distinct)),
        "train_count": int(len(train)),
        "test_count": int(len(test)),
        "component_identity_in_explanation": False,
        "example_overlap": False,
    }


def shuffled_explanation_control(
    explanations: np.ndarray,
    model_predictions: np.ndarray,
    *,
    class_count: int,
    train_fraction: float,
    max_iter: int,
    seed: int,
) -> dict[str, Any]:
    """The R5 null: rows of the explanation matrix permuted against the
    predictions they explain.

    Any I5 above this is attributable to the explanation carrying information
    about the decision. Any I5 at or below it is not evidence, whatever its
    absolute value.
    """
    generator = np.random.default_rng(seed + 81_200)
    permuted = explanations[generator.permutation(len(explanations))]
    return forward_simulation(
        permuted,
        model_predictions,
        class_count=class_count,
        train_fraction=train_fraction,
        max_iter=max_iter,
        seed=seed,
    )
