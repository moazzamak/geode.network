from __future__ import annotations

import unittest

import numpy as np

from experiments.common.v7_adaptation import (
    GaussianAdaptationTransaction,
    fit_gaussian_bundle,
)
from experiments.common.v7_protocol import ConfirmationEvent


class GaussianAdaptationTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(1)
        self.features = np.concatenate(
            [rng.normal(label * 3.0, 0.2, size=(40, 6)) for label in range(2)]
        )
        self.labels = np.repeat(np.arange(2), 40)
        self.parent = fit_gaussian_bundle(
            self.features,
            self.labels,
            rank=2,
            threshold=20.0,
        )

    def test_confirmation_is_required_and_rollback_is_exact(self) -> None:
        transaction = GaussianAdaptationTransaction(self.parent)
        with self.assertRaises(PermissionError):
            transaction.apply(
                confirmation=None,
                label=2,
                support=np.ones((10, 6)),
                rank=2,
                operation="native_gaussian",
            )
        confirmation = ConfirmationEvent(
            review_id="review-test",
            response="new_class",
            confirmed_label="2",
            confirmed_window=1,
        )
        child = transaction.apply(
            confirmation=confirmation,
            label=2,
            support=np.ones((10, 6)),
            rank=2,
            operation="native_gaussian",
        )
        self.assertEqual(child.class_order, (0, 1, 2))
        self.assertEqual(transaction.rollback().bundle_hash, self.parent.bundle_hash)
