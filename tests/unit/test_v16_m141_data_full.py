"""Focused tests for the M141 all-extension rule (no corpus, no GPU)."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.tier4.eval_v16_m140_data_extension import _extension_indices
from experiments.tier4.eval_v16_m141_data_full import (
    _all_extension_indices,
    _rest_extension_indices,
)


class AllExtensionTests(unittest.TestCase):
    def test_returns_every_unused_row_per_class_in_raw_order(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        subsample = np.array([1, 4, 6])
        ext = _all_extension_indices(labels, subsample, 3)
        self.assertEqual(ext.tolist(), [0, 2, 3, 5, 7, 8])

    def test_empty_when_all_used(self):
        labels = np.array([0, 0, 1, 1])
        subsample = np.array([0, 1, 2, 3])
        ext = _all_extension_indices(labels, subsample, 2)
        self.assertEqual(len(ext), 0)

    def test_first_69k_equals_m140_selection_rule(self):
        """The first 69,000 rows (per class, first 200 unused) must equal the
        M140 ext600 selection, because M141 reuses that sealed memmap."""
        rng = np.random.default_rng(11)
        classes = 3
        counts = np.array([620, 630, 640])
        labels = np.concatenate([np.full(c, k) for k, c in enumerate(counts)])
        # subsample: 400 per class -> first 400 raw rows of each class
        subsample = np.concatenate(
            [np.flatnonzero(labels == k)[:400] for k in range(classes)])
        ext = _all_extension_indices(labels, subsample, classes)
        # per class, the first 200 extension rows are raw positions 400..599
        base = 0
        for k in range(classes):
            block = ext[base:base + 200]
            expected = np.flatnonzero(labels == k)[400:600]
            self.assertTrue(np.array_equal(block, expected))
            base += counts[k] - 400


class RestExtensionTests(unittest.TestCase):
    def test_rest_is_everything_after_the_first_200_per_class(self):
        labels = np.array([0] * 620 + [1] * 630)
        subsample = np.concatenate([np.arange(400), np.arange(620, 1020)])
        rest = _rest_extension_indices(labels, subsample, 2, per_class_take=200)
        # class 0: unused raw rows 400..619; first 200 -> 400..599; rest 600..619
        expected0 = np.arange(600, 620)
        # class 1: unused raw rows 1020..1249; first 200 -> 1020..1219; rest 1220..1249
        expected1 = np.arange(1220, 1250)
        self.assertTrue(np.array_equal(rest, np.concatenate([expected0, expected1])))

    def test_ext600_indices_match_m140_selection(self):
        labels = np.array([0] * 620 + [1] * 630)
        subsample = np.concatenate([np.arange(400), np.arange(620, 1020)])
        ext600, shortfall = _extension_indices(labels, subsample, cap=600, classes=2)
        expected = np.concatenate([np.arange(400, 600), np.arange(1020, 1220)])
        self.assertEqual(shortfall, {})
        self.assertTrue(np.array_equal(ext600, expected))


if __name__ == "__main__":
    unittest.main()
