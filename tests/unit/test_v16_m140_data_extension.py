"""Focused tests for the M140 extension-index rule (no corpus, no GPU)."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.tier4.eval_v16_m140_data_extension import _extension_indices


class ExtensionIndexTests(unittest.TestCase):
    def test_takes_first_unused_rows_per_class(self):
        # raw labels: class 0 rows 0..7, class 1 rows 8..15
        labels = np.array([0] * 8 + [1] * 8)
        # subsample: 400 -> emulate 2 per class here: rows 1, 3 (class 0), 8, 10
        subsample = np.array([1, 3, 8, 10])
        ext, shortfall = _extension_indices(labels, subsample, cap=4, classes=2)
        # need 2 more per class: class 0 -> rows 0, 2; class 1 -> rows 9, 11
        self.assertEqual(sorted(ext.tolist()), [0, 2, 9, 11])
        self.assertEqual(shortfall, {})

    def test_records_shortfall_instead_of_dropping(self):
        labels = np.array([0] * 5 + [1] * 5)
        subsample = np.array([0, 1, 2, 5])
        ext, shortfall = _extension_indices(labels, subsample, cap=5, classes=2)
        # class 0: used 3, available 5 -> can add 2 (rows 3, 4) -> need 2 -> ok
        # class 1: used 1, available 5 -> can add 4 -> need 4 -> ok
        self.assertEqual(shortfall, {})
        self.assertEqual(sorted(ext.tolist()), [3, 4, 6, 7, 8, 9])

        subsample = np.array([0, 1, 2, 3, 4, 5])
        ext, shortfall = _extension_indices(labels, subsample, cap=5, classes=2)
        # class 0: all 5 used -> need 0; class 1: used 1 -> need 4, available 4 -> ok
        self.assertEqual(shortfall, {})
        self.assertEqual(sorted(ext.tolist()), [6, 7, 8, 9])

    def test_shortfall_when_class_exhausted(self):
        labels = np.array([0] * 4 + [1] * 8)
        subsample = np.array([0, 1, 2, 3, 4, 5])
        ext, shortfall = _extension_indices(labels, subsample, cap=5, classes=2)
        # class 0: used 4, available 4 -> need 1 -> shortfall 1
        # class 1: used 2 (rows 4, 5), need 3 -> rows 6, 7, 8
        self.assertEqual(shortfall, {0: 1})
        self.assertEqual(sorted(ext.tolist()), [6, 7, 8])


if __name__ == "__main__":
    unittest.main()
