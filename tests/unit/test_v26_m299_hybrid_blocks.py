"""Unit tests for the M299 helpers: per-block L2 norms, the eigen-route
backward instrument, and the selection-digest cache contract."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.tier4.eval_v26_m299_hybrid_blocks import (
    _column_norms,
    _digest_of,
    _eigen_backward,
    _load_cached_dino,
)


class TestColumnNorms(unittest.TestCase):

    def test_norms_match_numpy(self):
        rng = np.random.default_rng(0)
        ms = (rng.standard_normal((300, 8)) * 40).astype(np.float32)
        dino = (rng.standard_normal((300, 5)) * 2).astype(np.float32)
        n_ms, n_dino = _column_norms(ms, dino)
        self.assertLessEqual(float(np.max(np.abs(
            n_ms - np.sqrt(np.sum(ms.astype(np.float64) ** 2, axis=0))))),
            1e-6)
        self.assertLessEqual(float(np.max(np.abs(
            n_dino - np.sqrt(np.sum(dino.astype(np.float64) ** 2,
                                    axis=0))))), 1e-6)
        self.assertGreater(n_ms.min(), 0.0)
        self.assertGreater(n_dino.min(), 0.0)


class TestEigenBackward(unittest.TestCase):

    def test_clean_system_passes_raw(self):
        rng = np.random.default_rng(1)
        d, classes = 20, 10
        x = rng.standard_normal((200, d))
        s = x.T @ x
        vals, vecs = np.linalg.eigh(s)
        cross = rng.standard_normal((d, classes))
        lam = 1.0
        w = vecs @ ((1.0 / (vals + lam))[:, None] * (vecs.T @ cross))
        report = _eigen_backward(vals, vecs, w, cross, lam)
        self.assertTrue(report["backward_passed"], report)
        self.assertEqual(report["instrument"], "raw")
        self.assertEqual(report["dropped_components"], 0)

    def test_indefinite_with_drop_uses_truncated_instrument(self):
        rng = np.random.default_rng(2)
        d, classes = 16, 8
        x = rng.standard_normal((120, d))
        s = x.T @ x
        vals, vecs = np.linalg.eigh(s)
        # force one penalised eigenvalue to exactly zero at lam = 1.0:
        # M296d drops it (not strongly convex) and the instrument
        # switches to the truncated-system form
        vals0 = vals.copy()
        vals0[0] = -1.0   # vals0 + lam == 0 at lam = 1.0
        cross = rng.standard_normal((d, classes))
        penalised = vals0 + 1.0
        scale = max(abs(float(penalised[0])), abs(float(penalised[-1])))
        keep = penalised > max(0.0, scale * 1e-10)
        inv = np.where(keep, 1.0 / penalised, 0.0)
        w = vecs @ (inv[:, None] * (vecs.T @ cross))
        report = _eigen_backward(vals0, vecs, w, cross, 1.0)
        self.assertEqual(report["instrument"], "truncated_system")
        self.assertTrue(report["backward_passed"], report)
        self.assertEqual(report["dropped_components"], 1)


class TestCachedDinoContract(unittest.TestCase):

    def test_digest_mismatch_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            sel = np.arange(10, dtype=np.int64)
            np.save(d / "test_dino.npy", np.zeros((10, 4)))
            meta = {"selection_sha256": "0" * 64}
            (d / "test_meta.json").write_text(json.dumps(meta))
            with self.assertRaises(SystemExit):
                _load_cached_dino(d, "test", sel)

    def test_digest_match_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            sel = np.arange(10, dtype=np.int64)
            np.save(d / "test_dino.npy", np.ones((10, 4)))
            meta = {"selection_sha256": _digest_of(sel)}
            (d / "test_meta.json").write_text(json.dumps(meta))
            loaded = _load_cached_dino(d, "test", sel)
            self.assertEqual(loaded.shape, (10, 4))
            self.assertTrue(bool((loaded == 1.0).all()))


if __name__ == "__main__":
    unittest.main()
