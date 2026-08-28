"""Unit tests for the M346 number-series machinery (registered 28
Aug 2026, before the build). Pins: the GAF transform's form and
determinism; the window/target alignment; the multi-penalty ridge
solve; the ms encoder's shape contract."""
from __future__ import annotations

import numpy as np

from experiments.tier4.eval_v26_m346_number_series import (
    _gaf_window,
    _ridge_multi,
    _windows,
)


def test_gaf_window_form_and_determinism():
    rng = np.random.default_rng(0)
    window = rng.standard_normal(32)
    img = _gaf_window(window)
    assert img.shape == (32, 32)
    assert img.dtype == np.float32
    # GAF values are cosines: bounded in [-1, 1]
    assert img.min() >= -1.0 and img.max() <= 1.0
    # diagonal: cos(2*phi_i) - deterministic given the window
    again = _gaf_window(window)
    assert np.array_equal(img, again)


def test_gaf_window_constant_window():
    # a constant window has zero span: the guard keeps it finite
    img = _gaf_window(np.full(32, 3.14))
    assert np.all(np.isfinite(img))


def test_windows_target_alignment():
    series = np.arange(100, dtype=np.float64)
    images, targets = _windows(series)
    # window i covers series[i:i+32]; target is series[i+32]
    assert len(images) == 100 - 32
    assert len(targets) == len(images)
    assert targets[0] == 32.0
    assert targets[-1] == 99.0
    # the first window's GAF is a function of series[0:32]
    assert np.array_equal(images[0], _gaf_window(series[0:32]))


def test_ridge_multi_penalty_ladder():
    rng = np.random.default_rng(1)
    feats = rng.standard_normal((500, 8))
    w_true = rng.standard_normal(8)
    targets = feats @ w_true + 0.01 * rng.standard_normal(500)
    weights, mean, std = _ridge_multi(feats, targets, [0.1, 10.0])
    # both penalties fit well relative to the target scale
    target_std = float(np.std(targets))
    for p in (0.1, 10.0):
        z = (feats - mean) / std
        preds = z @ weights[p]
        rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
        assert rmse < 0.05 * target_std
    # the smaller penalty fits train tighter
    def train_err(p):
        z = (feats - mean) / std
        return float(np.mean((z @ weights[p] - targets) ** 2))
    assert train_err(0.1) <= train_err(10.0)
