"""Unit tests for the M300 RFF machinery (registered 28 Aug 2026,
before the build). Pins: determinism under the registered seed,
seed-sensitivity, the sqrt(2/D) scale, the projection shape, the
L2 row-normalisation form, and the chunked in-place projection
(the M300b run-1 memory fix: chunk size and out= must not change
results)."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.tier4.eval_v26_m300_rff_quickdraw import (
    _l2_normalise_rows,
    build_design,
    rff_params,
    rff_project,
)


def test_rff_params_deterministic_under_seed():
    om1, ph1 = rff_params(64, 128, 1.0, 20260828)
    om2, ph2 = rff_params(64, 128, 1.0, 20260828)
    assert np.array_equal(om1, om2)
    assert np.array_equal(ph1, ph2)


def test_rff_params_seed_sensitive():
    om1, _ = rff_params(64, 128, 1.0, 20260828)
    om2, _ = rff_params(64, 128, 1.0, 20260829)
    assert not np.array_equal(om1, om2)


def test_rff_params_shapes_and_scale():
    dim, n_feat, sigma = 64, 128, 2.0
    om, ph = rff_params(dim, n_feat, sigma, 20260828)
    assert om.shape == (dim, n_feat)
    assert ph.shape == (n_feat,)
    # omega ~ N(0, sigma^-2 I): std within a loose band of 1/sigma
    assert abs(float(om.std()) - 1.0 / sigma) < 0.05
    # phase in [0, 2pi)
    assert ph.min() >= 0.0 and ph.max() < 2.0 * np.pi


def test_rff_project_shape_and_dtype():
    om, ph = rff_params(64, 128, 1.0, 20260828)
    block = np.random.default_rng(0).standard_normal((32, 64)
                                                     ).astype(np.float32)
    out = rff_project(block, om, ph)
    assert out.shape == (32, 128)
    assert out.dtype == np.float32
    # cos output is bounded; sqrt(2/D) scaling keeps the variance
    # near 1/D per feature (the RFF variance property)
    assert abs(float(out.var()) - 1.0 / 128.0) < 0.05


def test_rff_project_deterministic():
    om, ph = rff_params(64, 128, 1.0, 20260828)
    rng = np.random.default_rng(1)
    block = rng.standard_normal((16, 64)).astype(np.float32)
    a = rff_project(block, om, ph)
    b = rff_project(block, om, ph)
    assert np.array_equal(a, b)


def test_l2_normalise_rows_unit_norm():
    rng = np.random.default_rng(2)
    block = rng.standard_normal((20, 16)).astype(np.float32)
    out = _l2_normalise_rows(block)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_l2_normalise_rows_zero_row_safe():
    block = np.zeros((2, 8), dtype=np.float32)
    out = _l2_normalise_rows(block)
    assert np.all(np.isfinite(out))


def test_rff_project_chunking_bitwise_equal():
    # the M300b memory fix: chunk size must not change the projection
    # beyond float32 round-off. BLAS accumulates a (1,d)@(d,D) GEMV
    # in a different order than a (n,d)@(d,D) GEMM, so cross-chunk
    # agreement is allclose, not bitwise; determinism at FIXED
    # chunking (the default is a pure function of D) is bitwise and
    # is pinned by test_rff_project_deterministic.
    om, ph = rff_params(64, 256, 1.0, 20260828)
    rng = np.random.default_rng(3)
    block = rng.standard_normal((100, 64)).astype(np.float32)
    full = rff_project(block, om, ph)
    for chunk in (1, 7, 64, 100, 1000):
        chunked = rff_project(block, om, ph, chunk_rows=chunk)
        # outputs are bounded by sqrt(2/D) ~ 0.088; 1e-5 absolute is
        # ~0.01% of range and far above the measured 8.8e-07 round-off
        assert np.allclose(full, chunked, rtol=0.0, atol=1e-5), \
            f"chunk={chunk}"
        assert float(np.abs(full - chunked).max()) < 1e-5, f"chunk={chunk}"


def test_rff_project_out_writes_in_place():
    om, ph = rff_params(64, 256, 1.0, 20260828)
    rng = np.random.default_rng(4)
    block = rng.standard_normal((50, 64)).astype(np.float32)
    out = np.full((50, 256), np.nan, dtype=np.float32)
    returned = rff_project(block, om, ph, out=out)
    assert returned is out
    assert np.array_equal(out, rff_project(block, om, ph))


def test_rff_project_out_validates_shape():
    om, ph = rff_params(64, 256, 1.0, 20260828)
    block = np.zeros((10, 64), dtype=np.float32)
    with pytest.raises(ValueError):
        rff_project(block, om, ph, out=np.zeros((10, 128),
                                                dtype=np.float32))
    with pytest.raises(ValueError):
        rff_project(block, om, ph, out=np.zeros((10, 256),
                                                dtype=np.float64))


def test_build_design_matches_concatenate():
    # [features, phi(features)] must be bitwise equal to the
    # concatenate form the sealed M300 run used
    om, ph = rff_params(48, 192, 0.5, 20260828)
    rng = np.random.default_rng(5)
    feats = rng.standard_normal((40, 48)).astype(np.float32)
    design = build_design(feats, om, ph)
    expected = np.concatenate(
        [feats, rff_project(feats, om, ph)], axis=1)
    assert design.shape == (40, 48 + 192)
    assert design.dtype == np.float32
    assert np.array_equal(design, expected)
    # the features block is copied, not aliased
    assert np.array_equal(design[:, :48], feats)
