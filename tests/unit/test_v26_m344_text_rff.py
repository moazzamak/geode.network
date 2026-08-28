"""Unit tests for the M344 text-RFF machinery (registered 28 Aug
2026, before the build). Pins: the streaming design-form ridge
matches the M262 closed form on the concatenated design; the
streaming scorer matches the direct scorer; the sealed-probe
reproduction path (weights/mean/std/classes round-trip)."""
from __future__ import annotations

import numpy as np

from experiments.tier4.eval_v26_m344_text_rff import (
    _predict,
    _ridge_probe,
    _ridge_probe_design,
    _score_design,
)
from experiments.tier4.eval_v26_m300_rff_quickdraw import (
    build_design,
    rff_params,
)


def _toy_problem(n: int = 400, d: int = 24, D: int = 96,
                 classes: int = 3, seed: int = 7):
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n, d)).astype(np.float32)
    w_true = rng.standard_normal((d, classes))
    logits = feats @ w_true
    labels = [int(i) for i in logits.argmax(axis=1)]
    omega, phase = rff_params(d, D, 0.5, 20260828)
    return feats, labels, omega, phase


def test_streaming_ridge_matches_closed_form():
    # the streaming design-form ridge must match the M262 closed
    # form applied to the materialised design matrix
    feats, labels, omega, phase = _toy_problem()
    design = build_design(feats, omega, phase)
    direct = _ridge_probe(design, labels, 1.0)
    streaming = _ridge_probe_design(feats, labels, 1.0, omega,
                                     phase)
    assert np.allclose(direct["weights"], streaming["weights"],
                       atol=1e-4)
    assert np.allclose(direct["bias"], streaming["bias"], atol=1e-4)
    assert np.allclose(direct["mean"], streaming["mean"], atol=1e-4)
    assert np.allclose(direct["std"], streaming["std"], atol=1e-4)
    assert direct["classes"] == streaming["classes"]


def test_streaming_ridge_chunk_invariance():
    # the streaming accumulation must not depend on the chunk size
    feats, labels, omega, phase = _toy_problem()
    a = _ridge_probe_design(feats, labels, 1.0, omega, phase,
                            chunk=64)
    b = _ridge_probe_design(feats, labels, 1.0, omega, phase,
                            chunk=400)
    assert np.allclose(a["weights"], b["weights"], atol=1e-8)
    assert np.allclose(a["bias"], b["bias"], atol=1e-8)


def test_score_design_matches_direct_scoring():
    feats, labels, omega, phase = _toy_problem()
    probe = _ridge_probe_design(feats, labels, 1.0, omega, phase)
    streamed = _score_design(probe, feats, omega, phase, labels,
                             chunk=50)
    design = build_design(feats, omega, phase)
    direct = float((_predict(probe, design)
                    == np.asarray(labels)).mean())
    assert streamed == direct


def test_score_design_chunk_invariance():
    feats, labels, omega, phase = _toy_problem()
    probe = _ridge_probe_design(feats, labels, 1.0, omega, phase)
    a = _score_design(probe, feats, omega, phase, labels, chunk=37)
    b = _score_design(probe, feats, omega, phase, labels, chunk=400)
    assert a == b
