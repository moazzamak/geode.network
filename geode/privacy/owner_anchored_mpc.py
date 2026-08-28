"""M322b — VOID reproduction of the registered construction (27 Aug 2026).

This module implements the REGISTERED M322b protocol FAITHFULLY, as
written in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` §8.22
BEFORE the M322e amendment, for one purpose only: mechanical
evidence of the cross-term defect. It is NOT shipped functionality.

The registered construction: the user draws a uniform mask ``z_U``,
sends ``m = z - z_U`` to the contributor's host C and a second
server S; the head is re-split ``W = W_U + W_C + W_S`` (likewise
``b``); each server computes ``W_i^T m + b_i`` and returns the
C-vector to the user only; the user adds ``W_U^T z + b_U``.

The reconstructed sum is

    s' = W^T z + b - (W_C + W_S)^T z_U

— the cross term ``(W_C + W_S)^T z_U`` is missing, and by the
M322e registration NO party can compute it without breaking a
privacy constraint. The harness
``experiments/tier4/eval_v26_m322_fhe_head.py`` reproduces the
failure and writes ``evidence_void_m322b_g1.json``.

The amended construction (M322e-A, FHE head) is registered in the
plan; its build awaits the registered HE library/scheme choice.
"""
from __future__ import annotations

import numpy as np


def registered_m322b_mask(z: np.ndarray,
                          rng: np.random.Generator
                          ) -> tuple[np.ndarray, np.ndarray]:
    """The registered device step: ``(z_U, m = z - z_U)``."""
    z = np.asarray(z, dtype=np.float64)
    z_U = rng.uniform(-1.0, 1.0, size=z.shape)
    return z_U, z - z_U


def registered_m322b_split(W: np.ndarray, b: np.ndarray,
                           rng: np.random.Generator) -> dict:
    """The registered head split: fresh masks, residual on S."""
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d, C = W.shape
    W_U = rng.uniform(-1.0, 1.0, size=(d, C))
    W_C = rng.uniform(-1.0, 1.0, size=(d, C))
    W_S = W - W_U - W_C
    b_U = rng.uniform(-1.0, 1.0, size=(C,))
    b_C = rng.uniform(-1.0, 1.0, size=(C,))
    b_S = b - b_U - b_C
    return {"W_U": W_U, "b_U": b_U, "W_C": W_C, "b_C": b_C,
            "W_S": W_S, "b_S": b_S}


def registered_m322b_server(W_i: np.ndarray, b_i: np.ndarray,
                            m: np.ndarray) -> np.ndarray:
    """The registered server step: ``W_i^T m + b_i``."""
    return (np.asarray(m, dtype=np.float64)
            @ np.asarray(W_i, dtype=np.float64)
            + np.asarray(b_i, dtype=np.float64))


def registered_m322b_combine(z: np.ndarray, W_U: np.ndarray,
                             b_U: np.ndarray, x_C: np.ndarray,
                             x_S: np.ndarray) -> np.ndarray:
    """The registered device step: ``W_U^T z + b_U + x_C + x_S``."""
    return (np.asarray(W_U, dtype=np.float64).T
            @ np.asarray(z, dtype=np.float64)
            + np.asarray(b_U, dtype=np.float64)
            + np.asarray(x_C, dtype=np.float64)
            + np.asarray(x_S, dtype=np.float64))
