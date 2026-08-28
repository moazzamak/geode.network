"""M306 - the canonical pinned CPU/float64 replay oracle.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` §8.31
(27 Aug 2026, before any build). The oracle reproduces sealed heads
bit-exactly under the registered numerics policy. It is
self-contained: no GPU, no experiments import, no wall clock inside
any hashed payload.

The numerics policy (registered, fixed text - a change is a new
oracle ID, never a silent edit):

1. The Gram and cross are accumulated in float64 from the sealed
   fp32 rows in fixed 4096-row chunks (the sealed M228
   accumulation path).
2. The standardiser's centre and scale are rounded to fp32 (the
   sealed path), then read back as float64.
3. The standardised normal-equation system is assembled in
   float64.
4. The sealed anchor head is the LU solve of the diagonal-
   penalised system (the sealed M228 convention).
5. The repaired solve (M296d) is symmetric-by-construction
   assembly, then Cholesky, then eigendecomposition (driver
   ``evd``) with the strong-convexity truncation, then SVD last.
6. No iterative training, no random seed, no GPU.
7. The canonical replay pins the BLAS thread configuration; the
   certificate records it with the package versions and the
   hardware signature. Only the head digest (never the hardware
   signature, never timings) enters a content hash.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass, field
from typing import Any

import numpy as np

POLICY_TEXT = (
    "GEODE numerics policy v1 (M306, 27 Aug 2026): "
    "1) Gram and cross accumulate in float64 from fp32 rows in fixed "
    "4096-row chunks (the sealed M228 path); "
    "2) standardiser centre and scale rounded to fp32, read back as "
    "float64; "
    "3) standardised normal equations assembled in float64; "
    "4) sealed anchor head = LU solve of the diagonal-penalised "
    "system; "
    "5) repaired solve = symmetric-by-construction assembly, then "
    "Cholesky, then eigh (driver evd) with the M296d strong-convexity "
    "truncation, then SVD last; "
    "6) no iterative training, no random seed, no GPU; "
    "7) canonical replay pins the BLAS thread configuration, recorded "
    "in the certificate."
)

SEALED_BLOCK = 4096        # accumulation chunk (the sealed path)
STRONG_CONVEX_CUTOFF = 1e-10   # M296d penalised-mode truncation
SVD_SOLVE_CUTOFF = 16 * np.finfo(np.float64).eps


# ----------------------------------------------------------------------
# Oracle registration by hash
# ----------------------------------------------------------------------

def _scipy_version() -> str:
    from importlib.metadata import version as _v
    return _v("scipy")


def package_versions() -> dict[str, str]:
    """The pinned package versions. They enter the oracle ID: a
    dependency change is a new oracle, never a silent drift."""
    return {"numpy": np.__version__, "scipy": _scipy_version()}


def thread_config() -> dict[str, Any]:
    """The BLAS thread configuration the canonical replay pins."""
    keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    return {key: os.environ.get(key) for key in keys} | {
        "os_cpu_count": os.cpu_count()}


def hardware_signature() -> dict[str, Any]:
    """What machine ran the replay. Recorded in certificates, NEVER
    hashed into a content digest (timing/volatile-field rule)."""
    sig: dict[str, Any] = {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "threads": thread_config(),
    }
    sig.update(package_versions())
    try:
        cfg = np.show_config(mode="dicts")
        for key in ("BLAS", "LAPACK"):
            section = cfg.get(key, {}) if isinstance(cfg, dict) else {}
            sig[f"{key.lower()}_name"] = section.get("name")
    except Exception:  # pragma: no cover - show_config is best-effort
        pass
    return sig


def _policy_payload(block: int = SEALED_BLOCK) -> bytes:
    payload = {
        "policy": POLICY_TEXT,
        "sealed_block": block,
        "versions": package_versions(),
    }
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def oracle_id(block: int = SEALED_BLOCK) -> str:
    """SHA-256 of the policy text, the pinned block size, and the
    pinned package versions. Every certificate carries it."""
    return hashlib.sha256(_policy_payload(block)).hexdigest()


# ----------------------------------------------------------------------
# The sealed accumulation path (bit-for-bit the M228 path)
# ----------------------------------------------------------------------

@dataclass
class SealedSystem:
    """The raw normal equations accumulated exactly as the sealed
    M228 schedule accumulates them: float64 Gram and cross from
    fp32 rows, fixed 4096-row chunks."""

    gram: np.ndarray
    column_sum: np.ndarray
    cross: np.ndarray
    class_count: np.ndarray
    rows: int

    @classmethod
    def accumulate(cls, features: np.ndarray, labels: np.ndarray,
                   classes: int = 345,
                   block: int = SEALED_BLOCK) -> "SealedSystem":
        width = features.shape[1]
        gram = np.zeros((width, width), dtype=np.float64)
        column_sum = np.zeros(width, dtype=np.float64)
        cross = np.zeros((width, classes), dtype=np.float64)
        class_count = np.zeros(classes, dtype=np.float64)
        for start in range(0, len(features), block):
            stop = min(start + block, len(features))
            part = np.asarray(features[start:stop], dtype=np.float64)
            targets = np.zeros((len(part), classes), dtype=np.float64)
            targets[np.arange(len(part)), labels[start:stop]] = 1.0
            gram += part.T @ part
            column_sum += part.sum(axis=0)
            cross += part.T @ targets
            class_count += targets.sum(axis=0)
        return cls(gram=gram, column_sum=column_sum, cross=cross,
                   class_count=class_count, rows=len(features))

    def standardiser(self) -> tuple[np.ndarray, np.ndarray]:
        """(centre, scale), fp32-rounded - the sealed convention."""
        centre = self.column_sum / self.rows
        variance = np.diag(self.gram) / self.rows - np.square(centre)
        scale = np.sqrt(np.maximum(variance, 0.0)) + 1e-8
        return (centre.astype(np.float32), scale.astype(np.float32))

    def standardised_system(self
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(centred, cross, intercept) in float64 - the sealed
        standardised normal equations, closed form."""
        centre_f32, scale_f32 = self.standardiser()
        centre = centre_f32.astype(np.float64)
        inverse = 1.0 / scale_f32.astype(np.float64)
        centred = self.gram - np.outer(self.column_sum, centre)
        centred *= inverse[:, None]
        centred *= inverse[None, :]
        cross = (self.cross
                 - np.outer(centre, self.class_count)) * inverse[:, None]
        intercept = self.class_count / self.rows
        return centred, cross, intercept


# ----------------------------------------------------------------------
# The sealed anchor head (LU path, M228 convention)
# ----------------------------------------------------------------------

def sealed_lu_head(centred: np.ndarray, cross: np.ndarray,
                   intercept: np.ndarray,
                   penalty: float) -> tuple[np.ndarray, np.ndarray]:
    """W = (centred + penalty I)^-1 cross, b = intercept - the
    sealed LU convention, bit-for-bit."""
    system = centred.copy()
    system.flat[:: system.shape[0] + 1] += penalty
    weights = np.linalg.solve(system, cross)
    return weights, intercept


# ----------------------------------------------------------------------
# The repaired solve (M296d chain)
# ----------------------------------------------------------------------

def symmetric_system(centred: np.ndarray) -> np.ndarray:
    """Upper triangle kept exactly as accumulated; the strict lower
    triangle is the bitwise mirror. G == G.T exactly."""
    return np.triu(centred) + np.triu(centred, k=1).T


def _eigh_solve(system: np.ndarray, cross: np.ndarray,
                penalty: float) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy import linalg as scipy_linalg

    vals, vecs = scipy_linalg.eigh(system, check_finite=False,
                                   driver="evd")
    scale_pen = max(abs(float(vals[0])), abs(float(vals[-1])))
    cutoff_pen = max(0.0, scale_pen * STRONG_CONVEX_CUTOFF)
    keep = vals > cutoff_pen
    inv = np.where(keep, 1.0 / vals, 0.0)
    weights = (vecs * inv[None, :]) @ (vecs.T @ cross)
    return weights, {
        "solve_path": "eigh_fallback",
        "eigh_fallback": {
            "cutoff_strong_convex": float(cutoff_pen),
            "nonpositive_modes_dropped": int((vals <= 0.0).sum()),
            "dropped_components": int((~keep).sum()),
        },
    }


def _svd_solve(system: np.ndarray, cross: np.ndarray
               ) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy import linalg as scipy_linalg

    u, s, vt = scipy_linalg.svd(system, check_finite=False)
    cutoff = float(s.max()) * SVD_SOLVE_CUTOFF
    keep = s > cutoff
    inv_s = np.where(keep, 1.0 / s, 0.0)
    weights = (vt.T * inv_s[None, :]) @ (u.T @ cross)
    return weights, {
        "solve_path": "svd_fallback",
        "svd_fallback": {
            "min_singular_value": float(s.min()),
            "max_singular_value": float(s.max()),
            "cutoff": cutoff,
            "dropped_components": int((~keep).sum()),
        },
    }


def repaired_head(centred: np.ndarray, cross: np.ndarray,
                  intercept: np.ndarray,
                  penalty: float) -> tuple[np.ndarray, dict[str, Any]]:
    """The M296d repaired solve: symmetric-by-construction assembly,
    Cholesky; on refusal eigendecomposition (driver evd) under the
    strong-convexity truncation; SVD last. Returns ([W; b], report)
    with the backward instrument, mirroring the M296 runner
    bit-for-bit."""
    from scipy import linalg as scipy_linalg

    system = symmetric_system(centred)
    width = system.shape[0]
    symmetric_to_bit = bool(np.array_equal(system, system.T))
    system.flat[:: width + 1] += penalty
    report: dict[str, Any] = {"symmetric_to_bit": symmetric_to_bit}
    try:
        cho = scipy_linalg.cho_factor(system, lower=True,
                                      check_finite=False)
        weights = scipy_linalg.cho_solve(cho, cross,
                                         check_finite=False)
        report["solve_path"] = "cholesky"
    except scipy_linalg.LinAlgError:
        try:
            weights, path = _eigh_solve(system, cross, penalty)
        except scipy_linalg.LinAlgError:
            weights, path = _svd_solve(system, cross)
        report.update(path)
    residual = system @ weights - cross
    denom = (float(np.max(np.abs(system))) * float(np.max(np.abs(weights)))
             + float(np.max(np.abs(cross))))
    backward = float(np.max(np.abs(residual))) / max(denom, 1e-300)
    report["backward_error"] = backward
    report["backward_passed"] = backward <= 1e-10
    return np.vstack([weights, intercept[None, :]]), report


# ----------------------------------------------------------------------
# Digests and certificates
# ----------------------------------------------------------------------

def head_digest(weights: np.ndarray, bias: np.ndarray) -> str:
    """SHA-256 over the exact array bytes (shape and dtype first).
    Never includes a wall clock or a hardware signature."""
    w = np.ascontiguousarray(weights, dtype=np.float64)
    b = np.ascontiguousarray(bias, dtype=np.float64)
    header = json.dumps(
        {"shape": list(w.shape), "bias_shape": list(b.shape)},
        sort_keys=True).encode("utf-8")
    return hashlib.sha256(header + w.tobytes() + b.tobytes()).hexdigest()


@dataclass
class ReplayCertificate:
    """The replay record: what was reproduced, under which oracle,
    on which machine, and whether it matches a registered digest."""

    oracle: str
    head_digest: str
    expected_digest: str | None
    bit_exact: bool
    solve_path: str
    hardware: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "oracle": self.oracle,
            "head_digest": self.head_digest,
            "expected_digest": self.expected_digest,
            "bit_exact": self.bit_exact,
            "solve_path": self.solve_path,
            "hardware": self.hardware,
            "detail": self.detail,
        }
