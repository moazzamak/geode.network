"""Unit tests for the M296 repaired head solver (symmetric Gram +
Cholesky/SVD + condition number).

The M180 sealed path assembles a standardised system whose two
triangles are only ~1e-16 symmetric (each entry computed once, the
lower-left in its own convention). M296 replaces assembly with a
bitwise-symmetric matrix and a Cholesky factor. These tests pin:
bitwise symmetry, agreement with a direct solve on the same symmetric
system, condition-number correctness, the SVD fallback on a system
Cholesky refuses, and decision-level agreement with the sealed LU path
on synthetic data.
"""
from __future__ import annotations

import unittest

import numpy as np

from experiments.tier4.eval_v26_m296_head_repair import (
    CLASSES,
    PENALTY,
    condition_report,
    solve_symmetric,
    symmetric_system,
    _eigh_solve,
    _svd_solve,
)
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator


def _random_centred(width: int, rng: np.random.Generator) -> np.ndarray:
    """A gram-like standardised system: PSD-plus-noise base with each
    off-diagonal entry computed independently (the M180 convention)."""
    base = rng.standard_normal((width, width))
    base = base @ base.T + np.eye(width) * PENALTY
    # emulate per-entry independent computation: add tiny asymmetric noise
    asym = rng.standard_normal((width, width)) * 1e-13
    asym = (asym - asym.T) / 2.0
    return base + asym


class TestSymmetricSystem(unittest.TestCase):

    def test_symmetric_to_bit(self):
        rng = np.random.default_rng(0)
        centred = _random_centred(64, rng)
        system = symmetric_system(centred)
        self.assertTrue(np.array_equal(system, system.T))

    def test_upper_triangle_preserved_diagonal_exact(self):
        rng = np.random.default_rng(1)
        centred = _random_centred(33, rng)
        system = symmetric_system(centred)
        self.assertTrue(np.array_equal(system, np.triu(centred)
                                       + np.triu(centred, k=1).T))
        self.assertTrue(np.array_equal(np.diag(system),
                                       np.diag(centred)))

    def test_lower_is_mirror_of_upper(self):
        rng = np.random.default_rng(2)
        centred = _random_centred(20, rng)
        system = symmetric_system(centred)
        self.assertTrue(np.array_equal(np.tril(system, k=-1),
                                       np.triu(system, k=1).T))


class TestSolveSymmetric(unittest.TestCase):

    def _system(self, width, seed):
        rng = np.random.default_rng(seed)
        centred = _random_centred(width, rng)
        cross = rng.standard_normal((width, CLASSES))
        intercept = rng.standard_normal(CLASSES)
        return centred, cross, intercept

    def test_cholesky_matches_direct_solve(self):
        centred, cross, intercept = self._system(48, 3)
        weights, report = solve_symmetric(centred, cross, intercept,
                                          PENALTY)
        system = symmetric_system(centred)
        system.flat[:: system.shape[0] + 1] += PENALTY
        direct = np.linalg.solve(system, cross)
        rel = float(np.max(np.abs(direct - weights[:-1]))
                    / max(float(np.max(np.abs(direct))), 1e-12))
        self.assertLessEqual(rel, 1e-11)
        self.assertEqual(report["solve_path"], "cholesky")
        self.assertTrue(report["symmetric_to_bit"])
        self.assertTrue(report["backward_passed"], report)
        self.assertLessEqual(
            float(np.max(np.abs(intercept - weights[-1]))), 1e-14)

    def test_condition_report_matches_numpy(self):
        centred, cross, intercept = self._system(36, 4)
        system = symmetric_system(centred)
        system.flat[:: system.shape[0] + 1] += PENALTY
        report = condition_report(system, PENALTY)
        cond_np = float(np.linalg.cond(system))
        self.assertLessEqual(abs(report["condition_number"] - cond_np)
                             / cond_np, 1e-10)
        self.assertLessEqual(report["effective_rank"],
                             system.shape[0])

    def test_eigh_fallback_on_singular_system(self):
        # a rank-deficient centred system with zero penalty: Cholesky
        # refuses, the eigendecomposition path must drop only the
        # mathematically-zero components and return the minimum-norm
        # solution.
        rng = np.random.default_rng(5)
        rows, cols = 8, 20
        x = rng.standard_normal((rows, cols))
        centred = x.T @ x          # singular PSD, no penalty
        cross = rng.standard_normal((cols, CLASSES))
        intercept = rng.standard_normal(CLASSES)
        weights, report = solve_symmetric(centred, cross, intercept,
                                          penalty=0.0)
        self.assertEqual(report["solve_path"], "eigh_fallback")
        expected = np.linalg.pinv(symmetric_system(centred)) @ cross
        rel = float(np.max(np.abs(expected - weights[:-1]))
                    / max(float(np.max(np.abs(expected))), 1e-12))
        self.assertLessEqual(rel, 1e-9)
        dropped = report["eigh_fallback"]["dropped_components"]
        self.assertEqual(dropped, cols - rows)
        self.assertTrue(report["backward_passed"], report)

    def test_svd_last_resort_solves(self):
        # the SVD path is a last resort behind eigh; tested directly.
        # It must drop only mathematically-zero singular values and
        # match the minimum-norm solution.
        rng = np.random.default_rng(8)
        rows, cols = 6, 15
        x = rng.standard_normal((rows, cols))
        system = x.T @ x
        cross = rng.standard_normal((cols, CLASSES))
        weights, path = _svd_solve(system, cross)
        self.assertEqual(path["solve_path"], "svd_fallback")
        expected = np.linalg.pinv(system) @ cross
        rel = float(np.max(np.abs(expected - weights))
                    / max(float(np.max(np.abs(expected))), 1e-12))
        self.assertLessEqual(rel, 1e-9)
        self.assertEqual(path["svd_fallback"]["dropped_components"],
                         cols - rows)

    def test_noise_floor_direction_dropped(self):
        # M296d: a penalised mode at v <= 0 is non-convex (the ridge
        # objective has no minimizer along it - the normal-equation
        # stationary point is a MAXIMUM); the solve must drop it, gate
        # via the truncated-system instrument, and match the explicit
        # truncated solve.
        rng = np.random.default_rng(12)
        d = 60
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        vals_clean = np.linspace(1.0, 100.0, d)
        system_clean = (q * vals_clean) @ q.T
        u = q[:, -1]
        # indefinite (Cholesky refuses): a strongly negative mode
        planted = system_clean - 150.0 * np.outer(u, u)
        cross = rng.standard_normal((d, CLASSES))
        intercept = rng.standard_normal(CLASSES)
        penalty = 1.0
        weights, report = solve_symmetric(planted - penalty * np.eye(d),
                                          cross, intercept, penalty)
        self.assertEqual(report["solve_path"], "eigh_fallback")
        self.assertEqual(report["eigh_fallback"]["dropped_components"], 1)
        self.assertTrue(report["backward_passed"], report)
        vals, vecs = np.linalg.eigh(planted)
        scale_pen = max(abs(float(vals[0])), abs(float(vals[-1])))
        keep = vals > max(0.0, scale_pen * 1e-10)
        inv = np.where(keep, 1.0 / vals, 0.0)
        expected = (vecs * inv[None, :]) @ (vecs.T @ cross)
        rel = float(np.max(np.abs(expected - weights[:-1]))
                    / max(float(np.max(np.abs(expected))), 1e-12))
        self.assertLessEqual(rel, 1e-8)

    def test_near_cancellation_mode_dropped(self):
        # M296d: the measured sealed-system signature - a Gram mode at
        # ~ -0.999 cancels the penalty to a penalised eigenvalue of
        # ~5.7e-4. The mode is below max(0, scale*1e-10) and must be
        # dropped; keeping it amplifies 1/v ~ 1750x (the measured
        # full-pivot catastrophe, 0.1266 vs LU 0.2421).
        rng = np.random.default_rng(13)
        d = 60
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        vals_clean = np.linspace(1.0, 100.0, d)
        system_clean = (q * vals_clean) @ q.T
        v = q[:, 0]
        u = q[:, -1]
        # unpenalised mode at ~ -1.0 -> penalised eigenvalue 1e-12 of
        # the scale (near-cancellation, below the strong-convexity
        # cutoff 1e-10*scale); plus a -50 mode so Cholesky refuses and
        # the eigh route fires
        planted = (system_clean
                   - (1.0 - 1e-12) * np.outer(v, v)
                   - 150.0 * np.outer(u, u))
        cross = rng.standard_normal((d, CLASSES))
        intercept = rng.standard_normal(CLASSES)
        penalty = 1.0
        weights, report = solve_symmetric(planted - penalty * np.eye(d),
                                          cross, intercept, penalty)
        self.assertEqual(report["solve_path"], "eigh_fallback")
        detail = report["eigh_fallback"]
        self.assertEqual(detail["dropped_components"], 2)
        self.assertEqual(detail["nonpositive_modes_dropped"], 1)
        self.assertTrue(report["backward_passed"], report)
        vals, vecs = np.linalg.eigh(planted)
        scale_pen = max(abs(float(vals[0])), abs(float(vals[-1])))
        keep = vals > max(0.0, scale_pen * 1e-10)
        inv = np.where(keep, 1.0 / vals, 0.0)
        expected = (vecs * inv[None, :]) @ (vecs.T @ cross)
        rel = float(np.max(np.abs(expected - weights[:-1]))
                    / max(float(np.max(np.abs(expected))), 1e-12))
        self.assertLessEqual(rel, 1e-8)

    def test_indefinite_system_flags_and_falls_back(self):
        # the sealed standardisation convention can assemble a system
        # whose smallest eigenvalue is negative; Cholesky must refuse,
        # the eigendecomposition path must solve it under M296d (the
        # non-convex negative mode contributes zero), and the report
        # must flag indefiniteness with the standard |lambda| condition
        # number.
        rng = np.random.default_rng(7)
        width = 24
        q, _ = np.linalg.qr(rng.standard_normal((width, width)))
        vals = np.linspace(1.0, 100.0, width)
        vals[0] = -5.0
        centred = (q * vals) @ q.T
        cross = rng.standard_normal((width, CLASSES))
        intercept = rng.standard_normal(CLASSES)
        weights, report = solve_symmetric(centred, cross, intercept,
                                          penalty=0.0)
        self.assertEqual(report["solve_path"], "eigh_fallback")
        cond = report["conditioning"]
        self.assertTrue(cond["indefinite"])
        self.assertAlmostEqual(cond["lambda_min"], -5.0, places=10)
        self.assertAlmostEqual(cond["condition_number"], 20.0, places=10)
        # M296d: the -5 mode is non-convex and dropped
        self.assertEqual(report["eigh_fallback"]["dropped_components"], 1)
        system = symmetric_system(centred)
        vals_c, vecs_c = np.linalg.eigh(system)
        scale_pen = max(abs(float(vals_c[0])), abs(float(vals_c[-1])))
        keep = vals_c > max(0.0, scale_pen * 1e-10)
        inv = np.where(keep, 1.0 / vals_c, 0.0)
        expected = (vecs_c * inv[None, :]) @ (vecs_c.T @ cross)
        rel = float(np.max(np.abs(expected - weights[:-1]))
                    / max(float(np.max(np.abs(expected))), 1e-12))
        self.assertLessEqual(rel, 1e-9)


class TestDecisionLevelAgreement(unittest.TestCase):

    def test_driver_evd_matches_evr(self):
        # M296b: the registered numerics-policy pin. On a clustered
        # indefinite synthetic system the two drivers must agree within
        # the registered tolerance, so the pin changes speed, not math.
        from scipy import linalg as scipy_linalg
        rng = np.random.default_rng(10)
        d = 120
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        # heavy clustering: many eigenvalues in [0, 1], one negative
        vals = np.concatenate([np.linspace(0.0, 1.0, d - 1), [-30.0]])
        system = (q * vals) @ q.T
        ev, _ = scipy_linalg.eigh(system, driver="ev")
        evd, _ = scipy_linalg.eigh(system, driver="evd")
        evr, _ = scipy_linalg.eigh(system, driver="evr")
        scale = max(abs(float(ev[-1])), 1.0)
        self.assertLessEqual(float(np.max(np.abs(evd - ev))) / scale,
                             1e-8)
        self.assertLessEqual(float(np.max(np.abs(evr - ev))) / scale,
                             1e-8)

    def test_repaired_matches_sealed_lu_path(self):
        # strong class signal so decision-level agreement is not
        # dominated by chance; the LU path sees the ~1e-16-asymmetric
        # system, the repaired path sees its bitwise mirror.
        rng = np.random.default_rng(6)
        n_rows, cols = 4000, 128
        features = (rng.standard_normal((n_rows, cols))
                    * 40 + 120).astype(np.float32)
        labels = np.clip((features[:, 0] / 18 + features[:, 1] / 9)
                         .astype(np.int64), 0, CLASSES - 1)
        acc = RidgeAccumulator(cols, CLASSES)
        for start in range(0, n_rows, 512):
            stop = min(start + 512, n_rows)
            acc.add(features[start:stop], labels[start:stop])
        standardiser = acc.standardiser()
        centred, cross, intercept = acc._standardised_system()

        lu_system = centred.copy()
        lu_system.flat[:: cols + 1] += PENALTY
        lu_weights = np.vstack([np.linalg.solve(lu_system, cross),
                                intercept[None, :]])
        repaired, report = solve_symmetric(centred, cross, intercept,
                                           PENALTY)
        self.assertEqual(report["solve_path"], "cholesky")

        xs = standardiser(features).astype(np.float64)
        lu_pred = np.argmax(xs @ lu_weights[:-1] + lu_weights[-1], axis=1)
        rep_pred = np.argmax(xs @ repaired[:-1] + repaired[-1], axis=1)
        agree = float((lu_pred == rep_pred).mean())
        self.assertGreaterEqual(agree, 0.999)
        acc_lu = float((lu_pred == labels).mean())
        acc_rep = float((rep_pred == labels).mean())
        self.assertLessEqual(abs(acc_lu - acc_rep), 0.001)


if __name__ == "__main__":
    unittest.main()
