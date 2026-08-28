"""M301-module and M320 unit tests."""
from __future__ import annotations

import unittest

import numpy as np

from geode.core.alignment import (
    AlignmentError,
    cca_align,
    cca_from_moments,
    orthogonal_procrustes,
)
from geode.core.feature_bus import (
    FeatureArtifact,
    FeatureBus,
    FeatureVersion,
    FeatureVersionError,
)


class TestProcrustes(unittest.TestCase):
    def test_exact_rotation_recovers_the_map(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(200, 8))
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        b = a @ q
        artifact = orthogonal_procrustes(a, b)
        # the objective drops to (numerically) zero and the map is
        # orthogonal
        self.assertLess(artifact.report["objective_after"], 1e-8)
        self.assertLess(artifact.report["orthogonality_residual"],
                        1e-10)
        self.assertTrue(artifact.report["objective_improves"])

    def test_orthogonality_is_checked_not_assumed(self):
        # a hand-built non-orthogonal "map" must fail the gate by
        # construction of the artifact class - here we check the
        # residual instrument directly on a degenerate input
        a = np.zeros((3, 3))
        b = np.zeros((3, 3))
        # zero cross-covariance: svd returns the identity, orthogonal
        artifact = orthogonal_procrustes(a, b)
        self.assertLess(artifact.report["orthogonality_residual"],
                        1e-10)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(AlignmentError):
            orthogonal_procrustes(np.zeros((4, 3)), np.zeros((4, 2)))

    def test_artifact_digest_is_stable(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(50, 6))
        b = rng.normal(size=(50, 6))
        artifact = orthogonal_procrustes(a, b)
        self.assertEqual(artifact.digest(), artifact.digest())
        self.assertEqual(len(artifact.digest()), 64)


class TestCca(unittest.TestCase):
    def test_highly_correlated_spaces_yield_high_canonical_corr(self):
        rng = np.random.default_rng(2)
        z = rng.normal(size=(500, 4))
        a = np.hstack([z, rng.normal(scale=0.2, size=(500, 2))])
        b = np.hstack([z[:, :1] * 2 + rng.normal(scale=0.2,
                                                 size=(500, 1)),
                       rng.normal(size=(500, 3))])
        artifact = cca_align(a, b, components=3)
        self.assertEqual(artifact.report["components"], 3)
        self.assertTrue(artifact.report["all_nonnegative"])
        self.assertGreater(artifact.report["canonical_correlations"][0],
                           0.9)

    def test_projected_spaces_are_decorrelated(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(300, 5))
        b = rng.normal(size=(300, 5))
        artifact = cca_align(a, b, components=4)
        self.assertTrue(artifact.report["decorrelated"])

    def test_cca_from_moments_matches_the_direct_construction(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(200, 6))
        b = rng.normal(size=(200, 5))
        n = len(a)
        centre_a = a - a.mean(axis=0, keepdims=True)
        centre_b = b - b.mean(axis=0, keepdims=True)
        cov_a = centre_a.T @ centre_a
        cov_b = centre_b.T @ centre_b
        cross = centre_a.T @ centre_b
        from_moments = cca_from_moments(cov_a, cov_b, cross,
                                        components=4, ridge=1e-8)
        direct = cca_align(a, b, components=4, ridge=1e-8)
        self.assertEqual(from_moments.report["components"], 4)
        self.assertAlmostEqual(
            from_moments.report["canonical_correlations"][0],
            direct.report["canonical_correlations"][0], places=10)


class TestFeatureBus(unittest.TestCase):
    def _bus(self) -> FeatureBus:
        bus = FeatureBus()
        bus.register(FeatureArtifact(
            version=FeatureVersion(encoder="dino-s", extraction="v3",
                                   preprocessing="l2"),
            digest="d1", path="f:/cache/v3.npy"))
        return bus

    def test_registered_version_resolves(self):
        bus = self._bus()
        artifact = bus.resolve(FeatureVersion(
            encoder="dino-s", extraction="v3", preprocessing="l2"))
        self.assertEqual(artifact.digest, "d1")

    def test_unregistered_version_refused(self):
        bus = self._bus()
        with self.assertRaises(FeatureVersionError):
            bus.resolve(FeatureVersion(encoder="dino-s",
                                       extraction="v4",
                                       preprocessing="l2"))

    def test_mismatched_preprocessing_refused(self):
        bus = self._bus()
        with self.assertRaises(FeatureVersionError):
            bus.resolve(FeatureVersion(encoder="dino-s",
                                       extraction="v3",
                                       preprocessing="none"))

    def test_digest_mutation_refused(self):
        bus = self._bus()
        with self.assertRaises(FeatureVersionError):
            bus.register(FeatureArtifact(
                version=FeatureVersion(encoder="dino-s",
                                       extraction="v3",
                                       preprocessing="l2"),
                digest="d2", path="f:/cache/v3_other.npy"))

    def test_registered_versions_listed(self):
        bus = self._bus()
        self.assertEqual(bus.registered_versions(),
                         [("dino-s", "v3", "l2")])


if __name__ == "__main__":
    unittest.main()
