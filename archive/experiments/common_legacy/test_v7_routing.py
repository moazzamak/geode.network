from __future__ import annotations

import unittest

import numpy as np

from experiments.common.v7_routing import fit_routing_profile, route_profiles


class EmpiricalRoutingTests(unittest.TestCase):
    def test_centroid_profiles_route_and_fail_closed(self) -> None:
        rng = np.random.default_rng(4)
        left = rng.normal(-3.0, 0.1, size=(40, 6))
        right = rng.normal(3.0, 0.1, size=(40, 6))
        profiles = (
            fit_routing_profile(
                "centroid_radius",
                left,
                np.zeros(40, dtype=np.int64),
                left,
                model_signature="left",
                representation_hash="a" * 64,
                rank=2,
                prototypes_per_class=4,
                quantile=0.95,
                seed=1,
            ),
            fit_routing_profile(
                "centroid_radius",
                right,
                np.ones(40, dtype=np.int64),
                right,
                model_signature="right",
                representation_hash="a" * 64,
                rank=2,
                prototypes_per_class=4,
                quantile=0.95,
                seed=1,
            ),
        )
        top1, shortlists, fallback = route_profiles(
            profiles,
            np.vstack([left[:1], right[:1], np.zeros((1, 6))]),
            shortlist_size=1,
        )
        self.assertEqual(top1.tolist()[:2], [0, 1])
        self.assertEqual(shortlists[:2], [(0,), (1,)])
        self.assertTrue(fallback[2])
