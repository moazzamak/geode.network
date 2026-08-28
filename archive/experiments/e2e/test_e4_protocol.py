import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.candidate_routing import exact_bound_routing
from src.sdf_engine import EllipsoidExpert, Expert
from src.shadow_routing import run_shadow_router
from experiments.e2e.e4_cifar_protocol import (
    E4Data,
    build_ood_partitions,
    load_config,
)
from experiments.e2e.run_e6_transfer_qualification import _load_config as load_e6_config
from experiments.e2e.e5_bundle_loader import ExplicitTransform


class E4ProtocolTests(unittest.TestCase):
    def test_config_requires_five_unique_confirmatory_seeds(self):
        config_path = Path("experiments/configs/e4_cifar_qualification.json")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload["seeds"] = [11, 23, 37, 53]
        with tempfile.TemporaryDirectory() as directory:
            invalid_path = Path(directory) / "config.json"
            invalid_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly five"):
                load_config(invalid_path)

    def test_near_and_far_ood_partitions_are_disjoint(self):
        data = E4Data(
            id_features=np.zeros((2, 2)),
            id_labels=np.array([0, 1]),
            id_source_indices=np.array([0, 1]),
            near_features=np.zeros((10, 2)),
            near_labels=np.arange(10),
            near_source_indices=np.arange(10),
            far_features=np.zeros((12, 2)),
            far_source_indices=np.arange(12),
            fingerprints={},
        )
        config = {
            "near_ood": {
                "official_train_count": 6,
                "feature_sample_seed": 42,
                "development_validation_fraction": 0.5,
            },
            "far_ood": {
                "split_seed": 42,
                "validation_count": 3,
                "risk_control_count": 3,
            },
        }

        partitions, audit = build_ood_partitions(data, config)

        for family in ("near", "far"):
            values = list(partitions[family].values())
            self.assertTrue(audit[family]["pairwise_disjoint"])
            for left in range(len(values)):
                for right in range(left + 1, len(values)):
                    self.assertEqual(
                        np.intersect1d(values[left], values[right]).size, 0,
                    )

    def test_shadow_router_never_controls_authoritative_outputs(self):
        models = {}
        for class_id, center in enumerate((0.0, 4.0)):
            expert = Expert(alpha=2.0)
            expert.add_ellipsoid(EllipsoidExpert(
                center=np.array([center, 0.0]), radii=np.ones(2),
            ))
            models[class_id] = [expert]
        points = np.array([[0.0, 0.0], [4.0, 0.0]])

        observation = run_shadow_router(
            models, points, exact_bound_routing,
            router_name="exact_bound", timing_repeats=1,
            score_scales={0: 0.5, 1: 2.0},
        )

        np.testing.assert_array_equal(observation.authoritative_predictions, [0, 1])
        self.assertFalse(observation.quality_gate_passed)
        self.assertFalse(observation.candidate_controls_outputs)
        self.assertEqual(observation.oracle_counters.exact_class_sdf_pairs, 4)

    def test_e6_config_requires_immutable_source_branch(self):
        config_path = Path("experiments/configs/e6_transfer_qualification.json")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload["source_forgetting_budget"] = 0.01
        with tempfile.TemporaryDirectory() as directory:
            invalid_path = Path(directory) / "config.json"
            invalid_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be zero"):
                load_e6_config(invalid_path)

    def test_e6_external_lane_cannot_alias_controlled_checkpoint(self):
        config_path = Path("experiments/configs/e6_transfer_qualification.json")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload["pretraining_lanes"]["external"]["status"] = "qualified"
        with tempfile.TemporaryDirectory() as directory:
            invalid_path = Path(directory) / "config.json"
            invalid_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "separately frozen checkpoint"):
                load_e6_config(invalid_path)

    def test_explicit_transform_uses_fitted_lda_output_width(self):
        transform = ExplicitTransform(
            pca_components=np.eye(2),
            pca_mean=np.zeros(2),
            pca_explained_variance=np.ones(2),
            lda_scalings=np.eye(2),
            lda_xbar=np.zeros(2),
            scaler_mean=np.zeros(1),
            scaler_scale=np.ones(1),
        )
        transformed = transform.transform(np.array([[2.0, 3.0]]))
        np.testing.assert_array_equal(transformed, [[2.0]])


if __name__ == "__main__":
    unittest.main()